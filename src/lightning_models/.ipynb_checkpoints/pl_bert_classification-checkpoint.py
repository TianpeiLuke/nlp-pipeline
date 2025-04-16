#!/usr/bin/env python3
import os  # Added os
import pandas as pd  # Added pandas
from datetime import datetime  # Added datetime
from typing import Dict, Union, List, Optional

import torch
import torch.nn as nn
import lightning.pytorch as pl
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,  # Added AutoTokenizer
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from torch.optim import AdamW  # Added AdamW
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # Added EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint  # Added ModelCheckpoint
from lightning.pytorch.utilities import rank_zero_only  # Corrected import

from .dist_utils import all_gather  # Added all_gather
from .pl_model_plots import compute_metrics  # Added compute_metrics


class TextBertClassificationConfig(BaseModel):
    text_name: str
    label_name: str
    tokenizer: str = "bert-base-cased"
    is_binary: bool = True
    num_classes: int = 2
    metric_choices: List[str] = Field(default_factory=lambda: ["accuracy", "f1_score"])
    weight_decay: float = 0.0
    warmup_steps: int = 0
    adam_epsilon: float = 1e-8
    lr: float = 2e-5
    run_scheduler: bool = True
    reinit_pooler: bool = False
    reinit_layers: int = 0
    model_path: str
    id_name: Optional[str] = None
    text_input_ids_key: str = "input_ids"
    text_attention_mask_key: str = "attention_mask"

    @field_validator("num_classes")  # Changed to field_validator
    @classmethod
    def validate_num_classes(cls, value, values):
        if values.get("is_binary") and value != 2:
            raise ValueError("For binary classification, num_classes must be 2")
        if not values.get("is_binary") and value < 2:
            raise ValueError("For multiclass classification, num_classes must be >= 2")
        return value


class TextBertClassification(pl.LightningModule):
    def __init__(self, config: Union[Dict, TextBertClassificationConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = TextBertClassificationConfig(**config)
        self.config = config

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            self.config.tokenizer,
            num_labels=self.config.num_classes,
            output_attentions=False,
            return_dict=True,
        )
        self._maybe_reinitialize()

        self.loss_op = None
        self.pred_lst, self.label_lst, self.id_lst = [], [], []
        self.test_output_folder, self.test_has_label = None, False
        self.save_hyperparameters()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch[self.config.text_name]
        attention_mask = batch[self.config.text_attention_mask]
        labels = batch.get(self.config.label_name)

        return self.bert(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def _maybe_reinitialize(self):
        if not self.config.reinit_pooler:
            return
        encoder = getattr(self.bert, "bert")
        encoder.pooler.dense.weight.data.normal_(
            mean=0.0, std=encoder.config.initializer_range
        )
        encoder.pooler.dense.bias.data.zero_()
        for p in encoder.pooler.parameters():
            p.requires_grad = True

        if self.config.reinit_layers > 0:
            for layer in encoder.encoder.layer[-self.config.reinit_layers :]:
                for module in layer.modules():
                    if isinstance(module, (nn.Linear, nn.Embedding)):
                        module.weight.data.normal_(
                            mean=0.0, std=encoder.config.initializer_range
                        )
                    elif isinstance(module, nn.LayerNorm):
                        module.bias.data.zero_()
                        module.weight.data.fill_(1.0)
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.data.zero_()

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_params = [
            {
                "params": [
                    p
                    for n, p in self.bert.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.bert.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            grouped_params, lr=self.config.lr, eps=self.config.adam_epsilon
        )
        scheduler = (
            get_linear_schedule_with_warmup
            if self.config.run_scheduler
            else get_constant_schedule_with_warmup
        )(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
            ModelCheckpoint(
                monitor="val_loss", mode="min", save_top_k=1, save_weights_only=True
            ),
        ]

    def add_loss_op(self, loss_op=None):
        if loss_op is None:
            if self.config.is_binary:
                class_weight = torch.tensor(
                    [1.0, self.config.get("pos_weight", 1.0)]
                )
                self.loss_op = nn.CrossEntropyLoss(
                    weight=torch.tensor(class_weight).to(self.device)
                )
            else:
                self.loss_op = nn.CrossEntropyLoss()
        else:
            self.loss_op = loss_op

    def run_epoch(self, batch, stage):
        input_ids = batch[self.config.text_name]
        attention_masks = batch[self.config.text_attention_mask]
        labels = batch.get(self.config.label_name) if stage != "pred" else None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, device=self.device)

        outputs = self(
            input_ids=input_ids, attention_mask=attention_masks, labels=labels
        )
        logits = outputs.logits
        loss = outputs.loss if labels is not None else None

        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.config.is_binary else preds
        return loss, preds, labels

    def _shared_step(self, batch, stage):
        loss, preds, labels = self.run_epoch(batch, stage)
        if stage != "pred":
            preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
            self.pred_lst.extend(preds.tolist())
            self.label_lst.extend(labels.tolist())
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "train")
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, "val")
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self.pred_lst.clear()
        self.label_lst.clear()

    def on_validation_epoch_end(self):
        self._log_epoch_metrics(stage="val")

    def test_step(self, batch, batch_idx):
        if self.config.label_name in batch:
            self.test_has_label = True
        loss, preds, labels = self._shared_step(
            batch, "test" if self.test_has_label else "pred"
        )
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        if self.config.id_name and self.config.id_name in batch:
            self.id_lst.extend(batch[self.config.id_name])
        return loss

    def on_test_epoch_start(self):
        self.pred_lst.clear()
        self.label_lst.clear()
        self.id_lst.clear()
        self.test_output_folder = os.path.join(
            self.config.model_path,
            f"{self.config.model_class}-{datetime.now():%Y-%m-%d-%H-%M-%S}",
        )

    @rank_zero_only
    def save_predictions_to_file(
        self, output_folder, id_name, pred_list, label_list=None
    ):
        os.makedirs(output_folder, exist_ok=True)
        df = pd.DataFrame({"prob": pred_list})
        if id_name:
            final_ids = sum(all_gather(self.id_lst), [])
            df[id_name] = final_ids
        if label_list is not None:
            df["label"] = label_list

        path = os.path.join(output_folder, "test_result.tsv")
        df.to_csv(path, sep="\t", index=False)
        print(f"Saved test results to {path}")

    def on_test_epoch_end(self):
        final_pred_lst = sum(all_gather(self.pred_lst), [])
        final_label_lst = (
            sum(all_gather(self.label_lst), []) if self.test_has_label else None
        )

        if self.test_has_label:
            preds_tensor = torch.tensor(final_pred_lst)
            labels_tensor = torch.tensor(final_label_lst)
            metrics = compute_metrics(
                preds_tensor,
                labels_tensor,
                self.config.metric_choices,
                self.config.task,
                self.config.num_classes,
                "test",
            )
            self.log_dict(metrics, sync_dist=True, prog_bar=True)

        self.save_predictions_to_file(
            self.test_output_folder,
            self.config.id_name,
            final_pred_lst,
            final_label_lst if self.test_has_label else None,
        )

    def _gather_and_flatten(self, data):
        gathered = all_gather(data)
        return [x for sublist in gathered for x in sublist]

    def _log_epoch_metrics(self, stage="val"):
        preds = self._gather_and_flatten(self.pred_lst)
        labels = self._gather_and_flatten(self.label_lst)
        metrics = compute_metrics(
            torch.tensor(preds).to(self.device),
            torch.tensor(labels).to(self.device),
            self.config.metric_choices,
            self.config.task,
            self.config.num_classes,
            stage,
        )
        self.log_dict(metrics, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            labels = batch[self.config.label_name]
            _, preds, labels = self.run_epoch(batch, "test")
        except KeyError:
            _, preds, labels = self.run_epoch(batch, "pred")

        return preds if labels is None else (preds, labels)

    def export_to_onnx(self, save_path, opset_version=11):
        dummy_input = {
            "input_ids": torch.randint(0, 100, (1, 128)),
            "attention_mask": torch.ones(1, 128).long(),
        }
        torch.onnx.export(
            self.bert,
            (dummy_input,),
            save_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
            opset_version=opset_version,
        )

    def export_to_torchscript(self, save_path):
        self.bert.eval()
        example_inputs = {
            "input_ids": torch.randint(0, 100, (1, 128)),
            "attention_mask": torch.ones(1, 128).long(),
        }
        traced_script_module = torch.jit.trace(self.bert, (example_inputs,))
        traced_script_module.save(save_path)