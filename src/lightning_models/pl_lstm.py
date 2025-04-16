# Save this as: bsm/lightning_models/pl_lstm.py
import os
from datetime import datetime
from typing import Union, List, Dict

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import lightning.pytorch as pl

from .pl_model_plots import compute_metrics
from .dist_utils import all_gather


class TextLSTM(pl.LightningModule):
    def __init__(self, config: Dict[str, Union[int, float, str, bool, List[str]]], vocab_size: int, word_embeddings: torch.FloatTensor):
        super().__init__()
        self.config = config
        self.model_class = 'lstm'

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        # Use configurable key names for text input
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_attention_mask_key = config.get("text_attention_mask_key", "attention_mask")
        self.text_name = config["text_name"] + "_processed_" + self.text_input_ids_key
        self.text_attention_mask = config["text_name"] + "_processed_" + self.text_attention_mask_key

        # Class info
        self.is_binary = config.get("is_binary", True)
        self.task = 'binary' if self.is_binary else 'multiclass'
        self.num_classes = 2 if self.is_binary else config.get("num_classes", 2)

        # Model parameters
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])
        self.hidden_dimension = config.get("hidden_common_dim", 100)
        self.num_layers = config.get("num_layers", 1)
        self.dropout_keep = config.get("dropout_keep", 0.5)
        self.max_sen_len = config.get("max_sen_len", 512)

        # Optimizer params
        self.lr = config.get("lr", 2e-5)
        self.weight_decay = config.get("weight_decay", 0.0)
        self.adam_epsilon = config.get("adam_epsilon", 1e-8)
        self.momentum = config.get("momentum", 0.9)
        self.run_scheduler = config.get("run_scheduler", True)

        # Embedding
        self.embed_size = config.get("embed_size", word_embeddings.shape[1])
        if vocab_size != word_embeddings.shape[0]:
            raise ValueError("Mismatch in vocab size and embedding shape")
        self.embeddings = nn.Embedding(vocab_size, self.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=config.get("is_embeddings_trainable", True))

        # LSTM + Linear
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_dimension,
            num_layers=self.num_layers,
            dropout=self.dropout_keep,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(2 * self.hidden_dimension, self.num_classes)

        # Loss
        class_weights = config.get("class_weights")
        if class_weights and len(class_weights) == self.num_classes:
            self.loss_op = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        else:
            self.loss_op = nn.CrossEntropyLoss()

        # Misc
        self.model_path = config.get("model_path", "./")
        self.test_output_folder = None
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt_type = self.config.get("optimizer", "SGD")
        if opt_type == 'Adam':
            return optim.AdamW(self.parameters(), lr=self.lr, eps=self.adam_epsilon, weight_decay=self.weight_decay)
        return optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        lstm_out, _ = self.lstm(x)
        out_fwd = lstm_out[range(len(x)), self.max_sen_len - 1, :self.hidden_dimension]
        out_rev = lstm_out[:, 0, self.hidden_dimension:]
        out_combined = torch.cat((out_fwd, out_rev), dim=1)
        return self.fc(out_combined)

    def run_epoch(self, batch, stage):
        text_ids = batch[self.text_name]
        labels = batch[self.label_name] if stage != 'pred' else None
        logits = self(text_ids)
        loss = self.loss_op(logits, labels) if stage != 'pred' else None
        probs = torch.softmax(logits, dim=1)
        preds = probs[:, 1] if self.is_binary else probs
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.run_epoch(batch, 'train')
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, 'val')
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return {"preds": preds.detach().cpu(), "labels": labels.detach().cpu()}

    def on_validation_epoch_end(self):
        outputs = self.trainer.callback_metrics
        preds = torch.cat([o["preds"] for o in self.trainer.logged_metrics.values() if "preds" in o], dim=0)
        labels = torch.cat([o["labels"] for o in self.trainer.logged_metrics.values() if "labels" in o], dim=0)
        metrics = compute_metrics(preds, labels, self.metric_choices, self.task, self.num_classes, stage="val")
        self.log_dict(metrics, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, 'test')
        self.log("test_loss", loss, sync_dist=True, prog_bar=True)
        output = {"preds": preds.detach().cpu()}
        if labels is not None:
            output["labels"] = labels.detach().cpu()
        if self.id_name:
            output["ids"] = batch[self.id_name]
        return output

    def on_test_epoch_end(self):
        preds, labels, ids = [], [], []
        for output in self.trainer.logged_metrics.values():
            if isinstance(output, dict):
                preds.extend(output.get("preds", []))
                labels.extend(output.get("labels", []))
                ids.extend(output.get("ids", []))
        df = pd.DataFrame({"prob": preds})
        if labels:
            df["label"] = labels
        if ids:
            df[self.id_name] = ids
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        folder = os.path.join(self.model_path, f"lstm-{timestamp}")
        os.makedirs(folder, exist_ok=True)
        df.to_csv(os.path.join(folder, "test_result.tsv"), sep='\t', index=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        _, preds, labels = self.run_epoch(batch, 'pred')
        return preds