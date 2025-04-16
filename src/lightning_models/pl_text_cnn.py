import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lightning.pytorch as pl
from typing import Dict, List, Union

from .pl_model_plots import compute_metrics


class TextCNN(pl.LightningModule):
    def __init__(
        self,
        config: Dict[str, Union[int, float, str, bool, List[str]]],
        vocab_size: int,
        word_embeddings: torch.FloatTensor
    ):
        super().__init__()
        self.config = config
        self.model_class = "cnn"

        # === Core configuration ===
        self.id_name = config.get("id_name", None)
        self.label_name = config["label_name"]
        # Use configurable key names for text input
        self.text_input_ids_key = config.get("text_input_ids_key", "input_ids")
        self.text_attention_mask_key = config.get("text_attention_mask_key", "attention_mask")
        self.text_name = config["text_name"] + "_processed_" + self.text_input_ids_key
        self.text_attention_mask = config["text_name"] + "_processed_" + self.text_attention_mask_key
        
        
        self.embed_size = config.get("embed_size", word_embeddings.shape[1])
        self.num_classes = config.get("num_classes", 2)
        self.is_binary = config.get("is_binary", True)
        self.task = "binary" if self.is_binary else "multiclass"
        self.metric_choices = config.get("metric_choices", ["accuracy", "f1_score"])
        self.dropout_keep = config.get("dropout_keep", 0.5)
        self.max_sen_len = config.get("max_sen_len", 512)
        self.kernel_size = config.get("kernel_size", [3, 5, 7])
        self.num_layers = config.get("num_layers", 2)
        self.num_channels = config.get("num_channels", [100, 100])
        self.hidden_common_dim = config.get("hidden_common_dim", 100)

        if vocab_size != word_embeddings.shape[0]:
            raise ValueError("Vocab size does not match embedding matrix")

        self.embeddings = nn.Embedding(vocab_size, self.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=config.get("is_embeddings_trainable", True))

        self.conv_output_dims = {
            k: self._compute_conv_output_dim(k, self.max_sen_len, self.num_layers)
            for k in self.kernel_size
        }

        self.conv_input_dims = {
            k: self._compute_conv_input_dim(self.embed_size, self.num_channels, self.num_layers)
            for k in self.kernel_size
        }

        self.convs = nn.ModuleList([
            self._build_conv_layers(k, self.num_layers, self.num_channels, self.conv_input_dims[k], self.conv_output_dims[k])
            for k in self.kernel_size
        ])

        self.output_text_dim = self.hidden_common_dim
        self.network = self._build_text_subnetwork(len(self.kernel_size), self.num_channels, self.output_text_dim)

        class_weights = torch.tensor(config.get("class_weights", [1.0] * self.num_classes))
        self.loss_op = nn.CrossEntropyLoss(weight=class_weights)

        self.save_hyperparameters()

    def _compute_conv_output_dim(self, kernel_size, input_dim, num_layers):
        for _ in range(num_layers):
            input_dim = input_dim - kernel_size + 1
            if _ < num_layers - 1:
                input_dim = (input_dim - kernel_size) // kernel_size + 1
        return input_dim

    def _compute_conv_input_dim(self, embed_size, num_channels, num_layers):
        return [embed_size] + num_channels[:-1]

    def _build_conv_layers(self, kernel_size, num_layers, num_channels, input_dims, output_dim):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv1d(input_dims[i], num_channels[i], kernel_size))
            layers.append(nn.ReLU())
            if i < num_layers - 1:
                layers.append(nn.MaxPool1d(kernel_size))
            else:
                layers.append(nn.MaxPool1d(output_dim))
        return nn.Sequential(*layers)

    def _build_text_subnetwork(self, num_kernels, num_channels, output_text_dim):
        return nn.Sequential(
            nn.Dropout(self.dropout_keep),
            nn.Linear(num_channels[-1] * num_kernels, output_text_dim)
        )

    def configure_optimizers(self):
        optimizer_type = self.config.get("optimizer_type", "SGD")
        lr = self.config.get("lr", 0.02)
        momentum = self.config.get("momentum", 0.9)

        if optimizer_type == "Adam":
            return optim.Adam(self.parameters(), lr=lr)
        return optim.SGD(self.parameters(), lr=lr, momentum=momentum)

    def forward(self, text_ids):
        x = self.embeddings(text_ids).permute(0, 2, 1)
        conv_outs = [conv(x).squeeze(2) for conv in self.convs]
        features = torch.cat(conv_outs, dim=1)
        return self.network(features)

    def run_epoch(self, batch, stage):
        text_ids = batch[self.text_name]
        labels = batch[self.label_name] if stage != 'pred' else None
        logits = self(text_ids)

        loss = self.loss_op(logits, labels) if labels is not None else None
        preds = torch.softmax(logits, dim=1)
        preds = preds[:, 1] if self.is_binary else preds

        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, 'train')
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        metrics = compute_metrics(preds, labels, self.metric_choices, self.task, self.num_classes, "train")
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return {"loss": loss, **metrics}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, 'val')
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        metrics = compute_metrics(preds, labels, self.metric_choices, self.task, self.num_classes, "val")
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.run_epoch(batch, 'test')
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        metrics = compute_metrics(preds, labels, self.metric_choices, self.task, self.num_classes, "test")
        self.log_dict(metrics, prog_bar=True, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        try:
            _ = batch[self.label_name]
            mode = "test"
        except KeyError:
            mode = "pred"
        _, preds, labels = self.run_epoch(batch, mode)
        return preds if mode == 'pred' else (preds, labels)