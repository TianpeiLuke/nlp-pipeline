# -*- coding: utf-8 -*-
import unittest
from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl


from src.lightning_models.pl_train import model_train  # Replace with actual import
from src.lightning_models.pl_multimodal_bert import MultimodalBert



class DummyModel(MultimodalBert):
    def __init__(self, config):
        super().__init__(config)

    def training_step(self, batch, batch_idx):
        return {"loss": torch.tensor(0.5, requires_grad=True)}

    def validation_step(self, batch, batch_idx):
        return {"val_loss": torch.tensor(0.4)}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class TestModelTrain(unittest.TestCase):

    def setUp(self):
        # Dummy dataset: features, labels
        x = torch.randn(16, 10)
        y = torch.randint(0, 2, (16,))
        dataset = TensorDataset(x, y)

        self.train_dataloader = DataLoader(dataset, batch_size=4)
        self.val_dataloader = DataLoader(dataset, batch_size=4)

        self.config = {
            "label_name": "label",
            "text_name": "text",
            "model_class": "multimodal_bert",
            "fp16": False,
            "gradient_clip_val": 0.1,
            "max_epochs": 1,
            "early_stop_patience": 1,
            "val_check_interval": 1.0
        }

    @patch("lightning_models.pl_multimodal_bert.MultimodalBert", DummyModel)  # Patch with dummy model
    def test_model_train_runs(self):
        model = DummyModel(self.config)

        trainer = model_train(
            model=model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            device="cpu",
            early_stop_metric="val_loss"
        )

        self.assertIsInstance(trainer, pl.Trainer)
        self.assertTrue(hasattr(trainer, "fit_loop"))


if __name__ == "__main__":
    unittest.main()
