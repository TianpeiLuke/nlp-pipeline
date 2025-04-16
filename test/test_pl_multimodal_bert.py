# bsm/test/test_pl_multimodal_bert.py

import unittest
import torch
from lightning.pytorch import seed_everything

from ..lightning_models.pl_multimodal_bert import MultimodalBert


class TestMultimodalBert(unittest.TestCase):

    def setUp(self):
        seed_everything(42)
        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "tab_field_list": ["feature1", "feature2"],
            "hidden_common_dim": 16,
            "is_binary": True,
            "num_classes": 2,
            "model_path": "./checkpoints",
            "tokenizer": "bert-base-uncased",
            "lr": 1e-4,
        }

        # Create dummy batch
        self.batch = {
            "dialogue_processed_input_ids": torch.randint(0, 1000, (4, 2, 16)),  # [B, C, T]
            "dialogue_processed_attention_mask": torch.ones(4, 2, 16, dtype=torch.long),
            "feature1": torch.rand(4),
            "feature2": torch.rand(4),
            "label": torch.tensor([0, 1, 0, 1]),
        }

        self.model = MultimodalBert(self.config)
        self.model.eval()

    def test_model_initialization(self):
        self.assertEqual(self.model.task, "binary")
        self.assertEqual(self.model.num_classes, 2)
        self.assertTrue(hasattr(self.model, "text_subnetwork"))
        self.assertTrue(hasattr(self.model, "tab_subnetwork"))

    def test_forward_pass(self):
        with torch.no_grad():
            output = self.model(self.batch)
        self.assertEqual(output.shape, (4, 2))  # [B, num_classes]

    def test_training_step(self):
        self.model.train()
        out = self.model.training_step(self.batch, batch_idx=0)
        self.assertIn("loss", out)
        self.assertIsInstance(out["loss"], torch.Tensor)
        self.assertGreaterEqual(out["loss"].item(), 0)

    def test_multiclass_support(self):
        config_mc = self.config.copy()
        config_mc["is_binary"] = False
        config_mc["num_classes"] = 3
        model_mc = MultimodalBert(config_mc)

        batch_mc = self.batch.copy()
        batch_mc["label"] = torch.tensor([0, 2, 1, 1])
        with torch.no_grad():
            output = model_mc(batch_mc)
        self.assertEqual(output.shape, (4, 3))


if __name__ == '__main__':
    unittest.main()