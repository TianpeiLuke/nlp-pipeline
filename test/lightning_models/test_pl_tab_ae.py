import unittest
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything

# Updated import path for your project structure
from src.lightning_models.pl_tab_ae import TabAE


class TestTabAE(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.config_dict = {
            "tab_field_list": ["f1", "f2", "f3"],
            "hidden_common_dim": 16
        }
        self.model = TabAE(self.config_dict)
        self.model.eval()

        self.batch_size = 4
        self.batch_dict = {
            "f1": torch.randn(self.batch_size),
            "f2": torch.randn(self.batch_size),
            "f3": torch.randn(self.batch_size)
        }
        self.tensor_input = torch.randn(self.batch_size, len(self.config_dict["tab_field_list"]))

    def test_forward_with_dict(self):
        out = self.model(self.batch_dict)
        self.assertEqual(out.shape, (self.batch_size, self.config_dict["hidden_common_dim"]))

    def test_forward_with_tensor(self):
        out = self.model(self.tensor_input)
        self.assertEqual(out.shape, (self.batch_size, self.config_dict["hidden_common_dim"]))

    def test_combine_tab_data_shape(self):
        combined = self.model.combine_tab_data(self.batch_dict)
        self.assertEqual(combined.shape, (self.batch_size, len(self.config_dict["tab_field_list"])))

    def test_missing_field_raises_keyerror(self):
        bad_batch = {
            "f1": torch.randn(self.batch_size),
            "f2": torch.randn(self.batch_size)
            # missing "f3"
        }
        with self.assertRaises(KeyError):
            self.model.combine_tab_data(bad_batch)

    def test_invalid_type_raises_typeerror(self):
        bad_batch = {
            "f1": "invalid type",
            "f2": torch.randn(self.batch_size),
            "f3": torch.randn(self.batch_size)
        }
        with self.assertRaises(TypeError):
            self.model.combine_tab_data(bad_batch)


if __name__ == '__main__':
    unittest.main()
