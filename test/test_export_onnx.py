import os
import unittest
import tempfile
import torch
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader
from typing import Dict, List, Union

from src.lightning_models.pl_multimodal_bert import MultimodalBert  # Adjust this import to your actual module
from src.lightning_models.pl_train import load_onnx_model, model_online_inference  # Also adjust if needed

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, sample_batch: Dict[str, Union[torch.Tensor, List]]):
        self.data = sample_batch
        self.length = sample_batch[list(sample_batch.keys())[0]].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] if isinstance(v, torch.Tensor) else v[idx] for k, v in self.data.items()}

class TestExportToONNX(unittest.TestCase):

    def test_onnx_export_and_inference(self):
        # Setup
        sample_batch = {
            'dialogue_processed_input_ids': torch.randint(0, 100, (4, 5, 32)),
            'dialogue_processed_attention_mask': torch.ones(4, 5, 32, dtype=torch.long),
            'ttm_conc_count': torch.rand(4, 1),
            'reversal_flag': torch.tensor([0, 1, 0, 1]),
        }

        config = {
            "text_name": "dialogue",
            "label_name": "reversal_flag",
            "tab_field_list": ["ttm_conc_count"],
            "hidden_common_dim": 16,
            "is_binary": True,
        }

        model = MultimodalBert(config)
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = os.path.join(tmpdir, "model.onnx")
            model.export_to_onnx(onnx_path, sample_batch)
            self.assertTrue(os.path.exists(onnx_path))

            # Load back and validate
            session = load_onnx_model(onnx_path)
            val_dataset = DummyDataset(sample_batch)
            val_loader = DataLoader(val_dataset, batch_size=2)

            # Validate inference
            predictions = model_online_inference(session, val_loader)
            self.assertEqual(predictions.shape[0], len(val_dataset))

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
