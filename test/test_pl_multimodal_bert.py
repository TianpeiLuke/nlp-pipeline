import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from lightning.pytorch import seed_everything, Trainer

# Import the class to be tested
from src.lightning_models.pl_multimodal_bert import MultimodalBert

# --- Mock Sub-networks ---
# Create mock versions of the dependencies to isolate the MultimodalBert class.
# This prevents downloading real models and makes tests fast and reliable.

class MockTextBertBase(torch.nn.Module):
    """A mock for the TextBertBase sub-network."""
    def __init__(self, config):
        super().__init__()
        # The output dimension is important for the final classifier layer
        self.output_text_dim = config.get("hidden_common_dim", 32)
        # Store needed config values to avoid dependency on 'config' being in the batch
        self.text_name = config.get("text_name", "text")
        self.input_ids_key_suffix = config.get("text_input_ids_key", "input_ids")
        self.full_input_key = f"{self.text_name}_processed_{self.input_ids_key_suffix}"

    def forward(self, batch):
        # Infer batch size from the expected input tensor using the stored key
        batch_size = batch[self.full_input_key].shape[0]
        # Return a dummy tensor with the correct shape [batch_size, output_dim]
        return torch.randn(batch_size, self.output_text_dim)

class MockTabAE(torch.nn.Module):
    """A mock for the TabAE sub-network."""
    def __init__(self, config):
        super().__init__()
        self.tab_field_list = config.get("tab_field_list", [])
        # Define a predictable output dimension
        self.output_tab_dim = config.get("hidden_common_dim", 16) 

    def combine_tab_data(self, batch):
        """Mocks the logic to combine tabular features into a single tensor."""
        if not self.tab_field_list:
            return None
        # Unsqueeze to add a feature dimension, then concatenate
        combined = [batch[field].unsqueeze(1).float() for field in self.tab_field_list]
        return torch.cat(combined, dim=1)

    def forward(self, tab_data):
        if tab_data is None:
            return None
        batch_size = tab_data.shape[0]
        # Return a dummy tensor with the correct shape
        return torch.randn(batch_size, self.output_tab_dim)


class TestMultimodalBert(unittest.TestCase):

    def setUp(self):
        """Set up a consistent environment for each test."""
        seed_everything(42)
        self.temp_dir = tempfile.mkdtemp() # For saving test artifacts

        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "id_name": "id",
            "tab_field_list": ["feature1", "feature2"],
            "hidden_common_dim": 16, # Used by mock models
            "is_binary": True,
            "num_classes": 2, # Will be ignored if is_binary is True
            "model_path": self.temp_dir,
            "lr": 1e-4,
            "metric_choices": ["accuracy", "f1_score"]
        }

        # Create a dummy batch of data with the expected structure and types
        self.batch = {
            "dialogue_processed_input_ids": torch.randint(0, 1000, (4, 16)), # [B, T], assuming no chunking
            "dialogue_processed_attention_mask": torch.ones(4, 16, dtype=torch.long),
            "feature1": torch.rand(4),
            "feature2": torch.rand(4),
            "label": torch.tensor([0, 1, 0, 1]),
            "id": ["id1", "id2", "id3", "id4"], # For testing output files
            "config": self.config # Pass config to mocks for forward pass testing
        }
        # Model initialization is now moved inside each test under a patch context

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test that the model initializes correctly based on the config."""
        # Patch the dependencies using a 'with' statement
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):
            
            # Initialize the model inside the patch context
            model = MultimodalBert(self.config)

            self.assertEqual(model.task, "binary")
            self.assertEqual(model.num_classes, 2)
            self.assertIsInstance(model.text_subnetwork, MockTextBertBase)
            self.assertIsInstance(model.tab_subnetwork, MockTabAE)
            
            expected_in_features = model.text_subnetwork.output_text_dim + model.tab_subnetwork.output_tab_dim
            self.assertEqual(model.final_merge_network[1].in_features, expected_in_features)

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):
            
            model = MultimodalBert(self.config)
            model.eval()
            
            with torch.no_grad():
                logits = model(self.batch)
            self.assertEqual(logits.shape, (4, 2))

    def test_training_step(self):
        """Test a single training step."""
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):

            model = MultimodalBert(self.config)
            # The optimizer needs the trainer to be mocked
            model.trainer = MagicMock()
            model.trainer.estimated_stepping_batches = 100
            model.train()

            output = model.training_step(self.batch, batch_idx=0)
            self.assertIn("loss", output)
            self.assertIsInstance(output["loss"], torch.Tensor)
            self.assertGreaterEqual(output["loss"].item(), 0)

    @patch('src.lightning_models.pl_multimodal_bert.all_gather')
    @patch('src.lightning_models.pl_multimodal_bert.compute_metrics')
    def test_validation_flow(self, mock_compute_metrics, mock_all_gather):
        """Test the full validation loop: start, step, and end."""
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):
            
            model = MultimodalBert(self.config)
            model.eval()
            
            model.on_validation_epoch_start()
            self.assertEqual(model.pred_lst, [])
            self.assertEqual(model.label_lst, [])

            model.validation_step(self.batch, batch_idx=0)
            self.assertEqual(len(model.pred_lst), 4)
            self.assertEqual(len(model.label_lst), 4)

            # Mock all_gather to simulate gathering from a single GPU
            mock_all_gather.side_effect = lambda x: [x]
            model.on_validation_epoch_end()
            mock_compute_metrics.assert_called_once()
        
    def test_test_step_and_epoch_end(self):
        """Test that test steps save results to a file correctly."""
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):

            model = MultimodalBert(self.config)
            # Mock the trainer and its global_rank property
            model.trainer = MagicMock()
            type(model.trainer).global_rank = unittest.mock.PropertyMock(return_value=0)
            model.eval()

            model.on_test_epoch_start()
            model.test_step(self.batch, batch_idx=0)
            model.on_test_epoch_end()
            
            self.assertTrue(model.test_output_folder.exists())
            output_files = list(model.test_output_folder.glob("*.tsv"))
            self.assertEqual(len(output_files), 1)
            
            df = pd.read_csv(output_files[0], sep="\t")
            self.assertEqual(len(df), 4)
            self.assertIn("prob", df.columns)
            self.assertIn("label", df.columns)
            self.assertIn("id", df.columns)

    def test_multiclass_support(self):
        """Test that the model adapts correctly for multiclass classification."""
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):

            config_mc = self.config.copy()
            config_mc["is_binary"] = False
            config_mc["num_classes"] = 4
            
            model_mc = MultimodalBert(config_mc)
            model_mc.eval()

            batch_mc = self.batch.copy()
            # Use the correct transformed label key for multiclass
            batch_mc[model_mc.label_name_transformed] = torch.tensor([0, 2, 1, 3])
            batch_mc["config"] = config_mc # Update the config in the batch
            
            with torch.no_grad():
                logits = model_mc(batch_mc)
            self.assertEqual(logits.shape, (4, 4))
            
            # The label key is different for multiclass, so remove the old one
            if 'label' in batch_mc and model_mc.label_name_transformed != 'label':
                del batch_mc['label']
            
            loss, preds, labels = model_mc.run_epoch(batch_mc, "val")
            self.assertEqual(preds.shape, (4, 4))
            # Loss should be a valid tensor, not None
            self.assertIsInstance(loss, torch.Tensor)

    def test_onnx_export(self):
        """Test the ONNX export functionality."""
        with patch('src.lightning_models.pl_multimodal_bert.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_bert.TabAE', MockTabAE):

            model = MultimodalBert(self.config)
            model.eval()
            save_path = Path(self.temp_dir) / "model.onnx"
            
            # The sample batch for ONNX export only needs the tensor inputs.
            # The 'config' key is no longer needed here as the mock is now self-sufficient.
            onnx_batch = {
                model.text_name: self.batch["dialogue_processed_input_ids"],
                model.text_attention_mask: self.batch["dialogue_processed_attention_mask"],
                "feature1": self.batch["feature1"],
                "feature2": self.batch["feature2"],
            }
            
            model.export_to_onnx(save_path, onnx_batch)
            
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)
        
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
