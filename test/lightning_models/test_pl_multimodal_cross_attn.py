import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from lightning.pytorch import seed_everything

# Import the class to be tested
from src.lightning_models.pl_multimodal_cross_attn import MultimodalBertCrossAttn

# --- Mock Sub-networks ---
# Create mock versions of the dependencies to isolate the model.

class MockTextBertBase(torch.nn.Module):
    """A mock for the TextBertBase sub-network."""
    def __init__(self, config):
        super().__init__()
        # For the cross-attention model, this must match hidden_common_dim
        self.output_text_dim = config.get("hidden_common_dim", 16)
        # Store keys needed for forward pass to avoid dependency on batch['config']
        text_name = config.get("text_name", "text")
        input_ids_suffix = config.get("text_input_ids_key", "input_ids")
        self.input_ids_key = f"{text_name}_processed_{input_ids_suffix}"

    def forward(self, batch):
        # Infer batch size from the input tensor using the stored key
        batch_size = batch[self.input_ids_key].shape[0]
        # Return a dummy tensor with the correct shape [batch_size, output_dim]
        return torch.randn(batch_size, self.output_text_dim)

class MockTabAE(torch.nn.Module):
    """A mock for the TabAE sub-network."""
    def __init__(self, config):
        super().__init__()
        self.tab_field_list = config.get("tab_field_list", [])
        # For the cross-attention model, this must match hidden_common_dim
        self.output_tab_dim = config.get("hidden_common_dim", 16)

    def combine_tab_data(self, batch):
        """Mocks the logic to combine tabular features into a single tensor."""
        if not self.tab_field_list:
            return None
        combined = [batch[field].unsqueeze(1).float() for field in self.tab_field_list]
        return torch.cat(combined, dim=1)

    def forward(self, tab_data):
        if tab_data is None:
            return None
        batch_size = tab_data.shape[0]
        # Return a dummy tensor with the correct shape
        return torch.randn(batch_size, self.output_tab_dim)


class TestMultimodalBertCrossAttn(unittest.TestCase):

    def setUp(self):
        """Set up a consistent environment for each test."""
        seed_everything(42)
        self.temp_dir = tempfile.mkdtemp()

        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "id_name": "id",
            "tab_field_list": ["feature1", "feature2"],
            "is_binary": True,
            "num_classes": 2,
            "hidden_common_dim": 16, # This must match mock model output dims
            "num_heads": 4, # For the cross-attention layer
            "model_path": self.temp_dir,
            "lr": 1e-4,
            "metric_choices": ["accuracy", "f1_score"]
        }

        # Create a dummy batch of data
        self.batch = {
            "dialogue_processed_input_ids": torch.randint(0, 1000, (4, 32)),
            "dialogue_processed_attention_mask": torch.ones(4, 32, dtype=torch.long),
            "feature1": torch.rand(4),
            "feature2": torch.rand(4),
            "label": torch.tensor([0, 1, 0, 1]),
            "id": ["id1", "id2", "id3", "id4"],
            "config": self.config # Pass config to mocks
        }
        # Model is now initialized within each test under a patch context

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test that the model initializes correctly based on the config."""
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            model = MultimodalBertCrossAttn(self.config)
            self.assertEqual(model.task, "binary")
            self.assertIsInstance(model.text_subnetwork, MockTextBertBase)
            self.assertIsInstance(model.tab_subnetwork, MockTabAE)
            self.assertIsNotNone(model.cross_att)
            # Check the final classifier's input dimension, which should be 2 * hidden_dim
            self.assertEqual(model.final_merge_network[0].in_features, self.config["hidden_common_dim"] * 2)
            self.assertEqual(model.final_merge_network[2].out_features, model.num_classes)

    def test_init_raises_error_on_dim_mismatch(self):
        """Test that an AssertionError is raised if sub-network dimensions don't match hidden_common_dim."""
        # Create specialized mocks with fixed output dimensions for this test
        class MockTextWithFixedDim(torch.nn.Module):
            def __init__(self, config):
                super().__init__()
                self.output_text_dim = 16 # Fixed dimension

        class MockTabWithFixedDim(MockTabAE):
            def __init__(self, config):
                super().__init__(config)
                self.output_tab_dim = 16 # Fixed dimension

        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextWithFixedDim), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabWithFixedDim):
            
            config_mismatch = self.config.copy()
            # Set hidden_common_dim to a value that does not match the fixed mock dimensions
            config_mismatch["hidden_common_dim"] = 32 
            
            with self.assertRaises(AssertionError):
                MultimodalBertCrossAttn(config_mismatch)

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            model = MultimodalBertCrossAttn(self.config)
            model.eval()
            with torch.no_grad():
                logits = model(self.batch)
            self.assertEqual(logits.shape, (4, 2)) # [B, num_classes]

    def test_training_step(self):
        """Test a single training step."""
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            model = MultimodalBertCrossAttn(self.config)
            model.trainer = MagicMock()
            model.trainer.estimated_stepping_batches = 100
            model.train()
            output = model.training_step(self.batch, batch_idx=0)
            self.assertIn("loss", output)
            self.assertIsInstance(output["loss"], torch.Tensor)

    @patch('src.lightning_models.pl_multimodal_cross_attn.all_gather')
    @patch('src.lightning_models.pl_multimodal_cross_attn.compute_metrics')
    def test_validation_flow(self, mock_compute_metrics, mock_all_gather):
        """Test the full validation loop: start, step, and end."""
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            model = MultimodalBertCrossAttn(self.config)
            model.on_validation_epoch_start()
            self.assertEqual(model.pred_lst, [])
            self.assertEqual(model.label_lst, [])

            model.validation_step(self.batch, batch_idx=0)
            self.assertEqual(len(model.pred_lst), 4)

            # Mock all_gather to simulate gathering from a single GPU
            mock_all_gather.side_effect = lambda x: [x]
            model.on_validation_epoch_end()
            mock_compute_metrics.assert_called_once()
            
    def test_test_step_and_epoch_end(self):
        """Test that test steps save results to a file correctly."""
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            model = MultimodalBertCrossAttn(self.config)
            model.trainer = MagicMock()
            type(model.trainer).global_rank = unittest.mock.PropertyMock(return_value=0)

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
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            config_mc = self.config.copy()
            config_mc["is_binary"] = False
            config_mc["num_classes"] = 4

            model_mc = MultimodalBertCrossAttn(config_mc)
            model_mc.trainer = MagicMock() # Needed for run_epoch
            model_mc.eval()

            batch_mc = self.batch.copy()
            # Use the correct key for multiclass labels
            batch_mc[model_mc.label_name_transformed] = torch.tensor([0, 2, 1, 3])
            
            with torch.no_grad():
                logits = model_mc(batch_mc)
            self.assertEqual(logits.shape, (4, 4))
            
            loss, preds, labels = model_mc.run_epoch(batch_mc, "val")
            self.assertEqual(preds.shape, (4, 4))
            self.assertIsInstance(loss, torch.Tensor)

    def test_onnx_export(self):
        """Test the ONNX export functionality."""
        with patch('src.lightning_models.pl_multimodal_cross_attn.TextBertBase', MockTextBertBase), \
             patch('src.lightning_models.pl_multimodal_cross_attn.TabAE', MockTabAE):
            model = MultimodalBertCrossAttn(self.config)
            model.eval()
            save_path = Path(self.temp_dir) / "model.onnx"
            
            # ONNX export batch doesn't need the 'config' key.
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
    unittest.main(argv=[''], exit=False)