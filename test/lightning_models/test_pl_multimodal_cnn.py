import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from lightning.pytorch import seed_everything

# Import the class to be tested
from src.lightning_models.pl_multimodal_cnn import MultimodalCNN

# --- Mock Sub-networks ---
# Create mock versions of the dependencies to isolate the MultimodalCNN class.

class MockTextCNN(torch.nn.Module):
    """A mock for the TextCNN sub-network."""
    def __init__(self, config, vocab_size, word_embeddings):
        super().__init__()
        # The output dimension is important for the final classifier layer
        self.output_text_dim = config.get("hidden_common_dim", 32)

    def forward(self, input_ids: torch.Tensor):
        # Infer batch size from the input tensor
        batch_size = input_ids.shape[0]
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


class TestMultimodalCNN(unittest.TestCase):

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
            "hidden_common_dim": 16,
            "model_path": self.temp_dir,
            "lr": 1e-3,
            "metric_choices": ["accuracy", "f1_score"],
            # Add keys required by the TextCNN sub-network
            "max_sen_len": 32,
            "kernel_size": [3],
            # FIX: The number of channels must match the number of layers.
            "num_layers": 2,
            "num_channels": [16, 16] 
        }

        # Dummy embeddings and vocab needed for MultimodalCNN __init__
        self.vocab_size = 100
        self.embed_size = 50
        self.word_embeddings = torch.randn(self.vocab_size, self.embed_size)

        # Create a dummy batch of data
        self.batch = {
            "dialogue_processed_input_ids": torch.randint(0, self.vocab_size, (4, self.config["max_sen_len"])),
            "feature1": torch.rand(4),
            "feature2": torch.rand(4),
            "label": torch.tensor([0, 1, 0, 1]),
            "id": ["id1", "id2", "id3", "id4"]
        }
        # Model initialization is moved into each test under a patch context

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test that the model initializes correctly based on the config."""
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            model = MultimodalCNN(self.config, self.vocab_size, self.word_embeddings)
            
            self.assertEqual(model.task, "binary")
            self.assertEqual(model.num_classes, 2)
            self.assertIsInstance(model.text_subnetwork, MockTextCNN)
            self.assertIsInstance(model.tab_subnetwork, MockTabAE)
            # Check if the final layer has the correct input and output dimensions
            expected_in_features = model.text_subnetwork.output_text_dim + model.tab_subnetwork.output_tab_dim
            self.assertEqual(model.final_merge_network[1].in_features, expected_in_features)
            self.assertEqual(model.final_merge_network[1].out_features, model.num_classes)

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            model = MultimodalCNN(self.config, self.vocab_size, self.word_embeddings)
            model.eval()
            with torch.no_grad():
                logits = model(self.batch)
            # The output shape should be [batch_size, num_classes]
            self.assertEqual(logits.shape, (4, 2))

    def test_training_step(self):
        """Test a single training step."""
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            model = MultimodalCNN(self.config, self.vocab_size, self.word_embeddings)
            model.trainer = MagicMock()
            model.train()

            output = model.training_step(self.batch, batch_idx=0)
            self.assertIn("loss", output)
            self.assertIsInstance(output["loss"], torch.Tensor)
            self.assertGreaterEqual(output["loss"].item(), 0)

    @patch('src.lightning_models.pl_multimodal_cnn.all_gather')
    @patch('src.lightning_models.pl_multimodal_cnn.compute_metrics')
    def test_validation_flow(self, mock_compute_metrics, mock_all_gather):
        """Test the full validation loop: start, step, and end."""
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            model = MultimodalCNN(self.config, self.vocab_size, self.word_embeddings)
            model.trainer = MagicMock()

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
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            model = MultimodalCNN(self.config, self.vocab_size, self.word_embeddings)
            model.trainer = MagicMock()
            # Mock the global_rank property, which is used for file naming
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
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            config_mc = self.config.copy()
            config_mc["is_binary"] = False
            config_mc["num_classes"] = 4
            
            model_mc = MultimodalCNN(config_mc, self.vocab_size, self.word_embeddings)
            model_mc.eval()

            batch_mc = self.batch.copy()
            batch_mc["label"] = torch.tensor([0, 2, 1, 3])
            
            with torch.no_grad():
                logits = model_mc(batch_mc)
            self.assertEqual(logits.shape, (4, 4))
            
            loss, preds, labels = model_mc.run_epoch(batch_mc, "val")
            self.assertEqual(preds.shape, (4, 4))
            self.assertIsInstance(loss, torch.Tensor)

    def test_onnx_export(self):
        """Test the ONNX export functionality."""
        with patch('src.lightning_models.pl_multimodal_cnn.TextCNN', MockTextCNN), \
             patch('src.lightning_models.pl_multimodal_cnn.TabAE', MockTabAE):
            model = MultimodalCNN(self.config, self.vocab_size, self.word_embeddings)
            model.eval()
            save_path = Path(self.temp_dir) / "model.onnx"
            
            # The export function expects a sample batch dictionary with all tensor inputs
            onnx_batch = {
                model.text_name: self.batch["dialogue_processed_input_ids"],
                "feature1": self.batch["feature1"],
                "feature2": self.batch["feature2"],
            }
            
            model.export_to_onnx(save_path, onnx_batch)
            
            self.assertTrue(save_path.exists())
            self.assertGreater(save_path.stat().st_size, 0)
        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)