# test/test_pl_lstm.py
import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from lightning.pytorch import seed_everything


# Import the class to be tested
from src.lightning_models.pl_lstm import TextLSTM

class TestTextLSTM(unittest.TestCase):

    def setUp(self):
        """Set up a consistent environment for each test."""
        seed_everything(42)
        self.temp_dir = tempfile.mkdtemp()

        # Config for the TextLSTM model
        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "id_name": "id",
            "is_binary": True,
            "num_classes": 2,
            "model_path": self.temp_dir,
            "lr": 1e-3,
            "max_sen_len": 32,
            "hidden_common_dim": 64, # Renamed hidden_dimension to hidden_common_dim for consistency
            "num_layers": 2, # Using more than one layer to test dropout
            "dropout_keep": 0.5,
            "is_embeddings_trainable": True,
            "metric_choices": ["accuracy", "f1_score"]
        }

        # Dummy embeddings and vocab
        self.vocab_size = 200
        self.embed_size = 100
        self.word_embeddings = torch.randn(self.vocab_size, self.embed_size)

        # Create a dummy batch of data
        self.batch = {
            "dialogue_processed_input_ids": torch.randint(0, self.vocab_size, (4, self.config["max_sen_len"])),
            "label": torch.tensor([0, 1, 0, 1]),
            "id": ["id1", "id2", "id3", "id4"]
        }

        # Initialize the model
        self.model = TextLSTM(self.config, self.vocab_size, self.word_embeddings)
        self.model.trainer = MagicMock() # Mock trainer for components that need it
        self.model.eval()

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        """Test that the model initializes correctly based on the config."""
        self.assertEqual(self.model.task, "binary")
        self.assertEqual(self.model.num_classes, 2)
        self.assertIsInstance(self.model.embeddings, torch.nn.Embedding)
        self.assertEqual(self.model.embeddings.weight.shape, (self.vocab_size, self.embed_size))
        # Check LSTM properties
        self.assertEqual(self.model.lstm.hidden_size, self.config["hidden_common_dim"])
        self.assertEqual(self.model.lstm.num_layers, self.config["num_layers"])
        self.assertTrue(self.model.lstm.bidirectional)
        # Check final linear layer output
        self.assertEqual(self.model.fc.out_features, self.config["num_classes"])
        self.assertEqual(self.model.fc.in_features, self.config["hidden_common_dim"] * 2) # Bidirectional

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        with torch.no_grad():
            logits = self.model(self.batch)
        # The output shape should be [batch_size, num_classes]
        self.assertEqual(logits.shape, (4, self.config["num_classes"]))

    def test_run_epoch_and_training_step(self):
        """Test the run_epoch logic and a single training step."""
        self.model.train()
        
        # Test training_step wrapper
        output = self.model.training_step(self.batch, batch_idx=0)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)
        self.assertTrue(output["loss"].requires_grad)

    @patch('src.lightning_models.dist_utils.all_gather')
    @patch('src.lightning_models.pl_model_plots.compute_metrics')
    def test_validation_flow(self, mock_compute_metrics, mock_all_gather):
        """Test the full validation loop: start, step, and end."""
        self.model.on_validation_epoch_start()
        self.assertEqual(self.model.pred_lst, [])
        self.assertEqual(self.model.label_lst, [])

        self.model.validation_step(self.batch, batch_idx=0)
        self.assertEqual(len(self.model.pred_lst), 4)
        self.assertEqual(len(self.model.label_lst), 4)

        # Mock all_gather to simulate gathering from a single GPU
        mock_all_gather.side_effect = lambda x: [x]
        self.model.on_validation_epoch_end()
        mock_compute_metrics.assert_called_once()
        
    def test_test_step_and_epoch_end(self):
        """Test that test steps save results to a file correctly."""
        # Mock the global_rank property, which is used for file naming
        type(self.model.trainer).global_rank = unittest.mock.PropertyMock(return_value=0)
        
        self.model.on_test_epoch_start()
        self.model.test_step(self.batch, batch_idx=0)
        self.model.on_test_epoch_end()
        
        self.assertTrue(self.model.test_output_folder.exists())
        output_files = list(self.model.test_output_folder.glob("*.tsv"))
        self.assertEqual(len(output_files), 1)
        
        df = pd.read_csv(output_files[0], sep="\t")
        self.assertEqual(len(df), 4)
        self.assertIn("prob", df.columns)
        self.assertIn("label", df.columns)
        self.assertIn("id", df.columns)

    def test_multiclass_support(self):
        """Test that the model adapts correctly for multiclass classification."""
        config_mc = self.config.copy()
        config_mc["is_binary"] = False
        config_mc["num_classes"] = 5
        
        model_mc = TextLSTM(config_mc, self.vocab_size, self.word_embeddings)
        model_mc.eval()

        batch_mc = self.batch.copy()
        batch_mc["label"] = torch.tensor([0, 2, 1, 4])
        
        with torch.no_grad():
            logits = model_mc(batch_mc)
        self.assertEqual(logits.shape, (4, 5))
        
        # The run_epoch method handles multiclass predictions
        loss, preds, labels = model_mc.run_epoch(batch_mc, "val")
        # For multiclass, preds should have shape [batch_size, num_classes]
        self.assertEqual(preds.shape, (4, 5))
        self.assertIsInstance(loss, torch.Tensor)

    def test_onnx_export(self):
        """Test the ONNX export functionality."""
        save_path = Path(self.temp_dir) / "model.onnx"
        
        # The export function expects a sample batch dictionary
        onnx_batch = {
            self.model.text_name: self.batch["dialogue_processed_input_ids"]
        }
        
        self.model.export_to_onnx(save_path, onnx_batch)
        
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 0)
        
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)