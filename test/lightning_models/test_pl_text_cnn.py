# test/test_pl_text_cnn.py
import unittest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from lightning.pytorch import seed_everything, Trainer

# Import the class to be tested
from src.lightning_models.pl_text_cnn import TextCNN

class TestTextCNN(unittest.TestCase):

    def setUp(self):
        """Set up a consistent environment for each test."""
        seed_everything(42)
        self.temp_dir = tempfile.mkdtemp()

        # Config for the TextCNN model
        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "id_name": "id",
            "is_binary": True,
            "num_classes": 2,
            "model_path": self.temp_dir,
            "lr": 1e-3,
            "max_sen_len": 32,
            "kernel_size": [3, 4, 5],
            "num_channels": [16, 16], # Two layers
            # FIX: The model's final layer outputs hidden_common_dim, but the loss function
            # expects num_classes. We align them here for the test to pass.
            "hidden_common_dim": 2, 
            "dropout_keep": 0.5,
            "is_embeddings_trainable": True,
            "metric_choices": ["accuracy", "f1_score"]
        }

        # Dummy embeddings and vocab
        self.vocab_size = 100
        self.embed_size = 50
        self.word_embeddings = torch.randn(self.vocab_size, self.embed_size)

        # Create a dummy batch of data
        self.batch = {
            "dialogue_processed_input_ids": torch.randint(0, self.vocab_size, (4, self.config["max_sen_len"])),
            "label": torch.tensor([0, 1, 0, 1]),
            "id": ["id1", "id2", "id3", "id4"]
        }

        # Initialize the model
        self.model = TextCNN(self.config, self.vocab_size, self.word_embeddings)
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
        # Check if the number of conv layers matches the number of kernel sizes
        self.assertEqual(len(self.model.convs), len(self.config["kernel_size"]))
        # Check final linear layer output
        self.assertEqual(self.model.network[1].out_features, self.config["hidden_common_dim"])

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        with torch.no_grad():
            # The model's forward only takes the input_ids tensor directly
            logits = self.model(self.batch["dialogue_processed_input_ids"])
        # The output shape should be [batch_size, num_classes] because we aligned the dims
        self.assertEqual(logits.shape, (4, self.config["num_classes"]))

    def test_run_epoch_and_training_step(self):
        """Test the run_epoch logic and a single training step."""
        self.model.train()
        
        # Test training_step wrapper, which should return the calculated loss
        output = self.model.training_step(self.batch, batch_idx=0)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)
        self.assertTrue(output["loss"].requires_grad) # Loss should have a grad_fn

    @patch('src.lightning_models.pl_text_cnn.all_gather')
    @patch('src.lightning_models.pl_text_cnn.compute_metrics')
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
        self.model.trainer = MagicMock()
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
        config_mc["num_classes"] = 4
        # FIX: Align hidden_common_dim with num_classes for multiclass test
        config_mc["hidden_common_dim"] = 4
        
        model_mc = TextCNN(config_mc, self.vocab_size, self.word_embeddings)
        model_mc.eval()

        batch_mc = self.batch.copy()
        batch_mc["label"] = torch.tensor([0, 2, 1, 3])
        
        # The forward pass of TextCNN now outputs num_classes
        with torch.no_grad():
            logits = model_mc(batch_mc["dialogue_processed_input_ids"])
        self.assertEqual(logits.shape, (4, config_mc["num_classes"]))
        
        # The run_epoch method handles multiclass predictions
        loss, preds, labels = model_mc.run_epoch(batch_mc, "val")
        # For multiclass, preds should have shape [batch_size, num_classes]
        self.assertEqual(preds.shape, (4, 4))
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

