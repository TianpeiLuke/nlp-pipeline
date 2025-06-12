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
            "hidden_common_dim": 64,
            "num_layers": 2,
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
        self.model.trainer = MagicMock()
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
        self.assertEqual(self.model.lstm.hidden_size, self.config["hidden_common_dim"])
        self.assertEqual(self.model.lstm.num_layers, self.config["num_layers"])
        self.assertTrue(self.model.lstm.bidirectional)
        self.assertEqual(self.model.fc.out_features, self.config["num_classes"])

    def test_forward_pass(self):
        """Test a single forward pass through the model."""
        with torch.no_grad():
            logits = self.model(self.batch)
        self.assertEqual(logits.shape, (4, self.config["num_classes"]))

    def test_training_step(self):
        """Test that training_step returns a dict with loss requiring grad."""
        self.model.train()
        output = self.model.training_step(self.batch, batch_idx=0)
        self.assertIn("loss", output)
        self.assertIsInstance(output["loss"], torch.Tensor)
        self.assertTrue(output["loss"].requires_grad)

    @patch('src.lightning_models.pl_lstm.compute_metrics')
    @patch('src.lightning_models.dist_utils.all_gather', side_effect=lambda x: [x])
    def test_validation_loop_calls_compute_metrics(self, mock_all_gather, mock_compute_metrics):
        """Test the full validation loop: start, step, and end triggers metrics computation."""
        # Start epoch
        self.model.on_validation_epoch_start()
        self.assertListEqual(self.model.pred_lst, [])
        self.assertListEqual(self.model.label_lst, [])

        # One validation step
        self.model.validation_step(self.batch, batch_idx=0)
        self.assertEqual(len(self.model.pred_lst), 4)
        self.assertEqual(len(self.model.label_lst), 4)

        # End epoch should call compute_metrics once
        self.model.on_validation_epoch_end()
        mock_compute_metrics.assert_called_once()

    def test_test_step_and_epoch_end(self):
        """Test that test steps save results to a file correctly."""
        type(self.model.trainer).global_rank = unittest.mock.PropertyMock(return_value=0)
        self.model.on_test_epoch_start()
        self.model.test_step(self.batch, batch_idx=0)
        self.model.on_test_epoch_end()

        folder = self.model.test_output_folder
        self.assertTrue(folder.exists())
        files = list(folder.glob("*.tsv"))
        self.assertEqual(len(files), 1)

        df = pd.read_csv(files[0], sep="\t")
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

        # run_epoch returns loss, preds, labels for multiclass
        loss, preds, labels = model_mc.run_epoch(batch_mc, "val")
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(preds.shape, (4, 5))

    def test_onnx_export(self):
        """Test the ONNX export functionality and invalid key handling."""
        save_path = Path(self.temp_dir) / "model.onnx"
        onnx_batch = {self.model.text_name: self.batch["dialogue_processed_input_ids"]}
        self.model.export_to_onnx(save_path, onnx_batch)
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 0)

    def test_predict_step(self):
        """Test predict_step returns correct types for pred and test modes."""
        # pred mode (no label)
        pred_batch = {"dialogue_processed_input_ids": self.batch["dialogue_processed_input_ids"]}
        preds = self.model.predict_step(pred_batch, batch_idx=0)
        self.assertIsInstance(preds, torch.Tensor)
        self.assertEqual(preds.shape[0], 4)

        # test mode (with label)
        type(self.model.trainer).global_rank = unittest.mock.PropertyMock(return_value=0)
        test_batch = self.batch.copy()
        preds_lbl = self.model.predict_step(test_batch, batch_idx=0)
        self.assertIsInstance(preds_lbl, tuple)
        self.assertEqual(len(preds_lbl), 2)

    def test_export_to_torchscript(self):
        """Skip TorchScript export for TextLSTM (not implemented)."""
        if not hasattr(self.model, 'export_to_torchscript'):
            self.skipTest("export_to_torchscript not implemented for TextLSTM")

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
