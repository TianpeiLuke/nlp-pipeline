import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import torch
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from lightning.pytorch import seed_everything
from pydantic import ValidationError

# Import the class to be tested
from src.lightning_models.pl_bert_classification import TextBertClassification, TextBertClassificationConfig

class DummyOutputs:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss

class DummyBertModel(torch.nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
    def __call__(self, input_ids=None, attention_mask=None, labels=None):
        batch_size = input_ids.shape[0]
        logits = torch.randn(batch_size, self.num_labels)
        loss = torch.tensor(0.123)
        return DummyOutputs(logits=logits, loss=loss)

class TestTextBertClassification(unittest.TestCase):
    def setUp(self):
        seed_everything(42)
        self.temp_dir = tempfile.mkdtemp()
        # Minimal config for TextBertClassification
        self.config_dict = {
            "text_name": "text",
            "label_name": "label",
            "tokenizer": "bert-base-uncased",
            "is_binary": True,
            "num_classes": 2,
            "model_path": self.temp_dir,
            "id_name": "id",
        }
        # Dummy batch
        self.batch = {
            "text": torch.randint(0, 100, (4, 16)),
            "attention_mask": torch.ones(4, 16, dtype=torch.long),
            "label": torch.tensor([0, 1, 0, 1]),
            "id": ["a", "b", "c", "d"],
        }

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch('src.lightning_models.pl_bert_classification.AutoModelForSequenceClassification')
    def test_initialization_and_forward(self, mock_pretrained):
        mock_pretrained.from_pretrained.return_value = DummyBertModel(num_labels=2)
        config = TextBertClassificationConfig(**self.config_dict)
        model = TextBertClassification(config)
        outputs = model(self.batch)
        self.assertTrue(hasattr(outputs, 'logits'))
        self.assertTrue(hasattr(outputs, 'loss'))
        self.assertEqual(outputs.logits.shape, (4, 2))

    @patch('src.lightning_models.pl_bert_classification.AutoModelForSequenceClassification')
    def test_run_epoch_and_training_step(self, mock_pretrained):
        mock_pretrained.from_pretrained.return_value = DummyBertModel(num_labels=2)
        model = TextBertClassification(self.config_dict)
        # stub trainer for scheduler
        model.trainer = MagicMock()
        model.trainer.estimated_stepping_batches = 10
        # inject missing task attribute for metrics calls
        object.__setattr__(model.config, 'task', 'binary')
        loss, preds, labels = model.run_epoch(self.batch, 'train')
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(preds.shape, (4,))
        self.assertTrue((labels == self.batch['label']).all())
        out = model.training_step(self.batch, 0)
        self.assertIn('loss', out)

    @patch('src.lightning_models.pl_bert_classification.AutoModelForSequenceClassification')
    @patch('src.lightning_models.pl_bert_classification.compute_metrics')
    @patch('src.lightning_models.pl_bert_classification.all_gather')
    def test_validation_loop_calls_compute_metrics(self, mock_all_gather, mock_compute_metrics, mock_pretrained):
        mock_pretrained.from_pretrained.return_value = DummyBertModel(num_labels=2)
        model = TextBertClassification(self.config_dict)
        object.__setattr__(model.config, 'task', 'binary')
        model.eval()
        model.on_validation_epoch_start()
        model.validation_step(self.batch, 0)
        mock_all_gather.side_effect = lambda x: [x]
        model.on_validation_epoch_end()
        mock_compute_metrics.assert_called_once()

    @patch('src.lightning_models.pl_bert_classification.AutoModelForSequenceClassification')
    def test_test_step_and_output_file(self, mock_pretrained):
        mock_pretrained.from_pretrained.return_value = DummyBertModel(num_labels=2)
        model = TextBertClassification(self.config_dict)
        object.__setattr__(model.config, 'task', 'binary')
        model.trainer = MagicMock()
        type(model.trainer).global_rank = PropertyMock(return_value=0)
        model.eval()
        model.on_test_epoch_start()
        model.test_step(self.batch, 0)
        model.on_test_epoch_end()
        folder = Path(model.test_output_folder)
        files = list(folder.glob('test_result.tsv'))
        self.assertTrue(files)
        df = pd.read_csv(files[0], sep='\t')
        self.assertIn('prob', df.columns)
        self.assertIn('id', df.columns)
        self.assertIn('label', df.columns)

    @patch('src.lightning_models.pl_bert_classification.AutoModelForSequenceClassification')
    def test_predict_step(self, mock_pretrained):
        mock_pretrained.from_pretrained.return_value = DummyBertModel(num_labels=2)
        model = TextBertClassification(self.config_dict)
        # missing label => returns tuple (preds, None)
        batch_pred = {"text": self.batch['text'], "attention_mask": self.batch['attention_mask']}
        preds_only = model.predict_step(batch_pred, 0)
        self.assertIsInstance(preds_only, tuple)
        self.assertIsInstance(preds_only[0], torch.Tensor)
        self.assertIsNone(preds_only[1])
        # with label => returns (preds, labels)
        full_output = model.predict_step(self.batch, 0)
        self.assertIsInstance(full_output, tuple)
        preds, labels = full_output
        self.assertIsInstance(preds, torch.Tensor)
        self.assertIsInstance(labels, torch.Tensor)

    @patch('src.lightning_models.pl_bert_classification.AutoModelForSequenceClassification')
    def test_export_to_onnx_and_torchscript(self, mock_pretrained):
        mock_pretrained.from_pretrained.return_value = DummyBertModel(num_labels=2)
        model = TextBertClassification(self.config_dict)
        model.eval()
        onnx_path = Path(self.temp_dir) / 'model.onnx'
        with patch('torch.onnx.export', lambda *args, **kwargs: open(onnx_path, 'wb').close()), \
             patch('onnx.load', lambda x: None), \
             patch('onnx.checker.check_model', lambda x: None):
            model.export_to_onnx(onnx_path)
            self.assertTrue(onnx_path.exists())
        ts_path = Path(self.temp_dir) / 'model.ts'
        with patch('torch.jit.trace', lambda m, x: MagicMock(save=lambda p: open(ts_path, 'wb').close())):
            model.export_to_torchscript(ts_path)
            self.assertTrue(ts_path.exists())

    def test_config_validation_binary(self):
        """Binary mode requires num_classes == 2"""
        bad = self.config_dict.copy()
        bad["num_classes"] = 3
        with self.assertRaises(ValidationError):
            TextBertClassificationConfig(**bad)

    def test_config_validation_multiclass(self):
        """Multiclass mode requires num_classes >= 2"""
        bad = self.config_dict.copy()
        bad["is_binary"] = False
        bad["num_classes"] = 1
        with self.assertRaises(ValidationError):
            TextBertClassificationConfig(**bad)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
