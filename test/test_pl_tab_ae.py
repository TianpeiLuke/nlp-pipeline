import unittest
import torch
import torch.nn as nn
from lightning.pytorch import seed_everything

# Updated import path for your project structure
from ..src.lightning_models.pl_tab_ae import TabAE


class TestTabAE(unittest.TestCase):

    def setUp(self):
        seed_everything(42)
        self.config = {
            'tab_field_list': ['feat1', 'feat2', 'feat3'],
            'label_name': 'target',
            'hidden_common_dim': 8,
            'is_binary': True,
            'optimizer_type': 'SGD',
            'lr': 0.01,
            'momentum': 0.9,
            'pos_weight': 1.0
        }

        self.model = TabAE(self.config)
        self.model.add_loss_op()
        self.model.eval()

        # Dummy batch of size 4
        self.batch = {
            'feat1': torch.tensor([0.1, 0.2, 0.3, 0.4]),
            'feat2': torch.tensor([1.0, 1.5, 1.7, 1.9]),
            'feat3': torch.tensor([0.5, 0.7, 0.2, 0.9]),
            'target': torch.tensor([0.0, 1.0, 0.0, 1.0])
        }

    def test_model_init(self):
        self.assertIsInstance(self.model.embedding_layer, nn.Sequential)
        self.assertIsInstance(self.model.classifier, nn.Linear)
        self.assertEqual(self.model.output_tab_dim, 8)
        self.assertEqual(self.model.num_classes, 2)

    def test_embedding_forward_pass(self):
        with torch.no_grad():
            embedding = self.model(self.batch)
        self.assertEqual(embedding.shape, (4, 8))  # [B, D]

    def test_classifier_forward_pass(self):
        with torch.no_grad():
            logits = self.model.forward_classify(self.batch)
        self.assertEqual(logits.shape, (4, 1))  # [B] for binary (squeezed)

    def test_run_epoch_loss(self):
        loss, preds, labels = self.model.run_epoch(self.batch, stage='train')
        self.assertIsNotNone(loss)
        self.assertEqual(preds.shape, labels.shape)
        self.assertTrue(torch.all((preds >= 0) & (preds <= 1)))  # sigmoid output range

    def test_training_step(self):
        loss = self.model.training_step(self.batch, batch_idx=0)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
