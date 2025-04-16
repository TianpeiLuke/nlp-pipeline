import unittest
import torch
from src.lightning_models.pl_bert import TextBertBase  # Adjust based on your actual file structure


class TestTextBertBase(unittest.TestCase):
    def setUp(self):
        self.config = {
            'text_name': 'dialogue',
            'label_name': 'reversal_flag',
            'hidden_common_dim': 128,
            'model_path': './checkpoints',
            'tokenizer': 'bert-base-uncased',
            'is_binary': True,
            'num_classes': 2
        }

        # Create dummy input data: B=4, C=2, T=16
        self.batch = {
            'dialogue_processed_input_ids': torch.randint(0, 1000, (4, 2, 16)),
            'dialogue_processed_attention_mask': torch.ones(4, 2, 16, dtype=torch.long),
            'reversal_flag': torch.tensor([0, 1, 0, 1])  # Needed for full run with labels
        }

        self.model = TextBertBase(self.config)

    def test_forward_shape(self):
        """Test output shape of forward pass"""
        output = self.model(self.batch)  # Shape: [B, D]
        self.assertEqual(output.shape, (4, self.config['hidden_common_dim']))

    def test_no_empty_chunks(self):
        """Ensure ValueError is raised if all attention is zero (invalid input)"""
        self.batch['dialogue_processed_attention_mask'].zero_()
        with self.assertRaises(ValueError):
            self.model(self.batch)


if __name__ == "__main__":
    unittest.main()
