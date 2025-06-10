# test/test_pl_train.py
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning.pytorch as pl
import tempfile
import shutil
import os
from pathlib import Path

# Assuming the files are in a structure that allows this import
# You may need to adjust the path based on your project structure
from src.lightning_models.pl_train import model_train

# A self-contained dummy model for testing the training function
# It inherits directly from pl.LightningModule to be independent of other model implementations.
class SimpleDummyModel(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        # A simple linear layer
        self.layer = torch.nn.Linear(10, 2)
        # The loss function
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        # A minimal training step
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # A minimal validation step that logs the metric we want to monitor
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss(y_hat, y)
        # Log the metric that EarlyStopping and ModelCheckpoint will monitor
        self.log("val/f1_score", 0.8, prog_bar=True) 
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # A standard optimizer
        return torch.optim.Adam(self.parameters(), lr=0.001)

class TestModelTrain(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and dummy data for each test."""
        # Create a temporary directory for logs and checkpoints
        self.temp_dir = tempfile.mkdtemp()
        
        # Control the checkpoint directory by setting the environment variable directly.
        # This is a more robust way to handle this dependency in tests than mocking.
        self.checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.environ["SM_CHECKPOINT_DIR"] = self.checkpoint_dir
        
        # Create a dummy dataset: 16 samples, 10 features each
        features = torch.randn(16, 10)
        labels = torch.randint(0, 2, (16,))
        dataset = TensorDataset(features, labels)

        # Create dataloaders
        self.train_dataloader = DataLoader(dataset, batch_size=4)
        self.val_dataloader = DataLoader(dataset, batch_size=4)

        # Configuration for the model_train function
        self.config = {
            "model_class": "simple_dummy",
            "fp16": False,
            "gradient_clip_val": 0.5,
            "max_epochs": 2, # Run for 2 epochs to ensure validation runs
            "early_stop_patience": 2,
            "val_check_interval": 1.0,
        }

    def tearDown(self):
        """Clean up the temporary directory and environment variable after each test."""
        shutil.rmtree(self.temp_dir)
        if "SM_CHECKPOINT_DIR" in os.environ:
            del os.environ["SM_CHECKPOINT_DIR"]

    @patch("src.lightning_models.pl_train.torch.cuda.is_available", return_value=False)
    def test_model_train_completes_and_checkpoints(self, mock_cuda_available):
        """
        Tests if the model_train function runs to completion and saves a checkpoint.
        This test now forces a CPU environment and directs checkpoints via an env var.
        """
        # Instantiate our simple dummy model
        model = SimpleDummyModel()

        # Call the training function
        # WORKAROUND: Use 'val_loss' instead of 'val/f1_score' because the pl_train.py script
        # creates an invalid filename if the metric contains a '/'.
        trainer = model_train(
            model=model,
            config=self.config,
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            device=1, # Forcing CPU, devices=1 means use 1 CPU core
            model_log_path=self.temp_dir,
            early_stop_metric="val_loss" 
        )

        # 1. Check if a trainer object is returned
        self.assertIsInstance(trainer, pl.Trainer)
        
        # 2. Check if the training process finished successfully
        self.assertEqual(trainer.state.status, 'finished')
        
        # 3. Check if the fit loop ran for the specified number of epochs
        self.assertEqual(trainer.current_epoch, self.config["max_epochs"])

        # 4. Check that a checkpoint file was created in the directory set by the env var
        self.assertTrue(os.path.exists(self.checkpoint_dir))
        
        # Find checkpoint files within the controlled directory
        checkpoints = list(Path(self.checkpoint_dir).glob("*.ckpt"))
        self.assertGreater(len(checkpoints), 0, "No checkpoint file was saved.")
        print(f"Test successful. Checkpoint saved at: {checkpoints[0]}")


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

