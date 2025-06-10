import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import torch
from torch.utils.tensorboard import SummaryWriter

from src.lightning_models.pl_model_plots import roc_metric_plot, pr_metric_plot

class TestMetricPlots(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.writer = SummaryWriter(log_dir=self.tmp_dir)
        self.global_step = 5

    def tearDown(self):
        self.writer.close()
        shutil.rmtree(self.tmp_dir)

    def test_roc_plot_binary(self):
        y_pred = torch.rand(100)
        y_true = torch.randint(0, 2, (100,))
        roc_metric_plot(
            y_pred, y_true, y_pred, y_true, self.tmp_dir,
            task="binary", num_classes=2,
            writer=self.writer, global_step=self.global_step
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "ROC-BSM.svg")))

    def test_pr_plot_binary(self):
        y_pred = torch.rand(100)
        y_true = torch.randint(0, 2, (100,))
        pr_metric_plot(
            y_pred, y_true, y_pred, y_true, self.tmp_dir,
            task="binary", num_classes=2, 
            writer=self.writer, global_step=self.global_step
        )
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, "PR-BSM.svg")))

    def test_roc_plot_multiclass(self):
        num_classes = 3
        y_pred = torch.softmax(torch.randn(100, num_classes), dim=1)
        y_true = torch.randint(0, num_classes, (100,))
        roc_metric_plot(
            y_pred, y_true, y_pred, y_true, self.tmp_dir,
            task="multiclass", num_classes=num_classes,
            writer=self.writer, global_step=self.global_step
        )
        for fname in ["ROC-BSM-ovr.svg", "ROC-BSM-macro.svg", "ROC-BSM-weighted.svg"]:
            self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, fname)))

    def test_pr_plot_multiclass(self):
        num_classes = 3
        y_pred = torch.softmax(torch.randn(100, num_classes), dim=1)
        y_true = torch.randint(0, num_classes, (100,))
        pr_metric_plot(
            y_pred, y_true, y_pred, y_true, self.tmp_dir,
            task="multiclass", num_classes=num_classes, 
            writer=self.writer, global_step=self.global_step
        )
        for fname in ["PR-BSM-ovr.svg", "PR-BSM-macro.svg", "PR-BSM-weighted.svg"]:
            self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, fname)))

if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)