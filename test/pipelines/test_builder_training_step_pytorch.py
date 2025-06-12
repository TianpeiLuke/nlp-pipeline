import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

# Import the builder class
from src.pipelines.builder_training_step_pytorch import PyTorchTrainingStepBuilder

class TestPyTorchTrainingStepBuilder(unittest.TestCase):
    def setUp(self):
        # Build a minimal config namespace
        self.config = SimpleNamespace()
        self.config.region = 'NA'
        self.config.current_date = '20250610'
        # Hyperparameters stub
        hp = SimpleNamespace()
        hp.get_config = lambda: {'param': 'value'}
        hp.serialize_config = lambda: {'param': 'value'}
        self.config.hyperparameters = hp
        # Required attributes
        self.config.training_entry_point = 'train.py'
        self.config.source_dir = 'src'
        self.config.training_instance_type = 'ml.p3.2xlarge'
        self.config.training_instance_count = 1
        self.config.training_volume_size = 50
        self.config.framework_version = '1.12.0'
        self.config.py_version = 'py38'
        self.config.input_path = 's3://bucket/input'
        self.config.output_path = 's3://bucket/output'
        # Mirror for logger usage
        self.config.instance_type = self.config.training_instance_type
        # Checkpoint logic
        self.config.has_checkpoint = lambda: False
        self.config.get_checkpoint_uri = lambda: 'unused'

        # Instantiate builder bypassing __init__
        self.builder = object.__new__(PyTorchTrainingStepBuilder)
        self.builder.config = self.config
        self.builder.session = MagicMock()
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        self.builder.aws_region = PyTorchTrainingStepBuilder.REGION_MAPPING['NA']

    def test_validate_configuration_missing_attr(self):
        # Missing required attrs should raise
        cfg2 = SimpleNamespace(region='NA')
        builder2 = object.__new__(PyTorchTrainingStepBuilder)
        builder2.config = cfg2
        builder2.session = None
        builder2.role = None
        builder2.notebook_root = Path('.')
        builder2.aws_region = builder2.REGION_MAPPING['NA']
        with self.assertRaises(ValueError):
            builder2.validate_configuration()

    def test_get_profiler_config(self):
        cfg = self.builder._create_profiler_config()
        self.assertEqual(cfg.system_monitor_interval_millis, 1000)

    def test_get_metric_definitions(self):
        metrics = self.builder._get_metric_definitions()
        names = [m['Name'] for m in metrics]
        expected = ['Train Loss', 'Validation Loss', 'Validation F1 Score', 'Validation AUC ROC']
        self.assertEqual(names, expected)

    def test_get_checkpoint_uri_default(self):
        # When no checkpoint present, fallback uses output_path, 'checkpoints', current_date
        self.config.has_checkpoint = lambda: False
        uri = self.builder._get_checkpoint_uri()
        self.assertEqual(uri, f"{self.config.output_path}/checkpoints/{self.config.current_date}")

    @patch('src.pipelines.builder_training_step_pytorch.PyTorch')
    @patch('src.pipelines.builder_training_step_pytorch.TrainingInput')
    def test_create_step_without_checkpoint(self, mock_training_input_cls, mock_pytorch_cls):
        # Simulate no existing checkpoint
        self.config.has_checkpoint = lambda: False
        # Setup TrainingInput stubs
        mock_training_input_cls.side_effect = lambda path: f"TI:{path}"
        # Stub estimator
        estimator = MagicMock()
        mock_pytorch_cls.return_value = estimator
        deps = ['step1', 'step2']

        step = self.builder.create_step(dependencies=deps)

        # Verify TrainingInput called for train, val, test paths
        values = step.inputs.values()
        self.assertEqual(len(values), 3)
        self.assertTrue(all(isinstance(v, str) and v.startswith('TI:') for v in values))

        # PyTorch estimator instantiation uses fallback checkpoint
        mock_pytorch_cls.assert_called_once()
        # TrainingStep attributes
        self.assertIsInstance(step, TrainingStep)
        self.assertEqual(step.estimator, estimator)
        self.assertEqual(step.depends_on, deps)
        expected_name = self.builder._get_step_name('PytorchTraining')
        self.assertEqual(step.name, expected_name)

    @patch('src.pipelines.builder_training_step_pytorch.PyTorch')
    @patch('src.pipelines.builder_training_step_pytorch.TrainingInput')
    def test_create_step_with_checkpoint(self, mock_training_input_cls, mock_pytorch_cls):
        # Simulate existing checkpoint
        self.config.has_checkpoint = lambda: True
        self.config.get_checkpoint_uri = lambda: 's3://bucket/ckpt'
        mock_training_input_cls.side_effect = lambda path: f"TI:{path}"
        estimator = MagicMock()
        mock_pytorch_cls.return_value = estimator

        step = self.builder.create_step()

        # Ensure TrainingInput still called
        mock_training_input_cls.assert_called()
        # Ensure estimator got the checkpoint argument
        _, kwargs = mock_pytorch_cls.call_args
        self.assertEqual(kwargs.get('checkpoint_s3_uri'), 's3://bucket/ckpt')

        self.assertIsInstance(step, TrainingStep)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
