import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep

# Import the builder class
from src.pipeline_steps.builder_training_step_pytorch import PyTorchTrainingStepBuilder

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
        # Add input_names and output_names
        self.config.input_names = {"input_path": "data"}
        self.config.output_names = {
            "model_output": "ModelArtifacts",
            "metrics_output": "TrainingMetrics",
            "training_job_name": "TrainingJobName"
        }

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

    def test_normalize_s3_uri(self):
        """Test that S3 URIs are correctly normalized."""
        # Add S3PathHandler mock
        with patch('src.pipeline_steps.builder_training_step_pytorch.S3PathHandler') as mock_handler:
            mock_handler.normalize.return_value = 's3://bucket/path'
            
            # Test with trailing slash
            uri = 's3://bucket/path/'
            normalized = self.builder._normalize_s3_uri(uri)
            self.assertEqual(normalized, 's3://bucket/path')
            mock_handler.normalize.assert_called_with(uri, "S3 URI")
            
            # Test with PipelineVariable
            pipeline_var = MagicMock()
            pipeline_var.expr = 's3://bucket/path/'
            normalized = self.builder._normalize_s3_uri(pipeline_var)
            self.assertEqual(normalized, 's3://bucket/path')
            mock_handler.normalize.assert_called_with('s3://bucket/path/', "S3 URI")
            
            # Test with Get expression
            get_expr = {'Get': 'Steps.ProcessingStep.ProcessingOutputConfig.Outputs["Output"].S3Output.S3Uri'}
            normalized = self.builder._normalize_s3_uri(get_expr)
            self.assertEqual(normalized, get_expr)
            # Should not call normalize for Get expressions
            mock_handler.normalize.assert_called_with('s3://bucket/path/', "S3 URI")

    @patch('src.pipeline_steps.builder_training_step_pytorch.TrainingInput')
    def test_get_training_inputs(self, mock_training_input_cls):
        """Test that training inputs are correctly constructed."""
        # Setup mock
        mock_training_input_cls.side_effect = lambda s3_data: f"TI:{s3_data}"
        
        # Test with input_path in inputs
        inputs = {
            "input_path": "s3://bucket/input"
        }
        
        # Mock S3PathHandler methods
        with patch('src.pipeline_steps.builder_training_step_pytorch.S3PathHandler') as mock_handler:
            mock_handler.normalize.return_value = 's3://bucket/input'
            mock_handler.is_valid.return_value = True
            
            training_inputs = self.builder._get_training_inputs(inputs)
            
            # Check that data channel is created
            self.assertIn('data', training_inputs)
            self.assertEqual(training_inputs['data'], "TI:s3://bucket/input")
            
        # Test with input_path in config
        self.builder.config.input_path = 's3://bucket/config_input'
        
        with patch('src.pipeline_steps.builder_training_step_pytorch.S3PathHandler') as mock_handler:
            mock_handler.normalize.return_value = 's3://bucket/config_input'
            mock_handler.is_valid.return_value = True
            
            training_inputs = self.builder._get_training_inputs({})
            
            # Check that data channel is created from config
            self.assertIn('data', training_inputs)
            self.assertEqual(training_inputs['data'], "TI:s3://bucket/config_input")

    def test_match_tabular_preprocessing_outputs(self):
        """Test that outputs from TabularPreprocessingStep are correctly matched."""
        # Create a mock TabularPreprocessingStep
        step = MagicMock()
        
        # Setup outputs for the step
        output = MagicMock()
        output.output_name = 'processed_data'
        output.destination = 's3://bucket/processed_data'
        step.outputs = [output]
        
        # Call _match_tabular_preprocessing_outputs
        inputs = {}
        matched = self.builder._match_tabular_preprocessing_outputs(inputs, step)
        
        # Check that inputs are correctly matched
        self.assertIn('inputs', matched)
        self.assertIn('input_path', inputs.get('inputs', {}))
        self.assertEqual(inputs['inputs']['input_path'], 's3://bucket/processed_data')

    @patch('src.pipeline_steps.builder_training_step_pytorch.PyTorch')
    @patch('src.pipeline_steps.builder_training_step_pytorch.TrainingInput')
    def test_create_step_without_checkpoint(self, mock_training_input_cls, mock_pytorch_cls):
        # Simulate no existing checkpoint
        self.config.has_checkpoint = lambda: False
        # Setup TrainingInput stubs
        mock_training_input_cls.side_effect = lambda s3_data: f"TI:{s3_data}"
        # Stub estimator
        estimator = MagicMock()
        mock_pytorch_cls.return_value = estimator
        deps = ['step1', 'step2']

        step = self.builder.create_step(dependencies=deps)

        # Verify TrainingInput called for data path
        values = step.inputs.values()
        self.assertEqual(len(values), 1)
        self.assertTrue(all(isinstance(v, str) and v.startswith('TI:') for v in values))

        # PyTorch estimator instantiation uses fallback checkpoint
        mock_pytorch_cls.assert_called_once()
        # TrainingStep attributes
        self.assertIsInstance(step, TrainingStep)
        self.assertEqual(step.estimator, estimator)
        self.assertEqual(step.depends_on, deps)
        expected_name = self.builder._get_step_name('PyTorchTraining')
        self.assertEqual(step.name, expected_name)

    @patch('src.pipeline_steps.builder_training_step_pytorch.PyTorch')
    @patch('src.pipeline_steps.builder_training_step_pytorch.TrainingInput')
    def test_create_step_with_checkpoint(self, mock_training_input_cls, mock_pytorch_cls):
        # Simulate existing checkpoint
        self.config.has_checkpoint = lambda: True
        self.config.get_checkpoint_uri = lambda: 's3://bucket/ckpt'
        mock_training_input_cls.side_effect = lambda s3_data: f"TI:{s3_data}"
        estimator = MagicMock()
        mock_pytorch_cls.return_value = estimator

        step = self.builder.create_step()

        # Ensure TrainingInput still called
        mock_training_input_cls.assert_called()
        # Ensure estimator got the checkpoint argument
        _, kwargs = mock_pytorch_cls.call_args
        self.assertEqual(kwargs.get('checkpoint_s3_uri'), 's3://bucket/ckpt')

        self.assertIsInstance(step, TrainingStep)

    @patch('src.pipeline_steps.builder_training_step_pytorch.PyTorch')
    @patch('src.pipeline_steps.builder_training_step_pytorch.TrainingInput')
    def test_create_step_with_dependencies(self, mock_training_input_cls, mock_pytorch_cls):
        """Test creating a step with dependencies that provide inputs."""
        # Setup mocks
        mock_estimator = MagicMock()
        mock_pytorch_cls.return_value = mock_estimator
        mock_training_input_cls.side_effect = lambda s3_data: f"TI:{s3_data}"
        
        # Create a mock dependency step
        dep_step = MagicMock()
        dep_step.name = 'TabularPreprocessingStep'
        
        # Setup outputs for the dependency step
        output = MagicMock()
        output.output_name = 'processed_data'
        output.destination = 's3://bucket/processed_data'
        dep_step.outputs = [output]
        
        # Call create_step with the dependency
        with patch('src.pipeline_steps.builder_training_step_pytorch.S3PathHandler') as mock_handler:
            mock_handler.normalize.return_value = 's3://bucket/processed_data'
            mock_handler.is_valid.return_value = True
            
            step = self.builder.create_step(dependencies=[dep_step])
        
        # Verify the step is created correctly
        self.assertIsInstance(step, TrainingStep)
        self.assertEqual(step.estimator, mock_estimator)
        self.assertEqual(step.depends_on, [dep_step])
        
        # Verify that _match_custom_properties was called and inputs were extracted
        self.assertIn('data', step.inputs)
        self.assertEqual(step.inputs['data'], "TI:s3://bucket/processed_data")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
