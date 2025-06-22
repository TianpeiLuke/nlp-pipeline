import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep
from sagemaker.inputs import TransformInput

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_batch_transform_step import BatchTransformStepBuilder
from src.pipeline_steps.config_batch_transform_step import BatchTransformStepConfig

class TestBatchTransformStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        # BasePipelineConfig expected attributes
        self.config.job_type = 'training'
        self.config.batch_input_location = 's3://bucket/input'
        self.config.batch_output_location = 's3://bucket/output'
        self.config.transform_instance_type = 'ml.m5.large'
        self.config.transform_instance_count = 1
        self.config.content_type = 'text/csv'
        self.config.accept = 'text/csv'
        self.config.split_type = 'Line'
        self.config.assemble_with = 'Line'
        self.config.input_filter = '$[1:]'
        self.config.output_filter = '$[-1]'
        self.config.join_source = 'Input'
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(BatchTransformStepBuilder)
        self.builder.config = self.config
        
        # Create a properly configured session mock
        session_mock = MagicMock()
        # Set sagemaker_config to an empty dict to pass validation
        session_mock.sagemaker_config = {}
        self.builder.session = session_mock
        
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        # Use a mock for the _get_step_name method from the base class
        self.builder._get_step_name = MagicMock(return_value='BatchTransform')

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_invalid_job_type(self):
        """Test that configuration validation fails with invalid job type."""
        self.config.job_type = 'invalid'
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()

    @patch('src.pipelines.builder_batch_transform_step.Transformer')
    def test_create_transformer(self, mock_transformer_cls):
        """Test that the transformer is created with the correct parameters."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        model_name = 'test-model'
        transformer = self.builder._create_transformer(model_name)
        
        # Verify Transformer was created with correct parameters
        mock_transformer_cls.assert_called_once_with(
            model_name=model_name,
            instance_type=self.config.transform_instance_type,
            instance_count=self.config.transform_instance_count,
            output_path=self.config.batch_output_location,
            accept=self.config.accept,
            assemble_with=self.config.assemble_with,
            sagemaker_session=self.builder.session
        )
        
        # Verify the returned transformer is our mock
        self.assertEqual(transformer, mock_transformer)

    @patch('src.pipelines.builder_batch_transform_step.Transformer')
    def test_create_step(self, mock_transformer_cls):
        """Test that the transform step is created with the correct parameters."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Create step
        model_name = 'test-model'
        step = self.builder.create_step(model_name=model_name)
        
        # Verify the TransformStep
        self.assertIsInstance(step, TransformStep)
        self.assertEqual(step.name, 'BatchTransform-Training')
        self.assertEqual(step.transformer, mock_transformer)
        
        # Verify the transform input
        self.assertIsInstance(step.inputs, TransformInput)
        self.assertEqual(step.inputs.data, self.config.batch_input_location)
        self.assertEqual(step.inputs.content_type, self.config.content_type)
        self.assertEqual(step.inputs.split_type, self.config.split_type)
        self.assertEqual(step.inputs.join_source, self.config.join_source)
        self.assertEqual(step.inputs.input_filter, self.config.input_filter)
        self.assertEqual(step.inputs.output_filter, self.config.output_filter)
        
        # Verify dependencies
        self.assertEqual(step.depends_on, [])
        
    @patch('src.pipelines.builder_batch_transform_step.Transformer')
    def test_create_step_with_dependencies(self, mock_transformer_cls):
        """Test that the transform step is created with dependencies."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies
        model_name = 'test-model'
        step = self.builder.create_step(model_name=model_name, dependencies=dependencies)
        
        # Verify dependencies
        self.assertEqual(step.depends_on, dependencies)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
