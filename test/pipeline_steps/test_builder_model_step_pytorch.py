import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
import datetime

from sagemaker.pytorch import PyTorchModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import Parameter

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_model_step_pytorch import PytorchModelStepBuilder
from src.pipeline_steps.config_model_step_pytorch import PytorchModelCreationConfig

class TestPytorchModelStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        
        # Required attributes for validation
        self.config.inference_entry_point = 'inference.py'
        self.config.source_dir = '/path/to/inference_scripts'
        self.config.inference_instance_type = 'ml.m5.large'
        self.config.framework_version = '1.10.0'
        self.config.py_version = 'py38'
        self.config.container_startup_health_check_timeout = 300
        self.config.container_memory_limit = 6144
        self.config.data_download_timeout = 900
        self.config.inference_memory_limit = 6144
        self.config.max_concurrent_invocations = 10
        self.config.max_payload_size = 6
        self.config.current_date = '2025-06-20T12:00:00Z'
        
        # Mock Path.exists to return True for script validation
        self.path_exists_patch = patch('pathlib.Path.exists', return_value=True)
        self.path_exists_patch.start()
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(PytorchModelStepBuilder)
        self.builder.config = self.config
        
        # Create a properly configured session mock
        session_mock = MagicMock()
        session_mock.sagemaker_config = {}
        self.builder.session = session_mock
        
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        self.builder.aws_region = 'us-west-2'
        
        # Mock methods from the base class
        self.builder._get_step_name = MagicMock(return_value='PytorchModel')

    def tearDown(self):
        """Clean up after each test."""
        self.path_exists_patch.stop()
        
    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_attribute(self):
        """Test that configuration validation fails with missing required attribute."""
        # Save original value
        original_value = self.config.inference_entry_point
        # Set to None to trigger validation error
        self.config.inference_entry_point = None
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original value
        self.config.inference_entry_point = original_value

    @patch('src.pipelines.builder_model_step_pytorch.image_uris')
    def test_get_image_uri(self, mock_image_uris):
        """Test that the image URI is retrieved correctly."""
        # Setup mock
        expected_uri = 'amazonaws.com/pytorch-inference:1.10.0-cpu-py38'
        mock_image_uris.retrieve.return_value = expected_uri
        
        # Get image URI
        image_uri = self.builder._get_image_uri()
        
        # Verify image_uris.retrieve was called with correct parameters
        mock_image_uris.retrieve.assert_called_once_with(
            framework="pytorch",
            region=self.builder.aws_region,
            version=self.config.framework_version,
            py_version=self.config.py_version,
            instance_type=self.config.inference_instance_type,
            image_scope="inference"
        )
        
        # Verify the returned URI is our mock
        self.assertEqual(image_uri, expected_uri)

    def test_create_env_config(self):
        """Test that environment configuration is created correctly."""
        env_config = self.builder._create_env_config()
        
        # Verify all required environment variables are present
        self.assertEqual(env_config['MMS_DEFAULT_RESPONSE_TIMEOUT'], '300')
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_LOG_LEVEL'], '20')
        self.assertEqual(env_config['SAGEMAKER_PROGRAM'], 'inference.py')
        self.assertEqual(env_config['SAGEMAKER_SUBMIT_DIRECTORY'], '/opt/ml/model/code')
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_MEMORY_LIMIT'], '6144')
        self.assertEqual(env_config['SAGEMAKER_MODEL_DATA_DOWNLOAD_TIMEOUT'], '900')
        self.assertEqual(env_config['SAGEMAKER_INFERENCE_MEMORY_LIMIT'], '6144')
        self.assertEqual(env_config['SAGEMAKER_MAX_CONCURRENT_INVOCATIONS'], '10')
        self.assertEqual(env_config['SAGEMAKER_MAX_PAYLOAD_IN_MB'], '6')
        self.assertEqual(env_config['AWS_REGION'], 'us-west-2')

    def test_create_env_config_missing_critical_value(self):
        """Test that _create_env_config raises ValueError with missing critical value."""
        # Save original value
        original_value = self.config.inference_entry_point
        # Set to empty string to trigger validation error
        self.config.inference_entry_point = ''
        
        with self.assertRaises(ValueError):
            self.builder._create_env_config()
            
        # Restore original value
        self.config.inference_entry_point = original_value

    @patch('src.pipelines.builder_model_step_pytorch.PyTorchModel')
    @patch('src.pipelines.builder_model_step_pytorch.image_uris')
    def test_create_pytorch_model(self, mock_image_uris, mock_pytorch_model_cls):
        """Test that the PyTorch model is created with the correct parameters."""
        # Setup mocks
        mock_image_uri = 'amazonaws.com/pytorch-inference:1.10.0-cpu-py38'
        mock_image_uris.retrieve.return_value = mock_image_uri
        
        mock_model = MagicMock()
        mock_pytorch_model_cls.return_value = mock_model
        
        # Create model
        model_data = 's3://bucket/model.tar.gz'
        model = self.builder._create_pytorch_model(model_data)
        
        # Verify PyTorchModel was created with correct parameters
        mock_pytorch_model_cls.assert_called_once()
        call_args = mock_pytorch_model_cls.call_args[1]
        
        # Check model name format (should contain 'bsm-rnr-model-' and date)
        self.assertTrue(call_args['name'].startswith('bsm-rnr-model-'))
        self.assertTrue(len(call_args['name']) <= 63)  # SageMaker name length limit
        
        self.assertEqual(call_args['model_data'], model_data)
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['entry_point'], 'inference.py')
        self.assertEqual(call_args['source_dir'], '/path/to/inference_scripts')
        self.assertEqual(call_args['framework_version'], '1.10.0')
        self.assertEqual(call_args['py_version'], 'py38')
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertEqual(call_args['image_uri'], mock_image_uri)
        
        # Verify environment variables
        env_vars = call_args['env']
        self.assertEqual(env_vars['SAGEMAKER_PROGRAM'], 'inference.py')
        self.assertEqual(env_vars['SAGEMAKER_CONTAINER_MEMORY_LIMIT'], '6144')
        
        # Verify the returned model is our mock
        self.assertEqual(model, mock_model)

    @patch('src.pipelines.builder_model_step_pytorch.PyTorchModel')
    @patch('src.pipelines.builder_model_step_pytorch.ModelStep')
    @patch('src.pipelines.builder_model_step_pytorch.Parameter')
    @patch('src.pipelines.builder_model_step_pytorch.image_uris')
    def test_create_step(self, mock_image_uris, mock_parameter_cls, mock_model_step_cls, mock_pytorch_model_cls):
        """Test that the model step is created with the correct parameters."""
        # Setup mocks
        mock_image_uri = 'amazonaws.com/pytorch-inference:1.10.0-cpu-py38'
        mock_image_uris.retrieve.return_value = mock_image_uri
        
        mock_model = MagicMock()
        mock_pytorch_model_cls.return_value = mock_model
        
        mock_parameter = MagicMock()
        mock_parameter_cls.return_value = mock_parameter
        
        mock_step_args = MagicMock()
        mock_model.create.return_value = mock_step_args
        
        mock_step = MagicMock()
        mock_model_step_cls.return_value = mock_step
        
        # Create step
        model_data = 's3://bucket/model.tar.gz'
        step = self.builder.create_step(model_data)
        
        # Verify Parameter was created with correct parameters
        mock_parameter_cls.assert_called_once_with(
            name="InferenceInstanceType",
            default_value='ml.m5.large'
        )
        
        # Verify model.create was called with correct parameters
        mock_model.create.assert_called_once_with(
            instance_type=mock_parameter,
            accelerator_type=None
        )
        
        # Verify ModelStep was created with correct parameters
        mock_model_step_cls.assert_called_once_with(
            name='PytorchModel',
            step_args=mock_step_args
        )
        
        # Verify model_artifacts_path was set correctly
        self.assertEqual(mock_step.model_artifacts_path, model_data)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipelines.builder_model_step_pytorch.PyTorchModel')
    @patch('src.pipelines.builder_model_step_pytorch.ModelStep')
    @patch('src.pipelines.builder_model_step_pytorch.Parameter')
    @patch('src.pipelines.builder_model_step_pytorch.image_uris')
    def test_create_step_with_dependencies(self, mock_image_uris, mock_parameter_cls, mock_model_step_cls, mock_pytorch_model_cls):
        """Test that the model step is created with dependencies."""
        # Setup mocks
        mock_image_uri = 'amazonaws.com/pytorch-inference:1.10.0-cpu-py38'
        mock_image_uris.retrieve.return_value = mock_image_uri
        
        mock_model = MagicMock()
        mock_pytorch_model_cls.return_value = mock_model
        
        mock_parameter = MagicMock()
        mock_parameter_cls.return_value = mock_parameter
        
        mock_step_args = MagicMock()
        mock_model.create.return_value = mock_step_args
        
        mock_step = MagicMock()
        mock_model_step_cls.return_value = mock_step
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies
        model_data = 's3://bucket/model.tar.gz'
        step = self.builder.create_step(model_data, dependencies=dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
        # Note: Dependencies are not directly passed to ModelStep in the implementation,
        # so we don't need to verify that here. This test is mainly to ensure the method
        # accepts dependencies parameter without error.

    def test_create_model_step_backward_compatibility(self):
        """Test that the old create_model_step method calls the new create_step method."""
        with patch.object(self.builder, 'create_step', return_value="step_created") as mock_create_step:
            # Call the old method
            result = self.builder.create_model_step(
                model_data='s3://bucket/model.tar.gz'
            )
            # Verify it called the new method
            mock_create_step.assert_called_once_with(
                's3://bucket/model.tar.gz'
            )
            self.assertEqual(result, "step_created")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
