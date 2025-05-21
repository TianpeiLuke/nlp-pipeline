import unittest
from unittest.mock import patch, Mock, MagicMock # MagicMock can be useful for chained calls
import os

# Assuming your PytorchModelStepBuilder and ModelConfig are in these locations:
# from your_module import PytorchModelStepBuilder, ModelConfig # Adjust as needed
from sagemaker.workflow.properties import Properties
from sagemaker import Session
from sagemaker.workflow.pipeline_context import _StepArguments # For spec'ing mock_model_create_output

from src.workflow.workflow_config import ModelConfig
from src.workflow.builder_model_step import PytorchModelStepBuilder


class TestPytorchModelStepBuilder(unittest.TestCase):
    def setUp(self):
        self.mock_config = Mock(spec=ModelConfig)
        self.mock_config.region = "NA"
        self.mock_config.inference_instance_type = "ml.m5.4xlarge"
        self.mock_config.framework_version = "2.1.0"
        self.mock_config.py_version = "py310"
        self.mock_config.source_dir = "./source_dir_for_inference"
        self.mock_config.current_date = "2025-05-20"
        self.mock_config.container_startup_health_check_timeout = 300
        self.mock_config.container_memory_limit = 6144
        self.mock_config.data_download_timeout = 900
        self.mock_config.inference_memory_limit = 6144
        self.mock_config.max_concurrent_invocations = 1
        self.mock_config.max_payload_size = 6
        self.mock_config.inference_entry_point = "inference.py"

        self.mock_sagemaker_session = Mock(spec=Session)
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"

        self.builder = PytorchModelStepBuilder(
            config=self.mock_config,
            sagemaker_session=self.mock_sagemaker_session,
            role=self.role
        )

    def test_create_env_config_success(self):
        env_config = self.builder._create_env_config()
        self.assertEqual(env_config['SAGEMAKER_PROGRAM'], self.mock_config.inference_entry_point)
        self.assertEqual(env_config['AWS_REGION'], "us-east-1")
        self.assertEqual(env_config['MMS_DEFAULT_RESPONSE_TIMEOUT'], "300")
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_LOG_LEVEL'], "20")
        self.assertEqual(env_config['SAGEMAKER_SUBMIT_DIRECTORY'], "/opt/ml/model/code")
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_MEMORY_LIMIT'], "6144")
        self.assertEqual(env_config['SAGEMAKER_MODEL_DATA_DOWNLOAD_TIMEOUT'], "900")
        self.assertEqual(env_config['SAGEMAKER_INFERENCE_MEMORY_LIMIT'], "6144")
        self.assertEqual(env_config['SAGEMAKER_MAX_CONCURRENT_INVOCATIONS'], "1")
        self.assertEqual(env_config['SAGEMAKER_MAX_PAYLOAD_IN_MB'], "6")
        self.assertEqual(len(env_config), 10)

    def test_create_env_config_missing_value_raises_error(self):
        """Test based on actual error: inference_entry_point cannot be empty."""
        original_entry_point = self.mock_config.inference_entry_point
        self.mock_config.inference_entry_point = "" # Set to empty

        # Expect the specific ValueError your code raises for this condition
        with self.assertRaisesRegex(ValueError, "ModelConfig must have 'inference_entry_point' defined for creating environment configuration."):
            self.builder._create_env_config()

        self.mock_config.inference_entry_point = original_entry_point # Restore

    def test_create_env_config_missing_attribute_raises_error(self):
        """Test for missing inference_entry_point attribute based on actual error."""
        config_without_entry_point = Mock(spec=ModelConfig)
        config_without_entry_point.region = "NA"
        # Set other attributes needed by _create_env_config
        config_without_entry_point.container_startup_health_check_timeout = 300
        config_without_entry_point.container_memory_limit = 6144
        config_without_entry_point.data_download_timeout = 900
        config_without_entry_point.inference_memory_limit = 6144
        config_without_entry_point.max_concurrent_invocations = 1
        config_without_entry_point.max_payload_size = 6
        # Deliberately do not set inference_entry_point. Mock will raise AttributeError on access.
        # Make the mock raise AttributeError when inference_entry_point is accessed if it's not set.
        # This simulates hasattr(self.config, 'inference_entry_point') being false
        # OR a direct access self.config.inference_entry_point raising AttributeError.
        # The PytorchModelStepBuilder's _create_env_config in the prompt seems to directly raise ValueError.
        
        # To make this Mock truly lack the attribute for a hasattr check, or to make direct access fail:
        del config_without_entry_point.inference_entry_point # Ensure it's not present

        builder_with_bad_config = PytorchModelStepBuilder(
            config=config_without_entry_point,
            sagemaker_session=self.mock_sagemaker_session,
            role=self.role
        )
        # Expect the specific ValueError your code raises
        with self.assertRaisesRegex(ValueError, "ModelConfig must have 'inference_entry_point' defined for creating environment configuration."):
            builder_with_bad_config._create_env_config()

    @patch('sagemaker.image_uris.retrieve')
    def test_get_image_uri(self, mock_retrieve):
        expected_uri = "pytorch-inference:2.1.0-py310-cpu"
        mock_retrieve.return_value = expected_uri
        uri = self.builder._get_image_uri()
        self.assertEqual(uri, expected_uri)
        mock_retrieve.assert_called_once_with(
            framework="pytorch", region="us-east-1",
            version=self.mock_config.framework_version, py_version=self.mock_config.py_version,
            instance_type=self.mock_config.inference_instance_type, image_scope="inference"
        )

    @patch('src.workflow.builder_model_step.PyTorchModel')
    @patch('src.workflow.builder_model_step.Parameter')
    @patch('src.workflow.builder_model_step.ModelStep')
    def test_create_model_step_correct_usage(self, mock_model_step_class, mock_parameter_class, mock_pytorch_model_class):
        """Test correct usage of create_model_step"""
        # Set pipeline_name in the config
        self.mock_config.pipeline_name = "MyPipeline"

        mock_model_data = Mock(spec=Properties)
        mock_pytorch_model_instance = mock_pytorch_model_class.return_value
        mock_model_create_output = Mock(spec=_StepArguments)
        mock_pytorch_model_instance.create.return_value = mock_model_create_output
        mock_parameter_instance = mock_parameter_class.return_value
        mock_parameter_instance.name = "InferenceInstanceType"
        mock_parameter_instance.default_value = self.mock_config.inference_instance_type

        # Define the mock return value for _create_env_config
        mock_env_config_return = {'SAGEMAKER_PROGRAM': 'inference.py', 'ENV_VAR': 'true'}

        with patch.object(self.builder, '_get_image_uri', return_value='mocked-image-uri') as mock_get_uri, \
            patch.object(self.builder, '_create_env_config', return_value=mock_env_config_return) as mock_get_env:
            returned_step = self.builder.create_model_step(mock_model_data)

            mock_parameter_class.assert_called_once_with(
                name="InferenceInstanceType",
                default_value=self.mock_config.inference_instance_type
            )
            expected_model_name_prefix = "bsm-rnr-model-2025-05-20"
            mock_pytorch_model_class.assert_called_once()
            args_pytorch_model, kwargs_pytorch_model = mock_pytorch_model_class.call_args
            self.assertTrue(kwargs_pytorch_model['name'].startswith(expected_model_name_prefix))
            self.assertEqual(kwargs_pytorch_model['model_data'], mock_model_data)
            self.assertEqual(kwargs_pytorch_model['role'], self.role)
            self.assertEqual(kwargs_pytorch_model['entry_point'], self.mock_config.inference_entry_point)
            self.assertEqual(kwargs_pytorch_model['source_dir'], self.mock_config.source_dir)
            self.assertEqual(kwargs_pytorch_model['env'], mock_env_config_return)
            self.assertEqual(kwargs_pytorch_model['image_uri'], 'mocked-image-uri')
            mock_pytorch_model_instance.create.assert_called_once()
            args_model_create, kwargs_model_create = mock_pytorch_model_instance.create.call_args
            self.assertEqual(kwargs_model_create['instance_type'], mock_parameter_instance)
            self.assertIsNone(kwargs_model_create['accelerator_type'])

            # Verify the ModelStep name
            expected_step_name = "MyPipeline-CreateModel"
            mock_model_step_class.assert_called_once_with(
                name=expected_step_name,
                step_args=mock_model_create_output
            )
            self.assertEqual(returned_step, mock_model_step_class.return_value)


if __name__ == '__main__':
    unittest.main()