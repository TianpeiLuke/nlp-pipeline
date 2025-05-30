import unittest
from unittest.mock import patch, Mock, MagicMock # MagicMock can be useful for chained calls
from pathlib import Path

# Assuming your PytorchModelStepBuilder and ModelConfig are in these locations:
# from your_module import PytorchModelStepBuilder, ModelConfig # Adjust as needed
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.pipeline_context import _StepArguments, PipelineSession # For spec'ing mock_model_create_output
from sagemaker.workflow.parameters import Parameter # For instance_type_param mock


from pipelines.config_model_step_pytorch import PytorchModelCreationConfig
from pipelines.builder_model_step_pytorch import PytorchModelStepBuilder


class TestPytorchModelStepBuilder(unittest.TestCase):
    def setUp(self):
        # Use PytorchModelCreationConfig for spec and provide all necessary fields
        self.mock_config = Mock(spec=PytorchModelCreationConfig)
        
        # Attributes from BasePipelineConfig
        self.mock_config.pipeline_name = "MyPipeline" # Used in _get_step_name
        self.mock_config.aws_region = "us-east-1"     # Used by builder.aws_region
        self.mock_config.current_date = "2025-05-20"  # Used in _create_pytorch_model for model name
        self.mock_config.source_dir = "./source_dir_for_inference" # Used by builder
        self.mock_config.framework_version = "2.1.0" # Used by builder
        self.mock_config.py_version = "py310"         # Used by builder

        # Attributes from PytorchModelCreationConfig
        self.mock_config.inference_instance_type = "ml.m5.4xlarge"
        self.mock_config.inference_entry_point = "inference.py"
        self.mock_config.initial_instance_count = 1 # Not directly used by builder, but part of config
        self.mock_config.container_startup_health_check_timeout = 300
        self.mock_config.container_memory_limit = 6144
        self.mock_config.data_download_timeout = 900
        self.mock_config.inference_memory_limit = 6144
        self.mock_config.max_concurrent_invocations = 1
        self.mock_config.max_payload_size = 6

        self.mock_sagemaker_session = Mock(spec=PipelineSession) # Use PipelineSession for spec
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"

        # Create a dummy source_dir and entry_point for Pydantic validation if source_dir is local
        self.source_dir_path = Path(self.mock_config.source_dir)
        self.entry_point_file_path = self.source_dir_path / self.mock_config.inference_entry_point
        self.source_dir_path.mkdir(parents=True, exist_ok=True)
        self.entry_point_file_path.touch(exist_ok=True)

        # Instantiate the builder AFTER setting up the config mock
        self.builder = PytorchModelStepBuilder(
            config=self.mock_config,
            sagemaker_session=self.mock_sagemaker_session,
            role=self.role
        )
    
    def tearDown(self):
        # Clean up dummy files and directory
        if self.entry_point_file_path.exists():
            self.entry_point_file_path.unlink()
        if self.source_dir_path.exists():
            # Ensure it's empty before trying to remove if other files were created
            try:
                self.source_dir_path.rmdir()
            except OSError:
                # Directory might not be empty if other tests create files
                pass


    def test_create_env_config_success(self):
        env_config = self.builder._create_env_config()
        self.assertEqual(env_config['SAGEMAKER_PROGRAM'], self.mock_config.inference_entry_point)
        self.assertEqual(env_config['AWS_REGION'], self.mock_config.aws_region) # Use mocked region
        self.assertEqual(env_config['MMS_DEFAULT_RESPONSE_TIMEOUT'], str(self.mock_config.container_startup_health_check_timeout))
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_LOG_LEVEL'], "20")
        self.assertEqual(env_config['SAGEMAKER_SUBMIT_DIRECTORY'], "/opt/ml/model/code")
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_MEMORY_LIMIT'], str(self.mock_config.container_memory_limit))
        self.assertEqual(env_config['SAGEMAKER_MODEL_DATA_DOWNLOAD_TIMEOUT'], str(self.mock_config.data_download_timeout))
        self.assertEqual(env_config['SAGEMAKER_INFERENCE_MEMORY_LIMIT'], str(self.mock_config.inference_memory_limit))
        self.assertEqual(env_config['SAGEMAKER_MAX_CONCURRENT_INVOCATIONS'], str(self.mock_config.max_concurrent_invocations))
        self.assertEqual(env_config['SAGEMAKER_MAX_PAYLOAD_IN_MB'], str(self.mock_config.max_payload_size))
        self.assertEqual(len(env_config), 10)

    def test_builder_init_empty_entry_point_raises_error(self):
        """Test that builder __init__ (via validate_configuration) fails if entry_point is empty."""
        self.mock_config.inference_entry_point = "" 
        with self.assertRaisesRegex(ValueError, "inference_entry_point cannot be empty"):
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        # Restore for other tests if mock_config is shared and mutable (it is here)
        self.mock_config.inference_entry_point = "inference.py"


    def test_builder_init_missing_attribute_raises_error(self):
        """Test that builder __init__ (via validate_configuration) fails if an attribute is missing."""
        # Create a new mock config that is truly missing an attribute
        config_missing_attr = Mock(spec=PytorchModelCreationConfig)
        # Set all other required attributes from self.mock_config
        for attr in ['pipeline_name', 'aws_region', 'current_date', 'source_dir', 
                     'framework_version', 'py_version', 'inference_instance_type',
                     'container_startup_health_check_timeout', 'container_memory_limit',
                     'data_download_timeout', 'inference_memory_limit',
                     'max_concurrent_invocations', 'max_payload_size']:
            setattr(config_missing_attr, attr, getattr(self.mock_config, attr))
        # 'inference_entry_point' is deliberately NOT set on config_missing_attr
        # To make hasattr return False, we can make the attribute a MagicMock that returns False for __bool__
        # or ensure it's not in the mock's _mock_children, but Pydantic spec helps here.
        # If spec is used, accessing a non-existent attribute raises AttributeError.
        # The builder's validate_configuration uses hasattr.

        # To truly simulate missing attribute for hasattr, we can't just not set it on a standard Mock.
        # A more direct way is to test validate_configuration with a dict-like object.
        # However, for Mock(spec=...), if an attribute is not set and accessed, it raises AttributeError.
        # The builder's check is `if not hasattr(self.config, attr) or getattr(self.config, attr) is None:`.
        # So we can set it to None.
        config_missing_attr.inference_entry_point = None


        with self.assertRaisesRegex(ValueError, "PytorchModelCreationConfig missing required attribute\(s\): inference_entry_point"):
            PytorchModelStepBuilder(
                config=config_missing_attr,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )

    @patch('sagemaker.image_uris.retrieve') # Path to where image_uris is used by the builder
    def test_get_image_uri(self, mock_retrieve):
        expected_uri = "pytorch-inference:2.1.0-py310-cpu" # Example
        mock_retrieve.return_value = expected_uri
        uri = self.builder._get_image_uri()
        self.assertEqual(uri, expected_uri)
        mock_retrieve.assert_called_once_with(
            framework="pytorch",
            region=self.mock_config.aws_region, # Use mocked region
            version=self.mock_config.framework_version,
            py_version=self.mock_config.py_version,
            instance_type=self.mock_config.inference_instance_type,
            image_scope="inference"
        )

    # Update patch paths based on where PyTorchModel, Parameter, ModelStep are imported
    # in your PytorchModelStepBuilder file.
    @patch('__main__.PyTorchModel') # Assuming PyTorchModel is defined/imported in the same file as the test (due to current context)
    @patch('sagemaker.workflow.parameters.Parameter') # Correct path for Parameter
    @patch('sagemaker.workflow.model_step.ModelStep') # Correct path for ModelStep (SagemakerModelStep in builder)
    def test_create_model_step_correct_usage(self, mock_sagemaker_model_step_class, mock_parameter_class, mock_pytorch_model_class):
        """Test correct usage of create_step (which create_model_step calls)"""
        # mock_config.pipeline_name is set in setUp
        
        # model_data can be a string (S3 URI) or a Properties object
        mock_model_data_s3_uri = "s3://mybucket/mymodel/model.tar.gz"

        mock_pytorch_model_instance = mock_pytorch_model_class.return_value
        # Ensure model_data is passed to create() if it's an S3 URI,
        # or if it's Properties, it's handled correctly by SageMaker SDK.
        # The model.create() method in SageMaker SDK usually doesn't take model_data directly.
        # model_data is part of the PyTorchModel constructor.
        
        mock_step_args = Mock(spec=_StepArguments) # From sagemaker.workflow.pipeline_context
        mock_pytorch_model_instance.create.return_value = mock_step_args
        
        # Mock the Parameter class return value
        mock_param_instance = Mock(spec=Parameter)
        mock_parameter_class.return_value = mock_param_instance

        # Define the mock return value for _create_env_config and _get_image_uri
        mock_env_config = {'SAGEMAKER_PROGRAM': 'inference.py', 'TEST_ENV': 'true'}
        mock_image_uri = 'mocked-pytorch-image-uri'

        with patch.object(self.builder, '_get_image_uri', return_value=mock_image_uri) as mock_get_uri_method, \
             patch.object(self.builder, '_create_env_config', return_value=mock_env_config) as mock_create_env_method:

            returned_step = self.builder.create_step(model_data=mock_model_data_s3_uri)

            # 1. Check Parameter creation
            mock_parameter_class.assert_called_once_with(
                name="InferenceInstanceType",
                default_value=self.mock_config.inference_instance_type
            )

            # 2. Check PyTorchModel instantiation
            expected_model_name_prefix = f"bsm-rnr-model-{self.mock_config.current_date.replace('-', '')}"
            mock_pytorch_model_class.assert_called_once()
            args_pymodel, kwargs_pymodel = mock_pytorch_model_class.call_args
            
            self.assertTrue(kwargs_pymodel['name'].startswith(expected_model_name_prefix))
            self.assertEqual(kwargs_pymodel['model_data'], mock_model_data_s3_uri)
            self.assertEqual(kwargs_pymodel['role'], self.role)
            self.assertEqual(kwargs_pymodel['entry_point'], self.mock_config.inference_entry_point)
            self.assertEqual(kwargs_pymodel['source_dir'], self.mock_config.source_dir)
            self.assertEqual(kwargs_pymodel['framework_version'], self.mock_config.framework_version)
            self.assertEqual(kwargs_pymodel['py_version'], self.mock_config.py_version)
            self.assertEqual(kwargs_pymodel['sagemaker_session'], self.mock_sagemaker_session)
            self.assertEqual(kwargs_pymodel['env'], mock_env_config)
            self.assertEqual(kwargs_pymodel['image_uri'], mock_image_uri)

            # 3. Check model.create() call
            mock_pytorch_model_instance.create.assert_called_once_with(
                instance_type=mock_param_instance, # Should be the Parameter instance
                accelerator_type=None
            )

            # 4. Check ModelStep instantiation
            # Assuming _get_step_name is: f"{self.config.pipeline_name}-Create{step_prefix}"
            expected_step_name = f"{self.mock_config.pipeline_name}-CreateModel"
            mock_sagemaker_model_step_class.assert_called_once_with(
                name=expected_step_name,
                step_args=mock_step_args
            )
            self.assertEqual(returned_step, mock_sagemaker_model_step_class.return_value)
            self.assertEqual(returned_step.model_artifacts_path, mock_model_data_s3_uri)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)