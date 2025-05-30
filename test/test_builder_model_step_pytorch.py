import unittest
from unittest.mock import patch, Mock, MagicMock # MagicMock can be useful for chained calls
from pathlib import Path
import tempfile

# Assuming your PytorchModelStepBuilder and ModelConfig are in these locations:
# from your_module import PytorchModelStepBuilder, ModelConfig # Adjust as needed
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.pipeline_context import _StepArguments, PipelineSession # For spec'ing mock_model_create_output
from sagemaker.workflow.parameters import Parameter # For instance_type_param mock


from pipelines.config_model_step_pytorch import PytorchModelCreationConfig
from pipelines.builder_model_step_pytorch import PytorchModelStepBuilder


class TestPytorchModelStepBuilder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory that will be cleaned up automatically
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_source_dir = Path(self.temp_dir_obj.name)

        self.mock_config = Mock(spec=PytorchModelCreationConfig)
        
        self.mock_config.pipeline_name = "MyPipeline"
        # THIS IS THE KEY CHANGE: Mock 'region' which StepBuilderBase uses
        self.mock_config.region = "NA" # Or "EU", "FE" if your REGION_MAPPING uses these
        # StepBuilderBase will then convert this to an aws_region like "us-east-1"
        # If StepBuilderBase expects config.aws_region directly, then mock that instead/additionally.
        # Based on the error, it's currently: self.aws_region = self.REGION_MAPPING.get(self.config.region)
        # So, self.config.region is needed.
        # Also, ensure self.REGION_MAPPING is defined in StepBuilderBase or accessible.
        # For the test, you might need to patch REGION_MAPPING on the builder instance if it's a class attribute.

        self.mock_config.current_date = "2025-05-20"
        
        self.mock_config.source_dir = str(self.temp_source_dir)
        self.mock_config.inference_entry_point = "inference_script.py"

        (self.temp_source_dir / self.mock_config.inference_entry_point).touch()

        self.mock_config.framework_version = "2.1.0"
        self.mock_config.py_version = "py310"
        self.mock_config.inference_instance_type = "ml.m5.4xlarge"
        self.mock_config.container_startup_health_check_timeout = 300
        self.mock_config.container_memory_limit = 6144
        self.mock_config.data_download_timeout = 900
        self.mock_config.inference_memory_limit = 6144
        self.mock_config.max_concurrent_invocations = 1
        self.mock_config.max_payload_size = 6
        
        self.mock_config.get_model_name = Mock(return_value=f"{self.mock_config.pipeline_name}-model-testfixed")

        self.mock_sagemaker_session = Mock(spec=PipelineSession)
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"

        # For StepBuilderBase to have self.REGION_MAPPING:
        # If REGION_MAPPING is a class attribute of StepBuilderBase:
        # Option 1: Define it in the stub StepBuilderBase in the test file
        # Option 2: Patch it if it's complex or from elsewhere
        # For this example, let's assume the StepBuilderBase stub used by the test has it.
        # If StepBuilderBase directly uses config.aws_region, then mock that instead of config.region.
        # The error points to self.config.region, so we mock that.

        self.builder = PytorchModelStepBuilder(
            config=self.mock_config,
            sagemaker_session=self.mock_sagemaker_session,
            role=self.role
        )
        
        # If self.REGION_MAPPING is needed for the above instantiation:
        # You might need to define it in the StepBuilderBase stub used in your test file.
        # For example, in your StepBuilderBase stub:
        # class StepBuilderBase:
        #     REGION_MAPPING = {"NA": "us-east-1", "EU": "eu-west-1", "FE": "ap-northeast-1"} # Example
        #     def __init__(self, config, ...):
        #         self.config = config
        #         ...
        #         self.aws_region = self.REGION_MAPPING.get(self.config.region)


    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_create_env_config_success(self):
        env_config = self.builder._create_env_config()
        self.assertEqual(env_config['SAGEMAKER_PROGRAM'], self.mock_config.inference_entry_point)
        self.assertEqual(env_config['AWS_REGION'], self.mock_config.aws_region)
        self.assertEqual(env_config['MMS_DEFAULT_RESPONSE_TIMEOUT'], str(self.mock_config.container_startup_health_check_timeout))
        self.assertEqual(len(env_config), 10) # Verify count of env vars

    def test_builder_init_empty_entry_point_raises_error(self):
        """Test that builder __init__ (via validate_configuration) fails if entry_point is empty."""
        # Modify the mock config attribute that the builder will use
        self.mock_config.inference_entry_point = "" 
        
        with self.assertRaisesRegex(ValueError, "inference_entry_point cannot be empty"):
            # Re-instantiate builder to trigger __init__ and its validation
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        # Restore for other tests if needed (though each test should ideally use a fresh mock or setup)
        self.mock_config.inference_entry_point = "inference_script.py"


    def test_builder_init_missing_required_attribute_raises_error(self):
        """Test that builder __init__ (via validate_configuration) fails if a required attribute is None."""
        self.mock_config.framework_version = None # Set a required attribute to None

        with self.assertRaisesRegex(ValueError, "PytorchModelCreationConfig missing required attribute\(s\): framework_version"):
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        # Restore for other tests
        self.mock_config.framework_version = "2.1.0"

    def test_builder_init_missing_entry_point_file_raises_error(self):
        """Test builder __init__ fails if entry point file doesn't exist in local source_dir."""
        # Remove the dummy entry point file
        entry_point_file = self.temp_source_dir / self.mock_config.inference_entry_point
        if entry_point_file.exists():
            entry_point_file.unlink()

        with self.assertRaisesRegex(FileNotFoundError, "Builder validation: Inference entry point script not found"):
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        # Re-create for other tests
        entry_point_file.touch()


    @patch('sagemaker.image_uris.retrieve')
    def test_get_image_uri(self, mock_retrieve):
        expected_uri = "pytorch-inference:2.1.0-py310-ml.m5.4xlarge" # Example
        mock_retrieve.return_value = expected_uri
        uri = self.builder._get_image_uri()
        self.assertEqual(uri, expected_uri)
        mock_retrieve.assert_called_once_with(
            framework="pytorch",
            region=self.mock_config.aws_region,
            version=self.mock_config.framework_version,
            py_version=self.mock_config.py_version,
            instance_type=self.mock_config.inference_instance_type,
            image_scope="inference"
        )

    @patch('__main__.PyTorchModel') 
    @patch('sagemaker.workflow.parameters.Parameter')
    @patch('sagemaker.workflow.model_step.ModelStep') 
    def test_create_model_step_correct_usage(self, mock_sagemaker_model_step_class, mock_parameter_class, mock_pytorch_model_class):
        mock_model_data_s3_uri = "s3://mybucket/mymodel/model.tar.gz"
        mock_pytorch_model_instance = mock_pytorch_model_class.return_value
        mock_step_args = Mock(spec=_StepArguments)
        mock_pytorch_model_instance.create.return_value = mock_step_args
        mock_param_instance = Mock(spec=Parameter)
        mock_parameter_class.return_value = mock_param_instance

        mock_env_config = {'SAGEMAKER_PROGRAM': 'inference_script.py', 'TEST_ENV': 'true'}
        mock_image_uri = 'mocked-pytorch-image-uri'

        # Use the mocked get_model_name from setUp
        expected_model_name = self.mock_config.get_model_name()


        with patch.object(self.builder, '_get_image_uri', return_value=mock_image_uri), \
             patch.object(self.builder, '_create_env_config', return_value=mock_env_config):

            returned_step = self.builder.create_step(model_data=mock_model_data_s3_uri)

            mock_parameter_class.assert_called_once_with(
                name="InferenceInstanceType",
                default_value=self.mock_config.inference_instance_type
            )
            
            mock_pytorch_model_class.assert_called_once()
            args_pymodel, kwargs_pymodel = mock_pytorch_model_class.call_args
            
            self.assertEqual(kwargs_pymodel['name'], expected_model_name) # Check against mocked name
            self.assertEqual(kwargs_pymodel['model_data'], mock_model_data_s3_uri)
            # ... other assertions for PyTorchModel instantiation ...

            mock_pytorch_model_instance.create.assert_called_once_with(
                instance_type=mock_param_instance,
                accelerator_type=None
            )

            expected_step_name = f"{self.mock_config.pipeline_name}-CreateModel" # Based on StepBuilderBase stub
            mock_sagemaker_model_step_class.assert_called_once_with(
                name=expected_step_name,
                step_args=mock_step_args
            )
            self.assertEqual(returned_step, mock_sagemaker_model_step_class.return_value)
            self.assertEqual(returned_step.model_artifacts_path, mock_model_data_s3_uri)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)