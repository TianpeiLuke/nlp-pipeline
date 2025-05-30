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
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_source_dir = Path(self.temp_dir_obj.name)

        # Use a real PytorchModelCreationConfig instance or a mock that closely adheres to its spec
        # For simplicity in controlling attributes for error tests, Mock is used here.
        self.mock_config = Mock(spec=PytorchModelCreationConfig)
        
        self.mock_config.pipeline_name = "MyPipeline"
        self.mock_config.region = "NA"  # For StepBuilderBase to map to aws_region
        self.mock_config.aws_region = "us-east-1" # Explicitly set for direct access if needed by config logic
                                                 # Ensure this is consistent with REGION_MAPPING["NA"]
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
        
        # Mock the get_model_name method directly on the mock_config object
        self.mock_config.get_model_name = Mock(return_value=f"{self.mock_config.pipeline_name}-model-{self.mock_config.current_date.replace('-', '')}fixed")


        self.mock_sagemaker_session = Mock(spec=PipelineSession)
        self.role = "arn:aws:iam::123456789012:role/SageMakerRole"

        self.builder = PytorchModelStepBuilder(
            config=self.mock_config,
            sagemaker_session=self.mock_sagemaker_session,
            role=self.role
        )
    
    def tearDown(self):
        self.temp_dir_obj.cleanup()

    def test_create_env_config_success(self):
        env_config = self.builder._create_env_config()
        self.assertEqual(env_config['SAGEMAKER_PROGRAM'], self.mock_config.inference_entry_point)
        self.assertEqual(env_config['AWS_REGION'], self.builder.aws_region) # Assert against builder's derived region
        self.assertEqual(env_config['MMS_DEFAULT_RESPONSE_TIMEOUT'], str(self.mock_config.container_startup_health_check_timeout))
        self.assertEqual(len(env_config), 10)

    def test_builder_init_empty_entry_point_raises_error(self):
        self.mock_config.inference_entry_point = "" 
        with self.assertRaisesRegex(ValueError, "inference_entry_point cannot be empty"):
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        self.mock_config.inference_entry_point = "inference_script.py" # Restore

    def test_builder_init_missing_required_attribute_raises_error(self):
        self.mock_config.framework_version = None 
        with self.assertRaisesRegex(ValueError, "PytorchModelCreationConfig missing required attribute\(s\): framework_version"):
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        self.mock_config.framework_version = "2.1.0" # Restore

    def test_builder_init_missing_entry_point_file_raises_error(self):
        entry_point_file = self.temp_source_dir / self.mock_config.inference_entry_point
        if entry_point_file.exists():
            entry_point_file.unlink()

        with self.assertRaisesRegex(FileNotFoundError, "Builder validation: Inference entry point script not found"):
            PytorchModelStepBuilder(
                config=self.mock_config,
                sagemaker_session=self.mock_sagemaker_session,
                role=self.role
            )
        entry_point_file.touch() # Re-create for other tests

    @patch('pipelines.builder_model_step_pytorch.image_uris.retrieve') # Adjust to YOUR builder's import of image_uris
    def test_get_image_uri(self, mock_retrieve):
        expected_uri = "pytorch-inference:2.1.0-py310-ml.m5.4xlarge"
        mock_retrieve.return_value = expected_uri
        uri = self.builder._get_image_uri()
        self.assertEqual(uri, expected_uri)
        mock_retrieve.assert_called_once_with(
            framework="pytorch",
            region=self.builder.aws_region, # Use builder's derived aws_region
            version=self.mock_config.framework_version,
            py_version=self.mock_config.py_version,
            instance_type=self.mock_config.inference_instance_type,
            image_scope="inference"
        )

    # Adjust these patch paths to where PyTorchModel, Parameter, and ModelStep
    # are imported within your PytorchModelStepBuilder's file.
    @patch('pipelines.builder_model_step_pytorch.PyTorchModel') 
    @patch('sagemaker.workflow.parameters.Parameter')
    @patch('pipelines.builder_model_step_pytorch.SagemakerModelStep') # Builder uses SagemakerModelStep
    def test_create_model_step_correct_usage(self, mock_sagemaker_model_step_class, mock_parameter_class, mock_pytorch_model_class):
        mock_model_data_s3_uri = "s3://mybucket/mymodel/model.tar.gz"
        mock_pytorch_model_instance = mock_pytorch_model_class.return_value
        mock_step_args = Mock(spec=_StepArguments)
        mock_pytorch_model_instance.create.return_value = mock_step_args
        mock_param_instance = Mock(spec=Parameter)
        mock_parameter_class.return_value = mock_param_instance
        mock_env_config = {'SAGEMAKER_PROGRAM': 'inference_script.py'}
        mock_image_uri = 'mocked-pytorch-image-uri'
        
        # Use the get_model_name mock from setUp
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
            self.assertEqual(kwargs_pymodel['name'], expected_model_name)
            # ... (other assertions for PyTorchModel)

            expected_step_name = f"{self.mock_config.pipeline_name}-CreateModel"
            mock_sagemaker_model_step_class.assert_called_once_with(
                name=expected_step_name,
                step_args=mock_step_args
            )
            self.assertEqual(returned_step.model_artifacts_path, mock_model_data_s3_uri)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)