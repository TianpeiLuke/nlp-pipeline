import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import sys

from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
from src.pipeline_steps.config_mims_payload_step import PayloadConfig, VariableType

class TestMIMSPayloadStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a valid config for the PayloadConfig
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "model_owner": "test-team",
            "model_registration_domain": "BuyerSellerMessaging",
            "model_registration_objective": "TestObjective",
            "source_model_inference_content_types": ["text/csv"],
            "source_model_inference_response_types": ["application/json"],
            "source_model_inference_output_variable_list": {"score": VariableType.NUMERIC},
            "source_model_inference_input_variable_list": {
                "feature1": VariableType.NUMERIC, 
                "feature2": VariableType.TEXT
            },
            "payload_script_path": None  # Optional
        }
        
        # Create a real PayloadConfig instance
        self.config = PayloadConfig(**self.valid_config_data)
        
        # Mock the generate_and_upload_payloads method at the module level
        self.patcher = patch('src.pipeline_steps.config_mims_payload_step.PayloadConfig.generate_and_upload_payloads')
        self.mock_gen_upload = self.patcher.start()
        self.mock_gen_upload.return_value = 's3://test-bucket/mods/payload/payload_test-pipeline_1.0.0_TestObjective.tar.gz'
        
        # Instantiate builder with the mocked config
        self.builder = MIMSPayloadStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.')
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._get_step_name = MagicMock(return_value='PayloadTestStep')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-payload-test')

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    @patch('src.pipeline_steps.builder_mims_payload_step.MIMSPayloadStepBuilder.validate_configuration')
    def test_init_calls_validate_configuration(self, mock_validate):
        """Test that __init__ calls validate_configuration."""
        # Create a new builder instance
        builder = MIMSPayloadStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.')
        )
        
        # Verify validate_configuration was called
        mock_validate.assert_called_once()

    def test_get_input_requirements(self):
        """Test that input requirements are returned correctly."""
        input_reqs = self.builder.get_input_requirements()
        self.assertIn("dependencies", input_reqs)
        self.assertEqual(input_reqs["dependencies"], "Optional list of dependent steps")

    def test_get_output_properties(self):
        """Test that output properties are returned correctly."""
        output_props = self.builder.get_output_properties()
        self.assertIn("payload_s3_uri", output_props)
        self.assertIn("payload_s3_key", output_props)

    @patch('src.pipeline_steps.builder_mims_payload_step.Lambda')
    @patch('src.pipeline_steps.builder_mims_payload_step.LambdaStep')
    def test_create_step_calls_generate_and_upload_payloads(self, mock_lambda_step_cls, mock_lambda_cls):
        """Test that create_step calls generate_and_upload_payloads on the config."""
        # Setup mock lambda function and step
        mock_lambda = MagicMock()
        mock_lambda_cls.return_value = mock_lambda
        mock_step = MagicMock()
        mock_lambda_step_cls.return_value = mock_step
        
        # Create step
        step = self.builder.create_step()
        
        # Verify generate_and_upload_payloads was called
        self.config.generate_and_upload_payloads.assert_called_once()
        
        # Verify Lambda was created with correct parameters
        mock_lambda_cls.assert_called_once()
        lambda_call_kwargs = mock_lambda_cls.call_args.kwargs
        self.assertEqual(lambda_call_kwargs['function_name'], 'test-pipeline-payload-reference')
        self.assertEqual(lambda_call_kwargs['execution_role_arn'], self.builder.role)
        
        # Verify LambdaStep was created with correct parameters
        mock_lambda_step_cls.assert_called_once()
        step_call_kwargs = mock_lambda_step_cls.call_args.kwargs
        self.assertEqual(step_call_kwargs['name'], 'PayloadTestStep')
        self.assertEqual(step_call_kwargs['lambda_func'], mock_lambda)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_mims_payload_step.Lambda')
    @patch('src.pipeline_steps.builder_mims_payload_step.LambdaStep')
    def test_create_step_with_dependencies(self, mock_lambda_step_cls, mock_lambda_cls):
        """Test that the lambda step is created with dependencies."""
        # Setup mock lambda function and step
        mock_lambda = MagicMock()
        mock_lambda_cls.return_value = mock_lambda
        mock_step = MagicMock()
        mock_lambda_step_cls.return_value = mock_step
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies
        step = self.builder.create_step(dependencies=dependencies)
        
        # Verify LambdaStep was created with dependencies
        mock_lambda_step_cls.assert_called_once()
        step_call_kwargs = mock_lambda_step_cls.call_args.kwargs
        self.assertEqual(step_call_kwargs['depends_on'], dependencies)

    @patch('src.pipeline_steps.builder_mims_payload_step.Lambda')
    @patch('src.pipeline_steps.builder_mims_payload_step.LambdaStep')
    def test_create_step_constructs_s3_key_if_none(self, mock_lambda_step_cls, mock_lambda_cls):
        """Test that the step constructs S3 key if not provided."""
        # Set sample_payload_s3_key to None
        self.config.sample_payload_s3_key = None
        
        # Create step
        self.builder.create_step()
        
        # Verify sample_payload_s3_key is no longer None
        self.assertIsNotNone(self.config.sample_payload_s3_key)
        self.assertTrue(self.config.sample_payload_s3_key.startswith('mods/payload/'))

    @patch('src.pipeline_steps.builder_mims_payload_step.MIMSPayloadStepBuilder.create_step')
    def test_create_payload_step_calls_create_step(self, mock_create_step):
        """Test that create_payload_step calls create_step."""
        # Setup mock return value
        mock_create_step.return_value = "step_created"
        
        # Call create_payload_step
        result = self.builder.create_payload_step(dependencies=None)
        
        # Verify create_step was called with dependencies
        mock_create_step.assert_called_once_with(None)
        self.assertEqual(result, "step_created")

    @patch('logging.Logger.info')
    @patch('logging.Logger.warning')
    @patch('src.pipeline_steps.builder_mims_payload_step.Lambda')
    @patch('src.pipeline_steps.builder_mims_payload_step.LambdaStep')
    def test_with_script_path(self, mock_lambda_step_cls, mock_lambda_cls, mock_warning, mock_info):
        """Test behavior when a script path is provided."""
        # Set a script path
        self.config.payload_script_path = 'path/to/script.py'
        
        # Create step
        self.builder.create_step()
        
        # Verify warning was logged
        mock_info.assert_any_call('Script path provided: path/to/script.py')
        mock_warning.assert_any_call('Using custom script path is not implemented, falling back to embedded methods')

    @patch('tempfile.TemporaryDirectory')
    @patch('src.pipeline_steps.config_mims_payload_step.PayloadConfig.save_payloads')
    @patch('src.pipeline_steps.config_mims_payload_step.PayloadConfig.upload_payloads_to_s3')
    @patch('src.pipeline_steps.builder_mims_payload_step.Lambda')
    @patch('src.pipeline_steps.builder_mims_payload_step.LambdaStep')
    def test_integration_with_config_methods(self, mock_lambda_step_cls, mock_lambda_cls, 
                                            mock_upload, mock_save, mock_temp_dir):
        """Test integration with config methods for generate_and_upload_payloads."""
        # Stop the patcher to restore the original method
        self.patcher.stop()
        
        # Mock temporary directory
        mock_temp_dir_instance = MagicMock()
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
        
        # Mock save_payloads
        mock_file_paths = [Path("/tmp/test/payload.csv")]
        mock_save.return_value = mock_file_paths
        
        # Mock upload_payloads_to_s3
        expected_s3_uri = f"s3://{self.config.bucket}/{self.config.sample_payload_s3_key}"
        mock_upload.return_value = expected_s3_uri
        
        # Setup mock lambda function and step
        mock_lambda = MagicMock()
        mock_lambda_cls.return_value = mock_lambda
        mock_step = MagicMock()
        mock_lambda_step_cls.return_value = mock_step
        
        # Create step
        step = self.builder.create_step()
        
        # Verify methods were called in the correct order
        mock_save.assert_called_once()
        mock_upload.assert_called_once_with(mock_file_paths)
        
        # Verify LambdaStep inputs contain the correct S3 URI
        mock_lambda_step_cls.assert_called_once()
        step_call_kwargs = mock_lambda_step_cls.call_args.kwargs
        self.assertEqual(step_call_kwargs['inputs']['payload_s3_uri'], expected_s3_uri)

    def tearDown(self):
        """Clean up after each test."""
        # Stop the patcher if it's active
        if hasattr(self, 'patcher'):
            self.patcher.stop()

if __name__ == '__main__':
    unittest.main()
