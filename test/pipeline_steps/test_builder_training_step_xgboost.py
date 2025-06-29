# test/test_builder_training_step_xgboost.py
import unittest
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.functions import Join
from botocore.exceptions import ClientError

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.s3_utils import S3PathHandler

class TestXGBoostTrainingStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a mock config with all required attributes
        self.config = SimpleNamespace()
        self.config.region = 'us-east-1'
        self.config.pipeline_name = 'test-pipeline'
        self.config.current_date = '2025-06-12'
        
        # Mock hyperparameters
        hp = SimpleNamespace()
        hp.model_dump = lambda: {'param': 'value'}
        hp.full_field_list = ['order_id', 'feature1', 'feature2', 'label']
        hp.tab_field_list = ['feature1', 'feature2']
        hp.cat_field_list = []
        hp.label_name = 'label'
        hp.id_name = 'order_id'
        self.config.hyperparameters = hp
        
        # Required attributes for validation
        self.config.training_entry_point = 'train_xgb.py'
        self.config.source_dir = 'src/training_scripts'
        self.config.training_instance_type = 'ml.m5.large'
        self.config.training_instance_count = 1
        self.config.training_volume_size = 30
        self.config.framework_version = '1.7-1'
        self.config.py_version = 'py3'
        self.config.input_path = 's3://bucket/input'
        self.config.output_path = 's3://bucket/output'
        self.config.hyperparameters_s3_uri = 's3://bucket/config'
        
        # Input and output names
        self.input_path_key = "Path containing train/val/test subdirectories"
        self.config.input_names = {
            "input_path": self.input_path_key,
            "config": "Path to configuration files including hyperparameters.json"
        }
        self.config.output_names = {
            "output_path": "S3 path for output model artifacts"
        }
        
        # Environment variables
        self.config.env = {"TEST_ENV_VAR": "test_value"}
        
        # Optional attributes
        self.config.checkpoint_path = 's3://bucket/checkpoints'
        
        # Add methods to the mock config
        self.config.get_checkpoint_uri = lambda: self.config.checkpoint_path
        self.config.has_checkpoint = lambda: self.config.checkpoint_path is not None

        # Create the builder instance
        self.builder = object.__new__(XGBoostTrainingStepBuilder)
        self.builder.config = self.config
        self.builder.session = MagicMock()
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder._get_step_name = MagicMock(return_value='XGBoostTrainingStep')
        self.builder.aws_region = 'us-east-1'
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='sanitized-name')
        self.builder._get_cache_config = MagicMock(return_value={'Enabled': True})
        self.builder._extract_param = MagicMock(side_effect=lambda kwargs, key, default=None: 
                                               kwargs.get(key, default))

    def test_validate_configuration_success(self):
        """Test that validate_configuration passes with a correctly populated config."""
        try:
            self.builder.validate_configuration()
        except ValueError:
            self.fail("validate_configuration() raised ValueError unexpectedly!")

    def test_validate_configuration_missing_attr(self):
        """Test that validate_configuration raises a ValueError if a required attribute is missing."""
        del self.config.training_entry_point
        with self.assertRaisesRegex(ValueError, "missing required attribute"):
            self.builder.validate_configuration()
            
    def test_validate_configuration_missing_input_names(self):
        """Test that validate_configuration raises a ValueError if required input names are missing."""
        self.config.input_names = {"wrong_key": "description"}
        with self.assertRaisesRegex(ValueError, "input_names must contain keys"):
            self.builder.validate_configuration()

    @patch('src.pipeline_steps.builder_training_step_xgboost.shutil.rmtree')
    @patch('src.pipeline_steps.builder_training_step_xgboost.tempfile.mkdtemp')
    @patch('src.pipeline_steps.builder_training_step_xgboost.S3Uploader.upload')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.pipeline_steps.builder_training_step_xgboost.json.dump')
    def test_prepare_hyperparameters_file_with_existing_file(self, mock_json_dump, mock_file, mock_s3_upload, mock_mkdtemp, mock_rmtree):
        """
        Test _prepare_hyperparameters_file when an existing file is found.
        """
        mock_mkdtemp.return_value = '/tmp/dummy_dir'
        mock_s3_client = MagicMock()
        self.builder.session.boto_session.client.return_value = mock_s3_client
        
        expected_s3_uri = 's3://bucket/config/hyperparameters.json'
        mock_s3_upload.return_value = expected_s3_uri

        s3_uri = self.builder._prepare_hyperparameters_file()

        # Check that head_object is called to check if file exists
        mock_s3_client.head_object.assert_called_once_with(Bucket='bucket', Key='config/hyperparameters.json')
        
        # In the current implementation, delete_object is not called
        mock_s3_client.delete_object.assert_not_called()
        
        # Check that the file is uploaded
        mock_s3_upload.assert_called_once()
        
        # Check that temporary directory is cleaned up
        mock_rmtree.assert_called_once_with(Path('/tmp/dummy_dir'))
        
        # Check that the returned URI is correct
        self.assertEqual(s3_uri, expected_s3_uri)

    @patch('src.pipeline_steps.builder_training_step_xgboost.shutil.rmtree')
    @patch('src.pipeline_steps.builder_training_step_xgboost.tempfile.mkdtemp')
    @patch('src.pipeline_steps.builder_training_step_xgboost.S3Uploader.upload')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.pipeline_steps.builder_training_step_xgboost.json.dump')
    def test_prepare_hyperparameters_no_existing_file(self, mock_json_dump, mock_file, mock_s3_upload, mock_mkdtemp, mock_rmtree):
        """
        Test _prepare_hyperparameters_file when no existing file is found (404 error).
        """
        mock_mkdtemp.return_value = '/tmp/dummy_dir'
        mock_s3_client = MagicMock()
        mock_s3_client.head_object.side_effect = ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        self.builder.session.boto_session.client.return_value = mock_s3_client
        
        self.builder._prepare_hyperparameters_file()
        
        # Check that head_object is called to check if file exists
        mock_s3_client.head_object.assert_called_once()
        
        # In the current implementation, delete_object is not called
        mock_s3_client.delete_object.assert_not_called()
        
        # Check that the file is uploaded
        mock_s3_upload.assert_called_once()
        
        # Check that temporary directory is cleaned up
        mock_rmtree.assert_called_once()

    @patch('src.pipeline_steps.builder_training_step_xgboost.XGBoost')
    def test_create_estimator(self, mock_xgboost_cls):
        """Test that the XGBoost estimator is created with the correct parameters."""
        self.builder._create_estimator()
        
        mock_xgboost_cls.assert_called_once()
        _, kwargs = mock_xgboost_cls.call_args
        
        # Check that the estimator is created with the correct parameters
        self.assertEqual(kwargs.get('entry_point'), self.config.training_entry_point)
        self.assertEqual(kwargs.get('source_dir'), self.config.source_dir)
        self.assertEqual(kwargs.get('framework_version'), self.config.framework_version)
        self.assertEqual(kwargs.get('py_version'), self.config.py_version)
        self.assertEqual(kwargs.get('role'), self.builder.role)
        self.assertEqual(kwargs.get('instance_type'), self.config.training_instance_type)
        self.assertEqual(kwargs.get('instance_count'), self.config.training_instance_count)
        self.assertEqual(kwargs.get('volume_size'), self.config.training_volume_size)
        self.assertEqual(kwargs.get('output_path'), self.config.output_path)
        self.assertEqual(kwargs.get('checkpoint_s3_uri'), self.config.checkpoint_path)
        
        # Check that environment variables are set correctly
        self.assertEqual(kwargs.get('environment'), {"TEST_ENV_VAR": "test_value"})

    def test_get_environment_variables(self):
        """Test that environment variables are correctly retrieved from config."""
        env_vars = self.builder._get_environment_variables()
        self.assertEqual(env_vars, {"TEST_ENV_VAR": "test_value"})
        
    def test_get_input_requirements(self):
        """Test that input requirements are correctly retrieved."""
        self.builder.COMMON_PROPERTIES = {
            "dependencies": "List of steps this step depends on",
            "enable_caching": "Whether to enable caching for this step"
        }
        
        input_reqs = self.builder.get_input_requirements()
        
        self.assertIn("inputs", input_reqs)
        self.assertIn("dependencies", input_reqs)
        self.assertIn("enable_caching", input_reqs)
        
    def test_get_output_properties(self):
        """Test that output properties are correctly retrieved."""
        output_props = self.builder.get_output_properties()
        
        self.assertIn("ModelArtifacts.S3ModelArtifacts", output_props)
        
    def test_normalize_s3_uri(self):
        """Test that S3 URIs are correctly normalized."""
        # Test with trailing slash
        uri = 's3://bucket/path/'
        normalized = self.builder._normalize_s3_uri(uri)
        self.assertEqual(normalized, 's3://bucket/path')
        
        # Test with PipelineVariable
        pipeline_var = MagicMock()
        pipeline_var.expr = 's3://bucket/path/'
        normalized = self.builder._normalize_s3_uri(pipeline_var)
        self.assertEqual(normalized, 's3://bucket/path')
        
        # Test with Get expression
        get_expr = {'Get': 'Steps.ProcessingStep.ProcessingOutputConfig.Outputs["Output"].S3Output.S3Uri'}
        normalized = self.builder._normalize_s3_uri(get_expr)
        self.assertEqual(normalized, get_expr)
        
    def test_validate_s3_uri(self):
        """Test that S3 URIs are correctly validated."""
        # Test with valid URI
        valid_uri = 's3://bucket/path'
        self.assertTrue(self.builder._validate_s3_uri(valid_uri))
        
        # Test with PipelineVariable
        pipeline_var = MagicMock()
        pipeline_var.expr = 's3://bucket/path'
        self.assertTrue(self.builder._validate_s3_uri(pipeline_var))
        
        # Test with Get expression
        get_expr = {'Get': 'Steps.ProcessingStep.ProcessingOutputConfig.Outputs["Output"].S3Output.S3Uri'}
        self.assertTrue(self.builder._validate_s3_uri(get_expr))
        
        # Test with invalid URI
        invalid_uri = 'not-an-s3-uri'
        with patch('src.pipeline_steps.builder_training_step_xgboost.S3PathHandler.is_valid', return_value=False):
            self.assertFalse(self.builder._validate_s3_uri(invalid_uri))
            
    def test_get_training_inputs(self):
        """Test that training inputs are correctly constructed."""
        # Setup input dictionary
        inputs = {
            "input_path": "s3://bucket/input",
            "hyperparameters_s3_uri": "s3://bucket/config/hyperparameters.json"
        }
        
        # Mock S3PathHandler methods
        with patch('src.pipeline_steps.builder_training_step_xgboost.S3PathHandler.get_name', return_value='hyperparameters.json'), \
             patch('src.pipeline_steps.builder_training_step_xgboost.Join', side_effect=lambda on, values: f"{values[0]}{on}{values[1]}"):
            
            training_inputs = self.builder._get_training_inputs(inputs)
            
            # Check that train, val, test channels are created
            self.assertIn('train', training_inputs)
            self.assertIn('val', training_inputs)
            self.assertIn('test', training_inputs)
            
            # Check that config channel is created
            self.assertIn('config', training_inputs)
            
    @patch('src.pipeline_steps.builder_training_step_xgboost.XGBoostTrainingStepBuilder._create_estimator')
    @patch('src.pipeline_steps.builder_training_step_xgboost.XGBoostTrainingStepBuilder._prepare_hyperparameters_file')
    @patch('src.pipeline_steps.builder_training_step_xgboost.XGBoostTrainingStepBuilder._get_training_inputs')
    def test_create_step(self, mock_get_training_inputs, mock_prepare_hp_file, mock_create_estimator):
        """Test the end-to-end creation of the TrainingStep."""
        # Setup mocks
        mock_prepare_hp_file.return_value = 's3://bucket/config/hyperparameters.json'
        mock_estimator = MagicMock()
        mock_create_estimator.return_value = mock_estimator
        
        # Mock training inputs
        mock_train_inputs = {
            'train': MagicMock(),
            'val': MagicMock(),
            'test': MagicMock(),
            'config': MagicMock()
        }
        mock_get_training_inputs.return_value = mock_train_inputs
        
        # Call create_step with inputs that match the expected format
        # The key should match the description in config.input_names
        step = self.builder.create_step(
            inputs={"inputs": {self.input_path_key: "s3://bucket/input"}},
            enable_caching=True
        )
        
        # Verify the step is created correctly
        self.assertIsInstance(step, TrainingStep)
        self.assertEqual(step.estimator, mock_estimator)
        self.assertEqual(step.inputs, mock_train_inputs)
        self.assertEqual(step.name, 'XGBoostTrainingStep')

    def test_match_custom_properties(self):
        """Test that custom properties are correctly matched from dependency steps."""
        # Create a mock dependency step
        dep_step = MagicMock()
        dep_step.name = 'TabularPreprocessingStep'
        
        # Setup outputs for the dependency step
        output = MagicMock()
        output.output_name = 'processed_data'
        output.destination = 's3://bucket/processed_data'
        dep_step.outputs = [output]
        
        # Call _match_custom_properties
        inputs = {}
        input_requirements = self.builder.get_input_requirements()
        matched = self.builder._match_custom_properties(inputs, input_requirements, dep_step)
        
        # Check that inputs are correctly matched
        self.assertIn('inputs', matched)
        # The key in the inputs dictionary should match the description in config.input_names
        self.assertIn(self.input_path_key, inputs.get('inputs', {}))
        self.assertEqual(inputs['inputs'][self.input_path_key], 's3://bucket/processed_data')
        
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
        # The key in the inputs dictionary should match the description in config.input_names
        self.assertIn(self.input_path_key, inputs.get('inputs', {}))
        self.assertEqual(inputs['inputs'][self.input_path_key], 's3://bucket/processed_data')
        
    def test_match_hyperparameter_outputs(self):
        """Test that outputs from HyperparameterPrepStep are correctly matched."""
        # Create a mock HyperparameterPrepStep
        step = MagicMock()
        
        # Setup properties for the step
        hyperparameters_s3_uri = 's3://bucket/config/hyperparameters.json'
        step.properties = MagicMock()
        step.properties.Outputs = {
            'hyperparameters_s3_uri': hyperparameters_s3_uri
        }
        
        # Call _match_hyperparameter_outputs
        inputs = {}
        matched = self.builder._match_hyperparameter_outputs(inputs, step)
        
        # Check that inputs are correctly matched
        self.assertIn('hyperparameters_s3_uri', matched)
        # Just verify that the key exists in the inputs dictionary
        self.assertIn('hyperparameters_s3_uri', inputs)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
