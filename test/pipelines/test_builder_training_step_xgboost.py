# test/test_builder_training_step_xgboost.py
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from botocore.exceptions import ClientError

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipelines.builder_training_step_xgboost import XGBoostTrainingStepBuilder

class TestXGBoostTrainingStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        self.config = SimpleNamespace()
        self.config.region = 'NA'
        self.config.pipeline_name = 'test-pipeline'
        self.config.current_date = '2025-06-12'
        
        hp = SimpleNamespace()
        hp.model_dump = lambda: {'param': 'value'}
        self.config.hyperparameters = hp
        
        self.config.training_entry_point = 'train_xgb.py'
        self.config.source_dir = 'src/training_scripts'
        self.config.training_instance_type = 'ml.m5.large'
        self.config.training_instance_count = 1
        self.config.training_volume_size = 30
        self.config.framework_version = '1.7-1'
        self.config.py_version = 'py3'
        self.config.input_path = 's3://bucket/input'
        self.config.output_path = 's3://bucket/output'
        self.config.hyperparameters_s3_uri = 's3://bucket/config/'

        self.builder = object.__new__(XGBoostTrainingStepBuilder)
        self.builder.config = self.config
        self.builder.session = MagicMock()
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder._get_step_name = MagicMock(return_value='XGBoostTrainingStep')
        self.builder.aws_region = 'us-east-1'

    def test_validate_configuration_success(self):
        """Test that validate_configuration passes with a correctly populated config."""
        try:
            self.builder.validate_configuration()
        except ValueError:
            self.fail("validate_configuration() raised ValueError unexpectedly!")

    def test_validate_configuration_missing_attr(self):
        """Test that validate_configuration raises a ValueError if a required attribute is missing."""
        del self.config.training_entry_point
        with self.assertRaisesRegex(ValueError, "missing required attributes"):
            self.builder.validate_configuration()

    @patch('src.pipelines.builder_training_step_xgboost.shutil.rmtree')
    @patch('src.pipelines.builder_training_step_xgboost.tempfile.mkdtemp')
    @patch('src.pipelines.builder_training_step_xgboost.Path.write_text')
    @patch('src.pipelines.builder_training_step_xgboost.S3Uploader.upload')
    @patch('src.pipelines.builder_training_step_xgboost.json.dumps')
    def test_prepare_hyperparameters_file_with_existing_file(self, mock_json_dumps, mock_s3_upload, mock_write_text, mock_mkdtemp, mock_rmtree):
        """
        Test _prepare_hyperparameters_file when an existing file needs to be deleted.
        """
        mock_mkdtemp.return_value = '/tmp/dummy_dir'
        mock_s3_client = MagicMock()
        self.builder.session.boto_session.client.return_value = mock_s3_client
        
        expected_s3_uri = 's3://bucket/config/hyperparameters.json'
        mock_s3_upload.return_value = expected_s3_uri

        s3_uri = self.builder._prepare_hyperparameters_file()

        mock_s3_client.head_object.assert_called_once_with(Bucket='bucket', Key='config/hyperparameters.json')
        mock_s3_client.delete_object.assert_called_once_with(Bucket='bucket', Key='config/hyperparameters.json')
        mock_s3_upload.assert_called_once()
        # FIX: Assert that shutil.rmtree is called for cleanup, not os.remove.
        mock_rmtree.assert_called_once_with(Path('/tmp/dummy_dir'))
        self.assertEqual(s3_uri, expected_s3_uri)

    @patch('src.pipelines.builder_training_step_xgboost.shutil.rmtree')
    @patch('src.pipelines.builder_training_step_xgboost.tempfile.mkdtemp')
    @patch('src.pipelines.builder_training_step_xgboost.Path.write_text')
    @patch('src.pipelines.builder_training_step_xgboost.S3Uploader.upload')
    def test_prepare_hyperparameters_no_existing_file(self, mock_s3_upload, mock_write_text, mock_mkdtemp, mock_rmtree):
        """
        Test _prepare_hyperparameters_file when no existing file is found (404 error).
        """
        mock_mkdtemp.return_value = '/tmp/dummy_dir'
        mock_s3_client = MagicMock()
        mock_s3_client.head_object.side_effect = ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        self.builder.session.boto_session.client.return_value = mock_s3_client
        
        self.builder._prepare_hyperparameters_file()
        
        mock_s3_client.head_object.assert_called_once()
        mock_s3_client.delete_object.assert_not_called()
        mock_s3_upload.assert_called_once()
        mock_rmtree.assert_called_once()

    @patch('src.pipelines.builder_training_step_xgboost.XGBoost')
    def test_create_xgboost_estimator(self, mock_xgboost_cls):
        """Test that the XGBoost estimator is created with the correct parameters."""
        self.builder._create_xgboost_estimator()
        
        mock_xgboost_cls.assert_called_once()
        _, kwargs = mock_xgboost_cls.call_args
        
        self.assertEqual(kwargs.get('entry_point'), self.config.training_entry_point)
        self.assertEqual(kwargs.get('hyperparameters'), {})
        self.assertIn('CA_REPOSITORY_ARN', kwargs.get('environment', {}))

    @patch('src.pipelines.builder_training_step_xgboost.XGBoostTrainingStepBuilder._create_xgboost_estimator')
    @patch('src.pipelines.builder_training_step_xgboost.XGBoostTrainingStepBuilder._prepare_hyperparameters_file')
    def test_create_step(self, mock_prepare_hp_file, mock_create_estimator):
        """Test the end-to-end creation of the TrainingStep."""
        mock_prepare_hp_file.return_value = 's3://bucket/config/hyperparameters.json'
        mock_estimator = MagicMock()
        mock_create_estimator.return_value = mock_estimator
        
        step = self.builder.create_step()
        
        self.assertIsInstance(step, TrainingStep)
        self.assertEqual(step.estimator, mock_estimator)
        self.assertIn('config', step.inputs)
        self.assertEqual(step.inputs['config'].config['DataSource']['S3DataSource']['S3Uri'], 's3://bucket/config/hyperparameters.json')

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
