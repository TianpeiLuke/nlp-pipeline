import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import logging
import jsonschema
from sagemaker.config.config import validate_sagemaker_config

from sagemaker.workflow.model_step import ModelStep
from botocore.exceptions import ClientError

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_model_step_xgboost import XGBoostModelStepBuilder

class TestXGBoostModelStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        logging.getLogger().setLevel(logging.CRITICAL)  # Suppress logging output
        self.config = SimpleNamespace()
        self.config.inference_entry_point = 'inference_xgb.py'
        self.config.source_dir = 'src/inference_scripts'
        self.config.inference_instance_type = 'ml.m5.large'
        self.config.framework_version = '1.7-1'
        self.config.container_startup_health_check_timeout = 300
        self.config.container_memory_limit = 2048
        self.config.data_download_timeout = 600
        self.config.inference_memory_limit = 1024
        self.config.max_concurrent_invocations = 10
        self.config.max_payload_size = 5
        self.config.current_date = '2025-06-12'

        self.builder = object.__new__(XGBoostModelStepBuilder)
        self.builder.config = self.config
        self.builder.session = MagicMock()
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.aws_region = 'us-east-1'
        self.builder._get_step_name = MagicMock(return_value='XGBoostModelStep')

    @patch('src.pipelines.builder_model_step_xgboost.image_uris.retrieve')
    def test_get_image_uri(self, mock_image_uris_retrieve):
        """Test that the correct image URI is retrieved."""
        mock_image_uris_retrieve.return_value = '123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1'
        image_uri = self.builder._get_image_uri()
        mock_image_uris_retrieve.assert_called_once_with(
            framework="xgboost",
            region=self.builder.aws_region,
            version=self.config.framework_version,
            instance_type=self.config.inference_instance_type,
            image_scope="inference"
        )
        self.assertEqual(image_uri, '123456789012.dkr.ecr.us-east-1.amazonaws.com/xgboost:1.7-1')

    def test_create_env_config(self):
        """Test that the environment configuration is created correctly."""
        env_config = self.builder._create_env_config()
        self.assertEqual(env_config['SAGEMAKER_PROGRAM'], self.config.inference_entry_point)
        self.assertEqual(env_config['SAGEMAKER_CONTAINER_MEMORY_LIMIT'], str(self.config.container_memory_limit))
        self.assertEqual(env_config['AWS_REGION'], self.builder.aws_region)

    @patch('src.pipelines.builder_model_step_xgboost.XGBoostModel')
    def test_create_xgboost_model(self, mock_xgboost_model_cls):
        """Test that the XGBoost model is created with the correct parameters."""
        mock_model_instance = MagicMock()
        mock_xgboost_model_cls.return_value = mock_model_instance

        model_data = 's3://bucket/model.tar.gz'
        model = self.builder._create_xgboost_model(model_data)

        mock_xgboost_model_cls.assert_called_once_with(
            name='xgb-model-2025-06-12',
            model_data=model_data,
            role=self.builder.role,
            entry_point=self.config.inference_entry_point,
            source_dir=self.config.source_dir,
            framework_version=self.config.framework_version,
            sagemaker_session=self.builder.session,
            env=self.builder._create_env_config(),
            image_uri=self.builder._get_image_uri()
        )
        self.assertEqual(model, mock_model_instance)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
