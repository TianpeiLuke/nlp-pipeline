import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e import XGBoostTrainEvaluateE2EPipelineBuilder
from src.pipelines.config_base import BasePipelineConfig
from src.pipelines.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipelines.config_model_step_xgboost import XGBoostModelCreationConfig
from src.pipelines.config_model_eval_step_xgboost import XGBoostModelEvaluationConfig

class TestXGBoostTrainEvaluateE2EPipelineBuilder(unittest.TestCase):
    def setUp(self):
        """Set up mocks and patches for testing."""
        # Mock the load_configs function
        self.load_configs_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        
        # Create mock configs
        self.mock_base_config = MagicMock(spec=BasePipelineConfig)
        self.mock_base_config.pipeline_name = "test-pipeline"
        self.mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        self.mock_base_config.region = "us-west-2"
        
        self.mock_xgb_train_cfg = MagicMock(spec=XGBoostTrainingConfig)
        self.mock_xgb_model_cfg = MagicMock(spec=XGBoostModelCreationConfig)
        self.mock_xgb_eval_cfg = MagicMock(spec=XGBoostModelEvaluationConfig)
        
        # Set up mock configs dictionary
        self.mock_configs = {
            'Base': self.mock_base_config,
            'XGBoostTrainingConfig': self.mock_xgb_train_cfg,
            'XGBoostModelCreationConfig': self.mock_xgb_model_cfg,
            'XGBoostModelEvaluationConfig': self.mock_xgb_eval_cfg,
        }
        
        # Configure load_configs to return our mock configs
        self.mock_load_configs.return_value = self.mock_configs
        
        # Mock the step builders
        self.xgb_train_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.XGBoostTrainingStepBuilder')
        self.mock_xgb_train_builder_cls = self.xgb_train_builder_patch.start()
        self.mock_xgb_train_builder = MagicMock()
        self.mock_xgb_train_builder_cls.return_value = self.mock_xgb_train_builder
        self.mock_xgb_train_builder.create_step.return_value = MagicMock(name="xgb_train_step")
        
        self.xgb_model_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.XGBoostModelStepBuilder')
        self.mock_xgb_model_builder_cls = self.xgb_model_builder_patch.start()
        self.mock_xgb_model_builder = MagicMock()
        self.mock_xgb_model_builder_cls.return_value = self.mock_xgb_model_builder
        self.mock_xgb_model_builder.create_step.return_value = MagicMock(name="xgb_model_step")
        
        self.xgb_eval_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.XGBoostModelEvaluationStepBuilder')
        self.mock_xgb_eval_builder_cls = self.xgb_eval_builder_patch.start()
        self.mock_xgb_eval_builder = MagicMock()
        self.mock_xgb_eval_builder_cls.return_value = self.mock_xgb_eval_builder
        self.mock_xgb_eval_builder.create_step.return_value = MagicMock(name="xgb_eval_step")
        
        # Mock Pipeline
        self.pipeline_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.Pipeline')
        self.mock_pipeline_cls = self.pipeline_patch.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline
        
        # Create the builder instance
        self.builder = XGBoostTrainEvaluateE2EPipelineBuilder(
            config_path="dummy/path/to/config.json",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            input_data_path="s3://test-bucket/input-data",
            output_path="s3://test-bucket/output"
        )

    def tearDown(self):
        """Clean up patches after each test."""
        self.load_configs_patch.stop()
        self.xgb_train_builder_patch.stop()
        self.xgb_model_builder_patch.stop()
        self.xgb_eval_builder_patch.stop()
        self.pipeline_patch.stop()

    def test_initialization(self):
        """Test that the builder initializes correctly."""
        # Verify load_configs was called with the correct parameters
        self.mock_load_configs.assert_called_once_with(
            "dummy/path/to/config.json", 
            ANY  # CONFIG_CLASSES dictionary
        )
        
        # Verify configs were extracted correctly
        self.assertEqual(self.builder.base_config, self.mock_base_config)
        self.assertEqual(self.builder.xgb_train_cfg, self.mock_xgb_train_cfg)
        self.assertEqual(self.builder.xgb_model_cfg, self.mock_xgb_model_cfg)
        self.assertEqual(self.builder.xgb_eval_cfg, self.mock_xgb_eval_cfg)
        
        # Verify input and output paths were set correctly
        self.assertEqual(self.builder.input_data_path, "s3://test-bucket/input-data")
        self.assertEqual(self.builder.output_path, "s3://test-bucket/output")

    def test_get_pipeline_parameters(self):
        """Test that _get_pipeline_parameters returns the correct parameters."""
        params = self.builder._get_pipeline_parameters()
        
        # Verify the parameters list contains the expected parameters
        self.assertEqual(len(params), 4)
        param_names = [p.name for p in params]
        self.assertIn("PipelineExecutionTempDir", param_names)
        self.assertIn("KMSEncryptionKey", param_names)
        self.assertIn("SecurityGroupId", param_names)
        self.assertIn("VPCEndpointSubnet", param_names)

    def test_create_xgboost_train_step(self):
        """Test that _create_xgboost_train_step creates a step correctly."""
        step = self.builder._create_xgboost_train_step()
        
        # Verify XGBoostTrainingStepBuilder was instantiated with correct parameters
        self.mock_xgb_train_builder_cls.assert_called_once_with(
            config=self.mock_xgb_train_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify input_path and output_path were set correctly
        self.assertEqual(self.mock_xgb_train_builder.config.input_path, "s3://test-bucket/input-data")
        self.assertEqual(self.mock_xgb_train_builder.config.output_path, "s3://test-bucket/output")
        
        # Verify create_step was called
        self.mock_xgb_train_builder.create_step.assert_called_once()
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_xgb_train_builder.create_step.return_value)

    def test_create_model_creation_step(self):
        """Test that _create_model_creation_step creates a step correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        step = self.builder._create_model_creation_step(mock_dependency)
        
        # Verify XGBoostModelStepBuilder was instantiated with correct parameters
        self.mock_xgb_model_builder_cls.assert_called_once_with(
            config=self.mock_xgb_model_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify create_step was called with correct parameters
        self.mock_xgb_model_builder.create_step.assert_called_once_with(
            model_data="s3://bucket/model.tar.gz",
            dependencies=[mock_dependency]
        )
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_xgb_model_builder.create_step.return_value)

    def test_create_model_evaluation_step(self):
        """Test that _create_model_evaluation_step creates a step correctly."""
        # Create mock dependency steps
        mock_model_step = MagicMock()
        mock_model_step.properties.ModelName = "test-model"
        
        step = self.builder._create_model_evaluation_step(mock_model_step)
        
        # Verify XGBoostModelEvaluationStepBuilder was instantiated with correct parameters
        self.mock_xgb_eval_builder_cls.assert_called_once_with(
            config=self.mock_xgb_eval_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify create_step was called with correct parameters
        self.mock_xgb_eval_builder.create_step.assert_called_once_with(
            model_name="test-model",
            input_data_path=self.builder.input_data_path,
            dependencies=[mock_model_step]
        )
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_xgb_eval_builder.create_step.return_value)

    def test_create_pipeline_flow(self):
        """Test that _create_pipeline_flow creates the full pipeline flow correctly."""
        steps = self.builder._create_pipeline_flow()
        
        # Verify all the step creation methods were called
        self.mock_xgb_train_builder.create_step.assert_called_once()
        self.mock_xgb_model_builder.create_step.assert_called_once()
        self.mock_xgb_eval_builder.create_step.assert_called_once()
        
        # Verify the returned steps list contains all the expected steps
        self.assertEqual(len(steps), 3)  # 3 steps: training + model creation + evaluation

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline correctly."""
        pipeline = self.builder.generate_pipeline()
        
        # Verify Pipeline was instantiated with correct parameters
        self.mock_pipeline_cls.assert_called_once()
        call_args = self.mock_pipeline_cls.call_args[1]
        
        self.assertEqual(call_args["name"], f"{self.mock_base_config.pipeline_name}-xgb-train-evaluate-e2e")
        self.assertEqual(len(call_args["parameters"]), 4)  # 4 pipeline parameters
        self.assertEqual(len(call_args["steps"]), 3)  # 3 steps: training + model creation + evaluation
        self.assertEqual(call_args["sagemaker_session"], self.builder.session)
        
        # Verify the returned pipeline is our mock
        self.assertEqual(pipeline, self.mock_pipeline)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
