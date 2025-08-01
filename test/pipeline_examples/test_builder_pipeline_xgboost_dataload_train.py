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
from pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train import XGBoostDataloadTrainPipelineBuilder
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig

class TestXGBoostDataloadTrainPipelineBuilder(unittest.TestCase):
    def setUp(self):
        """Set up mocks and patches for testing."""
        # Mock the load_configs function
        self.load_configs_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        
        # Mock BasePipelineConfig.get_step_name to return the class name directly
        self.get_step_name_patch = patch('src.pipeline_steps.config_base.BasePipelineConfig.get_step_name')
        self.mock_get_step_name = self.get_step_name_patch.start()
        self.mock_get_step_name.side_effect = lambda x: x  # Return the input directly
        
        # Create mock configs
        self.mock_base_config = MagicMock(spec=BasePipelineConfig)
        self.mock_base_config.pipeline_name = "test-pipeline"
        self.mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        self.mock_base_config.region = "us-west-2"
        
        self.mock_cradle_train_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_train_cfg.job_type = "training"
        
        self.mock_tp_train_cfg = MagicMock(spec=TabularPreprocessingConfig)
        self.mock_tp_train_cfg.job_type = "training"
        self.mock_tp_train_cfg.input_names = {"data_input": "DataInput"}
        self.mock_tp_train_cfg.output_names = {"processed_data": "ProcessedTabularData"}
        
        self.mock_xgb_train_cfg = MagicMock(spec=XGBoostTrainingConfig)
        
        # Create mock CradleDataLoadConfig for training and testing
        self.mock_cradle_test_cfg = MagicMock(spec=object)
        self.mock_cradle_test_cfg.job_type = 'calibration'
        
        # Create mock TabularPreprocessingConfig for testing
        self.mock_tp_test_cfg = MagicMock(spec=object)
        self.mock_tp_test_cfg.job_type = 'calibration'
        self.mock_tp_test_cfg.input_names = {'data_input': 'calibration_data'}
        self.mock_tp_test_cfg.output_names = {'processed_data': 'ProcessedTabularData'}
        
        # Set up mock configs dictionary
        self.mock_configs = {
            'Base': self.mock_base_config,
            'CradleDataLoadConfig_training': self.mock_cradle_train_cfg,
            'CradleDataLoadConfig_calibration': self.mock_cradle_test_cfg,
            'TabularPreprocessingConfig_training': self.mock_tp_train_cfg,
            'TabularPreprocessingConfig_calibration': self.mock_tp_test_cfg,
            'XGBoostTrainingConfig': self.mock_xgb_train_cfg,
        }
        
        # Configure load_configs to return our mock configs
        self.mock_load_configs.return_value = self.mock_configs
        
        # Mock the step builders
        self.cradle_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train.CradleDataLoadingStepBuilder')
        self.mock_cradle_builder_cls = self.cradle_builder_patch.start()
        self.mock_cradle_builder = MagicMock()
        self.mock_cradle_builder_cls.return_value = self.mock_cradle_builder
        self.mock_cradle_builder.create_step.return_value = MagicMock(name="cradle_step")
        self.mock_cradle_builder.get_request_dict.return_value = {"request": "data"}
        self.mock_cradle_builder.get_step_outputs.return_value = {"DataOutput": "s3://bucket/data"}
        
        self.tp_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train.TabularPreprocessingStepBuilder')
        self.mock_tp_builder_cls = self.tp_builder_patch.start()
        self.mock_tp_builder = MagicMock()
        self.mock_tp_builder_cls.return_value = self.mock_tp_builder
        self.mock_tp_builder.create_step.return_value = MagicMock(name="tp_step")
        
        self.xgb_train_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train.XGBoostTrainingStepBuilder')
        self.mock_xgb_train_builder_cls = self.xgb_train_builder_patch.start()
        self.mock_xgb_train_builder = MagicMock()
        self.mock_xgb_train_builder_cls.return_value = self.mock_xgb_train_builder
        self.mock_xgb_train_builder.create_step.return_value = MagicMock(name="xgb_train_step")
        
        # Mock Pipeline
        self.pipeline_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train.Pipeline')
        self.mock_pipeline_cls = self.pipeline_patch.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline
        
        # Mock OUTPUT_TYPE constants
        self.constants_patch = patch.multiple(
            'pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_dataload_train',
            OUTPUT_TYPE_DATA="DataOutput",
            OUTPUT_TYPE_METADATA="MetadataOutput",
            OUTPUT_TYPE_SIGNATURE="SignatureOutput"
        )
        self.constants_patch.start()
        
        # Patch isinstance to return True for our mocks
        self.original_isinstance = isinstance
        
        def patched_isinstance(obj, classinfo):
            from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
            from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
            from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
            
            # Check if obj is one of our mocks and classinfo is the corresponding class
            if hasattr(self, 'mock_cradle_train_cfg') and obj is self.mock_cradle_train_cfg and classinfo is CradleDataLoadConfig:
                return True
            if hasattr(self, 'mock_cradle_test_cfg') and obj is self.mock_cradle_test_cfg and classinfo is CradleDataLoadConfig:
                return True
            if hasattr(self, 'mock_tp_train_cfg') and obj is self.mock_tp_train_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if hasattr(self, 'mock_tp_test_cfg') and obj is self.mock_tp_test_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if hasattr(self, 'mock_xgb_train_cfg') and obj is self.mock_xgb_train_cfg and classinfo is XGBoostTrainingConfig:
                return True
            
            # Fall back to the original isinstance for other cases
            return self.original_isinstance(obj, classinfo)
        
        # Replace the built-in isinstance with our patched version
        self.builtins_patch = patch('builtins.isinstance', patched_isinstance)
        self.builtins_patch.start()
        
        # Create the builder instance
        self.builder = XGBoostDataloadTrainPipelineBuilder(
            config_path="dummy/path/to/config.json",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::123456789012:role/SageMakerRole"
        )

    def tearDown(self):
        """Clean up patches after each test."""
        self.load_configs_patch.stop()
        self.get_step_name_patch.stop()
        self.builtins_patch.stop()
        self.cradle_builder_patch.stop()
        self.tp_builder_patch.stop()
        self.xgb_train_builder_patch.stop()
        self.pipeline_patch.stop()
        self.constants_patch.stop()

    def test_initialization(self):
        """Test that the builder initializes correctly."""
        # Verify load_configs was called with the correct parameters
        self.mock_load_configs.assert_called_once_with(
            "dummy/path/to/config.json", 
            ANY  # CONFIG_CLASSES dictionary
        )
        
        # Verify configs were extracted correctly
        self.assertEqual(self.builder.base_config, self.mock_base_config)
        self.assertEqual(self.builder.cradle_train_cfg, self.mock_cradle_train_cfg)
        self.assertEqual(self.builder.tp_train_cfg, self.mock_tp_train_cfg)
        self.assertEqual(self.builder.xgb_train_cfg, self.mock_xgb_train_cfg)

    def test_validate_preprocessing_inputs_success(self):
        """Test that _validate_preprocessing_inputs succeeds with valid inputs."""
        # Should not raise any exceptions
        self.builder._validate_preprocessing_inputs()

    def test_validate_preprocessing_inputs_missing_required(self):
        """Test that _validate_preprocessing_inputs fails with missing required inputs."""
        # Modify the input_names to be missing a required input
        self.mock_tp_train_cfg.input_names = {}
        
        with self.assertRaises(ValueError):
            self.builder._validate_preprocessing_inputs()

    def test_validate_preprocessing_inputs_unknown_input(self):
        """Test that _validate_preprocessing_inputs fails with unknown inputs."""
        # Modify the input_names to include an unknown input
        self.mock_tp_train_cfg.input_names = {"data_input": "DataInput", "unknown_input": "UnknownInput"}
        
        with self.assertRaises(ValueError):
            self.builder._validate_preprocessing_inputs()

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

    def test_create_flow(self):
        """Test that _create_flow creates a flow correctly."""
        steps = self.builder._create_flow('training')
        
        # Verify CradleDataLoadingStepBuilder was instantiated with correct parameters
        self.mock_cradle_builder_cls.assert_called_once_with(
            config=self.mock_cradle_train_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify create_step was called
        self.mock_cradle_builder.create_step.assert_called_once()
        
        # Verify get_request_dict was called
        self.mock_cradle_builder.get_request_dict.assert_called_once()
        
        # Verify the returned steps list contains the expected steps
        self.assertEqual(len(steps), 2)  # 2 steps: data load + preprocessing

    def test_create_training_flow(self):
        """Test that _create_training_flow creates the full training flow correctly."""
        steps = self.builder._create_training_flow()
        
        # Verify all the step creation methods were called
        self.mock_cradle_builder.create_step.assert_called_once()
        self.mock_tp_builder.create_step.assert_called_once()
        self.mock_xgb_train_builder.create_step.assert_called_once()
        
        # Verify the returned steps list contains all the expected steps
        self.assertEqual(len(steps), 3)  # 3 steps: data load + preprocessing + training

    def test_create_calibration_flow(self):
        """Test that _create_calibration_flow creates a calibration flow correctly."""
        steps = self.builder._create_calibration_flow()
        
        # Verify the returned steps list contains the expected steps
        self.assertEqual(len(steps), 2)  # 2 steps: data load + preprocessing

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline correctly."""
        pipeline = self.builder.generate_pipeline()
        
        # Verify Pipeline was instantiated with correct parameters
        self.mock_pipeline_cls.assert_called_once()
        call_args = self.mock_pipeline_cls.call_args[1]
        
        self.assertEqual(call_args["name"], f"{self.mock_base_config.pipeline_name}-loadprep-train")
        self.assertEqual(len(call_args["parameters"]), 4)  # 4 pipeline parameters
        self.assertEqual(len(call_args["steps"]), 5)  # 5 steps: 3 for training flow + 2 for calibration flow
        self.assertEqual(call_args["sagemaker_session"], self.builder.session)
        
        # Verify the returned pipeline is our mock
        self.assertEqual(pipeline, self.mock_pipeline)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
