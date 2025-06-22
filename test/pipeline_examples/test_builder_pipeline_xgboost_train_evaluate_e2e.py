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
from src.pipelines.config_model_eval_step_xgboost import XGBoostModelEvalConfig

class TestXGBoostTrainEvaluateE2EPipelineBuilder(unittest.TestCase):
    def setUp(self):
        """Set up mocks and patches for testing."""
        # Mock the load_configs function
        self.load_configs_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        
        # Mock BasePipelineConfig.get_step_name to return the class name directly
        self.get_step_name_patch = patch('src.pipelines.config_base.BasePipelineConfig.get_step_name')
        self.mock_get_step_name = self.get_step_name_patch.start()
        self.mock_get_step_name.side_effect = lambda x: x  # Return the input directly
        
        # Create mock configs
        self.mock_base_config = MagicMock(spec=BasePipelineConfig)
        self.mock_base_config.pipeline_name = "test-pipeline"
        self.mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        self.mock_base_config.region = "us-west-2"
        
        self.mock_xgb_train_cfg = MagicMock(spec=XGBoostTrainingConfig)
        self.mock_xgb_model_cfg = MagicMock(spec=XGBoostModelCreationConfig)
        self.mock_xgb_eval_cfg = MagicMock(spec=XGBoostModelEvalConfig)
        self.mock_xgb_eval_cfg.processing_entry_point = 'evaluate.py'
        self.mock_xgb_eval_cfg.get_input_names.return_value = {
            'model_input': 'model_input',
            'eval_data_input': 'eval_data_input'
        }
        self.mock_xgb_eval_cfg.get_output_names.return_value = {
            'eval_output': 'eval_output',
            'metrics_output': 'metrics_output'
        }
        
        # Create mock CradleDataLoadConfig for training and calibration
        self.mock_cradle_train_cfg = MagicMock(spec=object)
        self.mock_cradle_train_cfg.job_type = 'training'
        
        self.mock_cradle_calib_cfg = MagicMock(spec=object)
        self.mock_cradle_calib_cfg.job_type = 'calibration'
        
        # Create mock TabularPreprocessingConfig for training and calibration
        self.mock_tp_train_cfg = MagicMock(spec=object)
        self.mock_tp_train_cfg.job_type = 'training'
        self.mock_tp_train_cfg.input_names = {'data_input': 'training_data'}
        self.mock_tp_train_cfg.output_names = {'processed_data': 'ProcessedTabularData'}
        
        self.mock_tp_calib_cfg = MagicMock(spec=object)
        self.mock_tp_calib_cfg.job_type = 'calibration'
        self.mock_tp_calib_cfg.input_names = {'data_input': 'calibration_data'}
        self.mock_tp_calib_cfg.output_names = {'processed_data': 'ProcessedTabularData'}
        
        # Create mock PackageStepConfig
        self.mock_package_cfg = MagicMock(spec=object)
        self.mock_package_cfg.get_instance_type = MagicMock(return_value='ml.m5.xlarge')
        
        # Create mock ModelRegistrationConfig
        self.mock_registration_cfg = MagicMock(spec=object)
        self.mock_registration_cfg.framework = 'xgboost'
        self.mock_registration_cfg.aws_region = 'us-west-2'
        self.mock_registration_cfg.framework_version = '1.0-1'
        self.mock_registration_cfg.py_version = 'py3'
        self.mock_registration_cfg.inference_instance_type = 'ml.m5.xlarge'
        self.mock_registration_cfg.region = 'us-west-2'
        
        # Create mock PayloadConfig
        self.mock_payload_cfg = MagicMock(spec=object)
        self.mock_payload_cfg.model_registration_domain = 'test-domain'
        self.mock_payload_cfg.model_registration_objective = 'test-objective'
        self.mock_payload_cfg.source_model_inference_content_types = ['application/json']
        self.mock_payload_cfg.source_model_inference_response_types = ['application/json']
        self.mock_payload_cfg.source_model_inference_input_variable_list = ['input1', 'input2']
        self.mock_payload_cfg.source_model_inference_output_variable_list = ['output1', 'output2']
        self.mock_payload_cfg.region = 'us-west-2'
        self.mock_payload_cfg.aws_region = 'us-west-2'
        self.mock_payload_cfg.model_owner = 'test-owner'
        self.mock_payload_cfg.inference_entry_point = 'inference.py'
        self.mock_payload_cfg.bucket = 'test-bucket'
        self.mock_payload_cfg.sample_payload_s3_key = 'test-payload.json'
        self.mock_payload_cfg.expected_tps = 10
        self.mock_payload_cfg.max_latency_in_millisecond = 100
        self.mock_payload_cfg.max_acceptable_error_rate = 0.01
        self.mock_payload_cfg.generate_and_upload_payloads = MagicMock()
        
        # Set up mock configs dictionary with correct key format
        self.mock_configs = {
            'Base': self.mock_base_config,
            'XGBoostTrainingConfig': self.mock_xgb_train_cfg,
            'XGBoostModelCreationConfig': self.mock_xgb_model_cfg,
            'XGBoostModelEvalConfig': self.mock_xgb_eval_cfg,
            'CradleDataLoadConfig_training': self.mock_cradle_train_cfg,
            'CradleDataLoadConfig_calibration': self.mock_cradle_calib_cfg,
            'TabularPreprocessingConfig_training': self.mock_tp_train_cfg,
            'TabularPreprocessingConfig_calibration': self.mock_tp_calib_cfg,
            'PackageStepConfig': self.mock_package_cfg,
            'ModelRegistrationConfig': self.mock_registration_cfg,
            'PayloadConfig': self.mock_payload_cfg,
        }
        
        # Configure load_configs to return our mock configs
        self.mock_load_configs.return_value = self.mock_configs
        
        # Mock the step builders
        self.xgb_train_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.XGBoostTrainingStepBuilder')
        self.mock_xgb_train_builder_cls = self.xgb_train_builder_patch.start()
        self.mock_xgb_train_builder = MagicMock()
        self.mock_xgb_train_builder_cls.return_value = self.mock_xgb_train_builder
        self.mock_xgb_train_builder.create_step.return_value = MagicMock(name="xgb_train_step")
        
        self.xgb_model_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.XGBoostModelEvalStepBuilder')
        self.mock_xgb_model_builder_cls = self.xgb_model_builder_patch.start()
        self.mock_xgb_model_builder = MagicMock()
        self.mock_xgb_model_builder_cls.return_value = self.mock_xgb_model_builder
        self.mock_xgb_model_builder.create_step.return_value = MagicMock(name="xgb_model_step")
        
        self.xgb_eval_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.XGBoostModelEvalStepBuilder')
        self.mock_xgb_eval_builder_cls = self.xgb_eval_builder_patch.start()
        self.mock_xgb_eval_builder = MagicMock()
        self.mock_xgb_eval_builder_cls.return_value = self.mock_xgb_eval_builder
        self.mock_xgb_eval_builder.create_step.return_value = MagicMock(name="xgb_eval_step")
        
        # Mock the packaging step builder
        self.packaging_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.MIMSPackagingStepBuilder')
        self.mock_packaging_builder_cls = self.packaging_builder_patch.start()
        self.mock_packaging_builder = MagicMock()
        self.mock_packaging_builder_cls.return_value = self.mock_packaging_builder
        self.mock_packaging_builder.create_packaging_step.return_value = MagicMock(name="packaging_step")
        
        # Mock the registration step builder
        self.registration_builder_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.ModelRegistrationStepBuilder')
        self.mock_registration_builder_cls = self.registration_builder_patch.start()
        self.mock_registration_builder = MagicMock()
        self.mock_registration_builder_cls.return_value = self.mock_registration_builder
        self.mock_registration_builder.create_step.return_value = MagicMock(name="registration_step")
        
        # Mock the retrieve function
        self.retrieve_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.retrieve')
        self.mock_retrieve = self.retrieve_patch.start()
        self.mock_retrieve.return_value = "test-image-uri"
        
        # Mock Pipeline
        self.pipeline_patch = patch('pipeline_examples.xgboost_atoz.builder_pipeline_xgboost_train_evaluate_e2e.Pipeline')
        self.mock_pipeline_cls = self.pipeline_patch.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline
        
        # Patch isinstance to return True for our mocks
        self.original_isinstance = isinstance
        
        def patched_isinstance(obj, classinfo):
            from src.pipelines.config_data_load_step_cradle import CradleDataLoadConfig
            from src.pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig
            from src.pipelines.config_training_step_xgboost import XGBoostTrainingConfig
            from src.pipelines.config_model_eval_step_xgboost import XGBoostModelEvalConfig
            from src.pipelines.config_mims_packaging_step import PackageStepConfig
            from src.pipelines.config_mims_registration_step import ModelRegistrationConfig
            from src.pipelines.config_mims_payload_step import PayloadConfig
            
            # Check if obj is one of our mocks and classinfo is the corresponding class
            if obj is self.mock_cradle_train_cfg and classinfo is CradleDataLoadConfig:
                return True
            if obj is self.mock_cradle_calib_cfg and classinfo is CradleDataLoadConfig:
                return True
            if obj is self.mock_tp_train_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if obj is self.mock_tp_calib_cfg and classinfo is TabularPreprocessingConfig:
                return True
            if obj is self.mock_xgb_train_cfg and classinfo is XGBoostTrainingConfig:
                return True
            if obj is self.mock_xgb_eval_cfg and classinfo is XGBoostModelEvalConfig:
                return True
            if obj is self.mock_package_cfg and classinfo is PackageStepConfig:
                return True
            if obj is self.mock_registration_cfg and classinfo is ModelRegistrationConfig:
                return True
            if obj is self.mock_payload_cfg and classinfo is PayloadConfig:
                return True
            
            # Fall back to the original isinstance for other cases
            return self.original_isinstance(obj, classinfo)
        
        # Replace the built-in isinstance with our patched version
        self.builtins_patch = patch('builtins.isinstance', patched_isinstance)
        self.builtins_patch.start()
        
        # Create the builder instance
        self.builder = XGBoostTrainEvaluateE2EPipelineBuilder(
            config_path="dummy/path/to/config.json",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            notebook_root=Path("/dummy/notebook/root")
        )

    def tearDown(self):
        """Clean up patches after each test."""
        self.load_configs_patch.stop()
        self.get_step_name_patch.stop()
        self.builtins_patch.stop()
        self.xgb_train_builder_patch.stop()
        self.xgb_model_builder_patch.stop()
        self.xgb_eval_builder_patch.stop()
        self.packaging_builder_patch.stop()
        self.registration_builder_patch.stop()
        self.retrieve_patch.stop()
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
        self.assertEqual(self.builder.xgb_eval_cfg, self.mock_xgb_eval_cfg)
        
        # Verify notebook_root was set correctly
        self.assertEqual(self.builder.notebook_root, Path("/dummy/notebook/root"))

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
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ProcessingOutputConfig.Outputs = {
            "ProcessedTabularData": MagicMock()
        }
        mock_dependency.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri = "s3://bucket/processed_data"
        
        step = self.builder._create_xgboost_train_step(mock_dependency)
        
        # Verify XGBoostTrainingStepBuilder was instantiated with correct parameters
        self.mock_xgb_train_builder_cls.assert_called_once_with(
            config=ANY,  # We're using a temp copy of the config
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify create_step was called with the dependency
        self.mock_xgb_train_builder.create_step.assert_called_once_with(dependencies=[mock_dependency])
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_xgb_train_builder.create_step.return_value)

    def test_create_packaging_step(self):
        """Test that _create_packaging_step creates a step correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        step = self.builder._create_packaging_step(mock_dependency)
        
        # Verify MIMSPackagingStepBuilder was instantiated with correct parameters
        self.mock_packaging_builder_cls.assert_called_once_with(
            config=self.mock_package_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role,
            notebook_root=self.builder.notebook_root
        )
        
        # Verify create_packaging_step was called with correct parameters
        self.mock_packaging_builder.create_packaging_step.assert_called_once_with(
            model_data="s3://bucket/model.tar.gz",
            dependencies=[mock_dependency]
        )
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_packaging_builder.create_packaging_step.return_value)

    def test_create_model_eval_step(self):
        """Test that _create_model_eval_step creates a step correctly."""
        # Create mock dependency steps
        mock_train_step = MagicMock()
        mock_train_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        mock_calib_preprocess_step = MagicMock()
        mock_calib_preprocess_step.properties.ProcessingOutputConfig.Outputs = {
            "ProcessedTabularData": MagicMock()
        }
        mock_calib_preprocess_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri = "s3://bucket/calib_data"
        
        # Mock the get_effective_source_dir and os.path.exists to avoid file system checks
        with patch.object(self.mock_xgb_eval_cfg, 'get_effective_source_dir', return_value='/dummy/source/dir'):
            with patch('os.path.exists', return_value=True):
                step = self.builder._create_model_eval_step(mock_train_step, mock_calib_preprocess_step)
        
        # Verify XGBoostModelEvalStepBuilder was instantiated with correct parameters
        self.mock_xgb_eval_builder_cls.assert_called_once_with(
            config=self.mock_xgb_eval_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role,
            notebook_root=self.builder.notebook_root
        )
        
        # Verify create_step was called with the correct inputs, outputs, and dependencies
        self.mock_xgb_eval_builder.create_step.assert_called_once()
        call_args = self.mock_xgb_eval_builder.create_step.call_args[1]
        
        # Check inputs
        self.assertIn('inputs', call_args)
        self.assertEqual(call_args['inputs']['model_input'], "s3://bucket/model.tar.gz")
        self.assertEqual(call_args['inputs']['eval_data_input'], "s3://bucket/calib_data")
        
        # Check outputs
        self.assertIn('outputs', call_args)
        self.assertTrue('eval_output' in call_args['outputs'])
        self.assertTrue('metrics_output' in call_args['outputs'])
        
        # Check dependencies
        self.assertIn('dependencies', call_args)
        self.assertEqual(call_args['dependencies'], [mock_train_step, mock_calib_preprocess_step])
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_xgb_eval_builder.create_step.return_value)

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline correctly."""
        # Mock the step creation methods to avoid complex setup
        with patch.object(self.builder, '_create_data_load_step') as mock_create_data_load:
            with patch.object(self.builder, '_create_tabular_preprocess_step') as mock_create_preprocess:
                with patch.object(self.builder, '_create_xgboost_train_step') as mock_create_train:
                    with patch.object(self.builder, '_create_packaging_step') as mock_create_packaging:
                        with patch.object(self.builder, '_create_registration_steps') as mock_create_registration:
                            with patch.object(self.builder, '_create_model_eval_step') as mock_create_eval:
                                # Set up return values for the mocked methods
                                mock_train_load = MagicMock(name="train_load_step")
                                mock_train_preprocess = MagicMock(name="train_preprocess_step")
                                mock_train = MagicMock(name="train_step")
                                mock_packaging = MagicMock(name="packaging_step")
                                mock_registration = [MagicMock(name="registration_step")]
                                mock_calib_load = MagicMock(name="calib_load_step")
                                mock_calib_preprocess = MagicMock(name="calib_preprocess_step")
                                mock_eval = MagicMock(name="eval_step")
                                
                                mock_create_data_load.side_effect = [mock_train_load, mock_calib_load]
                                mock_create_preprocess.side_effect = [mock_train_preprocess, mock_calib_preprocess]
                                mock_create_train.return_value = mock_train
                                mock_create_packaging.return_value = mock_packaging
                                mock_create_registration.return_value = mock_registration
                                mock_create_eval.return_value = mock_eval
                                
                                # Call the method under test
                                pipeline = self.builder.generate_pipeline()
        
        # Verify Pipeline was instantiated with correct parameters
        self.mock_pipeline_cls.assert_called_once()
        call_args = self.mock_pipeline_cls.call_args[1]
        
        self.assertEqual(call_args["name"], f"{self.mock_base_config.pipeline_name}-xgb-train-eval")
        self.assertEqual(len(call_args["parameters"]), 4)  # 4 pipeline parameters
        self.assertEqual(call_args["sagemaker_session"], self.builder.session)
        
        # Verify the returned pipeline is our mock
        self.assertEqual(pipeline, self.mock_pipeline)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
