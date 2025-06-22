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
from pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end import PytorchEndToEndPipelineBuilder
from src.pipelines.config_base import BasePipelineConfig
from src.pipelines.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipelines.config_training_step_pytorch import PytorchTrainingConfig
from src.pipelines.config_model_step_pytorch import PytorchModelCreationConfig
from src.pipelines.config_mims_packaging_step import PackageStepConfig
from src.pipelines.config_mims_registration_step import ModelRegistrationConfig
from src.pipelines.config_mims_payload_step import PayloadConfig

class TestPytorchEndToEndPipelineBuilder(unittest.TestCase):
    def setUp(self):
        """Set up mocks and patches for testing."""
        # Mock the load_configs function
        self.load_configs_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.load_configs')
        self.mock_load_configs = self.load_configs_patch.start()
        
        # Create mock configs
        self.mock_base_config = MagicMock(spec=BasePipelineConfig)
        self.mock_base_config.pipeline_name = "test-pipeline"
        self.mock_base_config.pipeline_s3_loc = "s3://test-bucket/test-pipeline"
        self.mock_base_config.region = "us-west-2"
        
        self.mock_cradle_train_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_train_cfg.job_type = "training"
        
        self.mock_cradle_test_cfg = MagicMock(spec=CradleDataLoadConfig)
        self.mock_cradle_test_cfg.job_type = "calibration"
        
        self.mock_tp_train_cfg = MagicMock(spec=TabularPreprocessingConfig)
        self.mock_tp_train_cfg.job_type = "training"
        self.mock_tp_train_cfg.input_names = {"data_input": "DataInput"}
        self.mock_tp_train_cfg.output_names = {"processed_data": "ProcessedTabularData"}
        
        self.mock_tp_test_cfg = MagicMock(spec=TabularPreprocessingConfig)
        self.mock_tp_test_cfg.job_type = "calibration"
        self.mock_tp_test_cfg.input_names = {"data_input": "DataInput"}
        self.mock_tp_test_cfg.output_names = {"processed_data": "ProcessedTabularData"}
        
        self.mock_pytorch_train_cfg = MagicMock(spec=PytorchTrainingConfig)
        self.mock_pytorch_model_cfg = MagicMock(spec=PytorchModelCreationConfig)
        self.mock_package_cfg = MagicMock(spec=PackageStepConfig)
        self.mock_registration_cfg = MagicMock(spec=ModelRegistrationConfig)
        self.mock_payload_cfg = MagicMock(spec=PayloadConfig)
        
        # Set up mock configs dictionary
        self.mock_configs = {
            'Base': self.mock_base_config,
            'CradleDataLoadConfig_training': self.mock_cradle_train_cfg,
            'CradleDataLoadConfig_calibration': self.mock_cradle_test_cfg,
            'TabularPreprocessingConfig_training': self.mock_tp_train_cfg,
            'TabularPreprocessingConfig_calibration': self.mock_tp_test_cfg,
            'PytorchTrainingConfig': self.mock_pytorch_train_cfg,
            'PytorchModelCreationConfig': self.mock_pytorch_model_cfg,
            'PackageStepConfig': self.mock_package_cfg,
            'ModelRegistrationConfig': self.mock_registration_cfg,
            'PayloadConfig': self.mock_payload_cfg
        }
        
        # Configure load_configs to return our mock configs
        self.mock_load_configs.return_value = self.mock_configs
        
        # Mock the step builders
        self.cradle_builder_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.CradleDataLoadingStepBuilder')
        self.mock_cradle_builder_cls = self.cradle_builder_patch.start()
        self.mock_cradle_builder = MagicMock()
        self.mock_cradle_builder_cls.return_value = self.mock_cradle_builder
        self.mock_cradle_builder.create_step.return_value = MagicMock(name="cradle_step")
        self.mock_cradle_builder.get_request_dict.return_value = {"request": "data"}
        
        self.tp_builder_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.TabularPreprocessingStepBuilder')
        self.mock_tp_builder_cls = self.tp_builder_patch.start()
        self.mock_tp_builder = MagicMock()
        self.mock_tp_builder_cls.return_value = self.mock_tp_builder
        self.mock_tp_builder.create_step.return_value = MagicMock(name="tp_step")
        
        self.pytorch_train_builder_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.PyTorchTrainingStepBuilder')
        self.mock_pytorch_train_builder_cls = self.pytorch_train_builder_patch.start()
        self.mock_pytorch_train_builder = MagicMock()
        self.mock_pytorch_train_builder_cls.return_value = self.mock_pytorch_train_builder
        self.mock_pytorch_train_builder.create_step.return_value = MagicMock(name="pytorch_train_step")
        
        self.pytorch_model_builder_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.PytorchModelStepBuilder')
        self.mock_pytorch_model_builder_cls = self.pytorch_model_builder_patch.start()
        self.mock_pytorch_model_builder = MagicMock()
        self.mock_pytorch_model_builder_cls.return_value = self.mock_pytorch_model_builder
        self.mock_pytorch_model_builder.create_step.return_value = MagicMock(name="pytorch_model_step")
        
        self.packaging_builder_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.MIMSPackagingStepBuilder')
        self.mock_packaging_builder_cls = self.packaging_builder_patch.start()
        self.mock_packaging_builder = MagicMock()
        self.mock_packaging_builder_cls.return_value = self.mock_packaging_builder
        self.mock_packaging_builder.create_packaging_step.return_value = MagicMock(name="packaging_step")
        
        self.registration_builder_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.ModelRegistrationStepBuilder')
        self.mock_registration_builder_cls = self.registration_builder_patch.start()
        self.mock_registration_builder = MagicMock()
        self.mock_registration_builder_cls.return_value = self.mock_registration_builder
        self.mock_registration_builder.create_step.return_value = MagicMock(name="registration_step")
        
        # Mock Pipeline
        self.pipeline_patch = patch('pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end.Pipeline')
        self.mock_pipeline_cls = self.pipeline_patch.start()
        self.mock_pipeline = MagicMock()
        self.mock_pipeline_cls.return_value = self.mock_pipeline
        
        # Mock OUTPUT_TYPE constants
        self.constants_patch = patch.multiple(
            'pipeline_examples.pytorch_bsm.builder_pipeline_pytorch_end_to_end',
            OUTPUT_TYPE_DATA="DataOutput",
            OUTPUT_TYPE_METADATA="MetadataOutput",
            OUTPUT_TYPE_SIGNATURE="SignatureOutput"
        )
        self.constants_patch.start()
        
        # Create the builder instance
        self.builder = PytorchEndToEndPipelineBuilder(
            config_path="dummy/path/to/config.json",
            sagemaker_session=MagicMock(),
            role="arn:aws:iam::123456789012:role/SageMakerRole",
            notebook_root=Path("/dummy/notebook/root")
        )

    def tearDown(self):
        """Clean up patches after each test."""
        self.load_configs_patch.stop()
        self.cradle_builder_patch.stop()
        self.tp_builder_patch.stop()
        self.pytorch_train_builder_patch.stop()
        self.pytorch_model_builder_patch.stop()
        self.packaging_builder_patch.stop()
        self.registration_builder_patch.stop()
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
        self.assertEqual(self.builder.cradle_test_cfg, self.mock_cradle_test_cfg)
        self.assertEqual(self.builder.tp_train_cfg, self.mock_tp_train_cfg)
        self.assertEqual(self.builder.tp_test_cfg, self.mock_tp_test_cfg)
        self.assertEqual(self.builder.pytorch_train_cfg, self.mock_pytorch_train_cfg)
        self.assertEqual(self.builder.pytorch_model_cfg, self.mock_pytorch_model_cfg)
        self.assertEqual(self.builder.package_cfg, self.mock_package_cfg)
        self.assertEqual(self.builder.registration_cfg, self.mock_registration_cfg)
        self.assertEqual(self.builder.payload_cfg, self.mock_payload_cfg)

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

    def test_create_data_load_step(self):
        """Test that _create_data_load_step creates a step correctly."""
        step = self.builder._create_data_load_step(self.mock_cradle_train_cfg)
        
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
        
        # Verify the step was stored in cradle_loading_requests
        self.assertIn(step.name, self.builder.cradle_loading_requests)
        self.assertEqual(self.builder.cradle_loading_requests[step.name], {"request": "data"})
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_cradle_builder.create_step.return_value)

    def test_create_tabular_preprocess_step(self):
        """Test that _create_tabular_preprocess_step creates a step correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ProcessingOutputConfig.Outputs = {
            "DataOutput": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data"))
        }
        
        step = self.builder._create_tabular_preprocess_step(self.mock_tp_train_cfg, mock_dependency)
        
        # Verify TabularPreprocessingStepBuilder was instantiated with correct parameters
        self.mock_tp_builder_cls.assert_called_once_with(
            config=self.mock_tp_train_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify create_step was called with correct parameters
        self.mock_tp_builder.create_step.assert_called_once_with(
            inputs={"DataInput": "s3://bucket/data"},
            outputs={"ProcessedTabularData": f"{self.mock_base_config.pipeline_s3_loc}/tabular_preprocessing/training"}
        )
        
        # Verify add_depends_on was called with the dependency step
        step.add_depends_on.assert_called_once_with([mock_dependency])
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_tp_builder.create_step.return_value)

    def test_create_pytorch_train_step(self):
        """Test that _create_pytorch_train_step creates a step correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ProcessingOutputConfig.Outputs = {
            "ProcessedTabularData": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/processed_data"))
        }
        
        step = self.builder._create_pytorch_train_step(mock_dependency)
        
        # Verify PyTorchTrainingStepBuilder was instantiated with correct parameters
        self.mock_pytorch_train_builder_cls.assert_called_once()
        
        # Verify input_path and output_path were set correctly
        self.assertEqual(self.mock_pytorch_train_builder.config.input_path, "s3://bucket/processed_data")
        self.assertEqual(self.mock_pytorch_train_builder.config.output_path, f"{self.mock_base_config.pipeline_s3_loc}/pytorch_model_artifacts")
        
        # Verify create_step was called with correct parameters
        self.mock_pytorch_train_builder.create_step.assert_called_once_with(dependencies=[mock_dependency])
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_pytorch_train_builder.create_step.return_value)

    def test_create_model_creation_step(self):
        """Test that _create_model_creation_step creates a step correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        step = self.builder._create_model_creation_step(mock_dependency)
        
        # Verify PytorchModelStepBuilder was instantiated with correct parameters
        self.mock_pytorch_model_builder_cls.assert_called_once()
        
        # Verify create_step was called with correct parameters
        self.mock_pytorch_model_builder.create_step.assert_called_once_with(
            model_data="s3://bucket/model.tar.gz",
            dependencies=[mock_dependency]
        )
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_pytorch_model_builder.create_step.return_value)

    def test_create_packaging_step(self):
        """Test that _create_packaging_step creates a step correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.model_artifacts_path = "s3://bucket/model.tar.gz"
        
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

    def test_create_registration_steps(self):
        """Test that _create_registration_steps creates steps correctly."""
        # Create a mock dependency step
        mock_dependency = MagicMock()
        mock_dependency.properties.ProcessingOutputConfig.Outputs = [
            MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/packaged_model"))
        ]
        
        steps = self.builder._create_registration_steps(mock_dependency)
        
        # Verify ModelRegistrationStepBuilder was instantiated with correct parameters
        self.mock_registration_builder_cls.assert_called_once_with(
            config=self.mock_registration_cfg,
            sagemaker_session=self.builder.session,
            role=self.builder.role
        )
        
        # Verify generate_and_upload_payloads was called
        self.mock_payload_cfg.generate_and_upload_payloads.assert_called_once()
        
        # Verify create_step was called with correct parameters
        self.mock_registration_builder.create_step.assert_called_once_with(
            packaging_step_output="s3://bucket/packaged_model",
            payload_s3_key=self.mock_payload_cfg.sample_payload_s3_key,
            dependencies=[mock_dependency],
            regions=[self.mock_base_config.region]
        )
        
        # Verify the returned steps list contains our mock
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0], self.mock_registration_builder.create_step.return_value)
        
        # Verify the step was stored in registration_configs
        self.assertIn(steps[0].name, self.builder.registration_configs)

    def test_create_training_flow(self):
        """Test that _create_training_flow creates the full training flow correctly."""
        steps = self.builder._create_training_flow()
        
        # Verify all the step creation methods were called
        self.mock_cradle_builder.create_step.assert_called_once()
        self.mock_tp_builder.create_step.assert_called()
        self.mock_pytorch_train_builder.create_step.assert_called_once()
        self.mock_pytorch_model_builder.create_step.assert_called_once()
        self.mock_packaging_builder.create_packaging_step.assert_called_once()
        self.mock_registration_builder.create_step.assert_called_once()
        
        # Verify the returned steps list contains all the expected steps
        self.assertEqual(len(steps), 6)  # 5 main steps + 1 registration step

    def test_create_calibration_flow(self):
        """Test that _create_calibration_flow creates the calibration flow correctly."""
        steps = self.builder._create_calibration_flow()
        
        # Verify the step creation methods were called
        self.mock_cradle_builder.create_step.assert_called()
        self.mock_tp_builder.create_step.assert_called()
        
        # Verify the returned steps list contains the expected steps
        self.assertEqual(len(steps), 2)

    def test_generate_pipeline(self):
        """Test that generate_pipeline creates a complete pipeline correctly."""
        pipeline = self.builder.generate_pipeline()
        
        # Verify the cradle_loading_requests and registration_configs were cleared
        self.assertEqual(len(self.builder.cradle_loading_requests), 2)  # 2 cradle steps (train + calibration)
        self.assertEqual(len(self.builder.registration_configs), 1)  # 1 registration step
        
        # Verify Pipeline was instantiated with correct parameters
        self.mock_pipeline_cls.assert_called_once()
        call_args = self.mock_pipeline_cls.call_args[1]
        
        self.assertEqual(call_args["name"], f"{self.mock_base_config.pipeline_name}-pytorch-e2e")
        self.assertEqual(len(call_args["parameters"]), 4)  # 4 pipeline parameters
        self.assertEqual(len(call_args["steps"]), 8)  # 6 training steps + 2 calibration steps
        self.assertEqual(call_args["sagemaker_session"], self.builder.session)
        
        # Verify the returned pipeline is our mock
        self.assertEqual(pipeline, self.mock_pipeline)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
