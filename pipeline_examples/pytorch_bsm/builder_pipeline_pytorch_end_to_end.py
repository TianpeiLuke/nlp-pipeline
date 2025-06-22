import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import Step, ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep

from src.pipeline_steps.utils import load_configs

# Config classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_pytorch import PytorchTrainingConfig
from src.pipeline_steps.config_model_step_pytorch import PytorchModelCreationConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.builder_training_step_pytorch import PyTorchTrainingStepBuilder
from src.pipeline_steps.builder_model_step_pytorch import PytorchModelStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder

# Common parameters
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir", default_value="/tmp")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMSEncryptionKey", default_value="")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId", default_value="")
VPC_SUBNET = ParameterString(name="VPCEndpointSubnet", default_value="")

import os
import importlib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamically import pipeline constants if the environment variable is set
SECUREAI_PIPELINE_CONSTANTS_MODULE = os.environ.get("SECUREAI_PIPELINE_CONSTANTS_MODULE")
OUTPUT_TYPE_DATA = OUTPUT_TYPE_METADATA = OUTPUT_TYPE_SIGNATURE = None
if SECUREAI_PIPELINE_CONSTANTS_MODULE:
    try:
        const_mod = importlib.import_module(SECUREAI_PIPELINE_CONSTANTS_MODULE)
        OUTPUT_TYPE_DATA      = getattr(const_mod, "OUTPUT_TYPE_DATA",      None)
        OUTPUT_TYPE_METADATA  = getattr(const_mod, "OUTPUT_TYPE_METADATA",  None)
        OUTPUT_TYPE_SIGNATURE = getattr(const_mod, "OUTPUT_TYPE_SIGNATURE", None)
        logger.info(f"Imported pipeline constants from {SECUREAI_PIPELINE_CONSTANTS_MODULE}")
    except ImportError as e:
        logger.error(f"Could not import pipeline constants: {e}")
else:
    logger.warning(
        "SECUREAI_PIPELINE_CONSTANTS_MODULE not set; "
        "pipeline constants (DATA, METADATA, SIGNATURE) unavailable."
    )

# Map JSON keys → Pydantic classes
CONFIG_CLASSES = {
    'BasePipelineConfig':         BasePipelineConfig,
    'CradleDataLoadConfig':       CradleDataLoadConfig,
    'ProcessingStepConfigBase':   ProcessingStepConfigBase,
    'TabularPreprocessingConfig': TabularPreprocessingConfig,
    'PytorchTrainingConfig':      PytorchTrainingConfig,
    'PytorchModelCreationConfig': PytorchModelCreationConfig,
    'PackageStepConfig':          PackageStepConfig,
    'ModelRegistrationConfig':    ModelRegistrationConfig,
    'PayloadConfig':              PayloadConfig
}

class PytorchEndToEndPipelineBuilder:
    """
    Builds an end-to-end pipeline for PyTorch model training and registration.
    Steps:
    1) Cradle Data Load (training set)
    2) Tabular Preprocessing (training set)
    3) PyTorch Training
    4) PyTorch Model Creation
    5) MIMS Packaging
    6) Model Registration
    7) Cradle Data Load (calibration set)
    8) Tabular Preprocessing (calibration set)
    """
    REQUIRED_INPUTS = {"data_input"}
    OPTIONAL_INPUTS = {"metadata_input", "signature_input"}

    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        logger.info(f"Loading configs from: {config_path}")
        self.configs = load_configs(config_path, CONFIG_CLASSES)
        self._extract_configs()
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        self.cradle_loading_requests: Dict[str, Dict[str, Any]] = {}
        self.registration_configs: Dict[str, Dict[str, Any]] = {}
        self._validate_preprocessing_inputs()
        logger.info(f"Initialized builder for: {self.base_config.pipeline_name}")

    def _validate_preprocessing_inputs(self):
        allowed_inputs = self.REQUIRED_INPUTS | self.OPTIONAL_INPUTS
        for cfg in [self.tp_train_cfg, self.tp_test_cfg]:
            missing = self.REQUIRED_INPUTS - set(cfg.input_names.keys())
            if missing:
                raise ValueError(f"TabularPreprocessing config missing required input channels: {missing}")
            unknown = set(cfg.input_names.keys()) - allowed_inputs
            if unknown:
                raise ValueError(f"TabularPreprocessing config contains unknown input channels: {unknown}")

    def _find_config_key(self, class_name: str, **attrs: Any) -> str:
        base = BasePipelineConfig.get_step_name(class_name)
        candidates = [
            step_name for step_name in self.configs
            if step_name.startswith(base + "_") and
               all(str(val) in step_name[len(base) + 1:].split("_") for val in attrs.values())
        ]
        if not candidates:
            raise ValueError(f"No config found for {class_name} with {attrs}")
        if len(candidates) > 1:
            raise ValueError(f"Multiple configs found for {class_name} with {attrs}: {candidates}")
        return candidates[0]

    def _extract_configs(self):
        self.base_config = self.configs['Base']
        self.cradle_train_cfg = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='training')]
        self.cradle_test_cfg  = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='calibration')]
        self.tp_train_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='training')]
        self.tp_test_cfg  = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='calibration')]
        pytorch_train_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, PytorchTrainingConfig):
                pytorch_train_config_instance = cfg
                break
        if not pytorch_train_config_instance:
            raise ValueError("Could not find a configuration of type PytorchTrainingConfig in the config file.")
        self.pytorch_train_cfg = pytorch_train_config_instance
        pytorch_model_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, PytorchModelCreationConfig):
                pytorch_model_config_instance = cfg
                break
        if not pytorch_model_config_instance:
            raise ValueError("Could not find a configuration of type PytorchModelCreationConfig in the config file.")
        self.pytorch_model_cfg = pytorch_model_config_instance
        package_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, PackageStepConfig):
                package_config_instance = cfg
                break
        if not package_config_instance:
            raise ValueError("Could not find a configuration of type PackageStepConfig")
        self.package_cfg = package_config_instance
        registration_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, ModelRegistrationConfig):
                registration_config_instance = cfg
                break
        if not registration_config_instance:
            raise ValueError("Could not find a configuration of type ModelRegistrationConfig")
        self.registration_cfg = registration_config_instance
        payload_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, PayloadConfig):
                payload_config_instance = cfg
                break
        if not payload_config_instance:
            raise ValueError("Could not find a configuration of type PayloadConfig")
        self.payload_cfg = payload_config_instance
        if not all(isinstance(c, CradleDataLoadConfig) for c in [self.cradle_train_cfg, self.cradle_test_cfg]):
            raise TypeError("Expected CradleDataLoadConfig for both training and calibration")
        if not all(isinstance(c, TabularPreprocessingConfig) for c in [self.tp_train_cfg, self.tp_test_cfg]):
            raise TypeError("Expected TabularPreprocessingConfig for both data types")
        if not isinstance(self.pytorch_train_cfg, PytorchTrainingConfig):
            raise TypeError("Expected PytorchTrainingConfig")
        if not isinstance(self.pytorch_model_cfg, PytorchModelCreationConfig):
            raise TypeError("Expected PytorchModelCreationConfig")
        if not isinstance(self.package_cfg, PackageStepConfig):
            raise TypeError("Expected PackageStepConfig")
        if not isinstance(self.registration_cfg, ModelRegistrationConfig):
            raise TypeError("Expected ModelRegistrationConfig")
        if not isinstance(self.payload_cfg, PayloadConfig):
            raise TypeError("Expected PayloadConfig")
        logger.info("Successfully extracted and validated all configurations")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

    # --- Step creation methods ---
    def _create_data_load_step(self, cradle_config: CradleDataLoadConfig) -> ProcessingStep:
        loader = CradleDataLoadingStepBuilder(config=cradle_config, sagemaker_session=self.session, role=self.role)
        step = loader.create_step()
        self.cradle_loading_requests[step.name] = loader.get_request_dict()
        return step

    def _create_tabular_preprocess_step(self, tp_config: TabularPreprocessingConfig, dependency_step: Step) -> ProcessingStep:
        prep_builder = TabularPreprocessingStepBuilder(config=tp_config, sagemaker_session=self.session, role=self.role)
        cradle_outputs = dependency_step.properties.ProcessingOutputConfig.Outputs
        inputs = {tp_config.input_names["data_input"]: cradle_outputs[OUTPUT_TYPE_DATA].S3Output.S3Uri}
        outputs = {tp_config.output_names["processed_data"]: f"{self.base_config.pipeline_s3_loc}/tabular_preprocessing/{tp_config.job_type}"}
        prep_step = prep_builder.create_step(inputs=inputs, outputs=outputs)
        prep_step.add_depends_on([dependency_step])
        return prep_step

    def _create_pytorch_train_step(self, dependency_step: Step) -> TrainingStep:
        # Force the model training region to be NA
        temp_train_cfg = self.pytorch_train_cfg.model_copy()
        temp_train_cfg.region = 'NA'
        temp_train_cfg.aws_region = 'us-east-1'
        logger.info(f"Force Model Training in aws region: {temp_train_cfg.aws_region}")
        pytorch_builder = PyTorchTrainingStepBuilder(config=temp_train_cfg, sagemaker_session=self.session, role=self.role)
        # Set the dynamic input_path from the previous step's output
        # The output of tabular preprocess step is a folder with train/val/test.parquet
        object.__setattr__(
            pytorch_builder.config,
            'input_path',
            dependency_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri
        )
        # Set the output path explicitly
        output_path = f"{self.base_config.pipeline_s3_loc}/pytorch_model_artifacts"
        object.__setattr__(pytorch_builder.config, 'output_path', output_path)
        return pytorch_builder.create_step(dependencies=[dependency_step])

    def _create_model_creation_step(self, dependency_step: Step) -> ModelStep:
        # Force the model creation region to be NA
        temp_model_cfg = self.pytorch_model_cfg.model_copy()
        temp_model_cfg.region = 'NA'
        temp_model_cfg.aws_region = 'us-east-1'
        logger.info(f"Force Model Creation in aws region: {temp_model_cfg.aws_region}")
        model_builder = PytorchModelStepBuilder(config=temp_model_cfg, sagemaker_session=self.session, role=self.role)
        return model_builder.create_step(
            model_data=dependency_step.properties.ModelArtifacts.S3ModelArtifacts,
            dependencies=[dependency_step]
        )

    def _create_packaging_step(self, dependency_step: Step) -> ProcessingStep:
        packaging_builder = MIMSPackagingStepBuilder(config=self.package_cfg, sagemaker_session=self.session, role=self.role, notebook_root=self.notebook_root)
        return packaging_builder.create_packaging_step(
            model_data=dependency_step.model_artifacts_path,
            dependencies=[dependency_step]
        )

    def _create_registration_steps(self, dependency_step: ProcessingStep) -> List[ProcessingStep]:
        registration_builder = ModelRegistrationStepBuilder(config=self.registration_cfg, sagemaker_session=self.session, role=self.role)
        self.payload_cfg.generate_and_upload_payloads()
        # The output S3 URI for the packaged model
        s3_uri = dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
        result = registration_builder.create_step(
            packaging_step_output=s3_uri,
            payload_s3_key=self.payload_cfg.sample_payload_s3_key,
            dependencies=[dependency_step],
            regions=[self.base_config.region]
        )
        # Store registration config for execution doc
        if isinstance(result, dict):
            for step in result.values():
                self.registration_configs[step.name] = {}  # Fill as needed
            return list(result.values())
        else:
            self.registration_configs[result.name] = {}
            return [result]

    def _create_training_flow(self) -> List[Step]:
        load_step = self._create_data_load_step(self.cradle_train_cfg)
        prep_step = self._create_tabular_preprocess_step(self.tp_train_cfg, load_step)
        train_step = self._create_pytorch_train_step(prep_step)
        model_step = self._create_model_creation_step(train_step)
        packaging_step = self._create_packaging_step(model_step)
        registration_steps = self._create_registration_steps(packaging_step)
        flow_steps = [load_step, prep_step, train_step, model_step, packaging_step] + registration_steps
        logger.info(f"Created training flow: {' -> '.join(s.name for s in flow_steps)}")
        return flow_steps

    def _create_calibration_flow(self) -> List[Step]:
        load_step = self._create_data_load_step(self.cradle_test_cfg)
        prep_step = self._create_tabular_preprocess_step(self.tp_test_cfg, load_step)
        logger.info(f"Created calibration flow: {load_step.name} -> {prep_step.name}")
        return [load_step, prep_step]

    def generate_pipeline(self) -> Pipeline:
        pipeline_name = f"{self.base_config.pipeline_name}-pytorch-e2e"
        logger.info(f"Building pipeline: {pipeline_name}")
        self.cradle_loading_requests.clear()
        self.registration_configs.clear()
        logger.info("Creating training flow...")
        training_steps = self._create_training_flow()
        logger.info("Creating calibration flow...")
        calibration_steps = self._create_calibration_flow()
        all_steps = training_steps + calibration_steps
        logger.info(f"Created pipeline with {len(all_steps)} steps")
        return Pipeline(
            name=pipeline_name,
            parameters=self._get_pipeline_parameters(),
            steps=all_steps,
            sagemaker_session=self.session
        )
