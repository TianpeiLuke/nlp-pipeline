import sys
import logging
import copy
from pathlib import Path
from typing import Optional, List, Union, Any, Dict

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, Step, TrainingStep
from sagemaker.workflow.functions import Join 
from sagemaker.image_uris import retrieve
from sagemaker.inputs import TrainingInput

from src.pipelines.utils import load_configs

# Config classes
from src.pipelines.config_base import BasePipelineConfig
from src.pipelines.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipelines.config_training_step_xgboost import XGBoostTrainingConfig 
from src.pipelines.config_model_step_xgboost import XGBoostModelCreationConfig
from src.pipelines.config_mims_packaging_step import PackageStepConfig
from src.pipelines.config_mims_registration_step import ModelRegistrationConfig
from src.pipelines.config_mims_payload_step import PayloadConfig


# Step builders
from src.pipelines.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipelines.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipelines.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipelines.builder_model_step_xgboost import XGBoostModelStepBuilder
from src.pipelines.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipelines.builder_mims_registration_step import ModelRegistrationStepBuilder



# Common parameters
from mods_workflow_core.utils.constants import (
    PIPELINE_EXECUTION_TEMP_DIR,
    KMS_ENCRYPTION_KEY_PARAM,
    SECURITY_GROUP_ID,
    VPC_SUBNET,
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import importlib
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
    'TabularPreprocessingConfig': TabularPreprocessingConfig,
    'XGBoostTrainingConfig':      XGBoostTrainingConfig,
    'XGBoostModelCreationConfig': XGBoostModelCreationConfig,
    'PackageStepConfig':          PackageStepConfig,
    'ModelRegistrationConfig':    ModelRegistrationConfig,
    'PayloadConfig':              PayloadConfig
}



# ────────────────────────────────────────────────────────────────────────────────
class MDSXGBoostPipelineBuilder:
    """
    Builds a pipeline that performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) XGBoost Model Creation
    5) MIMS Packaging
    6) Model Registration
    7) Data Loading (for calibration set)
    8) Tabular Preprocessing (for calibration set)
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
        logging.info(f"Loading configs from: {config_path}")
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
        """Validate input channels in preprocessing configs."""
        allowed_inputs = self.REQUIRED_INPUTS | self.OPTIONAL_INPUTS
        
        for cfg in [self.tp_train_cfg, self.tp_test_cfg]:
            missing = self.REQUIRED_INPUTS - set(cfg.input_names.keys())
            if missing:
                raise ValueError(f"TabularPreprocessing config missing required input channels: {missing}")
            
            unknown = set(cfg.input_names.keys()) - allowed_inputs
            if unknown:
                raise ValueError(f"TabularPreprocessing config contains unknown input channels: {unknown}")

    def _find_config_key(self, class_name: str, **attrs: Any) -> str:
        """Find the unique step_name for a config class with given attributes."""
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
        """Extract and validate all required configuration objects."""
        self.base_config = self.configs['Base']

        # Cradle Data Load configs
        self.cradle_train_cfg = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='training')]
        self.cradle_test_cfg  = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='calibration')]

        # Tabular Preprocessing configs
        self.tp_train_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='training')]
        self.tp_test_cfg  = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='calibration')]
        
        # XGBoost Training config
        xgb_train_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostTrainingConfig):
                xgb_train_config_instance = cfg
                break
        
        if not xgb_train_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostTrainingConfig in the config file.")
        
        self.xgb_train_cfg = xgb_train_config_instance

        # XGBoost Model Creation config
        xgb_model_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostModelCreationConfig):
                xgb_model_config_instance = cfg
                break
        
        if not xgb_model_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostModelCreationConfig in the config file.")
        
        self.xgb_model_cfg = xgb_model_config_instance
        
        # Package config
        package_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, PackageStepConfig):
                package_config_instance = cfg
                break
        if not package_config_instance:
            raise ValueError("Could not find a configuration of type PackageStepConfig")
        self.package_cfg = package_config_instance

        # Registration config
        registration_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, ModelRegistrationConfig):
                registration_config_instance = cfg
                break
        if not registration_config_instance:
            raise ValueError("Could not find a configuration of type ModelRegistrationConfig")
        self.registration_cfg = registration_config_instance

        # Payload config
        payload_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, PayloadConfig):
                payload_config_instance = cfg
                break
        if not payload_config_instance:
            raise ValueError("Could not find a configuration of type PayloadConfig")
        self.payload_cfg = payload_config_instance

        # Type checks
        if not all(isinstance(c, CradleDataLoadConfig) for c in [self.cradle_train_cfg, self.cradle_test_cfg]):
            raise TypeError("Expected CradleDataLoadConfig for both training and calibration")
        if not all(isinstance(c, TabularPreprocessingConfig) for c in [self.tp_train_cfg, self.tp_test_cfg]):
            raise TypeError("Expected TabularPreprocessingConfig for both data types")
        if not isinstance(self.xgb_train_cfg, XGBoostTrainingConfig):
            raise TypeError("Expected XGBoostTrainingConfig")
        if not isinstance(self.xgb_model_cfg, XGBoostModelCreationConfig):
            raise TypeError("Expected XGBoostModelCreationConfig")
        if not isinstance(self.package_cfg, PackageStepConfig):
            raise TypeError("Expected PackageStepConfig")
        if not isinstance(self.registration_cfg, ModelRegistrationConfig):
            raise TypeError("Expected ModelRegistrationConfig")
        if not isinstance(self.payload_cfg, PayloadConfig):
            raise TypeError("Expected PayloadConfig")

        # Log successful config extraction
        logger.info("Successfully extracted and validated all configurations")
        logger.debug("Extracted configurations:")
        logger.debug(f"- Base config: {type(self.base_config)}")
        logger.debug(f"- Cradle configs: {type(self.cradle_train_cfg)}, {type(self.cradle_test_cfg)}")
        logger.debug(f"- Preprocessing configs: {type(self.tp_train_cfg)}, {type(self.tp_test_cfg)}")
        logger.debug(f"- XGBoost training config: {type(self.xgb_train_cfg)}")
        logger.debug(f"- XGBoost model config: {type(self.xgb_model_cfg)}")
        logger.debug(f"- Package config: {type(self.package_cfg)}")
        logger.debug(f"- Registration config: {type(self.registration_cfg)}")
        logger.debug(f"- Payload config: {type(self.payload_cfg)}")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]
    
    # --- Methods for Creating Each Step ---

    def _create_data_load_step(self, cradle_config: CradleDataLoadConfig) -> ProcessingStep:
        """Creates a single CradleDataLoadingStep."""
        loader = CradleDataLoadingStepBuilder(config=cradle_config, sagemaker_session=self.session, role=self.role)
        step = loader.create_step()
        self.cradle_loading_requests[step.name] = loader.get_request_dict()
        return step

    def _create_tabular_preprocess_step(self, tp_config: TabularPreprocessingConfig, dependency_step: Step) -> ProcessingStep:
        """Creates a single TabularPreprocessingStep with a dependency."""
        prep_builder = TabularPreprocessingStepBuilder(config=tp_config, sagemaker_session=self.session, role=self.role)
        cradle_outputs = dependency_step.properties.ProcessingOutputConfig.Outputs
        
        inputs = {tp_config.input_names["data_input"]: cradle_outputs[OUTPUT_TYPE_DATA].S3Output.S3Uri}
        outputs = {tp_config.output_names["processed_data"]: f"{self.base_config.pipeline_s3_loc}/tabular_preprocessing/{tp_config.job_type}"}
        
        prep_step = prep_builder.create_step(inputs=inputs, outputs=outputs)
        prep_step.add_depends_on([dependency_step])
        return prep_step
    
    def _create_xgboost_train_step(self, dependency_step: Step) -> TrainingStep:
        """Creates the XGBoost TrainingStep, taking preprocessed data as input."""
        xgb_builder = XGBoostTrainingStepBuilder(config=self.xgb_train_cfg, sagemaker_session=self.session, role=self.role)
        
        # Set the dynamic input path from the previous step's output
        object.__setattr__(
            xgb_builder.config,
            'input_path',
            dependency_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri
        )
        return xgb_builder.create_step(dependencies=[dependency_step])
    
    def _create_model_creation_step(self, dependency_step: Step) -> ModelStep:
        """Creates the XGBoost Model Creation step."""
        model_builder = XGBoostModelStepBuilder(config=self.xgb_model_cfg, sagemaker_session=self.session, role=self.role)
        return model_builder.create_step(
            model_data=dependency_step.properties.ModelArtifacts.S3ModelArtifacts,
            dependencies=[dependency_step]
        )

    def _create_packaging_step(self, dependency_step: Step) -> ProcessingStep:
        """Creates the MIMS Packaging step."""
        packaging_builder = MIMSPackagingStepBuilder(config=self.package_cfg, sagemaker_session=self.session, role=self.role, notebook_root=self.notebook_root)
        return packaging_builder.create_packaging_step(
            model_data=dependency_step.properties.ModelArtifacts.S3ModelArtifacts,
            dependencies=[dependency_step]
        )
    
    def _create_registration_steps(self, dependency_step: ProcessingStep) -> List[ProcessingStep]:
        """Creates the Model Registration steps."""
        registration_builder = ModelRegistrationStepBuilder(config=self.registration_cfg, sagemaker_session=self.session, role=self.role)
        self.payload_cfg.generate_and_upload_payloads()
        
        image_uri = retrieve(
            framework="xgboost",
            region=self.xgb_model_cfg.aws_region,
            version=self.xgb_model_cfg.framework_version,
            py_version=self.xgb_model_cfg.py_version,
            instance_type=self.xgb_model_cfg.inference_instance_type,
            image_scope="inference"
        )
        
        s3_uri = dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
        result = registration_builder.create_step(
            packaging_step_output=s3_uri,
            payload_s3_key=self.payload_cfg.sample_payload_s3_key,
            dependencies=[dependency_step],
            regions=[self.base_config.region]
        )
        
        execution_doc_config = self._create_execution_doc_config(image_uri)

        # Store execution document configs
        if isinstance(result, dict):
            for step in result.values():
                self.registration_configs[step.name] = execution_doc_config
            return list(result.values())
        else:
            self.registration_configs[result.name] = execution_doc_config
            return [result]
        
    def _create_execution_doc_config(self, image_uri: str) -> Dict[str, Any]:
        """Helper to create the execution document configuration dictionary."""
        return {
            "model_domain": self.payload_cfg.model_registration_domain,
            "model_objective": self.payload_cfg.model_registration_objective,
            "source_model_inference_content_types": self.payload_cfg.source_model_inference_content_types,
            "source_model_inference_response_types": self.payload_cfg.source_model_inference_response_types,
            "source_model_inference_input_variable_list": self.payload_cfg.source_model_inference_input_variable_list,
            "source_model_inference_output_variable_list": self.payload_cfg.source_model_inference_output_variable_list,
            "model_registration_region": self.payload_cfg.region,
            "source_model_inference_image_arn": image_uri,
            "source_model_region": self.payload_cfg.aws_region,
            "model_owner": self.payload_cfg.model_owner,
            "source_model_environment_variable_map": {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": self.xgb_model_cfg.inference_entry_point,
                "SAGEMAKER_REGION": self.payload_cfg.aws_region,
                "SAGEMAKER_SUBMIT_DIRECTORY": '/opt/ml/model/code',
            },
            'load_testing_info_map': {
                "sample_payload_s3_bucket": self.payload_cfg.bucket,
                "sample_payload_s3_key": self.payload_cfg.sample_payload_s3_key,
                "expected_tps": self.payload_cfg.expected_tps,
                "max_latency_in_millisecond": self.payload_cfg.max_latency_in_millisecond,
                "instance_type_list": [self.package_cfg.get_instance_type()],
                "max_acceptable_error_rate": self.payload_cfg.max_acceptable_error_rate,
            },
        }

    # --- Pipeline Flow Construction ---
    def _create_training_flow(self) -> List[Step]:
        """Creates the full training flow by connecting individual step creation methods."""
        load_step = self._create_data_load_step(self.cradle_train_cfg)
        prep_step = self._create_tabular_preprocess_step(self.tp_train_cfg, load_step)
        train_step = self._create_xgboost_train_step(prep_step)
        model_step = self._create_model_creation_step(train_step)
        packaging_step = self._create_packaging_step(model_step)
        registration_steps = self._create_registration_steps(packaging_step)
        
        flow_steps = [load_step, prep_step, train_step, model_step, packaging_step] + registration_steps
        logging.info(f"Created training flow: {' -> '.join(s.name for s in flow_steps)}")
        return flow_steps

    def _create_calibration_flow(self) -> List[Step]:
        """Creates the calibration data flow."""
        load_step = self._create_data_load_step(self.cradle_test_cfg)
        prep_step = self._create_tabular_preprocess_step(self.tp_test_cfg, load_step)
        logging.info(f"Created calibration flow: {load_step.name} -> {prep_step.name}")
        return [load_step, prep_step]

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with both Cradle data loading and model registration configurations.
        """
        if not self.cradle_loading_requests or not self.registration_configs:
            raise ValueError("Pipeline must be generated before filling execution document.")

        if "PIPELINE_STEP_CONFIGS" not in execution_document:
            raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
    
        pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

        # Fill Cradle configurations
        for step_name, request_dict in self.cradle_loading_requests.items():
            if step_name not in pipeline_configs:
                logger.warning(f"Cradle step '{step_name}' not found in execution document")
                continue
            pipeline_configs[step_name]["STEP_CONFIG"] = request_dict
            logger.info(f"Updated execution config for Cradle step: {step_name}")

        # Fill Registration configurations
        for step_name, config in self.registration_configs.items():
            registration_step_name = f"Registration_{self.registration_cfg.region}"
            if registration_step_name not in pipeline_configs:
                logger.warning(f"Registration step '{registration_step_name}' not found in execution document")
                continue
            pipeline_configs[registration_step_name]["STEP_CONFIG"] = config
            logger.info(f"Updated execution config for registration step: {registration_step_name}")

        return execution_document

    def generate_pipeline(self) -> Pipeline:
        """Generate the complete pipeline with all steps."""
        pipeline_name = f"{self.base_config.pipeline_name}-xgb-e2e"
        logging.info(f"Building pipeline: {pipeline_name}")
        
        self.cradle_loading_requests.clear()
        self.registration_configs.clear()
        
        logging.info("Creating training flow...")
        training_steps = self._create_training_flow()
        logging.info("Creating calibration flow...")
        calibration_steps = self._create_calibration_flow()

        all_steps = training_steps + calibration_steps
        logging.info(f"Created pipeline with {len(all_steps)} steps")
        
        return Pipeline(
            name=pipeline_name,
            parameters=self._get_pipeline_parameters(),
            steps=all_steps,
            sagemaker_session=self.session
        )

