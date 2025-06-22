import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from sagemaker.image_uris import retrieve

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    Step
)

from src.pipelines.utils import load_configs

# Config classes
from src.pipelines.config_base import BasePipelineConfig
from src.pipelines.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipelines.config_processing_step_base import ProcessingStepConfigBase
from src.pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipelines.config_training_step_xgboost import XGBoostTrainingConfig 
from src.pipelines.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipelines.config_mims_packaging_step import PackageStepConfig
from src.pipelines.config_mims_registration_step import ModelRegistrationConfig
from src.pipelines.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipelines.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipelines.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipelines.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipelines.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from src.pipelines.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipelines.builder_mims_registration_step import ModelRegistrationStepBuilder


PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir", default_value="/tmp")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMSEncryptionKey", default_value="")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId", default_value="")
VPC_SUBNET = ParameterString(name="VPCEndpointSubnet", default_value="")

import os
import importlib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

CONFIG_CLASSES = {
    'BasePipelineConfig':         BasePipelineConfig,
    'CradleDataLoadConfig':       CradleDataLoadConfig,
    'ProcessingStepConfigBase':   ProcessingStepConfigBase,
    'TabularPreprocessingConfig': TabularPreprocessingConfig,
    'XGBoostTrainingConfig':      XGBoostTrainingConfig,
    'XGBoostModelEvalConfig':     XGBoostModelEvalConfig,
    'PackageStepConfig':          PackageStepConfig,
    'ModelRegistrationConfig':    ModelRegistrationConfig,
    'PayloadConfig':              PayloadConfig
}


# ────────────────────────────────────────────────────────────────────────────────
class XGBoostTrainEvaluateE2EPipelineBuilder:
    """
    Builds a pipeline that performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) Packaging
    5) MIMS Registration
    6) Data Loading (for calibration set)
    7) Tabular Preprocessing (for calibration set)
    8) Model Evaluation (on calibration set)
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
        """Validate input channels in preprocessing configs."""
        allowed_inputs = self.REQUIRED_INPUTS | self.OPTIONAL_INPUTS
        
        for cfg in [self.tp_train_cfg, self.tp_calib_cfg]:
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
        
        # Cradle Data Load configs
        self.cradle_train_cfg = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='training')]
        self.cradle_calib_cfg = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='calibration')]
        
        # Tabular Preprocessing configs
        self.tp_train_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='training')]
        self.tp_calib_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='calibration')]
        
        # XGBoost Training config
        xgb_train_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostTrainingConfig):
                xgb_train_config_instance = cfg
                break
        if not xgb_train_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostTrainingConfig in the config file.")
        self.xgb_train_cfg = xgb_train_config_instance
        
        # XGBoost Model Evaluation config
        xgb_eval_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostModelEvalConfig):
                xgb_eval_config_instance = cfg
                break
        if not xgb_eval_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostModelEvalConfig in the config file.")
        self.xgb_eval_cfg = xgb_eval_config_instance
        
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
        if not all(isinstance(c, CradleDataLoadConfig) for c in [self.cradle_train_cfg, self.cradle_calib_cfg]):
            raise TypeError("Expected CradleDataLoadConfig for both training and calibration")
        if not all(isinstance(c, TabularPreprocessingConfig) for c in [self.tp_train_cfg, self.tp_calib_cfg]):
            raise TypeError("Expected TabularPreprocessingConfig for both data types")
        if not isinstance(self.xgb_train_cfg, XGBoostTrainingConfig):
            raise TypeError("Expected XGBoostTrainingConfig")
        if not isinstance(self.package_cfg, PackageStepConfig):
            raise TypeError("Expected PackageStepConfig")
        if not isinstance(self.registration_cfg, ModelRegistrationConfig):
            raise TypeError("Expected ModelRegistrationConfig")
        if not isinstance(self.payload_cfg, PayloadConfig):
            raise TypeError("Expected PayloadConfig")
        if not isinstance(self.xgb_eval_cfg, XGBoostModelEvalConfig):
            raise TypeError("Expected XGBoostModelEvalConfig")

        # Log successful config extraction
        logger.info("Successfully extracted and validated all configurations")
        logger.debug("Extracted configurations:")
        logger.debug(f"- Base config: {type(self.base_config)}")
        logger.debug(f"- Cradle configs: {type(self.cradle_train_cfg)}, {type(self.cradle_calib_cfg)}")
        logger.debug(f"- Preprocessing configs: {type(self.tp_train_cfg)}, {type(self.tp_calib_cfg)}")
        logger.debug(f"- XGBoost training config: {type(self.xgb_train_cfg)}")
        logger.debug(f"- Package config: {type(self.package_cfg)}")
        logger.debug(f"- Registration config: {type(self.registration_cfg)}")
        logger.debug(f"- Payload config: {type(self.payload_cfg)}")
        logger.debug(f"- Model Evaluation config: {type(self.xgb_eval_cfg)}")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

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

    def _create_xgboost_train_step(self, dependency_step: Step) -> TrainingStep:
        temp_train_cfg = self.xgb_train_cfg.model_copy()
        temp_train_cfg.region = 'NA'
        temp_train_cfg.aws_region = 'us-east-1'
        logger.info(f"Force Model Training in aws region: {temp_train_cfg.aws_region}")
        xgb_builder = XGBoostTrainingStepBuilder(config=temp_train_cfg, sagemaker_session=self.session, role=self.role)
        object.__setattr__(
            xgb_builder.config,
            'input_path',
            dependency_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri
        )
        output_path = f"{self.base_config.pipeline_s3_loc}/xgboost_model_artifacts"
        object.__setattr__(xgb_builder.config, 'output_path', output_path)
        return xgb_builder.create_step(dependencies=[dependency_step])

    def _create_packaging_step(self, dependency_step: Step) -> ProcessingStep:
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
            framework=self.registration_cfg.framework,
            region=self.registration_cfg.aws_region,
            version=self.registration_cfg.framework_version,
            py_version=self.registration_cfg.py_version,
            instance_type=self.registration_cfg.inference_instance_type,
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

    def _create_model_eval_step(self, train_step: TrainingStep, calib_preprocess_step: ProcessingStep) -> ProcessingStep:
        """
        Create the model evaluation step, connecting training output and calibration preprocessing output.
        """
        eval_builder = XGBoostModelEvalStepBuilder(
            config=self.xgb_eval_cfg,
            sagemaker_session=self.session,
            role=self.role,
            notebook_root=self.notebook_root
        )

        # Get the effective source directory for code
        effective_source_dir = self.xgb_eval_cfg.get_effective_source_dir()
        if not effective_source_dir:
            raise ValueError("Neither processing_source_dir nor source_dir is specified for model evaluation")

        logger.info(f"Using code source directory: {effective_source_dir}")
        
        # Construct and validate the full path to the evaluation script
        eval_script_path = os.path.join(effective_source_dir, self.xgb_eval_cfg.processing_entry_point)
        if not os.path.exists(eval_script_path):
            raise ValueError(
                f"Evaluation script not found at: {eval_script_path}. "
                f"Make sure entry point {self.xgb_eval_cfg.processing_entry_point} exists in {effective_source_dir}"
            )

        logger.info(f"Found evaluation script at: {eval_script_path}")

        # Get input and output channel names from config class constants
        input_channels = self.xgb_eval_cfg.get_input_names()
        output_channels = self.xgb_eval_cfg.get_output_names()
    
        # Create inputs dictionary using fixed channel names
        inputs = {
            "model_input": train_step.properties.ModelArtifacts.S3ModelArtifacts,
            "eval_data_input": calib_preprocess_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri,
        }

        # Validate that all required input channels are provided
        missing_inputs = set(input_channels.keys()) - set(inputs.keys())
        if missing_inputs:
            raise ValueError(f"Missing required input channels: {missing_inputs}")

        # Create outputs dictionary using fixed channel names
        outputs = {
            "eval_output": f"{self.base_config.pipeline_s3_loc}/model_eval/eval_predictions",
            "metrics_output": f"{self.base_config.pipeline_s3_loc}/model_eval/eval_metrics"
        }

        # Validate that all required output channels are provided
        missing_outputs = set(output_channels.keys()) - set(outputs.keys())
        if missing_outputs:
            raise ValueError(f"Missing required output channels: {missing_outputs}")

        logger.info(f"Creating model evaluation step with inputs: {inputs}")
        logger.info(f"Creating model evaluation step with outputs: {outputs}")

        # Create the step with all inputs and outputs
        eval_step = eval_builder.create_step(
            inputs=inputs,
            outputs=outputs,
            dependencies=[train_step, calib_preprocess_step]
        )

        logger.info(f"Created model evaluation step: {eval_step.name}")
        return eval_step
    
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
                "SAGEMAKER_PROGRAM": self.payload_cfg.inference_entry_point,
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
        pipeline_name = f"{self.base_config.pipeline_name}-xgb-train-eval"
        logger.info(f"Building pipeline: {pipeline_name}")

        # --- Training flow ---
        train_load_step = self._create_data_load_step(self.cradle_train_cfg)
        train_preprocess_step = self._create_tabular_preprocess_step(self.tp_train_cfg, train_load_step)
        train_step = self._create_xgboost_train_step(train_preprocess_step)
        packaging_step = self._create_packaging_step(train_step)
        registration_steps = self._create_registration_steps(packaging_step)

        # --- Calibration flow ---
        calib_load_step = self._create_data_load_step(self.cradle_calib_cfg)
        calib_preprocess_step = self._create_tabular_preprocess_step(self.tp_calib_cfg, calib_load_step)

        # --- Model Evaluation step (connects training and calibration flows) ---
        model_eval_step = self._create_model_eval_step(train_step, calib_preprocess_step)

        all_steps = [
            train_load_step,
            train_preprocess_step,
            train_step,
            packaging_step,
            *registration_steps,
            calib_load_step,
            calib_preprocess_step,
            model_eval_step
        ]
        logger.info(f"Created pipeline with {len(all_steps)} steps")
        return Pipeline(
            name=pipeline_name,
            parameters=self._get_pipeline_parameters(),
            steps=all_steps,
            sagemaker_session=self.session
        )
