import sys
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
    Step
)
from sagemaker.image_uris import retrieve

from src.pipeline_steps.utils import load_configs

# Config classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig 
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
# Removed HyperparameterPrepStepBuilder import - no longer needed
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder

import os
import importlib
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common parameters
from mods.mods_template import MODSTemplate
from mods_workflow_core.utils.constants import (
    PIPELINE_EXECUTION_TEMP_DIR,
    KMS_ENCRYPTION_KEY_PARAM,
    PROCESSING_JOB_SHARED_NETWORK_CONFIG,
    SECURITY_GROUP_ID,
    VPC_SUBNET,
)
from secure_ai_sandbox_workflow_python_sdk.utils.constants import (
    OUTPUT_TYPE_DATA,
    OUTPUT_TYPE_METADATA,
    OUTPUT_TYPE_SIGNATURE,
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
# Load configs once so decorator metadata is available
# ────────────────────────────────────────────────────────────────────────────────
# Check Author and Pipeline Name
region = 'NA'
model_class = 'xgboost'
# Get the current file's directory
current_dir = Path(__file__).resolve().parent
logger.info(f"Current directory: {current_dir}")

# Load Config Locally
logger.info(f"Current region: {region}")
logger.info(f"Current directory: {current_dir}")
config_path = current_dir.parent.parent / 'pipeline_config' / f'config_{region}_{model_class}' / f'config_{region}_{model_class}.json'

if not config_path.exists():
    logger.error(f"Config file not found: {str(config_path)}")
    sys.exit(1)
logger.info(f"Using config file: {config_path}")

_all_configs = load_configs(config_path, CONFIG_CLASSES)
_base_config = _all_configs['Base']

AUTHOR           = _base_config.author
PIPELINE_DESC    = _base_config.pipeline_description
PIPELINE_VERSION = _base_config.pipeline_version

# ────────────────────────────────────────────────────────────────────────────────
@MODSTemplate(author=AUTHOR, description=PIPELINE_DESC, version=PIPELINE_VERSION)
class XGBoostTrainEvaluatePipelineBuilder:
    """
    Builds a pipeline that performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) Hyperparameter Preparation
    4) XGBoost Model Training
    5) Packaging
    6) Payload Sample Generation
    7) MIMS Registration
    8) Data Loading (for calibration set)
    9) Tabular Preprocessing (for calibration set)
    10) Model Evaluation (on calibration set)
    """
    # Constants removed - validation now handled by TabularPreprocessingConfig class

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
        
        logger.info(f"Initialized builder for: {self.base_config.pipeline_name}")

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
        
        # Hyperparameter prep config no longer needed - XGBoostTrainingStepBuilder handles this internally
        
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
            if type(cfg).__name__ == 'PackageStepConfig':
                package_config_instance = cfg
                break
        if not package_config_instance:
            raise ValueError("Could not find a configuration of type PackageStepConfig")
        self.package_cfg = package_config_instance
        
        # Registration config - use exact class check, not isinstance()
        registration_config_instance = None
        for key, cfg in self.configs.items():
            if type(cfg).__name__ == 'ModelRegistrationConfig':
                registration_config_instance = cfg
                break
        if not registration_config_instance:
            raise ValueError("Could not find a configuration of type ModelRegistrationConfig")
        self.registration_cfg = registration_config_instance
        
        # Payload config - use exact class check, not isinstance()
        payload_config_instance = None
        for key, cfg in self.configs.items():
            if type(cfg).__name__ == 'PayloadConfig':
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
        
        # Pass all three inputs from CradleDataLoadingStep to satisfy validation requirements
        # TabularPreprocessingConfig defines DATA as required, METADATA and SIGNATURE as optional
        # But StepBuilderBase._validate_inputs treats all input_names keys as required
        inputs = {
            "DATA": cradle_outputs[OUTPUT_TYPE_DATA].S3Output.S3Uri,
            "METADATA": cradle_outputs[OUTPUT_TYPE_METADATA].S3Output.S3Uri,
            "SIGNATURE": cradle_outputs[OUTPUT_TYPE_SIGNATURE].S3Output.S3Uri
        }
        
        # Get output descriptor from the standardized output_names
        processed_data_key = tp_config.output_names["processed_data"]
        outputs = {
            processed_data_key: f"{self.base_config.pipeline_s3_loc}/tabular_preprocessing/{tp_config.job_type}"
        }
        
        prep_step = prep_builder.create_step(inputs=inputs, outputs=outputs)
        prep_step.add_depends_on([dependency_step])
        return prep_step

    # _create_hyperparameter_prep_step method removed - XGBoostTrainingStepBuilder now handles hyperparameters internally

    def _create_xgboost_train_step(self, preprocess_step: Step) -> TrainingStep:
        # Create a copy of the training config to modify it
        temp_train_cfg = self.xgb_train_cfg.model_copy()
        
        # Force NA region (us-east-1) for model training
        temp_train_cfg.region = 'NA'
        temp_train_cfg.aws_region = 'us-east-1'
        logger.info(f"Force Model Training in aws region: {temp_train_cfg.aws_region}")
        
        # Create the XGBoost training step builder with the modified config
        xgb_builder = XGBoostTrainingStepBuilder(config=temp_train_cfg, sagemaker_session=self.session, role=self.role)
        
        # Get the preprocessing output path - this is a Pipeline variable that contains train/val/test subdirs
        output_name = self.tp_train_cfg.output_names["processed_data"]
        preproc_output_path = preprocess_step.properties.ProcessingOutputConfig.Outputs[output_name].S3Output.S3Uri
        
        # Set the output path for model artifacts
        output_path = f"{self.base_config.pipeline_s3_loc}/xgboost_model_artifacts"
        
        # Instead of setting on the config (which causes logging issues with Pipeline variables),
        # pass the paths as parameters to create_step
        logger.info("Creating XGBoost training step with preprocessing output as input")
        
        # Create and return the step with input/output paths as parameters
        # This avoids the problematic logging code path in the builder
        return xgb_builder.create_step(
            input_path=preproc_output_path,  # Pass as parameter instead of config attribute
            output_path=output_path,         # Pass as parameter instead of config attribute
            dependencies=[preprocess_step]
        )

    def _create_packaging_step(self, dependency_step: Step) -> ProcessingStep:
        # Create a copy of the package config to modify it
        package_cfg = self.package_cfg.model_copy()
        
        # Now create the builder with the updated config
        packaging_builder = MIMSPackagingStepBuilder(
            config=package_cfg, 
            sagemaker_session=self.session, 
            role=self.role, 
            notebook_root=self.notebook_root
        )
        
        # Get model data from dependency step
        model_data = dependency_step.properties.ModelArtifacts.S3ModelArtifacts
        
        # Set up inference scripts path - use source_dir as the inference scripts location
        inference_scripts_path = package_cfg.source_dir
        if not inference_scripts_path:
            # Fall back to notebook_root/inference if source_dir not specified
            inference_scripts_path = str(self.notebook_root / "inference") if self.notebook_root else "inference"
            logger.info(f"Using default inference scripts path: {inference_scripts_path}")
        
        # Get the output key from output_names VALUES - standard pattern for outputs
        output_key = package_cfg.output_names["packaged_model_output"]  # e.g. "PackagedModel"
        
        # Standard pattern - use VALUE from output_names as key in outputs dict
        outputs = {
            output_key: f"{self.base_config.pipeline_s3_loc}/packaged_model"
        }
        
        logger.info(f"Creating packaging step with direct parameter passing")
        
        # Create and return the step - Using direct parameter passing instead of nested inputs
        return packaging_builder.create_step(
            model_input=model_data,              # Pass parameter directly
            inference_scripts_input=inference_scripts_path,  # Pass parameter directly
            outputs=outputs,
            dependencies=[dependency_step]
        )

    def _create_payload_testing_step(self, train_step) -> ProcessingStep:
        """
        Create a payload testing step to generate and upload inference payloads.
        
        Args:
            train_step: The training step that produced the model and hyperparameters
            
        Returns:
            ProcessingStep: The payload testing step
        """
        # Create the builder
        payload_builder = MIMSPayloadStepBuilder(
            config=self.payload_cfg, 
            sagemaker_session=self.session, 
            role=self.role
        )
        
        # Get the model artifacts from training step
        model_uri = train_step.properties.ModelArtifacts.S3ModelArtifacts
        # Use .expr for logging Pipeline variables to avoid TypeError
        logger.info("Extracted model URI for payload step (expression reference)")
        
        # Always use the consistent logical name "model_input"
        model_key = "model_input"
        
        # Log what we're using (PayloadConfig should enforce correct script param name)
        logger.info(f"Using logical name '{model_key}' for model input")
        logger.info(f"Script parameter name from config: {self.payload_cfg.input_names.get(model_key, 'ModelArtifacts')}")
        
        # Create and return the step with the model input
        # Using the logical name "model_input" - this will be mapped to the script input name by the builder
        return payload_builder.create_step(
            model_input=model_uri,
            dependencies=[train_step]
        )

    def _create_registration_steps(self, packaging_step: ProcessingStep, payload_step: ProcessingStep) -> List[ProcessingStep]:
        """Creates the Model Registration steps."""
        # Ensure registration config has standardized input_names, following the pattern:
        # key = logical name, value = script input name
        if self.registration_cfg.input_names is None or len(self.registration_cfg.input_names) == 0:
            object.__setattr__(self.registration_cfg, 'input_names', {
                "packaged_model_output": "ModelPackage",  # Script input name
                "payload_sample": "PayloadData"           # Script input name
            })
            logger.info("Added standardized input_names to registration config")
        
        registration_builder = ModelRegistrationStepBuilder(config=self.registration_cfg, sagemaker_session=self.session, role=self.role)
        
        image_uri = retrieve(
            framework=self.registration_cfg.framework,
            region=self.registration_cfg.aws_region,
            version=self.registration_cfg.framework_version,
            py_version=self.registration_cfg.py_version,
            instance_type=self.registration_cfg.inference_instance_type,
            image_scope="inference"
        )
        
        # Get the model package output from the packaging step
        s3_uri = packaging_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
        
        # Get the payload sample from the payload testing step
        # ProcessingStep outputs must be accessed through ProcessingOutputConfig.Outputs
        payload_sample = payload_step.properties.ProcessingOutputConfig.Outputs["payload_sample"].S3Output.S3Uri
        
        # Avoid directly logging Pipeline variables
        logger.info("Retrieved model package output and payload sample references")
        
        # Use both packaging and payload steps as dependencies
        # Connect the packaged model and payload to the registration step
        result = registration_builder.create_step(
            packaged_model_output=s3_uri,
            payload_sample=payload_sample,
            dependencies=[packaging_step, payload_step],
            regions=[self.base_config.region]
        )
        
        execution_doc_config = self._create_execution_doc_config(image_uri)

        # Store execution document configs
        logger.info("Store execution document configs")
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
                f"Make sure 'model_evaluation_xgb.py' exists in {effective_source_dir}"
            )

        logger.info(f"Found evaluation script at: {eval_script_path}")

        # Get input and output channel names directly from config properties 
        # We need to use the original keys, not the description values
        
        # Create inputs dictionary using logical names from standardized input_names
        inputs = {
            # Use the logical names from the config's input_names
            "model_input": train_step.properties.ModelArtifacts.S3ModelArtifacts,
            "eval_data_input": calib_preprocess_step.properties.ProcessingOutputConfig.Outputs[self.tp_calib_cfg.output_names["processed_data"]].S3Output.S3Uri,
        }

        # Get the output descriptor VALUES from the config's output_names
        eval_output_key = self.xgb_eval_cfg.output_names["eval_output"]       # Should be "EvaluationResults"
        metrics_output_key = self.xgb_eval_cfg.output_names["metrics_output"] # Should be "EvaluationMetrics"
        
        # Create outputs dictionary using VALUES from output_names as keys
        # Following the standard pattern: VALUES used as keys in outputs dict
        outputs = {
            # Use the VALUES from output_names as keys
            eval_output_key: f"{self.base_config.pipeline_s3_loc}/model_eval/eval_predictions",
            metrics_output_key: f"{self.base_config.pipeline_s3_loc}/model_eval/eval_metrics"
        }
        
        # No need to validate output channels since we're using hardcoded keys

        # Log that we're creating the evaluation step (avoid logging Pipeline variables directly)
        logger.info("Creating model evaluation step with inputs and outputs")
        logger.info(f"Input keys: {list(inputs.keys())}")
        logger.info(f"Output keys: {list(outputs.keys())}")

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
            "model_domain": self.registration_cfg.model_registration_domain,
            "model_objective": self.registration_cfg.model_registration_objective,
            "source_model_inference_content_types": self.registration_cfg.source_model_inference_content_types,
            "source_model_inference_response_types": self.registration_cfg.source_model_inference_response_types,
            "source_model_inference_input_variable_list": self.registration_cfg.source_model_inference_input_variable_list,
            "source_model_inference_output_variable_list": self.registration_cfg.source_model_inference_output_variable_list,
            "model_registration_region": self.registration_cfg.region,
            "source_model_inference_image_arn": image_uri,
            "source_model_region": self.registration_cfg.aws_region,
            "model_owner": self.registration_cfg.model_owner,
            "source_model_environment_variable_map": {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": self.registration_cfg.inference_entry_point,
                "SAGEMAKER_REGION": self.registration_cfg.aws_region,
                "SAGEMAKER_SUBMIT_DIRECTORY": '/opt/ml/model/code',
            },
            'load_testing_info_map': {
                "sample_payload_s3_bucket": self.registration_cfg.bucket,
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
        
        # No hyperparameter preparation step - functionality integrated into XGBoostTrainingStep
        
        # Connect training step with preprocessing step only
        train_step = self._create_xgboost_train_step(train_preprocess_step)
        
        # Create packaging step using training step output
        packaging_step = self._create_packaging_step(train_step)
        
        # Create payload testing step
        payload_testing_step = self._create_payload_testing_step(train_step)
        
        # Connect registration to both packaging and payload testing steps
        registration_steps = self._create_registration_steps(packaging_step, payload_testing_step)

        # --- Calibration flow ---
        calib_load_step = self._create_data_load_step(self.cradle_calib_cfg)
        calib_preprocess_step = self._create_tabular_preprocess_step(self.tp_calib_cfg, calib_load_step)

        # --- Model Evaluation step (connects training and calibration flows) ---
        model_eval_step = self._create_model_eval_step(train_step, calib_preprocess_step)

        # Include all steps in the pipeline (no hyperparameter_prep_step)
        all_steps = [
            train_load_step,
            train_preprocess_step,
            train_step,
            packaging_step,
            payload_testing_step,
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
