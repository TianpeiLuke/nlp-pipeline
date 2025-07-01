from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import logging
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.image_uris import retrieve

from src.pipeline_steps.utils import load_configs
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate
from src.pipeline_builder.pipeline_dag import PipelineDAG

# Config classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_hyperparameter_prep_step import HyperparameterPrepConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig 
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.builder_hyperparameter_prep_step import HyperparameterPrepStepBuilder
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder

# Pipeline parameters
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir", default_value="/tmp")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMSEncryptionKey", default_value="")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId", default_value="")
VPC_SUBNET = ParameterString(name="VPCEndpointSubnet", default_value="")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import pipeline constants if available
import os
import importlib

# Define default constants with uppercase values
OUTPUT_TYPE_DATA = "DATA"
OUTPUT_TYPE_METADATA = "METADATA"
OUTPUT_TYPE_SIGNATURE = "SIGNATURE"

# Try to import from module if available
SECUREAI_PIPELINE_CONSTANTS_MODULE = os.environ.get("SECUREAI_PIPELINE_CONSTANTS_MODULE")
if SECUREAI_PIPELINE_CONSTANTS_MODULE:
    try:
        const_mod = importlib.import_module(SECUREAI_PIPELINE_CONSTANTS_MODULE)
        # These will override the defaults if available
        OUTPUT_TYPE_DATA      = getattr(const_mod, "OUTPUT_TYPE_DATA", OUTPUT_TYPE_DATA)
        OUTPUT_TYPE_METADATA  = getattr(const_mod, "OUTPUT_TYPE_METADATA", OUTPUT_TYPE_METADATA)
        OUTPUT_TYPE_SIGNATURE = getattr(const_mod, "OUTPUT_TYPE_SIGNATURE", OUTPUT_TYPE_SIGNATURE)
        logger.info(f"Imported pipeline constants from {SECUREAI_PIPELINE_CONSTANTS_MODULE}")
    except ImportError as e:
        logger.error(f"Could not import pipeline constants: {e}")
        logger.info("Using default uppercase constants: DATA, METADATA, SIGNATURE")
else:
    logger.info("Using default uppercase constants: DATA, METADATA, SIGNATURE")

# Config classes mapping
CONFIG_CLASSES = {
    'BasePipelineConfig':         BasePipelineConfig,
    'CradleDataLoadConfig':       CradleDataLoadConfig,
    'ProcessingStepConfigBase':   ProcessingStepConfigBase,
    'TabularPreprocessingConfig': TabularPreprocessingConfig,
    'HyperparameterPrepConfig':   HyperparameterPrepConfig,
    'XGBoostTrainingConfig':      XGBoostTrainingConfig,
    'XGBoostModelEvalConfig':     XGBoostModelEvalConfig,
    'PackageStepConfig':          PackageStepConfig,
    'ModelRegistrationConfig':    ModelRegistrationConfig,
    'PayloadConfig':              PayloadConfig
}


class XGBoostTrainEvaluateE2ETemplateBuilder:
    """
    Template-based builder for XGBoost Train-Evaluate E2E pipeline.
    
    This pipeline performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) Packaging
    5) MIMS Registration
    6) Data Loading (for calibration set)
    7) Tabular Preprocessing (for calibration set)
    8) Model Evaluation (on calibration set)
    """
    REQUIRED_INPUTS = {"DATA"}
    OPTIONAL_INPUTS = {"METADATA", "SIGNATURE"}

    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        logger.info(f"Loading configs from: {config_path}")
        self.configs = load_configs(config_path, CONFIG_CLASSES)
        self.base_config = self.configs.get('Base')
        if not self.base_config:
            raise ValueError("Base configuration not found in config file")
            
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        
        self.cradle_loading_requests: Dict[str, Dict[str, Any]] = {}
        self.registration_configs: Dict[str, Dict[str, Any]] = {}
        
        # Validate preprocessing inputs
        self._validate_preprocessing_inputs()
        
        logger.info(f"Initialized builder for: {self.base_config.pipeline_name}")
        
    def _validate_preprocessing_inputs(self):
        """Validate input channels in preprocessing configs."""
        allowed_inputs = self.REQUIRED_INPUTS | self.OPTIONAL_INPUTS
        
        # Find preprocessing configs
        tp_configs = [cfg for name, cfg in self.configs.items() 
                     if isinstance(cfg, TabularPreprocessingConfig)]
        
        if len(tp_configs) < 2:
            raise ValueError("Expected at least two TabularPreprocessingConfig instances")
            
        for cfg in tp_configs:
            # Create a temporary builder to validate the configuration
            temp_builder = TabularPreprocessingStepBuilder(
                config=cfg,
                sagemaker_session=self.session,
                role=self.role,
                notebook_root=self.notebook_root
            )
            
            # Get input requirements from the builder
            input_reqs = temp_builder.get_input_requirements()
            
            # Check for required inputs
            missing = self.REQUIRED_INPUTS - set(input_reqs.keys())
            if missing:
                raise ValueError(f"TabularPreprocessing config missing required input channels: {missing}")
            
            # Check for unknown inputs
            unknown = set(input_reqs.keys()) - allowed_inputs - {"outputs", "enable_caching", "dependencies"}
            if unknown:
                raise ValueError(f"TabularPreprocessing config contains unknown input channels: {unknown}")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """Create a mapping from step types to builder classes."""
        return {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "TabularPreprocessing": TabularPreprocessingStepBuilder,
            "HyperparameterPrep": HyperparameterPrepStepBuilder,
            "XGBoostTraining": XGBoostTrainingStepBuilder,
            "Package": MIMSPackagingStepBuilder,
            "Payload": MIMSPayloadStepBuilder,
            "Registration": ModelRegistrationStepBuilder,
            "XGBoostModelEval": XGBoostModelEvalStepBuilder,
        }

    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """Create a mapping from step names to config instances."""
        config_map = {}
        
        # Find configs by type and job_type attribute
        cradle_configs = {
            getattr(cfg, 'job_type', 'unknown'): cfg 
            for _, cfg in self.configs.items() 
            if isinstance(cfg, CradleDataLoadConfig)
        }
        
        tp_configs = {
            getattr(cfg, 'job_type', 'unknown'): cfg 
            for _, cfg in self.configs.items() 
            if isinstance(cfg, TabularPreprocessingConfig)
        }
        
        # Add training flow steps
        config_map["train_data_load"] = cradle_configs.get('training')
        config_map["train_preprocess"] = tp_configs.get('training')
        
        # Find single instance configs (removing hyperparameter_prep)
        for cfg_type, step_name in [
            (XGBoostTrainingConfig, "xgboost_train"),
            (PackageStepConfig, "model_packaging"),
            (PayloadConfig, "payload_test"),
            (ModelRegistrationConfig, "model_registration"),
            (XGBoostModelEvalConfig, "model_evaluation")
        ]:
            # Use exact type matching (type(cfg) is cfg_type) instead of isinstance()
            # This prevents subclasses (like PayloadConfig) from matching when looking for parent class (ModelRegistrationConfig)
            instances = [cfg for _, cfg in self.configs.items() if type(cfg) is cfg_type]
            if instances:
                config_map[step_name] = instances[0]
        
        # Add calibration flow steps
        config_map["calib_data_load"] = cradle_configs.get('calibration')
        config_map["calib_preprocess"] = tp_configs.get('calibration')
        
        # Validate all required configs are present
        missing_configs = [name for name, cfg in config_map.items() if cfg is None]
        if missing_configs:
            raise ValueError(f"Missing required configurations: {missing_configs}")
        
        return config_map

    def _create_execution_doc_config(self, image_uri: str) -> Dict[str, Any]:
        """
        Helper to create the execution document configuration dictionary.
        
        Args:
            image_uri: The URI of the inference image to use
            
        Returns:
            Dictionary with execution document configuration
        """
        # Find needed configs
        registration_cfg = next((cfg for _, cfg in self.configs.items() 
                               if isinstance(cfg, ModelRegistrationConfig)), None)
        payload_cfg = next((cfg for _, cfg in self.configs.items() 
                           if isinstance(cfg, PayloadConfig)), None)
        package_cfg = next((cfg for _, cfg in self.configs.items() 
                           if isinstance(cfg, PackageStepConfig)), None)
        
        if not registration_cfg or not payload_cfg or not package_cfg:
            raise ValueError("Missing required configs for execution document")
        
        return {
            "model_domain": registration_cfg.model_registration_domain,
            "model_objective": registration_cfg.model_registration_objective,
            "source_model_inference_content_types": registration_cfg.source_model_inference_content_types,
            "source_model_inference_response_types": registration_cfg.source_model_inference_response_types,
            "source_model_inference_input_variable_list": registration_cfg.source_model_inference_input_variable_list,
            "source_model_inference_output_variable_list": registration_cfg.source_model_inference_output_variable_list,
            "model_registration_region": registration_cfg.region,
            "source_model_inference_image_arn": image_uri,
            "source_model_region": registration_cfg.aws_region,
            "model_owner": registration_cfg.model_owner,
            "source_model_environment_variable_map": {
                "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
                "SAGEMAKER_PROGRAM": registration_cfg.inference_entry_point,
                "SAGEMAKER_REGION": registration_cfg.aws_region,
                "SAGEMAKER_SUBMIT_DIRECTORY": '/opt/ml/model/code',
            },
            'load_testing_info_map': {
                "sample_payload_s3_bucket": registration_cfg.bucket,
                "sample_payload_s3_key": payload_cfg.sample_payload_s3_key,
                "expected_tps": payload_cfg.expected_tps,
                "max_latency_in_millisecond": payload_cfg.max_latency_in_millisecond,
                "instance_type_list": [package_cfg.get_instance_type() if hasattr(package_cfg, 'get_instance_type') else package_cfg.processing_instance_type_small],
                "max_acceptable_error_rate": payload_cfg.max_acceptable_error_rate,
            },
        }

    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create the DAG structure for the pipeline."""
        dag = PipelineDAG()
        
        # Add nodes (removing hyperparameter_prep)
        dag.add_node("train_data_load")
        dag.add_node("train_preprocess")
        dag.add_node("xgboost_train")
        dag.add_node("model_packaging")
        dag.add_node("model_registration")
        dag.add_node("payload_test")
        dag.add_node("calib_data_load")
        dag.add_node("calib_preprocess")
        dag.add_node("model_evaluation")
        
        # Add edges for training flow
        dag.add_edge("train_data_load", "train_preprocess")
        dag.add_edge("train_preprocess", "xgboost_train")
        dag.add_edge("xgboost_train", "model_packaging")
        dag.add_edge("xgboost_train", "payload_test")  # Direct connection from training to payload
        dag.add_edge("model_packaging", "model_registration")
        dag.add_edge("payload_test", "model_registration")
        
        # Add edges for calibration flow
        dag.add_edge("calib_data_load", "calib_preprocess")
        
        # Connect model evaluation to both flows
        dag.add_edge("xgboost_train", "model_evaluation")
        dag.add_edge("calib_preprocess", "model_evaluation")
        
        return dag

    def _store_registration_step_configs(self, template: PipelineBuilderTemplate) -> None:
        """
        Store execution document configs for registration steps.
        
        Args:
            template: The pipeline builder template containing the step instances
        """
        try:
            # Find registration steps
            registration_steps = []
            for step_name, step_instance in template.step_instances.items():
                if "registration" in step_name.lower() or "modelregistration" in str(type(step_instance)).lower():
                    registration_steps.append(step_instance)
                    logger.info(f"Found registration step: {step_name}")
            
            if not registration_steps:
                logger.warning("No registration steps found in pipeline")
                return
            
            # Try to retrieve the image URI for registration configs
            registration_cfg = next((cfg for _, cfg in self.configs.items() 
                                   if isinstance(cfg, ModelRegistrationConfig)), None)
            if not registration_cfg:
                logger.warning("No ModelRegistrationConfig found, skipping execution doc config")
                return
            
            # Get image URI
            try:
                image_uri = retrieve(
                    framework=registration_cfg.framework,
                    region=registration_cfg.aws_region,
                    version=registration_cfg.framework_version,
                    py_version=registration_cfg.py_version,
                    instance_type=registration_cfg.inference_instance_type,
                    image_scope="inference"
                )
                logger.info(f"Retrieved image URI: {image_uri}")
            except Exception as e:
                logger.warning(f"Could not retrieve image URI: {e}")
                image_uri = "image-uri-placeholder"  # Use placeholder for template
            
            # Create execution document config
            exec_config = self._create_execution_doc_config(image_uri)
            
            # Store configs for all registration steps found
            for step in registration_steps:
                if hasattr(step, 'name'):
                    self.registration_configs[step.name] = exec_config
                    logger.info(f"Stored execution doc config for registration step: {step.name}")
                elif isinstance(step, dict):
                    for name, s in step.items():
                        self.registration_configs[s.name] = exec_config
                        logger.info(f"Stored execution doc config for registration step: {s.name}")
        except Exception as e:
            logger.warning(f"Failed to store registration step configs: {e}")

    def generate_pipeline(self) -> Pipeline:
        """
        Build and return a SageMaker Pipeline object using the template.
        """
        pipeline_name = f"{self.base_config.pipeline_name}-xgb-train-eval"
        logger.info(f"Building pipeline: {pipeline_name}")
        
        # Create the DAG
        dag = self._create_pipeline_dag()
        
        # Create the config map
        config_map = self._create_config_map()
        
        # Create the step builder map
        step_builder_map = self._create_step_builder_map()
        
        # Create the template
        template = PipelineBuilderTemplate(
            dag=dag,
            config_map=config_map,
            step_builder_map=step_builder_map,
            sagemaker_session=self.session,
            role=self.role,
            pipeline_parameters=self._get_pipeline_parameters(),
            notebook_root=self.notebook_root,
        )
        
        # Property path registrations are now handled by the respective step builders
        # This follows the architectural principle that step builders should be responsible
        # for registering their own property paths, maintaining separation of concerns
        
        # NOTE: The redundant registration for packaging step has been removed from here
        # as builder_mims_packaging_step.py already implements comprehensive property path
        # registrations at module level and in the _match_custom_properties method
        
        # Generate the pipeline
        pipeline = template.generate_pipeline(pipeline_name)
        
        # Import Cradle data loading requests from the template
        self.cradle_loading_requests = template.cradle_loading_requests
        
        # Store registration step configs
        self._store_registration_step_configs(template)
        
        return pipeline

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with both Cradle data loading and model registration configurations.
        
        This method is preserved from the original implementation to maintain compatibility.
        """
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

        # Find registration config
        registration_cfg = next(
            (cfg for _, cfg in self.configs.items() if isinstance(cfg, ModelRegistrationConfig)), 
            None
        )
        
        # Fill Registration configurations
        if registration_cfg:
            for step_name, config in self.registration_configs.items():
                registration_step_name = f"Registration_{registration_cfg.region}"
                if registration_step_name not in pipeline_configs:
                    logger.warning(f"Registration step '{registration_step_name}' not found in execution document")
                    continue
                pipeline_configs[registration_step_name]["STEP_CONFIG"] = config
                logger.info(f"Updated execution config for registration step: {registration_step_name}")

        return execution_document
