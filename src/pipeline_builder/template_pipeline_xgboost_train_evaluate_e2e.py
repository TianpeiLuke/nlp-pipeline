from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import logging
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession

from src.pipeline_steps.utils import load_configs
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate
from src.pipeline_builder.pipeline_dag import PipelineDAG

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
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
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

# Config classes mapping
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

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """Create a mapping from step types to builder classes."""
        return {
            "CradleDataLoadStep": CradleDataLoadingStepBuilder,
            "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
            "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
            "PackagingStep": MIMSPackagingStepBuilder,
            "RegistrationStep": ModelRegistrationStepBuilder,
            "ModelEvalStep": XGBoostModelEvalStepBuilder,
        }

    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """Create a mapping from step names to config instances."""
        config_map = {}
        
        # Add training flow steps
        config_map["train_data_load"] = self.cradle_train_cfg
        config_map["train_preprocess"] = self.tp_train_cfg
        config_map["xgboost_train"] = self.xgb_train_cfg
        config_map["model_packaging"] = self.package_cfg
        config_map["model_registration"] = self.registration_cfg
        
        # Add calibration flow steps
        config_map["calib_data_load"] = self.cradle_calib_cfg
        config_map["calib_preprocess"] = self.tp_calib_cfg
        
        # Add model evaluation step
        config_map["model_evaluation"] = self.xgb_eval_cfg
        
        return config_map

    def _create_pipeline_dag(self) -> PipelineDAG:
        """Create the DAG structure for the pipeline."""
        dag = PipelineDAG()
        
        # Add nodes
        dag.add_node("train_data_load")
        dag.add_node("train_preprocess")
        dag.add_node("xgboost_train")
        dag.add_node("model_packaging")
        dag.add_node("model_registration")
        dag.add_node("calib_data_load")
        dag.add_node("calib_preprocess")
        dag.add_node("model_evaluation")
        
        # Add edges for training flow
        dag.add_edge("train_data_load", "train_preprocess")
        dag.add_edge("train_preprocess", "xgboost_train")
        dag.add_edge("xgboost_train", "model_packaging")
        dag.add_edge("model_packaging", "model_registration")
        
        # Add edges for calibration flow
        dag.add_edge("calib_data_load", "calib_preprocess")
        
        # Connect model evaluation to both flows
        dag.add_edge("xgboost_train", "model_evaluation")
        dag.add_edge("calib_preprocess", "model_evaluation")
        
        return dag

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
        
        # Generate the pipeline
        return template.generate_pipeline(pipeline_name)

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

        # Fill Registration configurations
        for step_name, config in self.registration_configs.items():
            registration_step_name = f"Registration_{self.registration_cfg.region}"
            if registration_step_name not in pipeline_configs:
                logger.warning(f"Registration step '{registration_step_name}' not found in execution document")
                continue
            pipeline_configs[registration_step_name]["STEP_CONFIG"] = config
            logger.info(f"Updated execution config for registration step: {registration_step_name}")

        return execution_document
