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

from src.pipelines.utils import load_configs

# Config classes
from src.pipelines.config_base import BasePipelineConfig
from src.pipelines.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipelines.config_processing_step_base import ProcessingStepConfigBase
from src.pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipelines.config_training_step_xgboost import XGBoostTrainingConfig 
from src.pipelines.config_model_step_xgboost import XGBoostModelCreationConfig
from src.pipelines.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipelines.config_mims_packaging_step import PackageStepConfig
from src.pipelines.config_mims_registration_step import ModelRegistrationConfig
from src.pipelines.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipelines.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipelines.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipelines.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipelines.builder_model_step_xgboost import XGBoostModelStepBuilder
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
    'XGBoostModelCreationConfig': XGBoostModelCreationConfig,
    'XGBoostModelEvalConfig':     XGBoostModelEvalConfig,
    'PackageStepConfig':          PackageStepConfig,
    'ModelRegistrationConfig':    ModelRegistrationConfig,
    'PayloadConfig':              PayloadConfig
}

class XGBoostTrainEvaluatePipelineBuilder:
    """
    Builds a pipeline that performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) XGBoost Model Creation
    5) Model Evaluation (on calibration set)
    6) Data Loading (for calibration set)
    7) Tabular Preprocessing (for calibration set)
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
        self.cradle_train_cfg = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='training')]
        self.cradle_calib_cfg = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='calibration')]
        self.tp_train_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='training')]
        self.tp_calib_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='calibration')]
        xgb_train_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostTrainingConfig):
                xgb_train_config_instance = cfg
                break
        if not xgb_train_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostTrainingConfig in the config file.")
        self.xgb_train_cfg = xgb_train_config_instance
        xgb_model_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostModelCreationConfig):
                xgb_model_config_instance = cfg
                break
        if not xgb_model_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostModelCreationConfig in the config file.")
        self.xgb_model_cfg = xgb_model_config_instance
        xgb_eval_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostModelEvalConfig):
                xgb_eval_config_instance = cfg
                break
        if not xgb_eval_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostModelEvalConfig in the config file.")
        self.xgb_eval_cfg = xgb_eval_config_instance
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

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

    def _create_data_load_step(self, cradle_config: CradleDataLoadConfig) -> ProcessingStep:
        loader = CradleDataLoadingStepBuilder(config=cradle_config, sagemaker_session=self.session, role=self.role)
        return loader.create_step()

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

    def _create_model_creation_step(self, dependency_step: Step) -> ModelStep:
        temp_model_cfg = self.xgb_model_cfg.model_copy()
        temp_model_cfg.region = 'NA'
        temp_model_cfg.aws_region = 'us-east-1'
        logger.info(f"Force Model Creation in aws region: {temp_model_cfg.aws_region}")
        model_builder = XGBoostModelStepBuilder(config=temp_model_cfg, sagemaker_session=self.session, role=self.role)
        return model_builder.create_step(
            model_data=dependency_step.properties.ModelArtifacts.S3ModelArtifacts,
            dependencies=[dependency_step]
        )

    def _create_packaging_step(self, dependency_step: Step) -> ProcessingStep:
        packaging_builder = MIMSPackagingStepBuilder(config=self.package_cfg, sagemaker_session=self.session, role=self.role, notebook_root=self.notebook_root)
        return packaging_builder.create_packaging_step(
            model_data=dependency_step.properties.ModelArtifacts.S3ModelArtifacts,
            dependencies=[dependency_step]
        )

    def _create_registration_steps(self, dependency_step: ProcessingStep) -> List[ProcessingStep]:
        registration_builder = ModelRegistrationStepBuilder(config=self.registration_cfg, sagemaker_session=self.session, role=self.role)
        self.payload_cfg.generate_and_upload_payloads()
        s3_uri = dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
        result = registration_builder.create_step(
            packaging_step_output=s3_uri,
            payload_s3_key=self.payload_cfg.sample_payload_s3_key,
            dependencies=[dependency_step],
            regions=[self.base_config.region]
        )
        if isinstance(result, dict):
            return list(result.values())
        else:
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
        # Inputs: model artifacts from training, eval data from calibration preprocessing
        inputs = {
            self.xgb_eval_cfg.input_names["model_input"]: train_step.properties.ModelArtifacts.S3ModelArtifacts,
            self.xgb_eval_cfg.input_names["eval_data_input"]: calib_preprocess_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri
        }
        outputs = {
            self.xgb_eval_cfg.output_names["eval_output"]: f"{self.base_config.pipeline_s3_loc}/model_eval/eval_predictions",
            self.xgb_eval_cfg.output_names["metrics_output"]: f"{self.base_config.pipeline_s3_loc}/model_eval/eval_metrics"
        }
        eval_step = eval_builder.create_step(inputs=inputs, outputs=outputs)
        eval_step.add_depends_on([train_step, calib_preprocess_step])
        return eval_step

    def generate_pipeline(self) -> Pipeline:
        pipeline_name = f"{self.base_config.pipeline_name}-xgb-train-eval"
        logger.info(f"Building pipeline: {pipeline_name}")

        # --- Training flow ---
        train_load_step = self._create_data_load_step(self.cradle_train_cfg)
        train_preprocess_step = self._create_tabular_preprocess_step(self.tp_train_cfg, train_load_step)
        train_step = self._create_xgboost_train_step(train_preprocess_step)
        model_creation_step = self._create_model_creation_step(train_step)
        packaging_step = self._create_packaging_step(model_creation_step)
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
            model_creation_step,
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
