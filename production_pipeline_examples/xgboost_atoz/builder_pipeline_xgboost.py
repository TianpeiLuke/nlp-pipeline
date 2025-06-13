# File: pipelines/builder_load_preprocess_train_pipeline.py

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
from sagemaker.inputs import TrainingInput

from mods.mods_template import MODSTemplate
from pipelines.utils import load_configs

# Config classes
from pipelines.config_base import BasePipelineConfig
from pipelines.config_cradle_data_load import CradleDataLoadConfig
from pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig
from pipelines.config_training_step_xgboost import XGBoostTrainingConfig 

# Step builders
from pipelines.builder_cradle_data_load_step import CradleDataLoadingStepBuilder
from pipelines.builder_tabular_preprocessing import TabularPreprocessingStepBuilder
from pipelines.builder_training_step_xgboost import XGBoostTrainingStepBuilder

# Common parameters
from mods_workflow_core.utils.constants import (
    PIPELINE_EXECUTION_TEMP_DIR,
    KMS_ENCRYPTION_KEY_PARAM,
    SECURITY_GROUP_ID,
    VPC_SUBNET,
)
from secure_ai_sandbox_workflow_python_sdk.utils.constants import (
    OUTPUT_TYPE_DATA,
    OUTPUT_TYPE_METADATA,
    OUTPUT_TYPE_SIGNATURE,
)

# Map JSON keys → Pydantic classes
CONFIG_CLASSES = {
    'BasePipelineConfig':         BasePipelineConfig,
    'CradleDataLoadConfig':       CradleDataLoadConfig,
    'TabularPreprocessingConfig': TabularPreprocessingConfig,
    'XGBoostTrainingConfig':      XGBoostTrainingConfig,
}

# ────────────────────────────────────────────────────────────────────────────────
# Load configs once so decorator metadata is available
# ────────────────────────────────────────────────────────────────────────────────
region = 'NA'
model_class = 'xgboost'
here = Path(__file__).resolve().parent
cfg_file = here / 'pipeline_config' / f'config_{region}_{model_class}.json'
if not cfg_file.exists():
    logging.error(f"Config file not found: {cfg_file}")
    sys.exit(1)

_all_configs = load_configs(str(cfg_file), CONFIG_CLASSES)
_base_config = _all_configs['Base']
AUTHOR           = _base_config.author
PIPELINE_DESC    = _base_config.pipeline_description
PIPELINE_VERSION = _base_config.pipeline_version

# ────────────────────────────────────────────────────────────────────────────────
@MODSTemplate(author=AUTHOR, description=PIPELINE_DESC, version=PIPELINE_VERSION)
class MDSXGBoostPipelineBuilder:
    """
    Builds a pipeline that performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) Data Loading (for testing set)
    5) Tabular Preprocessing (for testing set)
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
        self.session       = sagemaker_session
        self.role          = role
        self.notebook_root = notebook_root or Path.cwd()
        
        self.cradle_loading_requests: Dict[str, Dict[str, Any]] = {}
        self._validate_preprocessing_inputs()
        
        logging.info(f"Initialized builder for: {self.base_config.pipeline_name}")

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
        self.cradle_test_cfg  = self.configs[self._find_config_key('CradleDataLoadConfig', job_type='testing')]

        # Tabular Preprocessing configs
        self.tp_train_cfg = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='training')]
        self.tp_test_cfg  = self.configs[self._find_config_key('TabularPreprocessingConfig', job_type='testing')]
        
        xgb_train_config_instance = None
        for key, cfg in self.configs.items():
            if isinstance(cfg, XGBoostTrainingConfig):
                xgb_train_config_instance = cfg
                break
        
        if not xgb_train_config_instance:
            raise ValueError("Could not find a configuration of type XGBoostTrainingConfig in the config file.")
        
        self.xgb_train_cfg = xgb_train_config_instance

        # Type checks
        if not all(isinstance(c, CradleDataLoadConfig) for c in [self.cradle_train_cfg, self.cradle_test_cfg]):
            raise TypeError("Expected CradleDataLoadConfig for both training and testing")
        if not all(isinstance(c, TabularPreprocessingConfig) for c in [self.tp_train_cfg, self.tp_test_cfg]):
            raise TypeError("Expected TabularPreprocessingConfig for both data types")
        if not isinstance(self.xgb_train_cfg, XGBoostTrainingConfig):
            raise TypeError("Expected XGBoostTrainingConfig")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

    def _create_training_flow(self) -> List[Step]:
        """
        Create a data loading, preprocessing, and training flow for the 'training' job type.
        """
        job_type = 'training'
        
        # 1) Cradle data loading
        loader = CradleDataLoadingStepBuilder(
            config=self.cradle_train_cfg, sagemaker_session=self.session, role=self.role
        )
        load_step = loader.create_step()
        self.cradle_loading_requests[load_step.name] = loader.get_request_dict()

        # 2) Tabular preprocessing
        prep_builder = TabularPreprocessingStepBuilder(
            config=self.tp_train_cfg, sagemaker_session=self.session, role=self.role
        )
        cradle_outputs = loader.get_step_outputs(load_step)
        inputs = { self.tp_train_cfg.input_names["data_input"]: cradle_outputs[OUTPUT_TYPE_DATA] }
        
        # Define the single output channel for the preprocessed data directory
        prep_output_name = self.tp_train_cfg.output_names["processed_data"]
        prefix = f"{self.base_config.pipeline_s3_loc}/tabular_preprocessing/{job_type}"
        
        # FIX: The outputs dictionary now only defines the single required output.
        outputs = {
            prep_output_name: prefix,
        }
        
        prep_step = prep_builder.create_step(inputs=inputs, outputs=outputs)
        prep_step.add_depends_on([load_step])

        # 3) XGBoost Training
        xgb_builder = XGBoostTrainingStepBuilder(
            config=self.xgb_train_cfg, sagemaker_session=self.session, role=self.role
        )
        
        # Set the training step's input path to be the S3 URI from the preprocessing step's output
        object.__setattr__(
            xgb_builder.config,
            'input_path',
            prep_step.properties.ProcessingOutputConfig.Outputs[prep_output_name].S3Output.S3Uri
        )
        
        # The builder's create_step method now handles everything.
        xgb_train_step = xgb_builder.create_step(dependencies=[prep_step])

        logging.info(f"Created training flow: {load_step.name} -> {prep_step.name} -> {xgb_train_step.name}")
        return [load_step, prep_step, xgb_train_step]

    def _create_testing_flow(self) -> List[Step]:
        """Create a data loading and preprocessing flow for the 'testing' job type."""
        return self._create_flow('testing')

    def _create_flow(self, job_type: str) -> List[Step]:
        """Generic flow for loading and preprocessing."""
        cradle_cfg = self.cradle_train_cfg if job_type == 'training' else self.cradle_test_cfg
        tp_cfg = self.tp_train_cfg if job_type == 'training' else self.tp_test_cfg

        loader = CradleDataLoadingStepBuilder(config=cradle_cfg, sagemaker_session=self.session, role=self.role)
        load_step = loader.create_step()
        self.cradle_loading_requests[load_step.name] = loader.get_request_dict()
        
        prep = TabularPreprocessingStepBuilder(config=tp_cfg, sagemaker_session=self.session, role=self.role)
        cradle_outputs = loader.get_step_outputs(load_step)
        inputs = {tp_cfg.input_names["data_input"]: cradle_outputs[OUTPUT_TYPE_DATA]}
        
        prefix = f"{self.base_config.pipeline_s3_loc}/tabular_preprocessing/{job_type}"
        
        # FIX: Define only the single required output.
        outputs = {
            tp_cfg.output_names["processed_data"]: prefix
        }
        
        prep_step = prep.create_step(inputs=inputs, outputs=outputs)
        prep_step.add_depends_on([load_step])

        return [load_step, prep_step]

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """Fill in the execution document with stored Cradle data loading configurations."""
        if not self.cradle_loading_requests:
            raise ValueError("No Cradle loading requests found. Call generate_pipeline() first.")
        if "PIPELINE_STEP_CONFIGS" not in execution_document:
            raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")

        for step_name, request_dict in self.cradle_loading_requests.items():
            if step_name not in execution_document["PIPELINE_STEP_CONFIGS"]:
                raise KeyError(f"Step '{step_name}' not found in execution document")
            execution_document["PIPELINE_STEP_CONFIGS"][step_name]["STEP_CONFIG"] = request_dict
        return execution_document

    def generate_pipeline(self) -> Pipeline:
        """Generate the complete pipeline with all steps."""
        pipeline_name = f"{self.base_config.pipeline_name}-loadprep-train"
        logging.info(f"Building pipeline: {pipeline_name}")
        
        self.cradle_loading_requests.clear()
        
        logging.info("Creating training flow...")
        steps = self._create_training_flow()
        logging.info("Creating testing flow...")
        steps += self._create_testing_flow()

        logging.info(f"Created pipeline with {len(steps)} steps")
        return Pipeline(
            name=pipeline_name,
            parameters=self._get_pipeline_parameters(),
            steps=steps,
            sagemaker_session=self.session
        )
