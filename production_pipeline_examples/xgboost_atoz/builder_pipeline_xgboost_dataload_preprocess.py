# File: pipelines/builder_load_preprocess_pipeline.py

import sys
import logging
from pathlib import Path
from typing import Optional, List, Union

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, Step

from mods.mods_template import MODSTemplate
from pipelines.utils import load_configs

# Config classes
from pipelines.config_base import BasePipelineConfig
from pipelines.config_cradle_data_load import CradleDataLoadConfig
from pipelines.config_tabular_preprocessing_step import TabularPreprocessingConfig

# Step builders
from pipelines.builder_cradle_data_load_step import CradleDataLoadingStepBuilder
from pipelines.builder_tabular_preprocessing import TabularPreprocessingStepBuilder

# Common parameters (VPC, KMS, etc.)
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
    'BasePipelineConfig':          BasePipelineConfig,
    'CradleDataLoadConfig':        CradleDataLoadConfig,
    'TabularPreprocessingConfig':  TabularPreprocessingConfig,
}

# ────────────────────────────────────────────────────────────────────────────────
# Load configs once so decorator metadata is available
# ────────────────────────────────────────────────────────────────────────────────
region   = 'NA'
model_class = 'xgboost'
here     = Path(__file__).resolve().parent
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
class DataLoadPreprocessPipelineBuilder:
    """
    1) CradleDataLoading (training)
    2) TabularPreprocessing (training)
    3) CradleDataLoading (testing)
    4) TabularPreprocessing (testing)
    """
    # Required input channels for TabularPreprocessing
    REQUIRED_INPUTS = {"data_input"}
    # Optional input channels
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
        
        # Initialize dictionary to store Cradle loading requests
        self.cradle_loading_requests: Dict[str, Dict[str, Any]] = {}
        # Validate TabularPreprocessing input channels
        self._validate_preprocessing_inputs()
        
        logging.info(f"Initialized builder for: {self.base_config.pipeline_name}")

    def _validate_preprocessing_inputs(self):
        """
        Validate input channels in preprocessing configs:
        - Required channels must be present
        - Only allowed optional channels can be specified
        """
        allowed_inputs = self.REQUIRED_INPUTS | self.OPTIONAL_INPUTS
        
        for cfg in [self.tp_train_cfg, self.tp_test_cfg]:
            # Check required channels
            missing = self.REQUIRED_INPUTS - set(cfg.input_names.keys())
            if missing:
                raise ValueError(
                    f"TabularPreprocessing config missing required input channels: {missing}"
                )
            
            # Check for unknown channels
            unknown = set(cfg.input_names.keys()) - allowed_inputs
            if unknown:
                raise ValueError(
                    f"TabularPreprocessing config contains unknown input channels: {unknown}. "
                    f"Allowed channels are: {allowed_inputs}"
                )

    def _find_config_key(self, class_name: str, **attrs: Any) -> str:
        """
        Find the unique step_name for configs of type `class_name`
        that have all of the given attribute=value pairs in their suffix.
        """
        base = BasePipelineConfig.get_step_name(class_name)
        candidates = []
        for step_name in self.configs:
            if not step_name.startswith(base + "_"):
                continue
            # extract suffix parts
            parts = step_name[len(base) + 1 :].split("_")
            if all(str(val) in parts for val in attrs.values()):
                candidates.append(step_name)

        if not candidates:
            raise ValueError(f"No config found for {class_name} with {attrs}")
        if len(candidates) > 1:
            raise ValueError(f"Multiple configs found for {class_name} with {attrs}: {candidates}")
        return candidates[0]

    def _extract_configs(self):
        # Base
        self.base_config = self.configs['Base']

        # CradleDataLoadConfig instances
        train_dl_key = self._find_config_key('CradleDataLoadConfig', job_type='training')
        test_dl_key  = self._find_config_key('CradleDataLoadConfig', job_type='testing')

        self.cradle_train_cfg = self.configs[train_dl_key]
        self.cradle_test_cfg  = self.configs[test_dl_key]

        # TabularPreprocessingConfig instances
        train_tp_key = self._find_config_key('TabularPreprocessingConfig', job_type='training')
        test_tp_key  = self._find_config_key('TabularPreprocessingConfig', job_type='testing')

        self.tp_train_cfg = self.configs[train_tp_key]
        self.tp_test_cfg  = self.configs[test_tp_key]

        # Type checks
        if not isinstance(self.cradle_train_cfg, CradleDataLoadConfig) or \
           not isinstance(self.cradle_test_cfg,  CradleDataLoadConfig):
            raise TypeError("Expected CradleDataLoadConfig for both training and testing")
        if not isinstance(self.tp_train_cfg, TabularPreprocessingConfig) or \
           not isinstance(self.tp_test_cfg,  TabularPreprocessingConfig):
            raise TypeError("Expected TabularPreprocessingConfig for both data types")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        return [
            PIPELINE_EXECUTION_TEMP_DIR,
            KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID,
            VPC_SUBNET,
        ]

    def _create_flow(self, job_type: str) -> List[Any]:
        """
        Create a data loading and preprocessing flow for a given job type.
        
        Args:
            job_type: Either 'training' or 'testing'
            
        Returns:
            List containing the CradleDataLoading and TabularPreprocessing steps
        """
        # Select appropriate configs
        cradle_cfg = self.cradle_train_cfg if job_type == 'training' else self.cradle_test_cfg
        tp_cfg = self.tp_train_cfg if job_type == 'training' else self.tp_test_cfg

        # 1) Cradle data loading
        loader = CradleDataLoadingStepBuilder(
            config=cradle_cfg, 
            sagemaker_session=self.session, 
            role=self.role
        )
        load_step = loader.create_step()
        self.cradle_loading_requests[load_step.name] = loader.get_request_dict()

        # 2) Tabular preprocessing
        prep = TabularPreprocessingStepBuilder(
            config=tp_cfg, 
            sagemaker_session=self.session, 
            role=self.role
        )
        
        cradle_outputs = loader.get_step_outputs(load_step)
        inputs = {tp_cfg.input_names["data_input"]: cradle_outputs[OUTPUT_TYPE_DATA]}
        
        # Define the single output channel for the preprocessed data directory
        prep_output_name = tp_cfg.output_names["processed_data"]
        prefix = f"{self.base_config.pipeline_s3_loc}/tabular_preprocessing/{job_type}"
        
        # Define only the single required output
        outputs = {
            prep_output_name: prefix
        }
        
        prep_step = prep.create_step(inputs=inputs, outputs=outputs)
        prep_step.add_depends_on([load_step])

        logging.info(f"Created {job_type} flow: {load_step.name} -> {prep_step.name}")
        return [load_step, prep_step]

    def _create_training_flow(self):
        """Create training data loading and preprocessing flow."""
        return self._create_flow('training')

    def _create_testing_flow(self):
        """Create testing data loading and preprocessing flow."""
        return self._create_flow('testing')

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with stored Cradle data loading configurations.

        Args:
            execution_document (Dict[str, Any]): The execution document to be filled.

        Returns:
            Dict[str, Any]: The updated execution document.

        Raises:
            ValueError: If pipeline hasn't been generated yet (no stored requests)
            KeyError: If execution document doesn't have expected structure
        """
        if not self.cradle_loading_requests:
            raise ValueError(
                "No Cradle loading requests found. "
                "Make sure to call generate_pipeline() before fill_execution_document()"
            )

        if "PIPELINE_STEP_CONFIGS" not in execution_document:
            raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")

        pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]
        
        for step_name, request_dict in self.cradle_loading_requests.items():
            if step_name not in pipeline_configs:
                raise KeyError(f"Step '{step_name}' not found in execution document")
            
            if "STEP_CONFIG" not in pipeline_configs[step_name]:
                pipeline_configs[step_name]["STEP_CONFIG"] = {}
            
            pipeline_configs[step_name]["STEP_CONFIG"] = request_dict

        return execution_document

    def generate_pipeline(self) -> Pipeline:
        """
        Generate the complete pipeline with all steps.

        Returns:
            Pipeline: The configured SageMaker pipeline

        Note:
            This also clears and rebuilds the cradle_loading_requests dictionary.
        """
        pipeline_name = f"{self.base_config.pipeline_name}-loadprep"
        logging.info(f"Building pipeline: {pipeline_name}")
    
        # Clear any existing requests
        self.cradle_loading_requests.clear()
        logging.info("Cleared existing Cradle loading requests")

        # Create steps
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
