import sys
import logging
from pathlib import Path
from typing import Optional, List, Union, Any, Dict

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, Step

from src.pipeline_steps.utils import load_configs

# Config classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig

# Step builders
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

# Common parameters
#from mods_workflow_core.utils.constants import (
#    PIPELINE_EXECUTION_TEMP_DIR,
#    KMS_ENCRYPTION_KEY_PARAM,
#    SECURITY_GROUP_ID,
#    VPC_SUBNET,
#)
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir", default_value="/tmp")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMSEncryptionKey", default_value="")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId", default_value="")
VPC_SUBNET = ParameterString(name="VPCEndpointSubnet", default_value="")


# Map JSON keys → Pydantic classes
CONFIG_CLASSES = {
    'BasePipelineConfig':          BasePipelineConfig,
    'CradleDataLoadConfig':        CradleDataLoadConfig,
    'ProcessingStepConfigBase':   ProcessingStepConfigBase,
    'TabularPreprocessingConfig':  TabularPreprocessingConfig,
}

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


# ────────────────────────────────────────────────────────────────────────────────
class XGBoostDataloadPreprocessPipelineBuilder:
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
        # Ensure tp_cfg has both the input_names attribute and get_input_names method for compatibility
        if not hasattr(tp_cfg, 'input_names'):
            tp_cfg.input_names = tp_cfg.get_input_names()
            
        inputs = {tp_cfg.input_names["data_input"]: cradle_outputs[OUTPUT_TYPE_DATA]}
        
        # Ensure tp_cfg has both the output_names attribute and get_output_names method for compatibility
        if not hasattr(tp_cfg, 'output_names'):
            tp_cfg.output_names = tp_cfg.get_output_names()
            
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
