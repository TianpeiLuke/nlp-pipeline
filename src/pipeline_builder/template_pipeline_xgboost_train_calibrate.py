"""
Template-based builder for XGBoost Training and Calibration pipeline.

This template creates a simplified pipeline that performs:
1) Data Loading (for training set) using Cradle
2) Tabular Preprocessing (for training set)
3) XGBoost Model Training
4) Model Calibration
"""

from typing import Dict, List, Any, Optional, Type
from pathlib import Path
import logging
import os
import importlib

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.image_uris import retrieve
from sagemaker.network import NetworkConfig

# Import base template
from .pipeline_template_base import PipelineTemplateBase

# Import dependencies for DAG and step builders
from .pipeline_assembler import PipelineAssembler
from ..pipeline_dag.base_dag import PipelineDAG
from ..pipeline_deps.registry_manager import RegistryManager
from ..pipeline_deps.dependency_resolver import UnifiedDependencyResolver
from ..pipeline_steps.utils import load_configs

# Config classes
from ..pipeline_steps.config_base import BasePipelineConfig
from ..pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from ..pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from ..pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from ..pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig 
from ..pipeline_steps.config_model_calibration_step import ModelCalibrationConfig

# Step builders
from ..pipeline_steps.builder_step_base import StepBuilderBase
from ..pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from ..pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from ..pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from ..pipeline_steps.builder_model_calibration_step import ModelCalibrationStepBuilder
from ..pipeline_registry.step_names import STEP_NAMES

# Import constants from core library (these are the parameters that will be wrapped)
try:
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        PROCESSING_JOB_SHARED_NETWORK_CONFIG,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    )
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported constants from mods_workflow_core")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import constants from mods_workflow_core, using local definitions")
    # Define pipeline parameters locally if import fails - match exact definitions from original module
    PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="EXECUTION_S3_PREFIX")
    KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMS_ENCRYPTION_KEY_PARAM")
    SECURITY_GROUP_ID = ParameterString(name="SECURITY_GROUP_ID")
    VPC_SUBNET = ParameterString(name="VPC_SUBNET")
    # Also create the network config as defined in the original module
    PROCESSING_JOB_SHARED_NETWORK_CONFIG = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=[SECURITY_GROUP_ID],
        subnets=[VPC_SUBNET],
        encrypt_inter_container_traffic=True,
    )

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define default constants with uppercase values
OUTPUT_TYPE_DATA = "DATA"
OUTPUT_TYPE_METADATA = "METADATA"
OUTPUT_TYPE_SIGNATURE = "SIGNATURE"


class XGBoostTrainCalibrateTemplate(PipelineTemplateBase):
    """
    Template-based builder for XGBoost Training and Calibration pipeline.
    
    This pipeline performs:
    1) Data Loading (for training set) using Cradle
    2) Tabular Preprocessing (for training set) 
    3) XGBoost Model Training
    4) Model Calibration
    """
    # Define required and optional inputs for preprocessing steps
    REQUIRED_INPUTS = {"DATA"}
    OPTIONAL_INPUTS = {"METADATA", "SIGNATURE"}
    
    # Define config classes used by this template
    CONFIG_CLASSES = {
        'BasePipelineConfig':         BasePipelineConfig,
        'CradleDataLoadConfig':       CradleDataLoadConfig,
        'ProcessingStepConfigBase':   ProcessingStepConfigBase,
        'TabularPreprocessingConfig': TabularPreprocessingConfig,
        'XGBoostTrainingConfig':      XGBoostTrainingConfig,
        'ModelCalibrationConfig':     ModelCalibrationConfig
    }

    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        """
        Initialize XGBoost Train-Calibrate template.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        # Call parent constructor with dependencies
        super().__init__(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        
        logger.info(f"Initialized XGBoost Training and Calibration template for: {self._get_pipeline_name()}")

    def _validate_configuration(self) -> None:
        """
        Perform lightweight validation of configuration structure.
        
        This validates the presence of required configurations and basic structural
        requirements without duplicating dependency validation handled by the resolver.
        
        Raises:
            ValueError: If configuration structure is invalid
        """
        # Find training preprocessing config
        tp_configs = [cfg for name, cfg in self.configs.items() 
                     if isinstance(cfg, TabularPreprocessingConfig)]
        
        # Check for presence of training config
        training_config = next((cfg for cfg in tp_configs if getattr(cfg, 'job_type', None) == 'training'), None)
        if not training_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='training'")
        
        # Check for required XGBoost training config
        xgboost_configs = [cfg for _, cfg in self.configs.items() if isinstance(cfg, XGBoostTrainingConfig)]
        if not xgboost_configs:
            raise ValueError("No XGBoost training configuration found")
        if len(xgboost_configs) > 1:
            raise ValueError("Multiple XGBoost training configurations found, expected exactly one")
        
        # Check for required model calibration config
        calibration_configs = [cfg for _, cfg in self.configs.items() if isinstance(cfg, ModelCalibrationConfig)]
        if not calibration_configs:
            raise ValueError("No model calibration configuration found")
        if len(calibration_configs) > 1:
            raise ValueError("Multiple model calibration configurations found, expected exactly one")
                
        logger.info("Basic configuration structure validation passed")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """
        Get pipeline parameters.
        
        Returns:
            List of pipeline parameters
        """
        return [
            PIPELINE_EXECUTION_TEMP_DIR, KMS_ENCRYPTION_KEY_PARAM,
            SECURITY_GROUP_ID, VPC_SUBNET,
        ]

    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Create a mapping from step types to builder classes.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        # Use step names from centralized registry to ensure consistency
        return {
            STEP_NAMES["CradleDataLoading"]["spec_type"]: CradleDataLoadingStepBuilder,
            STEP_NAMES["TabularPreprocessing"]["spec_type"]: TabularPreprocessingStepBuilder,
            STEP_NAMES["XGBoostTraining"]["spec_type"]: XGBoostTrainingStepBuilder,
            STEP_NAMES["ModelCalibration"]["spec_type"]: ModelCalibrationStepBuilder,
        }

    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """
        Create a mapping from step names to config instances.
        
        Returns:
            Dictionary mapping step names to configurations
        """
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
        config_map["training_data_load"] = cradle_configs.get('training')
        config_map["training_preprocess"] = tp_configs.get('training')
        
        # Find single instance configs
        xgboost_configs = [cfg for _, cfg in self.configs.items() if isinstance(cfg, XGBoostTrainingConfig)]
        if xgboost_configs:
            config_map["xgboost_train"] = xgboost_configs[0]
        
        # Add calibration config
        calibration_configs = [cfg for _, cfg in self.configs.items() if isinstance(cfg, ModelCalibrationConfig)]
        if calibration_configs:
            config_map["model_calibration"] = calibration_configs[0]
        
        # Validate all required configs are present
        missing_configs = [name for name, cfg in config_map.items() if cfg is None]
        if missing_configs:
            raise ValueError(f"Missing required configurations: {missing_configs}")
        
        return config_map

    def _create_pipeline_dag(self) -> PipelineDAG:
        """
        Create the DAG structure for the pipeline.
        
        Returns:
            PipelineDAG instance
        """
        dag = PipelineDAG()
        
        # Add all nodes as specified
        dag.add_node("training_data_load")   # Data load for training
        dag.add_node("training_preprocess")  # Tabular preprocessing for training
        dag.add_node("xgboost_train")        # XGBoost training step
        dag.add_node("model_calibration")    # Model calibration step
        
        # Create the edges for our DAG
        dag.add_edge("training_data_load", "training_preprocess")
        dag.add_edge("training_preprocess", "xgboost_train")
        dag.add_edge("xgboost_train", "model_calibration")
        
        return dag
    
    def _store_pipeline_metadata(self, assembler: PipelineAssembler) -> None:
        """
        Store pipeline metadata from template.
        
        This method stores Cradle data loading requests for use in filling execution documents.
        
        Args:
            assembler: PipelineAssembler instance
        """
        # Store Cradle data loading requests
        if hasattr(assembler, 'cradle_loading_requests'):
            self.pipeline_metadata['cradle_loading_requests'] = assembler.cradle_loading_requests
    
    def _get_pipeline_name(self) -> str:
        """
        Get pipeline name.
        
        Returns:
            Pipeline name
        """
        return f"{self.base_config.pipeline_name}-xgb-train-calibrate"

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with pipeline metadata.
        
        This method fills the execution document with Cradle data loading requests.
        
        Args:
            execution_document: Execution document to fill
            
        Returns:
            Updated execution document
        """
        if "PIPELINE_STEP_CONFIGS" not in execution_document:
            raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
    
        pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

        # Fill Cradle configurations
        cradle_requests = self.pipeline_metadata.get('cradle_loading_requests', {})
        for step_name, request_dict in cradle_requests.items():
            if step_name not in pipeline_configs:
                logger.warning(f"Cradle step '{step_name}' not found in execution document")
                continue
            pipeline_configs[step_name]["STEP_CONFIG"] = request_dict
            logger.info(f"Updated execution config for Cradle step: {step_name}")

        return execution_document
