"""
Template-based builder for XGBoost Train-Evaluate E2E pipeline without MIMS Registration.

This template creates a pipeline that performs:
1) Data Loading (for training set)
2) Tabular Preprocessing (for training set)
3) XGBoost Model Training
4) Packaging
5) Data Loading (for calibration set)
6) Tabular Preprocessing (for calibration set)
7) Model Evaluation (on calibration set)
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
from ..pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from ..pipeline_steps.config_package_step import PackageConfig
from ..pipeline_steps.config_payload_step import PayloadConfig
from ..pipeline_steps.config_registration_step import RegistrationConfig

# Step builders
from ..pipeline_steps.builder_step_base import StepBuilderBase
from ..pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from ..pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from ..pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from ..pipeline_steps.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from ..pipeline_steps.builder_package_step import PackageStepBuilder
from ..pipeline_steps.builder_payload_step import PayloadStepBuilder
from ..pipeline_registry.step_names import STEP_NAMES

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import constants from core library (these are the parameters that will be wrapped)
try:
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        PROCESSING_JOB_SHARED_NETWORK_CONFIG,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    )
    logger.info("Successfully imported constants from mods_workflow_core")
except ImportError:
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

# Define default constants with uppercase values
OUTPUT_TYPE_DATA = "DATA"
OUTPUT_TYPE_METADATA = "METADATA"
OUTPUT_TYPE_SIGNATURE = "SIGNATURE"


class XGBoostTrainEvaluateNoRegistrationTemplate(PipelineTemplateBase):
    """
    Template-based builder for XGBoost Train-Evaluate E2E pipeline without MIMS Registration.
    
    This pipeline performs:
    1) Data Loading (for training set)
    2) Tabular Preprocessing (for training set)
    3) XGBoost Model Training
    4) Packaging
    5) Data Loading (for calibration set)
    6) Tabular Preprocessing (for calibration set)
    7) Model Evaluation (on calibration set)
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
        'XGBoostModelEvalConfig':     XGBoostModelEvalConfig,
        'PackageConfig':              PackageConfig,
        'PayloadConfig':              PayloadConfig,
        'RegistrationConfig':         RegistrationConfig  # Added to support configs that may include it
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
        Initialize XGBoost Train-Evaluate E2E template without Registration.
        
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
        
        logger.info(f"Initialized XGBoost Train-Evaluate E2E template (No Registration) for: {self._get_pipeline_name()}")

    def _validate_configuration(self) -> None:
        """
        Perform lightweight validation of configuration structure.
        
        This validates the presence of required configurations and basic structural
        requirements without duplicating dependency validation handled by the resolver.
        
        Raises:
            ValueError: If configuration structure is invalid
        """
        # Find preprocessing configs
        tp_configs = [cfg for name, cfg in self.configs.items() 
                     if isinstance(cfg, TabularPreprocessingConfig)]
        
        if len(tp_configs) < 2:
            raise ValueError("Expected at least two TabularPreprocessingConfig instances")
            
        # Check for presence of training and calibration configs
        training_config = next((cfg for cfg in tp_configs if getattr(cfg, 'job_type', None) == 'training'), None)
        if not training_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='training'")
            
        calibration_config = next((cfg for cfg in tp_configs if getattr(cfg, 'job_type', None) == 'calibration'), None)
        if not calibration_config:
            raise ValueError("No TabularPreprocessingConfig found with job_type='calibration'")
        
        # Check for required single-instance configs
        for config_type, name in [
            (XGBoostTrainingConfig, "XGBoost training"),
            (PackageConfig, "model packaging"),
            (PayloadConfig, "payload testing"),
            (XGBoostModelEvalConfig, "model evaluation")
        ]:
            instances = [cfg for _, cfg in self.configs.items() if type(cfg) is config_type]
            if not instances:
                raise ValueError(f"No {name} configuration found")
            if len(instances) > 1:
                raise ValueError(f"Multiple {name} configurations found, expected exactly one")
                
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
            STEP_NAMES["Package"]["spec_type"]: PackageStepBuilder,
            STEP_NAMES["Payload"]["spec_type"]: PayloadStepBuilder,
            STEP_NAMES["XGBoostModelEval"]["spec_type"]: XGBoostModelEvalStepBuilder,
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
        config_map["train_data_load"] = cradle_configs.get('training')
        config_map["train_preprocess"] = tp_configs.get('training')
        
        # Find single instance configs (removing hyperparameter_prep)
        for cfg_type, step_name in [
            (XGBoostTrainingConfig, "xgboost_train"),
            (PackageConfig, "model_packaging"),
            (PayloadConfig, "payload_test"),
            (XGBoostModelEvalConfig, "model_evaluation")
        ]:
            # Use exact type matching (type(cfg) is cfg_type) instead of isinstance()
            # This prevents subclasses (like PayloadConfig) from matching when looking for parent class
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

    def _create_pipeline_dag(self) -> PipelineDAG:
        """
        Create the DAG structure for the pipeline.
        
        Returns:
            PipelineDAG instance
        """
        dag = PipelineDAG()
        
        # Add all nodes - no model_registration node
        dag.add_node("train_data_load")    # Data load for training
        dag.add_node("train_preprocess")   # Tabular preprocessing for training
        dag.add_node("xgboost_train")      # XGBoost training step
        dag.add_node("model_packaging")    # Package step
        dag.add_node("payload_test")       # Payload step
        dag.add_node("calib_data_load")    # Data load for calibration
        dag.add_node("calib_preprocess")   # Tabular preprocessing for calibration
        dag.add_node("model_evaluation")   # Model evaluation step
        
        # Training flow
        dag.add_edge("train_data_load", "train_preprocess")
        dag.add_edge("train_preprocess", "xgboost_train")
        
        # Output flow - no connection to registration
        dag.add_edge("xgboost_train", "model_packaging")
        dag.add_edge("xgboost_train", "payload_test")
        
        # Calibration flow
        dag.add_edge("calib_data_load", "calib_preprocess")
        
        # Evaluation flow
        dag.add_edge("xgboost_train", "model_evaluation")
        dag.add_edge("calib_preprocess", "model_evaluation")
        
        return dag
    
    def _store_pipeline_metadata(self, assembler: PipelineAssembler) -> None:
        """
        Store pipeline metadata from template.
        
        This method stores Cradle data loading requests for use in filling execution documents.
        Registration step metadata is not stored since this template does not include registration.
        It also logs information about property references that were handled during pipeline assembly.
        
        Args:
            assembler: PipelineAssembler instance
        """
        # Store Cradle data loading requests
        if hasattr(assembler, 'cradle_loading_requests'):
            self.pipeline_metadata['cradle_loading_requests'] = assembler.cradle_loading_requests
            
        # Log property reference handling for debugging
        if hasattr(assembler, 'steps'):
            property_ref_count = 0
            for step_name, step in assembler.steps.items():
                # Check if step has inputs that might be property references
                if hasattr(step, 'inputs') and step.inputs:
                    for input_item in step.inputs:
                        if hasattr(input_item, 'source') and not isinstance(input_item.source, str):
                            property_ref_count += 1
            
            if property_ref_count > 0:
                logger.info(f"Pipeline contains {property_ref_count} property references that benefit from automatic handling")
    
    def _get_pipeline_name(self) -> str:
        """
        Get pipeline name.
        
        Returns:
            Pipeline name
        """
        return f"{self.base_config.pipeline_name}-xgb-train-eval-no-registration"

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with pipeline metadata.
        
        This method fills the execution document with Cradle data loading requests.
        Model registration configurations are not included as this template does not
        include registration.
        
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
