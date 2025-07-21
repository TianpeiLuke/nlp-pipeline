"""
Template-based builder for model deployment pipeline using DummyTraining step.

This template creates a pipeline that performs:
1) DummyTraining (makes a pretrained model available)
2) Packaging (prepares the model for deployment)
3) Payload Testing (validates the model works correctly)
4) MIMS Registration (registers the model in the Model Management System)

All steps run in parallel where possible to optimize pipeline execution time.
"""

from typing import Dict, Optional, List, Any, Type
from pathlib import Path
import logging
import time
from datetime import datetime
import os
import importlib

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.network import NetworkConfig
from sagemaker.image_uris import retrieve

# Import base template
from .pipeline_template_base import PipelineTemplateBase

# Import dependencies for DAG and step builders
from .pipeline_assembler import PipelineAssembler
from ..pipeline_dag.base_dag import PipelineDAG
from ..pipeline_deps.registry_manager import RegistryManager
from ..pipeline_deps.dependency_resolver import UnifiedDependencyResolver
from ..pipeline_steps.utils import load_configs

# Set up logging
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

# Import all configs
from ..pipeline_steps.config_base import BasePipelineConfig
from ..pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from ..pipeline_steps.config_dummy_training_step import DummyTrainingConfig
from ..pipeline_steps.config_mims_packaging_step import PackageStepConfig
from ..pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from ..pipeline_steps.config_mims_payload_step import PayloadConfig

# Import step builders
from ..pipeline_steps.builder_step_base import StepBuilderBase
from ..pipeline_steps.builder_dummy_training_step import DummyTrainingStepBuilder
from ..pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from ..pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
from ..pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder
from ..pipeline_registry.step_names import STEP_NAMES


class DummyTrainingModelRegistrationTemplate(PipelineTemplateBase):
    """
    Template-based builder for model deployment pipeline using DummyTraining step.
    
    This pipeline is designed for deploying a pre-trained model without retraining.
    It consists of four steps:
    
    1. DummyTraining Step: 
        - Takes a pretrained model.tar.gz file
        - Makes it available for downstream steps by copying it to an S3 location
        - Bypasses the actual training process
    
    2. Packaging Step:
        - Unpacks the model.tar.gz
        - Injects source code and dependencies
        - Repacks the model.tar.gz with the additional files
    
    3. Payload Test Step:
        - Tests the model with sample payloads to ensure it works correctly
        - Validates the model's input and output formats
    
    4. MIMS Model Registration Step:
        - Registers the model in the Model Management System (MMS) using MIMS
        - Makes the model available for deployment
    """
    # Define config classes used by this template
    CONFIG_CLASSES = {
        'BasePipelineConfig': BasePipelineConfig,
        'DummyTrainingConfig': DummyTrainingConfig,
        'ProcessingStepConfigBase': ProcessingStepConfigBase,
        'PackageStepConfig': PackageStepConfig,
        'ModelRegistrationConfig': ModelRegistrationConfig,
        'PayloadConfig': PayloadConfig
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
        Initialize DummyTraining model registration template.
        
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
        
        # Initialize the payload config
        try:
            logger.info("Generating and uploading payloads")
            payload_config = next((cfg for _, cfg in self.configs.items() if isinstance(cfg, PayloadConfig)), None)
            if payload_config:
                payload_config.generate_and_upload_payloads()
                logger.info("Successfully generated and uploaded payloads")
            else:
                logger.warning("No PayloadConfig found, skipping payload generation")
        except Exception as e:
            logger.error(f"Error generating and uploading payloads: {e}")
            raise ValueError(f"Failed to generate and upload payloads: {e}") from e
        
        # Storage for pipeline metadata
        self.model_s3_path = None
        
        logger.info(f"Initialized DummyTraining model registration template for: {self._get_pipeline_name()}")
        
    def _validate_configuration(self) -> None:
        """
        Perform lightweight validation of configuration structure.
        
        This validates the presence of required configurations and basic structural
        requirements without duplicating dependency validation handled by the resolver.
        
        Raises:
            ValueError: If configuration structure is invalid
        """
        required_steps = ['Base', 'DummyTraining', 'Package', 'Registration', 'Payload']
        missing_steps = [step for step in required_steps if step not in self.configs]
        if missing_steps:
            raise ValueError(f"Missing required configurations for steps: {missing_steps}")

        # Check for required single-instance configs
        for config_type, name in [
            (DummyTrainingConfig, "DummyTraining"),
            (PackageStepConfig, "model packaging"),
            (PayloadConfig, "payload testing"),
            (ModelRegistrationConfig, "model registration"),
        ]:
            instances = [cfg for _, cfg in self.configs.items() if isinstance(cfg, config_type)]
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
            PIPELINE_EXECUTION_TEMP_DIR,
            KMS_ENCRYPTION_KEY_PARAM,
            VPC_SUBNET,
            SECURITY_GROUP_ID,
        ]
        

    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Create a mapping from step types to builder classes.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        # Use step names from centralized registry to ensure consistency
        return {
            STEP_NAMES["DummyTraining"]["spec_type"]: DummyTrainingStepBuilder,
            STEP_NAMES["Package"]["spec_type"]: MIMSPackagingStepBuilder,
            STEP_NAMES["Payload"]["spec_type"]: MIMSPayloadStepBuilder,
            STEP_NAMES["Registration"]["spec_type"]: ModelRegistrationStepBuilder,
        }
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """
        Create a mapping from step names to config instances.
        
        Returns:
            Dictionary mapping step names to configurations
        """
        config_map = {}
        
        # Find single instance configs
        for cfg_type, step_name in [
            (DummyTrainingConfig, "dummy_training"),
            (PackageStepConfig, "model_packaging"),
            (PayloadConfig, "payload_test"),
            (ModelRegistrationConfig, "model_registration"),
        ]:
            # Find instances of each config type
            instances = [cfg for _, cfg in self.configs.items() if isinstance(cfg, cfg_type)]
            if instances:
                config_map[step_name] = instances[0]
        
        # Validate all required configs are present
        missing_configs = [name for name, cfg in config_map.items() if cfg is None]
        if missing_configs:
            raise ValueError(f"Missing required configurations: {missing_configs}")
            
        # Update DummyTraining config with model path if we have one
        if self.model_s3_path and "dummy_training" in config_map:
            # Create a copy of the config and update it
            dummy_config = config_map["dummy_training"].model_copy()
            dummy_config.pretrained_model_path = self.model_s3_path
            config_map["dummy_training"] = dummy_config
            logger.info(f"Updated DummyTraining config with model path: {self.model_s3_path}")
        
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
                               if isinstance(cfg, ModelRegistrationConfig) and not isinstance(cfg, PayloadConfig)), None)
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
        """
        Create the DAG structure for the pipeline.
        
        Returns:
            PipelineDAG instance
        """
        dag = PipelineDAG()
        
        # Add all nodes
        dag.add_node("dummy_training")    # DummyTraining step
        dag.add_node("model_packaging")   # Package step
        dag.add_node("model_registration") # MIMS registration step
        dag.add_node("payload_test")      # Payload step
        
        # Define the edges
        dag.add_edge("dummy_training", "model_packaging")
        dag.add_edge("dummy_training", "payload_test")
        dag.add_edge("model_packaging", "model_registration")
        dag.add_edge("payload_test", "model_registration")
        
        return dag
    
    def _store_pipeline_metadata(self, assembler: PipelineAssembler) -> None:
        """
        Store pipeline metadata from template.
        
        This method stores registration step configurations for use in filling execution documents.
        
        Args:
            assembler: PipelineAssembler instance
        """
        # Find registration steps
        try:
            registration_steps = []
            for step_name, step_instance in assembler.step_instances.items():
                if "registration" in step_name.lower() or "modelregistration" in str(type(step_instance)).lower():
                    registration_steps.append(step_instance)
                    logger.info(f"Found registration step: {step_name}")
            
            if not registration_steps:
                logger.warning("No registration steps found in pipeline")
                return
            
            # Try to retrieve the image URI for registration configs
            registration_cfg = next((cfg for _, cfg in self.configs.items() 
                                   if isinstance(cfg, ModelRegistrationConfig) and not isinstance(cfg, PayloadConfig)), None)
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
            registration_configs = {}
            for step in registration_steps:
                if hasattr(step, 'name'):
                    registration_configs[step.name] = exec_config
                    logger.info(f"Stored execution doc config for registration step: {step.name}")
                elif isinstance(step, dict):
                    for name, s in step.items():
                        registration_configs[s.name] = exec_config
                        logger.info(f"Stored execution doc config for registration step: {s.name}")
            
            # Store in pipeline metadata
            self.pipeline_metadata['registration_configs'] = registration_configs
            
        except Exception as e:
            logger.warning(f"Failed to store registration step configs: {e}")
    
    def _get_pipeline_name(self) -> str:
        """
        Get pipeline name.
        
        Returns:
            Pipeline name
        """
        return f"{self.base_config.pipeline_name}-dummy-training-reg"
        
    def generate_pipeline(self) -> Pipeline:
        """
        Create deployment pipeline.
        
        This method creates a pipeline using the configurations provided during initialization.
        
        Returns:
            SageMaker pipeline
            
        Raises:
            ValueError: If pipeline generation fails
        """
        # Call parent generate_pipeline which will use our _create_pipeline_dag, 
        # _create_config_map and _store_pipeline_metadata methods
        return super().generate_pipeline()

    def fill_execution_document(self, execution_document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fill in the execution document with pipeline metadata.
        
        This method fills the execution document with model registration configurations.
        
        Args:
            execution_document: Execution document to fill
            
        Returns:
            Updated execution document
        """
        if "PIPELINE_STEP_CONFIGS" not in execution_document:
            raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")
    
        pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

        # Find registration config
        registration_cfg = next(
            (cfg for _, cfg in self.configs.items() 
             if isinstance(cfg, ModelRegistrationConfig) and not isinstance(cfg, PayloadConfig)), 
            None
        )
        
        # Fill Registration configurations
        if registration_cfg:
            registration_configs = self.pipeline_metadata.get('registration_configs', {})
            for step_name, config in registration_configs.items():
                registration_step_name = f"Registration_{registration_cfg.region}"
                if registration_step_name not in pipeline_configs:
                    logger.warning(f"Registration step '{registration_step_name}' not found in execution document")
                    continue
                pipeline_configs[registration_step_name]["STEP_CONFIG"] = config
                logger.info(f"Updated execution config for registration step: {registration_step_name}")

        return execution_document


# Example usage
if __name__ == "__main__":
    # This is just an example and won't be executed when imported
    config_path = "path/to/config.json"
    
    template = DummyTrainingModelRegistrationTemplate(
        config_path=config_path,
        # sagemaker_session and role would be provided in actual usage
    )
    
    pipeline = template.generate_pipeline()
    # pipeline.upsert()  # To create or update the pipeline in SageMaker
