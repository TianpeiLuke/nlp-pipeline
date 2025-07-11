from typing import Dict, Optional, List
from pathlib import Path
import logging
import time
from datetime import datetime

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.network import NetworkConfig

# Import the pipeline builder template
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate, PipelineDAG
from src.pipeline_steps.utils import load_configs

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

# Import all Configs
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_model_step_pytorch import PytorchModelCreationConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

# Import Builders
from src.pipeline_steps.builder_model_step_pytorch import PytorchModelStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder

CONFIG_CLASSES = {
    'BasePipelineConfig': BasePipelineConfig,
    'PytorchModelCreationConfig': PytorchModelCreationConfig,
    'ProcessingStepConfigBase': ProcessingStepConfigBase,
    'PackageStepConfig': PackageStepConfig,
    'ModelRegistrationConfig': ModelRegistrationConfig,
    'PayloadConfig': PayloadConfig
}


class TemplatePytorchPipelineBuilder:
    """
    Builder for model deployment pipeline without training using the PipelineBuilderTemplate.
    
    This pipeline is designed for deploying a pre-trained PyTorch model without retraining.
    It consists of four sequentially connected steps:
    
    1. Model Step: 
        - Creates the model object in SageMaker using a pretrained model.tar.gz
        - Configures the model with the appropriate environment and entry point
    
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
    
    This pipeline builder uses the PipelineBuilderTemplate to create a SageMaker Pipeline
    that orchestrates these steps in the correct order.
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize the PyTorch model registration pipeline builder.
        
        Args:
            config_path: Path to the configuration file
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
            
        Raises:
            ValueError: If required configurations are missing
            TypeError: If configurations are of the wrong type
        """
        start_time = time.time()
        logger.info(f"Loading configs from: {config_path}")
        
        try:
            self.configs = load_configs(config_path, CONFIG_CLASSES)
            logger.info(f"Loaded {len(self.configs)} configurations")
        except Exception as e:
            logger.error(f"Error loading configurations: {e}")
            raise ValueError(f"Failed to load configurations from {config_path}: {e}") from e
        
        # Extract and validate configs
        self._validate_and_extract_configs()
        
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        
        # Initialize the payload config
        try:
            logger.info("Generating and uploading payloads")
            self.payload_config.generate_and_upload_payloads()
            logger.info("Successfully generated and uploaded payloads")
        except Exception as e:
            logger.error(f"Error generating and uploading payloads: {e}")
            raise ValueError(f"Failed to generate and upload payloads: {e}") from e

        elapsed_time = time.time() - start_time
        logger.info(f"Initialized TemplatePipelineBuilder for pipeline: {self.base_config.pipeline_name} in {elapsed_time:.2f} seconds")

    def _validate_and_extract_configs(self):
        """
        Extract and validate individual configs.
        
        This method checks that all required configurations are present and of the correct type.
        It extracts the configurations into instance variables for easier access.
        
        Raises:
            ValueError: If required configurations are missing
            TypeError: If configurations are of the wrong type
        """
        required_steps = ['Base', 'PytorchModel', 'Package', 'Registration', 'Payload']
        missing_steps = [step for step in required_steps if step not in self.configs]
        if missing_steps:
            raise ValueError(f"Missing required configurations for steps: {missing_steps}")

        self.base_config = self.configs['Base']
        self.model_config = self.configs['PytorchModel']
        self.package_config = self.configs['Package']
        self.registration_config = self.configs['Registration']
        self.payload_config = self.configs['Payload']

        # Validate config types
        if not isinstance(self.model_config, PytorchModelCreationConfig):
            raise TypeError(f"Expected PytorchModelCreationConfig, got {type(self.model_config)}")
        if not isinstance(self.package_config, PackageStepConfig):
            raise TypeError(f"Expected PackageStepConfig, got {type(self.package_config)}")
        if not isinstance(self.registration_config, ModelRegistrationConfig):
            raise TypeError(f"Expected ModelRegistrationConfig, got {type(self.registration_config)}")
        if not isinstance(self.payload_config, PayloadConfig):
            raise TypeError(f"Expected PayloadConfig, got {type(self.payload_config)}")
            
        logger.info("All required configurations are present and of the correct type")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """Get pipeline parameters."""
        return [
            PIPELINE_EXECUTION_TEMP_DIR,
            KMS_ENCRYPTION_KEY_PARAM,
            VPC_SUBNET,
            SECURITY_GROUP_ID,
        ]

    def validate_model_path(self, model_s3_path: str) -> None:
        """
        Validate the provided model S3 path.
        
        Args:
            model_s3_path: S3 path to the model artifact
            
        Raises:
            ValueError: If the S3 path is invalid
        """
        if not model_s3_path:
            raise ValueError("Model S3 path cannot be empty")
            
        if not model_s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {model_s3_path}. Must start with 's3://'")
            
        if not model_s3_path.endswith('.tar.gz'):
            logger.warning(f"Model path {model_s3_path} does not end with .tar.gz")
            
        # Check for basic S3 path structure (bucket/key)
        parts = model_s3_path[5:].split('/', 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"Invalid S3 path structure: {model_s3_path}. Expected format: s3://bucket/key")
            
        logger.info(f"Validated model S3 path: {model_s3_path}")

    def _prepare_model_config(self, model_s3_path: str):
        """
        Prepare the model config with the provided model path.
        
        This method creates a copy of the model configuration and updates it with
        the provided model path and standardized region settings.
        
        Args:
            model_s3_path: S3 path to the model artifact
            
        Returns:
            Updated model configuration
        """
        # Force the model region to be NA
        model_config_copy = self.model_config.model_copy()
        model_config_copy.region = 'NA'
        model_config_copy.aws_region = 'us-east-1'
        model_config_copy.model_data = model_s3_path
        
        logger.info(f"Prepared model config with data: {model_s3_path}")
        logger.info(f"Model aws region: {model_config_copy.aws_region}")
        
        return model_config_copy

    def generate_pipeline(self, model_s3_path: str) -> Pipeline:
        """
        Create deployment pipeline using an existing model.
        
        This method:
        1. Validates the model S3 path
        2. Creates a DAG defining the pipeline structure
        3. Prepares the configurations for each step
        4. Uses the PipelineBuilderTemplate to generate the pipeline
        
        Args:
            model_s3_path: S3 path to the model artifact
            
        Returns:
            SageMaker pipeline
            
        Raises:
            ValueError: If the model path is invalid or if pipeline generation fails
        """
        start_time = time.time()
        logger.info(f"Creating deployment pipeline: {self.base_config.pipeline_name}")
        
        # Validate the model path
        self.validate_model_path(model_s3_path)
        
        try:
            # Define the DAG structure
            nodes = ["PytorchModel", "Package", "Payload", "Registration"]
            edges = [
                ("PytorchModel", "Package"),
                ("Package", "Payload"),
                ("Payload", "Registration")
            ]
            
            logger.info(f"Creating DAG with {len(nodes)} nodes and {len(edges)} edges")
            dag = PipelineDAG(nodes=nodes, edges=edges)
            
            # Create config map with prepared model config
            logger.info("Preparing model configuration")
            model_config = self._prepare_model_config(model_s3_path)
            
            config_map = {
                "PytorchModel": model_config,
                "Package": self.package_config,
                "Payload": self.payload_config,
                "Registration": self.registration_config
            }
            logger.info(f"Created config map with {len(config_map)} entries")
            
            # Create step builder map
            step_builder_map = {
                "PytorchModel": PytorchModelStepBuilder,
                "Package": MIMSPackagingStepBuilder,
                "Payload": MIMSPayloadStepBuilder,
                "Registration": ModelRegistrationStepBuilder
            }
            
            # Create the pipeline builder template
            logger.info("Creating pipeline builder template")
            pipeline_builder = PipelineBuilderTemplate(
                dag=dag,
                config_map=config_map,
                step_builder_map=step_builder_map,
                sagemaker_session=self.session,
                role=self.role,
                pipeline_parameters=self._get_pipeline_parameters(),
                notebook_root=self.notebook_root
            )
            
            # Generate the pipeline
            logger.info(f"Generating pipeline: {self.base_config.pipeline_name}")
            pipeline = pipeline_builder.generate_pipeline(self.base_config.pipeline_name)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Generated pipeline {self.base_config.pipeline_name} in {elapsed_time:.2f} seconds")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"Failed to generate pipeline: {str(e)}")
            raise ValueError(f"Failed to generate pipeline: {str(e)}") from e


# Example usage
if __name__ == "__main__":
    # This is just an example and won't be executed when imported
    config_path = "path/to/config.json"
    model_s3_path = "s3://bucket/path/to/model.tar.gz"
    
    builder = TemplatePytorchPipelineBuilder(
        config_path=config_path,
        # sagemaker_session and role would be provided in actual usage
    )
    
    pipeline = builder.generate_pipeline(model_s3_path)
    # pipeline.upsert()  # To create or update the pipeline in SageMaker
