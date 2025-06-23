from typing import Dict, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession

# Import the pipeline builder template
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate, PipelineDAG
from src.pipeline_steps.utils import load_configs

# Common parameters
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir", default_value="/tmp")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMSEncryptionKey", default_value="")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId", default_value="")
VPC_SUBNET = ParameterString(name="VPCEndpointSubnet", default_value="")

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

logger = logging.getLogger(__name__)


class TemplatePytorchPipelineBuilder:
    """
    Builder for model deployment pipeline without training using the PipelineBuilderTemplate.
    
    The MODS Pipeline consists of three sequentially connected steps
    1. Model Step: 
        - This step create the model object in SageMaker using a pretrained model.tar.gz
    2. Packaging Step
        - This step would unpack the model.tar.gz, inject source code locally and repack the model.tar.gz
    3. MIMS Model Registration Step
        - This step would register the model in MMS using MIMS
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """Initialize pipeline builder."""
        logger.info(f"Loading configs from: {config_path}")
        self.configs = load_configs(config_path, CONFIG_CLASSES)
        
        # Extract and validate configs
        self._validate_and_extract_configs()
        
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        
        # Initialize the payload config
        self.payload_config.generate_and_upload_payloads()

        logger.info(f"Initialized TemplatePipelineBuilder for pipeline: {self.base_config.pipeline_name}")

    def _validate_and_extract_configs(self):
        """Extract and validate individual configs."""
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
        if not isinstance(self.package_config, PackageStepConfig):
            raise TypeError(f"Expected PackageStepConfig, got {type(self.package_config)}")
        if not isinstance(self.registration_config, ModelRegistrationConfig):
            raise TypeError(f"Expected ModelRegistrationConfig, got {type(self.registration_config)}")

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """Get pipeline parameters."""
        return [
            PIPELINE_EXECUTION_TEMP_DIR,
            KMS_ENCRYPTION_KEY_PARAM,
            VPC_SUBNET,
            SECURITY_GROUP_ID,
        ]

    def validate_model_path(self, model_s3_path: str) -> None:
        """Validate the provided model S3 path."""
        if not model_s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {model_s3_path}. Must start with 's3://'")
        if not model_s3_path.endswith('.tar.gz'):
            logger.warning(f"Model path {model_s3_path} does not end with .tar.gz")
        logger.info(f"Validated model S3 path: {model_s3_path}")

    def _prepare_model_config(self, model_s3_path: str):
        """Prepare the model config with the provided model path."""
        # Force the model region to be NA
        model_config_copy = self.model_config.model_copy()
        model_config_copy.region = 'NA'
        model_config_copy.aws_region = 'us-east-1'
        model_config_copy.model_data = model_s3_path
        logger.info(f"Model aws region: {model_config_copy.aws_region}")
        return model_config_copy

    def generate_pipeline(self, model_s3_path: str) -> Pipeline:
        """Create deployment pipeline using existing model."""
        logger.info(f"Creating deployment pipeline: {self.base_config.pipeline_name}")
        
        self.validate_model_path(model_s3_path)
        
        try:
            # Define the DAG structure
            nodes = ["CreatePytorchModelStep", "PackagingStep", "PayloadStep", "RegistrationStep"]
            edges = [
                ("CreatePytorchModelStep", "PackagingStep"),
                ("PackagingStep", "PayloadStep"),
                ("PayloadStep", "RegistrationStep")
            ]
            
            dag = PipelineDAG(nodes=nodes, edges=edges)
            
            # Create config map with prepared model config
            config_map = {
                "CreatePytorchModelStep": self._prepare_model_config(model_s3_path),
                "PackagingStep": self.package_config,
                "PayloadStep": self.payload_config,
                "RegistrationStep": self.registration_config
            }
            
            # Create step builder map
            step_builder_map = {
                "CreatePytorchModelStep": PytorchModelStepBuilder,
                "PackagingStep": MIMSPackagingStepBuilder,
                "PayloadStep": MIMSPayloadStepBuilder,
                "RegistrationStep": ModelRegistrationStepBuilder
            }
            
            # Create the pipeline builder template
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
            return pipeline_builder.generate_pipeline(self.base_config.pipeline_name)
            
        except Exception as e:
            logger.error(f"Failed to generate pipeline: {str(e)}")
            raise


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
