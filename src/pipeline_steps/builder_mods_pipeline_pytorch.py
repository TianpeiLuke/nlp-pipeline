from typing import Dict, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession

from mods_workflow_core.utils.constants import (
    PIPELINE_EXECUTION_TEMP_DIR,
    KMS_ENCRYPTION_KEY_PARAM,
    PROCESSING_JOB_SHARED_NETWORK_CONFIG,
    SECURITY_GROUP_ID,
    VPC_SUBNET,
)


from .utils import load_configs

# Import all Configs
from .config_base import BasePipelineConfig
from .config_training_step_pytorch import PytorchTrainingConfig
from .config_model_step_pytorch import PytorchModelCreationConfig
from .config_processing_step_base import ProcessingStepConfigBase
from .config_mims_packaging_step import PackageStepConfig
from .config_mims_registration_step import ModelRegistrationConfig
from .config_mims_payload_step import PayloadConfig

# Import Builders
from .builder_model_step_pytorch import PytorchModelStepBuilder
from .builder_mims_packaging_step import MIMSPackagingStepBuilder
from .builder_mims_registration_step import ModelRegistrationStepBuilder

CONFIG_CLASSES = {
        'BasePipelineConfig': BasePipelineConfig,
        'PytorchTrainingConfig': PytorchTrainingConfig,
        'PytorchModelCreationConfig': PytorchModelCreationConfig,
        'ProcessingStepConfigBase': ProcessingStepConfigBase,
        'PackageStepConfig': PackageStepConfig,
        'ModelRegistrationConfig': ModelRegistrationConfig,
        'PayloadConfig': PayloadConfig
    }


logger = logging.getLogger(__name__)


class PytorchPipelineBuilder:
    """
    Builder for model deployment pipeline without training.
    
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

        logger.info(f"Initialized PipelineBuilder for pipeline: {self.base_config.pipeline_name}")

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

    def _create_model_step(self, model_s3_path: str):
        """Create model step using provided model S3 path."""
        logger.info(f"Creating model step with model from: {model_s3_path}")
        
        # Force the model region to be NA
        self.model_config.region = 'NA'
        self.model_config.aws_region = 'us-east-1'
        logger.info(f"Model aws region: {self.model_config.aws_region}")

        model_builder = PytorchModelStepBuilder(
            config=self.model_config,
            sagemaker_session=self.session,
            role=self.role
        )
        return model_builder.create_model_step(model_data=model_s3_path)

    def _create_packaging_step(self, model_step):
        """Create MIMS packaging step."""
        logger.info("Creating MIMS packaging step")
        
        # Ensure processing script exists
        script_path = self.package_config.get_script_path()
        if script_path:
            logger.info(f"Using packaging script: {script_path}")
        else:
            raise ValueError("No packaging script path found in config")

        packaging_builder = MIMSPackagingStepBuilder(
            config=self.package_config,
            sagemaker_session=self.session,
            role=self.role,
            notebook_root=self.notebook_root
        )
        
        return packaging_builder.create_packaging_step(
            model_data=model_step.model_artifacts_path,
            dependencies=[model_step]
        )

    def _create_registration_steps(self, packaging_step: ProcessingStep) -> Dict[str, ProcessingStep]:
        """Create model registration steps."""
        logger.info("Creating registration steps")
        
        # Generate and upload payloads first
        logger.info("Generating and uploading payloads")
        try:
            self.payload_config.generate_and_upload_payloads()
        except Exception as e:
            logger.error(f"Failed to generate/upload payloads: {str(e)}")
            raise

        registration_builder = ModelRegistrationStepBuilder(
            config=self.registration_config,
            sagemaker_session=self.session,
            role=self.role
        )

        try:
            # Get output name from package config
            output_name = self.package_config.output_names
            logger.info(f"Looking for output with name: {output_name}")
        
            # Get the S3 URI directly from the first output
            outputs = packaging_step.properties.ProcessingOutputConfig.Outputs
            s3_uri = outputs[0].S3Output.S3Uri
            
            # Log using expr
            logger.info(f"Using output expression: {outputs[0].expr}")

            '''
            return registration_builder.create_step(
                packaging_step_output=s3_uri,
                payload_s3_key=self.payload_config.sample_payload_s3_key,
                dependencies=[packaging_step],
                regions=[self.base_config.region]
            )
            '''
            
            result = registration_builder.create_step(
                packaging_step_output=s3_uri,
                payload_s3_key=self.payload_config.sample_payload_s3_key,
                dependencies=[packaging_step],
                regions=[self.base_config.region]
            )

            # Always return a list of steps
            if isinstance(result, dict):
                return list(result.values())
            else:
                return [result]

        except Exception as e:
            logger.error(f"Error in creating registration steps: {str(e)}")
            logger.error("Available packaging step outputs:")
            for output in outputs:
                logger.error(f"- {output.OutputName}: {output.S3Output.S3Uri}")
            raise

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

    def generate_pipeline(self, model_s3_path: str) -> Pipeline:
        """Create deployment pipeline using existing model."""
        logger.info(f"Creating deployment pipeline: {self.base_config.pipeline_name}")
        
        self.validate_model_path(model_s3_path)

        try:
            # Create steps
            model_step = self._create_model_step(model_s3_path)
            packaging_step = self._create_packaging_step(model_step)
            registration_steps = self._create_registration_steps(packaging_step)

            # Combine all steps
            steps = [
                model_step,
                packaging_step,
                *registration_steps  # Unpack the list of registration steps
            ]

            # Create pipeline
            return Pipeline(
                name=self.base_config.pipeline_name,
                parameters=self._get_pipeline_parameters(),
                steps=steps,
                sagemaker_session=self.session
            )

        except Exception as e:
            logger.error(f"Failed to generate pipeline: {str(e)}")
            raise