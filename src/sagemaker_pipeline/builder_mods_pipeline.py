from typing import Dict, Optional, List
from pathlib import Path
import logging
from datetime import datetime

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline_context import PipelineSession

from .utils import load_configs

# Import all Configs
from .config_base import BasePipelineConfig
from .config_training_step import TrainingConfig
from .config_model_step import ModelCreationConfig
from .config_processing_step_base import ProcessingStepConfigBase
from .config_mims_packaging_step import PackageStepConfig
from .config_mims_registration_step import ModelRegistrationConfig

# Import Builders
from .builder_model_step import PytorchModelStepBuilder
from .builder_mims_packaging_step import MIMSPackagingStepBuilder
from .builder_mims_registration_step import ModelRegistrationStepBuilder

CONFIG_CLASSES = {
        'BasePipelineConfig': BasePipelineConfig,
        'TrainingConfig': TrainingConfig,
        'ModelCreationConfig': ModelCreationConfig,
        'ProcessingStepConfigBase': ProcessingStepConfigBase,
        'PackageStepConfig': PackageStepConfig,
        'ModelRegistrationConfig': ModelRegistrationConfig
    }


logger = logging.getLogger(__name__)


class PipelineBuilder:
    """Builder for model deployment pipeline without training"""

    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize pipeline builder.
        
        Args:
            config_path: Path to the JSON config file
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
        """
        # Load configs from JSON
        self.configs = load_configs(config_path, CONFIG_CLASSES)
        
        # Extract individual configs
        self.base_config = self.configs['Base']
        self.model_config = self.configs['Model']
        self.package_config = self.configs['Package']
        self.registration_config = self.configs['Registration']

        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()

        logger.info(f"Initializing PipelineBuilder for pipeline: {self.base_config.pipeline_name}")

    def _create_model_step(self, model_s3_path: str):
        """Create model step using provided model S3 path."""
        logger.info(f"Creating model step with model from: {model_s3_path}")
        model_builder = PytorchModelStepBuilder(
            config=self.model_config,
            sagemaker_session=self.session,
            role=self.role
        )
        return model_builder.create_model_step(model_data=model_s3_path)

    def _create_packaging_step(self, model_step):
        """Create MIMS packaging step"""
        logger.info("Creating MIMS packaging step")
        packaging_builder = MIMSPackagingStepBuilder(
            config=self.package_config,
            sagemaker_session=self.session,
            role=self.role,
            notebook_root=self.notebook_root  # Added notebook_root
        )
        return packaging_builder.create_packaging_step(
            model_data=model_step.model_artifacts_path,
            dependencies=[model_step]
        )

    def _create_registration_steps(self, packaging_step: ProcessingStep) -> Dict[str, ProcessingStep]:
        """Create model registration steps"""
        logger.info("Creating registration steps")
        registration_builder = ModelRegistrationStepBuilder(
            config=self.registration_config,
            sagemaker_session=self.session,
            role=self.role
        )
        
        return registration_builder.create_registration_steps(
            packaging_step_output=packaging_step.properties.ProcessingOutputConfig.Outputs[
                "output"
            ].S3Uri,
            dependencies=[packaging_step],
            regions=[self.base_config.region]  # Use region from base config
        )

    def _get_pipeline_parameters(self) -> List[ParameterString]:
        """Get pipeline parameters"""
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
        logger.info(f"Validated model S3 path: {model_s3_path}")

    def create_pipeline(self, model_s3_path: str) -> Pipeline:
        """
        Create deployment pipeline using existing model.
        
        Args:
            model_s3_path: S3 path to the model artifacts
            
        Returns:
            SageMaker Pipeline instance
        """
        logger.info(f"Creating deployment pipeline: {self.base_config.pipeline_name}")
        
        # Validate model path
        self.validate_model_path(model_s3_path)

        # Create steps
        model_step = self._create_model_step(model_s3_path)
        packaging_step = self._create_packaging_step(model_step)
        registration_steps = self._create_registration_steps(packaging_step)

        # Combine all steps
        steps = [
            model_step,
            packaging_step,
            *registration_steps.values()
        ]

        # Create and return pipeline
        return Pipeline(
            name=self.base_config.pipeline_name,
            parameters=self._get_pipeline_parameters(),
            steps=steps,
            sagemaker_session=self.session
        )
