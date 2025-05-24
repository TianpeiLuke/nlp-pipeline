from typing import Dict, Optional, List, Union
from pathlib import Path
import logging

from sagemaker.processing import ProcessingInput
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import Properties


from secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step import (
    MimsModelRegistrationProcessingStep,
)

from .config_mims_registration_step import ModelRegistrationConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

class ModelRegistrationStepBuilder(StepBuilderBase):
    """Builder for model registration steps"""

    def __init__(
        self,
        config: ModelRegistrationConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        super().__init__(config, sagemaker_session, role, notebook_root)
        self.config: ModelRegistrationConfig = config  # Type hint for IDE support

    def validate_configuration(self) -> None:
        """Validate required configuration settings"""
        # Most validations are handled by ModelRegistrationConfig
        # Additional business logic validations can be added here
        if not self.config.source_model_inference_output_variable_list:
            raise ValueError("At least one output variable must be defined")

        logger.info(
            f"Validated registration configuration for region {self.config.region} "
            f"with {len(self.config.source_model_inference_output_variable_list)} output variables"
        )

    def _get_processing_inputs(
        self, 
        packaging_step_output: Union[str, Properties],
        performance_metadata_location: Optional[str] = None
    ) -> List[ProcessingInput]:
        """
        Get processing inputs for registration step.
        
        Args:
            packaging_step_output: S3 path or Properties object from packaging step
            performance_metadata_location: Optional path to performance metadata
            
        Returns:
            List of ProcessingInput objects
        """
        # Create model input without string formatting of Properties object
        model_input = ProcessingInput(
            source=packaging_step_output,
            destination="/opt/ml/processing/input/model",
            s3_data_distribution_type="FullyReplicated",
            s3_input_mode="File"
        )
        
        inputs = [model_input]

        if performance_metadata_location:
            inputs.append(
                ProcessingInput(
                    source=performance_metadata_location,
                    destination="/opt/ml/processing/input/performance",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )

        return inputs

    def _validate_regions(self, regions: List[str]) -> None:
        """Validate region codes"""
        invalid_regions = [r for r in regions if r not in self.REGION_MAPPING]
        if invalid_regions:
            raise ValueError(
                f"Invalid region(s): {invalid_regions}. "
                f"Must be one of: {', '.join(self.REGION_MAPPING.keys())}"
            )

    def create_step(
        self,
        packaging_step_output: Union[str, Properties],
        dependencies: Optional[List[Step]] = None,
        performance_metadata_location: Optional[str] = None,
        regions: Optional[List[str]] = None,
    ) -> Union[Step, Dict[str, Step]]:
        """
        Create registration steps for specified regions.
        
        Args:
            packaging_step_output: S3 path or Properties object from packaging step
            dependencies: List of dependent steps
            performance_metadata_location: Optional path to performance metadata
            regions: List of regions (defaults to config region)
            
        Returns:
            Single step or dictionary mapping region codes to registration steps
        """
        regions = regions or [self.config.region]
        self._validate_regions(regions)

        # Log the packaging step output reference instead of the value
        if isinstance(packaging_step_output, Properties):
            logger.info("Creating registration steps with packaging output from Properties")
        else:
            logger.info(f"Creating registration steps with packaging output path: {packaging_step_output}")

        registration_steps = {}
        for region in regions:
            step_name = f"Registration_{region}"
            logger.info(f"Creating registration step for region {region}")
            
            try:
                step = MimsModelRegistrationProcessingStep(
                    step_name=step_name,
                    role=self.role,
                    sagemaker_session=self.session,
                    processing_input=self._get_processing_inputs(
                        packaging_step_output,
                        performance_metadata_location
                    ),
                    depends_on=dependencies or []
                )
                registration_steps[region] = step
                logger.info(f"Created registration step for {region}")
            except Exception as e:
                logger.error(f"Failed to create registration step for {region}: {str(e)}")
                raise

        # If only one region, return the step directly
        if len(regions) == 1:
            return registration_steps[regions[0]]
        return registration_steps

    # Maintain backwards compatibility
    def create_registration_steps(
        self,
        packaging_step_output: str,
        dependencies: Optional[List[Step]] = None,
        performance_metadata_location: Optional[str] = None,
        regions: Optional[List[str]] = None
    ) -> Dict[str, ProcessingStep]:
        """Backwards compatible method for creating registration steps"""
        result = self.create_step(
            packaging_step_output=packaging_step_output,
            dependencies=dependencies,
            performance_metadata_location=performance_metadata_location,
            regions=regions
        )
        # Ensure we always return a dictionary for backwards compatibility
        if isinstance(result, dict):
            return result
        return {self.config.region: result}
