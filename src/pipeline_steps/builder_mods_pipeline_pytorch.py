from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import Step
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class ModsPipelinePyTorchBuilder(StepBuilderBase):
    """
    Builder for a MODS Pipeline for PyTorch models.
    This class is responsible for configuring and creating a SageMaker Pipeline
    that includes all the steps for a PyTorch model workflow.
    
    Note: This is a special builder that doesn't create a single step, but rather
    a complete pipeline. It's included in the step builders for consistency.
    """

    def __init__(
        self,
        config: Any,  # No specific config class for this builder
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the MODS pipeline.

        Args:
            config: Configuration object containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Pipeline.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific builder are present and valid before attempting to build the pipeline.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating ModsPipelinePyTorchBuilder configuration...")
        
        # This builder doesn't have specific configuration requirements
        # as it's more of a coordinator for other builders
        
        logger.info("ModsPipelinePyTorchBuilder configuration validation succeeded.")
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # This builder doesn't have specific input requirements
        # as it's more of a coordinator for other builders
        return {}
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # This builder doesn't have specific output properties
        # as it's more of a coordinator for other builders
        return {}
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to ModsPipelinePyTorch builder.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        # This builder doesn't have specific custom properties to match
        return set()
    
    def create_pipeline(self, **kwargs) -> Pipeline:
        """
        Creates a SageMaker Pipeline for a PyTorch model workflow.
        This method is a placeholder for the actual implementation, which would
        create and configure all the steps for a PyTorch model workflow.

        Args:
            **kwargs: Keyword arguments for configuring the pipeline.

        Returns:
            A configured sagemaker.workflow.pipeline.Pipeline instance.
        """
        logger.info("Creating MODS Pipeline for PyTorch...")
        
        # This is a placeholder for the actual implementation
        # In a real implementation, this would create and configure all the steps
        # for a PyTorch model workflow
        
        # Example of what this might look like:
        # 1. Create data loading step
        # 2. Create preprocessing step
        # 3. Create training step
        # 4. Create model step
        # 5. Create evaluation step
        # 6. Create packaging step
        # 7. Create registration step
        # 8. Connect all steps in a pipeline
        
        # For now, just return a placeholder pipeline
        pipeline = Pipeline(
            name="PyTorch-MODS-Pipeline",
            steps=[],
            sagemaker_session=self.session
        )
        
        logger.info(f"Created Pipeline with name: {pipeline.name}")
        return pipeline
    
    def create_step(self, **kwargs) -> Step:
        """
        This method is not applicable for this builder, as it creates a pipeline, not a step.
        It's included for compatibility with the StepBuilderBase interface.

        Raises:
            NotImplementedError: This method is not implemented for this builder.
        """
        raise NotImplementedError(
            "ModsPipelinePyTorchBuilder creates a pipeline, not a step. Use create_pipeline() instead."
        )
