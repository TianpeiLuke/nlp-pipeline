from typing import Optional, List, Union, Dict, Any, Set
from pathlib import Path
import logging

from sagemaker.transformer import Transformer
from sagemaker.workflow.steps import TransformStep, Step
from sagemaker.inputs import TransformInput
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.pipeline_context import PipelineSession

from .config_batch_transform_step import BatchTransformStepConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

class BatchTransformStepBuilder(StepBuilderBase):
    """
    Builder for creating a SageMaker Batch Transform step in a workflow.
    """

    def __init__(
        self,
        config: BatchTransformStepConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        if not isinstance(config, BatchTransformStepConfig):
            raise ValueError(
                "BatchTransformStepBuilder requires a BatchTransformStepConfig instance."
            )
        super().__init__(config=config, sagemaker_session=sagemaker_session, role=role, notebook_root=notebook_root)
        self.config: BatchTransformStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validate that all required transform settings are provided.
        """
        # batch_input_location and batch_output_location already validated by Pydantic
        # transform_instance_type and count likewise
        if self.config.job_type not in {"training", "testing", "validation", "calibration"}:
            raise ValueError(f"Unsupported job_type: {self.config.job_type}")
        logger.info(f"BatchTransformStepBuilder configuration for '{self.config.job_type}' validated.")

    def _create_transformer(self, model_name: Union[str, Properties]) -> Transformer:
        """
        Instantiate the SageMaker Transformer object.
        """
        return Transformer(
            model_name=model_name,
            instance_type=self.config.transform_instance_type,
            instance_count=self.config.transform_instance_count,
            output_path=self.config.batch_output_location,
            accept=self.config.accept,
            assemble_with=self.config.assemble_with,
            sagemaker_session=self.session,
        )

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get base input requirements and add additional ones
        input_reqs = super().get_input_requirements()
        input_reqs["dependencies"] = self.COMMON_PROPERTIES["dependencies"]
        input_reqs["model_name"] = "Name of the SageMaker model to use for batch transform"
        input_reqs["enable_caching"] = self.COMMON_PROPERTIES["enable_caching"]
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Define the output properties for batch transform
        output_props = {
            "transform_output": "S3 location of the batch transform output"
        }
        # Add any output names from config if they exist and are not None
        if hasattr(self.config, "output_names") and self.config.output_names is not None:
            output_props.update({k: v for k, v in self.config.output_names.items()})
        return output_props
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to BatchTransform step.
        
        This method looks for:
        1. model_name from a ModelStep
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Look for model_name from a ModelStep
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ModelName"):
            try:
                model_name = prev_step.properties.ModelName
                if "model_name" in input_requirements:
                    inputs["model_name"] = model_name
                    matched_inputs.add("model_name")
                    logger.info(f"Found model_name from ModelStep: {getattr(prev_step, 'name', str(prev_step))}")
            except AttributeError as e:
                logger.warning(f"Could not extract model_name from step: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> TransformStep:
        """
        Create a TransformStep for a batch transform.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - model_name: The name of the SageMaker model (string or Properties) (required)
                - dependencies: Optional list of Pipeline Step dependencies
                - enable_caching: Whether to enable caching for this step (default: True)

        Returns:
            TransformStep: configured batch transform step.
        """
        # Extract parameters
        model_name = self._extract_param(kwargs, 'model_name')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Validate required parameters
        if not model_name:
            raise ValueError("model_name must be provided")

        # Build the transformer
        transformer = self._create_transformer(model_name)

        # Configure the transform job inputs
        transform_input = TransformInput(
            data=self.config.batch_input_location,
            content_type=self.config.content_type,
            split_type=self.config.split_type,
            join_source=self.config.join_source,
            input_filter=self.config.input_filter,
            output_filter=self.config.output_filter,
        )

        # Name the step using registry or fallback
        step_name = f"{self._get_step_name('BatchTransform')}-{self.config.job_type.capitalize()}"

        transform_step = TransformStep(
            name=step_name,
            transformer=transformer,
            inputs=transform_input,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching) if enable_caching else None
        )

        logger.info(f"Created TransformStep with name: {step_name}")
        return transform_step
