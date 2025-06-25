from typing import Dict, Optional, Any, List, Set, Union
from pathlib import Path
import logging
import os
import importlib

from sagemaker.workflow.steps import Step
from sagemaker.processing import ProcessingInput
from sagemaker.workflow.properties import Properties

# Import the customized step
from secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step import (
    MimsModelRegistrationProcessingStep,
)

from .config_mims_registration_step import ModelRegistrationConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class ModelRegistrationStepBuilder(StepBuilderBase):
    """
    Builder for a Model Registration ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that registers a model with MIMS.
    """

    def __init__(
        self,
        config: ModelRegistrationConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model registration step.

        Args:
            config: A ModelRegistrationConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, ModelRegistrationConfig):
            raise ValueError(
                "ModelRegistrationStepBuilder requires a ModelRegistrationConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: ModelRegistrationConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating ModelRegistrationConfig...")
        
        # Validate required attributes
        required_attrs = [
            'processing_instance_type',
            'processing_instance_count',
            'processing_volume_size',
            'processing_entry_point',
            'processing_source_dir',
            'region',
            'model_name',
            'model_version'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelRegistrationConfig missing required attribute: {attr}")
        
        # Validate input and output names
        if "model_package_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'model_package_input'")
        
        if "payload_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'payload_input'")
        
        if "registration_output" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'registration_output'")
        
        logger.info("ModelRegistrationConfig validation succeeded.")


    def _get_processing_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'model_package_input', 'payload_input')
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Get the input keys from config
        model_package_key = self.config.input_names["model_package_input"]
        payload_key = self.config.input_names["payload_input"]
        
        # Check if inputs is empty or doesn't contain the required keys
        if not inputs:
            raise ValueError(f"Inputs dictionary is empty. Must supply S3 URIs for '{model_package_key}' and '{payload_key}'")
        
        if model_package_key not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{model_package_key}' in 'inputs'")
        
        if payload_key not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{payload_key}' in 'inputs'")

        # Define the input channels
        processing_inputs = [
            ProcessingInput(
                source=inputs[model_package_key],
                destination="/opt/ml/processing/input/model",
                s3_data_distribution_type="FullyReplicated",
                s3_input_mode="File"
            )
        ]
        
        # Add payload input if available
        if payload_key in inputs:
            processing_inputs.append(
                ProcessingInput(
                    source=inputs[payload_key],
                    destination="/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        
        return processing_inputs

        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
            "dependencies": self.COMMON_PROPERTIES["dependencies"],
            "performance_metadata_location": "Optional S3 location of performance metadata file"
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Get output properties from config's output_names
        return {k: v for k, v in (self.config.output_names or {}).items()}
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to ModelRegistration step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Look for model package output from a MIMSPackagingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our model_package_input
                model_package_key = self.config.input_names.get("model_package_input")
                if model_package_key:
                    # Look for an output with a name that contains 'model_package'
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and "model_package" in output.output_name.lower():
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if model_package_key not in inputs.get("inputs", {}):
                                inputs["inputs"][model_package_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found model package from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract model package from step: {e}")
                
        # Look for payload output from a MIMSPayloadStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our payload_input
                payload_key = self.config.input_names.get("payload_input")
                if payload_key:
                    # Look for an output with a name that contains 'payload'
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and "payload" in output.output_name.lower():
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if payload_key not in inputs.get("inputs", {}):
                                inputs["inputs"][payload_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found payload from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract payload from step: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> Step:
        """
        Creates a specialized MimsModelRegistrationProcessingStep for the pipeline.
        This method orchestrates the assembly of the inputs and configuration
        into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - dependencies: Optional list of steps that this step depends on.
                - performance_metadata_location: Optional S3 location of performance metadata file.
                  If not provided, no performance metadata will be used.

        Returns:
            A configured MimsModelRegistrationProcessingStep instance.
        """
        logger.info("Creating MimsModelRegistrationProcessingStep...")

        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        performance_metadata_location = self._extract_param(kwargs, 'performance_metadata_location')
        
        # Validate required parameters
        if not inputs:
            raise ValueError("inputs must be provided")

        # Get processing inputs
        processing_inputs = self._get_processing_inputs(inputs)

        # Create step name
        step_name = f"{self._get_step_name('ModelRegistration')}-{self.config.region}"
        
        # Create the specialized step
        try:
            registration_step = MimsModelRegistrationProcessingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session,
                processing_input=processing_inputs,
                performance_metadata_location=performance_metadata_location,
                depends_on=dependencies or []
            )
            
            logger.info(f"Created MimsModelRegistrationProcessingStep with name: {registration_step.name}")
            return registration_step
            
        except Exception as e:
            logger.error(f"Error creating MimsModelRegistrationProcessingStep: {e}")
            raise ValueError(f"Failed to create MimsModelRegistrationProcessingStep: {e}") from e
