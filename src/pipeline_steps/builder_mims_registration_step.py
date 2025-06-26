from typing import Dict, Optional, Any, List, Set, Union
from pathlib import Path
import logging

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
        
        # Validate required attributes that are actually defined in the config
        required_attrs = [
            'region',
            'model_registration_domain',
            'model_registration_objective',
            'framework',
            'inference_instance_type',
            'inference_entry_point',
            'source_model_inference_content_types',
            'source_model_inference_response_types',
            'source_model_inference_input_variable_list',
            'source_model_inference_output_variable_list'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelRegistrationConfig missing required attribute: {attr}")
        
        # Validate input names
        if "packaged_model_output" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'packaged_model_output'")
        
        if "payload_sample" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'payload_sample'")
        
        # Registration step has no outputs, so no validation needed for output_names
        
        logger.info("ModelRegistrationConfig validation succeeded.")

    def _get_processing_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'packaged_model_output', 'payload_s3_key')
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Get the input keys from config
        model_package_key = "packaged_model_output"
        payload_sample_key = "payload_sample"  
        
        # For backward compatibility
        payload_key = "payload_s3_key"
        payload_uri_key = "payload_s3_uri"
        
        # Check if inputs is empty
        if not inputs:
            raise ValueError(f"Inputs dictionary is empty. Must supply '{model_package_key}' and '{payload_sample_key}'")
        
        # Validate required model package input
        if model_package_key not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{model_package_key}' in 'inputs'")
        
        # Validate we have at least one form of payload input
        if (payload_sample_key not in inputs and 
            payload_key not in inputs and payload_uri_key not in inputs):
            raise ValueError(f"Must supply an S3 URI for either '{payload_sample_key}', '{payload_key}', or '{payload_uri_key}' in 'inputs'")

        # Define the input channels
        processing_inputs = [
            ProcessingInput(
                source=inputs[model_package_key],
                destination="/opt/ml/processing/input/model",
                s3_data_distribution_type="FullyReplicated",
                s3_input_mode="File"
            )
        ]
        
        # Add payload input - prefer new payload_sample key if available
        if payload_sample_key in inputs:
            processing_inputs.append(
                ProcessingInput(
                    source=inputs[payload_sample_key],
                    destination="/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        # Fallback to old keys for backward compatibility
        elif payload_key in inputs:
            processing_inputs.append(
                ProcessingInput(
                    source=inputs[payload_key],
                    destination="/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        elif payload_uri_key in inputs:
            processing_inputs.append(
                ProcessingInput(
                    source=inputs[payload_uri_key],
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
        
        Note: The MimsModelRegistrationProcessingStep does not produce any accessible outputs.
        The step registers the model in MIMS as a side effect but doesn't create any
        output properties that can be referenced by subsequent steps.
        
        Returns:
            Empty dictionary since this step doesn't produce any outputs
        """
        # Registration step has no outputs
        return {}
        
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
                # Check if the step has an output that matches our packaged_model_output
                model_package_key = "packaged_model_output"
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
                
        # Look for payload outputs from a PayloadStep through Properties.Outputs
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "Outputs"):
            try:
                # First try to get payload_sample
                if "payload_sample" in prev_step.properties.Outputs:
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    inputs["inputs"]["payload_sample"] = prev_step.properties.Outputs["payload_sample"]
                    matched_inputs.add("inputs")
                    logger.info(f"Found payload_sample from step outputs: {getattr(prev_step, 'name', str(prev_step))}")
                
                # We no longer need payload_metadata for registration
                
                # Fallback to old output names for backward compatibility
                if "payload_s3_key" in prev_step.properties.Outputs:
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    inputs["inputs"]["payload_s3_key"] = prev_step.properties.Outputs["payload_s3_key"]
                    matched_inputs.add("inputs")
                    logger.info(f"Found payload_s3_key from step outputs: {getattr(prev_step, 'name', str(prev_step))}")
                
                # Also try to get payload_s3_uri if available
                if "payload_s3_uri" in prev_step.properties.Outputs:
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    inputs["inputs"]["payload_s3_uri"] = prev_step.properties.Outputs["payload_s3_uri"]
                    matched_inputs.add("inputs")
                    logger.info(f"Found payload_s3_uri from step outputs: {getattr(prev_step, 'name', str(prev_step))}")
            except (AttributeError, KeyError) as e:
                logger.warning(f"Could not extract payload from step outputs: {e}")
                
        # Fallback to old method of looking through outputs for payload 
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our payload key
                payload_sample_key = "payload_sample"
                
                # Look for an output with a name that contains 'payload'
                for output in prev_step.outputs:
                    if hasattr(output, "output_name") and "payload" in output.output_name.lower():
                        if "inputs" not in inputs:
                            inputs["inputs"] = {}
                        
                        # Try to match to a specific output name
                        if "sample" in output.output_name.lower() and payload_sample_key not in inputs.get("inputs", {}):
                            inputs["inputs"][payload_sample_key] = output.destination
                            matched_inputs.add("inputs")
                            logger.info(f"Found payload sample from step outputs: {getattr(prev_step, 'name', str(prev_step))}")
                        # Fallback to using the first payload output found
                        elif payload_sample_key not in inputs.get("inputs", {}):
                            inputs["inputs"][payload_sample_key] = output.destination
                            matched_inputs.add("inputs")
                            logger.info(f"Found generic payload from step outputs: {getattr(prev_step, 'name', str(prev_step))}")
                            break
            except AttributeError as e:
                logger.warning(f"Could not extract payload from step outputs: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> Step:
        """
        Creates a specialized MimsModelRegistrationProcessingStep for the pipeline.
        This method orchestrates the assembly of the inputs and configuration
        into a single, executable pipeline step.

        Note: The MimsModelRegistrationProcessingStep does not define property files (outputs)
        that can be referenced by subsequent steps in the pipeline. It registers the model in MIMS
        as a side effect but doesn't produce output artifacts that can be accessed through properties.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - OR individual parameters:
                  - packaged_model_output: S3 URI of the packaged model
                  - payload_s3_key: S3 key for the payload
                  - payload_s3_uri: S3 URI for the payload (alternative to payload_s3_key)
                - dependencies: Optional list of steps that this step depends on.
                - performance_metadata_location: Optional S3 location of performance metadata file.
                  If not provided, no performance metadata will be used.
                - regions: Optional list of regions to register the model in.

        Returns:
            A configured MimsModelRegistrationProcessingStep instance that registers the model in MIMS.
        """
        logger.info("Creating MimsModelRegistrationProcessingStep...")

        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        performance_metadata_location = self._extract_param(kwargs, 'performance_metadata_location')
        
        # Check if individual input parameters were provided instead of 'inputs' dictionary
        packaged_model_output = self._extract_param(kwargs, 'packaged_model_output')
        payload_s3_key = self._extract_param(kwargs, 'payload_s3_key')
        payload_s3_uri = self._extract_param(kwargs, 'payload_s3_uri')
        
        # Extract new payload parameter
        payload_sample = self._extract_param(kwargs, 'payload_sample')
        
        # If individual parameters were provided, build the inputs dictionary
        if not inputs and (packaged_model_output or payload_sample or payload_s3_key or payload_s3_uri):
            inputs = {}
            if packaged_model_output:
                inputs["packaged_model_output"] = packaged_model_output
            if payload_sample:
                inputs["payload_sample"] = payload_sample
            if payload_s3_key:
                inputs["payload_s3_key"] = payload_s3_key
            if payload_s3_uri:
                inputs["payload_s3_uri"] = payload_s3_uri
        
        # Validate required parameters
        if not inputs:
            raise ValueError("Either 'inputs' dictionary or individual 'packaged_model_output' and 'payload_s3_key'/'payload_s3_uri' must be provided")

        # Get processing inputs
        processing_inputs = self._get_processing_inputs(inputs)

        # Create step name
        step_name = f"{self._get_step_name('Registration')}-{self.config.region}"
        
        # Create the specialized step
        try:
            registration_step = MimsModelRegistrationProcessingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session,
                processing_input=processing_inputs,  # This parameter name matches the expected signature
                performance_metadata_location=performance_metadata_location,
                depends_on=dependencies or []
            )
            
            logger.info(f"Created MimsModelRegistrationProcessingStep with name: {registration_step.name}")
            return registration_step
            
        except Exception as e:
            logger.error(f"Error creating MimsModelRegistrationProcessingStep: {e}")
            raise ValueError(f"Failed to create MimsModelRegistrationProcessingStep: {e}") from e
