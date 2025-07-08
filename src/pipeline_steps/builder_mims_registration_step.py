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

# Import the registration specification
try:
    from ..pipeline_step_specs.registration_spec import REGISTRATION_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    REGISTRATION_SPEC = None
    SPEC_AVAILABLE = False

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
            
        # Use the registration specification if available
        spec = REGISTRATION_SPEC if SPEC_AVAILABLE else None
        
        super().__init__(
            config=config,
            spec=spec,
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
        
        # Note: Input validation now handled by specification
        
        logger.info("ModelRegistrationConfig validation succeeded.")

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.
        
        This method creates ProcessingInput objects for each dependency defined in the specification.
        
        Args:
            inputs: Input data sources keyed by logical name
            
        Returns:
            List of ProcessingInput objects
            
        Raises:
            ValueError: If no specification is available or required inputs are missing
        """
        if not self.spec:
            # Fallback to legacy method if no specification available
            return self._get_processing_inputs_legacy(inputs)
            
        processing_inputs = []
        
        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name
            
            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue
                
            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            
            # Create ProcessingInput for MIMS step
            container_path = "/opt/ml/processing/input/model" if logical_name == "PackagedModel" else "/opt/ml/processing/mims_payload"
            
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path,
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
            
        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> None:
        """
        Get outputs for the step.
        
        Registration step has no outputs - it registers the model as a side effect.
        
        Args:
            outputs: Output destinations (unused for registration step)
            
        Returns:
            None - registration step produces no outputs
        """
        return None

    def _get_processing_inputs_legacy(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Legacy method for backward compatibility when no specification is available.
        """
        # Simplified legacy logic for backward compatibility
        if not inputs:
            raise ValueError("Inputs dictionary is empty")
        
        processing_inputs = []
        
        # Handle PackagedModel input
        if "PackagedModel" in inputs:
            processing_inputs.append(
                ProcessingInput(
                    input_name="PackagedModel",
                    source=inputs["PackagedModel"],
                    destination="/opt/ml/processing/input/model",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        
        # Handle payload inputs (multiple possible keys for backward compatibility)
        payload_keys = ["GeneratedPayloadSamples", "payload_s3_key", "payload_s3_uri"]
        for key in payload_keys:
            if key in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name="PayloadSamples",
                        source=inputs[key],
                        destination="/opt/ml/processing/mims_payload",
                        s3_data_distribution_type="FullyReplicated",
                        s3_input_mode="File"
                    )
                )
                break
        
        return processing_inputs
        
    
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
        
        # Extract parameters using standardized keys
        packaged_model = self._extract_param(kwargs, 'PackagedModel')
        generated_payload_samples = self._extract_param(kwargs, 'GeneratedPayloadSamples')  # Updated key
        payload_sample = self._extract_param(kwargs, 'payload_sample')  # Keep for backward compatibility
        
        # If individual parameters were provided, build the inputs dictionary
        if not inputs and (packaged_model or packaged_model_output or generated_payload_samples or payload_sample or payload_s3_key or payload_s3_uri):
            inputs = {}
            # Prefer the new key name if provided
            if packaged_model:
                inputs["PackagedModel"] = packaged_model
            # Fall back to old key name for backward compatibility
            elif packaged_model_output:
                inputs["PackagedModel"] = packaged_model_output  # Use new key for internal consistency
                
            # Prefer the new standardized key name for payload
            if generated_payload_samples:
                inputs["GeneratedPayloadSamples"] = generated_payload_samples
            # Fall back to old key name for backward compatibility
            elif payload_sample:
                inputs["payload_sample"] = payload_sample
                
            # Keep support for older payload formats
            if payload_s3_key:
                inputs["payload_s3_key"] = payload_s3_key
            if payload_s3_uri:
                inputs["payload_s3_uri"] = payload_s3_uri
        
        # Validate required parameters
        if not inputs:
            raise ValueError("Either 'inputs' dictionary or individual 'packaged_model_output' and 'payload_s3_key'/'payload_s3_uri' must be provided")

        # Handle inputs - use specification-driven approach
        final_inputs = {}
        
        # If dependencies are provided, extract inputs from them using UnifiedDependencyResolver
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                final_inputs.update(extracted_inputs)
                logger.info(f"Extracted inputs from dependencies: {list(extracted_inputs.keys())}")
            except Exception as e:
                logger.warning(f"Failed to extract inputs from dependencies: {e}")
                
        # Add explicitly provided inputs (overriding any extracted ones)
        final_inputs.update(inputs)
        
        # Get processing inputs using specification-driven method
        processing_inputs = self._get_inputs(final_inputs)

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
            
            # Attach specification to the step for future reference
            if hasattr(self, 'spec') and self.spec:
                setattr(registration_step, '_spec', self.spec)
            
            logger.info(f"Created MimsModelRegistrationProcessingStep with name: {registration_step.name}")
            return registration_step
            
        except Exception as e:
            logger.error(f"Error creating MimsModelRegistrationProcessingStep: {e}")
            raise ValueError(f"Failed to create MimsModelRegistrationProcessingStep: {e}") from e
