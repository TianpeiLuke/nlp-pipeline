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
        Creates a MimsModelRegistrationProcessingStep using specification-driven approach.
        
        This simplified method leverages the specification and dependency resolver to automatically
        handle input resolution, eliminating complex parameter handling logic.
        
        Args:
            **kwargs: Keyword arguments including:
                - dependencies: List of upstream steps (preferred approach)
                - inputs: Dictionary of input mappings (fallback)
                - performance_metadata_location: Optional S3 location of performance metadata
                
        Returns:
            A configured MimsModelRegistrationProcessingStep instance
        """
        logger.info("Creating MimsModelRegistrationProcessingStep...")
        
        # Extract core parameters
        dependencies = kwargs.get('dependencies', [])
        performance_metadata_location = kwargs.get('performance_metadata_location')
        
        # Use specification-driven input resolution
        inputs = {}
        if dependencies:
            inputs = self.extract_inputs_from_dependencies(dependencies)
            logger.info(f"Extracted inputs from dependencies: {list(inputs.keys())}")
        
        # Allow manual input override/supplement
        inputs.update(kwargs.get('inputs', {}))
        
        # Handle legacy parameter formats for backward compatibility
        legacy_inputs = self._handle_legacy_parameters(kwargs)
        inputs.update(legacy_inputs)
        
        # Validate we have required inputs
        if not inputs:
            raise ValueError("No inputs provided. Either specify 'dependencies' or 'inputs'.")
        
        # Get processing inputs using specification-driven method
        processing_inputs = self._get_inputs(inputs)
        
        # Create step with clean, simple logic
        step_name = f"{self._get_step_name('Registration')}-{self.config.region}"
        
        try:
            registration_step = MimsModelRegistrationProcessingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session,
                processing_input=processing_inputs,
                performance_metadata_location=performance_metadata_location,
                depends_on=dependencies
            )
            
            # Attach specification for future reference
            if self.spec:
                setattr(registration_step, '_spec', self.spec)
            
            logger.info(f"Created MimsModelRegistrationProcessingStep: {registration_step.name}")
            return registration_step
            
        except Exception as e:
            logger.error(f"Error creating MimsModelRegistrationProcessingStep: {e}")
            raise ValueError(f"Failed to create MimsModelRegistrationProcessingStep: {e}") from e

    def _handle_legacy_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle legacy parameter formats for backward compatibility.
        
        Args:
            kwargs: Original keyword arguments
            
        Returns:
            Dictionary of normalized inputs
        """
        legacy_inputs = {}
        
        # Handle various legacy parameter names
        legacy_mappings = {
            'packaged_model_output': 'PackagedModel',
            'PackagedModel': 'PackagedModel',
            'packaged_model': 'PackagedModel',
            'GeneratedPayloadSamples': 'GeneratedPayloadSamples',
            'generated_payload_samples': 'GeneratedPayloadSamples',
            'payload_sample': 'GeneratedPayloadSamples',
            'payload_s3_key': 'GeneratedPayloadSamples',
            'payload_s3_uri': 'GeneratedPayloadSamples'
        }
        
        for legacy_key, standard_key in legacy_mappings.items():
            if legacy_key in kwargs and kwargs[legacy_key]:
                legacy_inputs[standard_key] = kwargs[legacy_key]
                logger.info(f"Mapped legacy parameter '{legacy_key}' to '{standard_key}'")
        
        return legacy_inputs
