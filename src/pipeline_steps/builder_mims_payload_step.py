from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from .config_mims_payload_step import PayloadConfig
from .builder_step_base import StepBuilderBase
from ..pipeline_deps.registry_manager import RegistryManager
from ..pipeline_deps.dependency_resolver import UnifiedDependencyResolver

# Import the payload specification
try:
    from ..pipeline_step_specs.payload_spec import PAYLOAD_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    PAYLOAD_SPEC = None
    SPEC_AVAILABLE = False
    
logger = logging.getLogger(__name__)


class MIMSPayloadStepBuilder(StepBuilderBase):
    """
    Builder for a MIMS Payload Generation ProcessingStep.
    
    This implementation uses the specification-driven approach where dependencies, outputs,
    and script contract are defined in the payload specification.
    """

    def __init__(
        self,
        config: PayloadConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional["RegistryManager"] = None,
        dependency_resolver: Optional["UnifiedDependencyResolver"] = None
    ):
        """
        Initializes the builder with a specific configuration for the MIMS payload step.

        Args:
            config: A PayloadConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
            registry_manager: Optional registry manager for dependency injection
            dependency_resolver: Optional dependency resolver for dependency injection
        """
        if not isinstance(config, PayloadConfig):
            raise ValueError(
                "MIMSPayloadStepBuilder requires a PayloadConfig instance."
            )
            
        # Use the payload specification if available
        spec = PAYLOAD_SPEC if SPEC_AVAILABLE else None
        
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: PayloadConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        self.log_info("Validating PayloadConfig...")
        
        # Make sure bucket is set
        if not hasattr(self.config, 'bucket') or not self.config.bucket:
            raise ValueError("PayloadConfig missing required attribute: bucket")
        
        # Make sure sample_payload_s3_key is set or can be constructed
        if not hasattr(self.config, 'sample_payload_s3_key') or not self.config.sample_payload_s3_key:
            try:
                self.config.ensure_payload_path()
            except Exception as e:
                raise ValueError(f"Could not construct payload path: {e}")
        
        # Validate other required attributes
        required_attrs = [
            'pipeline_name',
            'source_model_inference_content_types',
            'processing_instance_count', 
            'processing_volume_size'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PayloadConfig missing required attribute: {attr}")
                
        self.log_info("PayloadConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Use processing_instance_type_large when use_large_processing_instance is True
        # Otherwise use processing_instance_type_small
        instance_type = self.config.processing_instance_type_large if self.config.use_large_processing_instance else self.config.processing_instance_type_small
        
        # Get framework version
        framework_version = getattr(self.config, 'processing_framework_version', "1.0-1")
        
        return SKLearnProcessor(
            framework_version=framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker("PayloadGeneration"),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.

        Returns:
            A dictionary of environment variables.
        """        
        env_vars = {
            "PIPELINE_NAME": self.config.pipeline_name,
            "REGION": self.config.region,
        }
        
        # Add content types
        if hasattr(self.config, 'source_model_inference_content_types'):
            env_vars["CONTENT_TYPES"] = ",".join(self.config.source_model_inference_content_types)
            
        # Add optional configurations
        for key, env_key in [
            ('default_numeric_value', 'DEFAULT_NUMERIC_VALUE'),
            ('default_string_value', 'DEFAULT_STRING_VALUE'),
            ('sample_payload_s3_key', 'PAYLOAD_S3_KEY'),
            ('bucket', 'BUCKET_NAME')
        ]:
            if hasattr(self.config, key) and getattr(self.config, key) is not None:
                env_vars[env_key] = str(getattr(self.config, key))
        
        self.log_info("Payload environment variables: %s", env_vars)
        return env_vars

    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Get inputs for the step using specification and contract.
        
        This method creates ProcessingInput objects for each dependency defined in the specification.
        
        Args:
            inputs: Input data sources keyed by logical name
            
        Returns:
            List of ProcessingInput objects
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
            
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
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
            else:
                raise ValueError(f"No container path found for input: {logical_name}")
                
            # Use the input value directly - property references are handled by PipelineAssembler
            processing_inputs.append(
                ProcessingInput(
                    input_name=logical_name,
                    source=inputs[logical_name],
                    destination=container_path
                )
            )
            
        return processing_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Get outputs for the step using specification and contract.
        
        This method creates ProcessingOutput objects for each output defined in the specification.
        
        Args:
            outputs: Output destinations keyed by logical name
            
        Returns:
            List of ProcessingOutput objects
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
            
        processing_outputs = []
        
        # Process each output in the specification
        for _, output_spec in self.spec.outputs.items():
            logical_name = output_spec.logical_name
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_output_paths:
                container_path = self.contract.expected_output_paths[logical_name]
            else:
                raise ValueError(f"No container path found for output: {logical_name}")
                
            # Try to find destination in outputs
            destination = None
            
            # Look in outputs by logical name
            if logical_name in outputs:
                destination = outputs[logical_name]
            else:
                # Generate destination from config
                destination = f"{self.config.pipeline_s3_loc}/payload/{logical_name}"
                self.log_info("Using generated destination for '%s': %s", logical_name, destination)
            
            processing_outputs.append(
                ProcessingOutput(
                    output_name=logical_name,
                    source=container_path,
                    destination=destination
                )
            )
            
        return processing_outputs

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.

        Returns:
            A list of strings representing the command-line arguments.
        """
        # If there are custom script arguments in the config, use those
        if hasattr(self.config, 'processing_script_arguments') and self.config.processing_script_arguments:
            return self.config.processing_script_arguments
            
        # Return a standard argument to ensure we don't return an empty list
        self.log_info("Using default arguments for payload generation")
        return ["--mode", "standard"]
        
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the specification-driven approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Input data sources keyed by logical name
                - outputs: Output destinations keyed by logical name
                - dependencies: Optional list of steps that this step depends on
                - enable_caching: A boolean indicating whether to cache the results of this step

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        self.log_info("Creating MIMS Payload ProcessingStep...")

        # Extract parameters
        inputs_raw = kwargs.get('inputs', {})
        outputs = kwargs.get('outputs', {})
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        # Handle inputs
        inputs = {}
        
        # If dependencies are provided, extract inputs from them
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                self.log_warning("Failed to extract inputs from dependencies: %s", e)
                
        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)
        
        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_inputs(inputs)
        proc_outputs = self._get_outputs(outputs)
        job_args = self._get_job_arguments()

        # Get step name from spec or construct one
        step_name = getattr(self.spec, 'step_type', None) or "PayloadGeneration"
        
        # Get full script path from config or contract
        script_path = self.config.get_script_path()
        if not script_path and self.contract:
            script_path = self.contract.entry_point
        
        # Create step
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        
        # Attach specification to the step for future reference
        if hasattr(self, 'spec') and self.spec:
            setattr(step, '_spec', self.spec)
            
        self.log_info("Created ProcessingStep with name: %s", step.name)
        return step
