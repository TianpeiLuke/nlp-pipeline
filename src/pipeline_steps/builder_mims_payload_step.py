from typing import Dict, Optional, Any, Set, List
from pathlib import Path
import logging
import os

from sagemaker.workflow.steps import Step, ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

from .config_mims_payload_step import PayloadConfig
from .builder_step_base import StepBuilderBase

# Register property paths for MIMS Payload step outputs
# Following standard pattern: use VALUE from output_names as property name key
StepBuilderBase.register_property_path(
    "PayloadStep", 
    "GeneratedPayloadSamples",                                            # OUTPUT DESCRIPTOR (Value from output_names)
    "properties.ProcessingOutputConfig.Outputs['GeneratedPayloadSamples'].S3Output.S3Uri"  # Runtime path
)

# Keep backward compatibility path using logical name
StepBuilderBase.register_property_path(
    "PayloadStep", 
    "GeneratedPayloadSamples",                                            # OUTPUT DESCRIPTOR (Value from output_names)
    "properties.ProcessingOutputConfig.Outputs['payload_sample'].S3Output.S3Uri"  # Runtime path with logical name
)

StepBuilderBase.register_property_path(
    "PayloadStep",
    "PayloadMetadata",                                                    # OUTPUT DESCRIPTOR (Value from output_names)
    "properties.ProcessingOutputConfig.Outputs['PayloadMetadata'].S3Output.S3Uri"
)

# Register variants for better matching
for step_type in ["Payload", "MIMSPayloadStepBuilder", "ProcessingStep"]:
    StepBuilderBase.register_property_path(
        step_type, 
        "GeneratedPayloadSamples",                                       
        "properties.ProcessingOutputConfig.Outputs['GeneratedPayloadSamples'].S3Output.S3Uri"  
    )
    StepBuilderBase.register_property_path(
        step_type, 
        "GeneratedPayloadSamples",                                     
        "properties.ProcessingOutputConfig.Outputs['payload_sample'].S3Output.S3Uri"  
    )

logger = logging.getLogger(__name__)


class MIMSPayloadStepBuilder(StepBuilderBase):
    """
    Builder for a MIMS Payload ProcessingStep.
    This class is responsible for configuring and creating a SageMaker Processing Step
    that generates payload samples for MIMS model registration using a processing script.
    """

    def __init__(
        self,
        config: PayloadConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the MIMS payload step.

        Args:
            config: A PayloadConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the Processing job.
            notebook_root: The root directory of the notebook environment.
        """
        if not isinstance(config, PayloadConfig):
            raise ValueError(
                "MIMSPayloadStepBuilder requires a PayloadConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: PayloadConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating PayloadConfig...")
        
        # Validate required attributes
        required_attrs = [
            'expected_tps', 
            'max_latency_in_millisecond',
            'max_acceptable_error_rate',
            'model_registration_domain',
            'model_registration_objective',
            'source_model_inference_content_types',
            'source_model_inference_response_types',
            'source_model_inference_input_variable_list',
            'source_model_inference_output_variable_list'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PayloadConfig missing required attribute: {attr}")
        
        # Validate bucket and ensure S3 path for payload exists
        if not hasattr(self.config, 'bucket') or not self.config.bucket:
            raise ValueError("PayloadConfig must have a 'bucket' attribute")
        
        # Make sure sample_payload_s3_key is set or can be constructed
        if not hasattr(self.config, 'sample_payload_s3_key') or not self.config.sample_payload_s3_key:
            try:
                self.config.ensure_payload_path()
            except Exception as e:
                raise ValueError(f"Could not construct payload path: {e}")
        
        logger.info("PayloadConfig validation succeeded.")

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # This step requires model artifacts as input
        input_reqs = {
            "model_input": "S3 URI of the model artifacts from training",
            "dependencies": self.COMMON_PROPERTIES["dependencies"],
            "enable_caching": self.COMMON_PROPERTIES["enable_caching"]
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        return {
            "payload_sample": "Directory containing the generated payload samples",
            "payload_metadata": "Directory containing the payload metadata"
        }
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to MIMSPayload step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Check if this is a TrainingStep with model artifacts
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ModelArtifacts"):
            if hasattr(prev_step.properties.ModelArtifacts, "S3ModelArtifacts"):
                model_uri = prev_step.properties.ModelArtifacts.S3ModelArtifacts
                
                # Always use the consistent logical name "model_input" for the inputs dictionary
                model_key = "model_input"  # Use fixed logical name
                
                # Initialize inputs dict if needed
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                    
                # Add model artifact to inputs
                inputs["inputs"][model_key] = model_uri
                matched_inputs.add("inputs")
                logger.info("Matched model artifacts from training step (reference)")
        
        return matched_inputs

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        
        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Get the instance type - default to ml.m5.large if not specified
        instance_type = getattr(self.config, 'processor_instance_type', "ml.m5.large")
        
        return SKLearnProcessor(
            framework_version="0.23-1",  # Use a common stable version
            role=self.role,
            instance_type=instance_type,
            instance_count=1,
            volume_size_in_gb=getattr(self.config, 'processing_volume_size', 30),
            base_job_name=self._sanitize_name_for_sagemaker(self._get_step_name('Payload')),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects using the standardized helper methods.
        
        Args:
            inputs: A dictionary with model_input key (already normalized by create_step)
        
        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Important: inputs should already be normalized by create_step
        # Do not normalize again to prevent losing direct parameters
        
        # Always use the logical name "model_input" for consistency
        model_key = "model_input"
        
        # Validate required inputs - check for the logical name as the key
        if model_key not in inputs:
            raise ValueError(f"{model_key} is required for MIMSPayload step")
            
        # Get the script parameter name that will be used
        script_input_name = self.config.input_names.get(model_key, "ModelArtifacts")
        logger.info(f"Using logical key '{model_key}' mapped to script parameter name '{script_input_name}'")
        logger.info(f"Available input keys: {list(inputs.keys())}")
        logger.info(f"Creating processing input with logical key '{model_key}' -> script parameter '{script_input_name}'")
        
        # Confirm the mapping will work correctly with some extra logging
        logger.info(f"Input value to be used: {type(inputs[model_key]).__name__} (not logging actual value)")
        
        # Use standard helper method to create ProcessingInput with the logical name
        # The StepBuilderBase._create_standard_processing_input will use input_names to map logical -> script names
        return [
            self._create_standard_processing_input(
                model_key,  # Use the logical name here
                inputs,     # Use the inputs directly, no second normalization
                "/opt/ml/processing/input/model"
            )
        ]

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        Unlike other step builders, this one generates its own output paths
        when no output paths are provided.
        
        Args:
            outputs: Optional dictionary with output specifications.
                    If empty, default paths will be generated.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        # Determine S3 prefix for outputs
        base_s3_uri = f"{self.config.pipeline_s3_loc}/payload" if hasattr(self.config, 'pipeline_s3_loc') else f"s3://{self.config.bucket}/mods/payload/{self.config.pipeline_name}"
        
        # Create an outputs dictionary if none was provided
        generated_outputs = {}
        
        # Check if we have output_names defined
        if hasattr(self.config, 'output_names') and self.config.output_names:
            # Use output_names VALUES as keys (standard pattern)
            if "payload_sample" in self.config.output_names:
                sample_key = self.config.output_names["payload_sample"]
                generated_outputs[sample_key] = f"{base_s3_uri}/payload_sample"
            else:
                generated_outputs["payload_sample"] = f"{base_s3_uri}/payload_sample"
                
            if "payload_metadata" in self.config.output_names:
                metadata_key = self.config.output_names["payload_metadata"]
                generated_outputs[metadata_key] = f"{base_s3_uri}/payload_metadata"
            else:
                generated_outputs["payload_metadata"] = f"{base_s3_uri}/payload_metadata"
        else:
            # No output_names defined, use default keys
            generated_outputs["payload_sample"] = f"{base_s3_uri}/payload_sample"
            generated_outputs["payload_metadata"] = f"{base_s3_uri}/payload_metadata"
            
        # If outputs were provided, use those instead of generated ones
        if outputs:
            generated_outputs.update(outputs)
            
        # Create ProcessingOutput objects
        processing_outputs = []
        
        if "payload_sample" in generated_outputs or (hasattr(self.config, 'output_names') and 
                                                   "payload_sample" in self.config.output_names and
                                                   self.config.output_names["payload_sample"] in generated_outputs):
            # Try to use standard approach if possible
            if hasattr(self.config, 'output_names') and "payload_sample" in self.config.output_names:
                processing_outputs.append(
                    self._create_standard_processing_output(
                        "payload_sample",
                        generated_outputs,
                        "/opt/ml/processing/output/payload_sample"
                    )
                )
            else:
                # Fall back to direct construction
                processing_outputs.append(
                    ProcessingOutput(
                        output_name="payload_sample",
                        source="/opt/ml/processing/output/payload_sample",
                        destination=generated_outputs["payload_sample"]
                    )
                )
                
        if "payload_metadata" in generated_outputs or (hasattr(self.config, 'output_names') and 
                                                     "payload_metadata" in self.config.output_names and
                                                     self.config.output_names["payload_metadata"] in generated_outputs):
            # Try to use standard approach if possible
            if hasattr(self.config, 'output_names') and "payload_metadata" in self.config.output_names:
                processing_outputs.append(
                    self._create_standard_processing_output(
                        "payload_metadata",
                        generated_outputs,
                        "/opt/ml/processing/output/payload_metadata"
                    )
                )
            else:
                # Fall back to direct construction
                processing_outputs.append(
                    ProcessingOutput(
                        output_name="payload_metadata",
                        source="/opt/ml/processing/output/payload_metadata",
                        destination=generated_outputs["payload_metadata"]
                    )
                )
                
        return processing_outputs

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs the environment variables for the processing job.
        
        Returns:
            A dictionary of environment variables.
        """
        # Set up required environment variables
        env = {
            "CONTENT_TYPES": ",".join(self.config.source_model_inference_content_types),
            "DEFAULT_NUMERIC_VALUE": str(self.config.default_numeric_value),
            "DEFAULT_TEXT_VALUE": self.config.default_text_value
        }
        
        # Add special field values if available
        if hasattr(self.config, 'special_field_values') and self.config.special_field_values:
            for field_name, value in self.config.special_field_values.items():
                env[f"SPECIAL_FIELD_{field_name.upper()}"] = value
                
        return env

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the command-line arguments for the processing script.
        
        Returns:
            A list of strings representing the command-line arguments.
        """
        # SageMaker requires at least one command-line argument
        # Adding a dummy argument to satisfy SageMaker validation requirement
        return ["--dummy-arg", "dummy-value"]

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates a SageMaker ProcessingStep for the pipeline.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - model_input: S3 URI of the model artifacts from training (can be passed directly)
                - inputs: Dictionary that may contain model_input (alternative to direct parameter)
                  Can be nested (e.g., {'inputs': {'model_input': uri}}) or flat (e.g., {'model_input': uri})
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.
                
        Returns:
            A configured ProcessingStep instance.
        """
        # Extract common parameters
        inputs_raw = self._extract_param(kwargs, 'inputs')
        model_input_direct = self._extract_param(kwargs, 'model_input')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        logger.info("Creating MIMSPayload ProcessingStep...")
        
        # Get the step name
        step_name = self._get_step_name('Payload')
        
        # Normalize inputs - handles both nested and flat structures
        # This ensures inputs work regardless of how they're passed
        inputs = self._normalize_inputs(inputs_raw)
        
        # Always use the consistent logical name "model_input" for the inputs dictionary
        # This is the key that _get_processor_inputs will look for
        model_key = "model_input"  # Use fixed logical name
        
        # The script parameter name will be determined in _get_processor_inputs using config.input_names
        script_param_name = self.config.input_names.get(model_key, "ModelArtifacts")
        logger.info(f"Using logical key '{model_key}' for inputs dictionary")
        logger.info(f"This will map to script parameter '{script_param_name}' in processor inputs")
        
        if model_input_direct is not None:
            inputs[model_key] = model_input_direct
            logger.info("Using directly provided model_input parameter (reference)")
        else:
            logger.info("No direct model_input parameter provided")
        
        # Auto-detect inputs from dependencies if we still need the model input
        if model_key not in inputs and dependencies:
            input_requirements = self.get_input_requirements()
            
            # Extract inputs from dependencies
            for dep_step in dependencies:
                # Create temporary dictionary to collect inputs from matching
                temp_inputs = {}
                matched = self._match_custom_properties(temp_inputs, input_requirements, dep_step)
                
                if matched:
                    # Normalize any nested inputs from the matching
                    normalized_deps = self._normalize_inputs(temp_inputs)
                    # Add to our main inputs
                    inputs.update(normalized_deps)
                    logger.info(f"Found inputs from dependency step: {getattr(dep_step, 'name', None)}")
                    
        # Log the normalized inputs for debugging
        logger.debug(f"Normalized inputs: {list(inputs.keys())}")
                    
        # Verify we have the required model input
        if model_key not in inputs:
            raise ValueError(f"Required model input '{model_key}' not found in inputs or dependencies")
        
        # Create the processing step using helper methods
        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs({})  # No custom outputs needed
        job_args = self._get_job_arguments()

        # Get script path using existing config fields
        script_path = None
        if hasattr(self.config, 'get_script_path'):
            script_path = self.config.get_script_path()
            
        # Fall back to default path if not available
        if script_path is None:
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "pipeline_scripts",
                "mims_payload.py"
            )
            
        logger.info(f"Using script path: {script_path}")
        
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            job_arguments=job_args,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
