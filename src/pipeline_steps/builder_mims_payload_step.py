from typing import Dict, Optional, Any, Set, List
from pathlib import Path
import logging
import os

from sagemaker.workflow.steps import Step, ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

from .config_mims_payload_step import PayloadConfig
from .builder_step_base import StepBuilderBase

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
                
                # Get the model_input key from config
                model_key = self.config.input_names.get("model_input", "model_input")
                
                # Initialize inputs dict if needed
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                    
                # Add model artifact to inputs
                inputs["inputs"][model_key] = model_uri
                matched_inputs.add("inputs")
                logger.info(f"Matched model artifacts from training step: {model_uri}")
        
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
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        
        Args:
            inputs: A dictionary with model_input in the inputs subdictionary
        
        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Get the model input key from config
        model_key = self.config.input_names.get("model_input", "model_input")
        
        # Validate model_input is provided
        if "inputs" not in inputs or model_key not in inputs["inputs"]:
            raise ValueError(f"{model_key} is required for MIMSPayload step")
        
        model_uri = inputs["inputs"][model_key]
        
        # Set up the processor inputs
        return [
            ProcessingInput(
                source=model_uri,
                destination="/opt/ml/processing/input/model/model.tar.gz"
            )
        ]

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        
        Args:
            outputs: Optional dictionary with output specifications (not used)

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        # Determine S3 prefix for outputs
        base_s3_uri = f"{self.config.pipeline_s3_loc}/payload" if hasattr(self.config, 'pipeline_s3_loc') else f"s3://{self.config.bucket}/mods/payload/{self.config.pipeline_name}"
        
        return [
            ProcessingOutput(
                output_name="payload_sample",
                source="/opt/ml/processing/output/payload_sample",
                destination=f"{base_s3_uri}/payload_sample"
            ),
            ProcessingOutput(
                output_name="payload_metadata", 
                source="/opt/ml/processing/output/payload_metadata",
                destination=f"{base_s3_uri}/payload_metadata"
            )
        ]

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
                - model_input: S3 URI of the model artifacts from training
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.
                
        Returns:
            A configured ProcessingStep instance.
        """
        # Extract common parameters
        model_input = self._extract_param(kwargs, 'model_input', None)
        dependencies = self._extract_param(kwargs, 'dependencies', None)
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        logger.info("Creating MIMSPayload ProcessingStep...")
        
        # Get the step name
        step_name = self._get_step_name('Payload')
        
        # Create inputs dictionary
        inputs = {}
        
        # If model_input was provided directly, add it to inputs
        if model_input is not None:
            model_key = self.config.input_names.get("model_input", "model_input")
            inputs["inputs"] = {model_key: model_input}
        
        # Auto-detect inputs from dependencies if no direct inputs and we have dependencies
        if not inputs and dependencies:
            input_requirements = self.get_input_requirements()
            
            # Extract both regular inputs and model_input from dependencies
            for dep_step in dependencies:
                matched = self._match_custom_properties(inputs, input_requirements, dep_step)
                if matched:
                    logger.info(f"Found inputs from dependency: {getattr(dep_step, 'name', str(dep_step))}")
                    
        # Verify we have the required model input
        model_key = self.config.input_names.get("model_input", "model_input")
        if "inputs" not in inputs or model_key not in inputs.get("inputs", {}):
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
