from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from .config_mims_packaging_step import PackageStepConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class MIMSPackagingStepBuilder(StepBuilderBase):
    """
    Builder for a MIMS Packaging ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that packages a model for MIMS registration.
    """

    def __init__(
        self,
        config: PackageStepConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the MIMS packaging step.

        Args:
            config: A PackageStepConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, PackageStepConfig):
            raise ValueError(
                "MIMSPackagingStepBuilder requires a PackageStepConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: PackageStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating PackageStepConfig...")
        
        # Validate processing attributes from the base class
        required_attrs = [
            'processing_instance_count',
            'processing_volume_size',
            'processing_entry_point',
            'processing_source_dir',
            'processing_framework_version',
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PackageStepConfig missing required attribute: {attr}")
        
        # Validate instance type settings
        if not hasattr(self.config, 'processing_instance_type_large'):
            raise ValueError("Missing required attribute: processing_instance_type_large")
        if not hasattr(self.config, 'processing_instance_type_small'):
            raise ValueError("Missing required attribute: processing_instance_type_small")
        if not hasattr(self.config, 'use_large_processing_instance'):
            raise ValueError("Missing required attribute: use_large_processing_instance")
        
        # Validate input and output names
        if "model_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'model_input'")
        
        if "inference_scripts_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'inference_scripts_input'")
        
        if "packaged_model_output" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'packaged_model_output'")
        
        logger.info("PackageStepConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type and framework version.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Get the appropriate instance type based on use_large_processing_instance
        instance_type = self.config.processing_instance_type_large if self.config.use_large_processing_instance else self.config.processing_instance_type_small
        
        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('MIMSPackaging')}"
            ),
            sagemaker_session=self.session,
            env={},  # No environment variables needed for this script
        )

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'model_input')
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Check if inputs is empty
        if not inputs:
            raise ValueError("Inputs dictionary is empty. Must supply required inputs.")
        
        # Validate required inputs against KEYS in input_names (standard pattern)
        required_input_keys = ["model_input", "inference_scripts_input"]
        missing_inputs = [key for key in required_input_keys if key not in inputs]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {', '.join(missing_inputs)}")

        # Define the input channels using script input names from VALUES in input_names
        processing_inputs = [
            ProcessingInput(
                input_name=self.config.input_names["model_input"],  # Script expects this name
                source=inputs["model_input"],  # Use LOGICAL name for lookup
                destination="/opt/ml/processing/input/model"
            ),
            ProcessingInput(
                input_name=self.config.input_names["inference_scripts_input"],  # Script name
                source=inputs["inference_scripts_input"],  # Use LOGICAL name for lookup
                destination="/opt/ml/processing/input/script"
            )
        ]
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        This defines the S3 location where the results of the processing job will be stored.

        Args:
            outputs: A dictionary mapping the logical output channel name ('packaged_model_output')
                     to its S3 destination URI.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        # Get VALUE from output_names (standard pattern for outputs)
        key_out = self.config.output_names["packaged_model_output"]
        
        # Check if outputs contains the required VALUE (standard pattern)
        if not outputs or key_out not in outputs:
            raise ValueError(f"Must supply an S3 URI for '{key_out}' in 'outputs'")
        
        # Define the output for the model package
        processing_outputs = [
            ProcessingOutput(
                output_name=key_out,
                source="/opt/ml/processing/output",
                destination=outputs[key_out]
            )
        ]
        
        return processing_outputs

        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
            "outputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.output_names or {}).keys()])} S3 paths",
            "enable_caching": self.COMMON_PROPERTIES["enable_caching"]
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
        Match custom properties specific to MIMSPackaging step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        step_name = getattr(prev_step, 'name', str(prev_step))
        
        # Try to find ModelArtifacts.S3ModelArtifacts (standard SageMaker property)
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ModelArtifacts"):
            try:
                # Extract model artifacts using the standard property path
                model_artifacts = prev_step.properties.ModelArtifacts.S3ModelArtifacts
                
                # Use logical name (KEY from input_names) as key in inputs
                model_input_key = "model_input"  # This is the logical name (KEY)
                inputs[model_input_key] = model_artifacts  # Set at top level using logical name
                matched_inputs.add(model_input_key)
                logger.info(f"Found model artifacts from step {step_name}")
            except AttributeError as e:
                logger.warning(f"Error getting ModelArtifacts.S3ModelArtifacts: {e}")
        
        # Fall back to model_data property if it exists
        elif hasattr(prev_step, "model_data"):
            model_input_key = "model_input"  # This is the logical name (KEY)
            inputs[model_input_key] = prev_step.model_data
            matched_inputs.add(model_input_key)
            logger.info(f"Found model_data from step {step_name}")
        
        # Add inference_scripts_input using the logical name (KEY)
        inference_scripts_key = "inference_scripts_input"  # This is the logical name (KEY)
        if inference_scripts_key not in inputs:
            inference_scripts_path = self.config.source_dir
            if not inference_scripts_path:
                # Fall back to notebook_root/inference
                inference_scripts_path = str(self.notebook_root / "inference") if self.notebook_root else "inference"
            
            inputs[inference_scripts_key] = inference_scripts_path  # Set using logical name
            matched_inputs.add(inference_scripts_key)
            logger.info(f"Using inference scripts path: {inference_scripts_path}")
        
        return matched_inputs
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline.
        This method orchestrates the assembly of the processor, inputs, outputs, and
        script arguments into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                  Can be nested (e.g., {'inputs': {'model_input': uri}}) or flat (e.g., {'model_input': uri}).
                - model_input: Direct parameter for model input URI (alternative to nested inputs).
                - inference_scripts_input: Direct parameter for inference scripts path (alternative to nested inputs).
                - outputs: A dictionary mapping output channel names to their S3 destinations.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating MIMSPackaging ProcessingStep...")

        # Extract parameters
        inputs_raw = self._extract_param(kwargs, 'inputs')
        outputs = self._extract_param(kwargs, 'outputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Normalize inputs - handles both nested and flat structures
        # This ensures inputs work regardless of whether they're passed as:
        # 1. inputs={'model_input': uri, 'inference_scripts_input': path}
        # 2. inputs={'inputs': {'model_input': uri, 'inference_scripts_input': path}}
        # 3. model_input=uri, inference_scripts_input=path (direct kwargs)
        inputs = self._normalize_inputs(inputs_raw)
        
        # Add direct parameters if provided (overriding any in inputs)
        for param in ['model_input', 'inference_scripts_input']:
            if param in kwargs:
                inputs[param] = kwargs[param]
                logger.debug(f"Added direct parameter: {param}")
        
        # Log the normalized inputs for debugging
        logger.debug(f"Normalized inputs: {list(inputs.keys())}")
        
        # Validate required parameters
        if not inputs:
            raise ValueError("inputs must be provided")
        if not outputs:
            raise ValueError("outputs must be provided")

        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        # Remove job_arguments completely as the script doesn't need any arguments

        step_name = self._get_step_name('Package')
        
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            # job_arguments parameter removed
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
