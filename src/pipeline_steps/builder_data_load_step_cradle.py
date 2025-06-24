from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging
import os
import json
import importlib

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, Step

from .config_data_load_step_cradle import CradleDataLoadConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

# Import constants from the same module used by the data loading step
try:
    from secure_ai_sandbox_workflow_python_sdk.utils.constants import (
        OUTPUT_TYPE_DATA,
        OUTPUT_TYPE_METADATA,
        OUTPUT_TYPE_SIGNATURE,
    )
except ImportError:
    # Fallback to dynamic import if the direct import fails
    SECUREAI_PIPELINE_CONSTANTS_MODULE = os.environ.get("SECUREAI_PIPELINE_CONSTANTS_MODULE")
    OUTPUT_TYPE_DATA = OUTPUT_TYPE_METADATA = OUTPUT_TYPE_SIGNATURE = None
    if SECUREAI_PIPELINE_CONSTANTS_MODULE:
        try:
            const_mod = importlib.import_module(SECUREAI_PIPELINE_CONSTANTS_MODULE)
            OUTPUT_TYPE_DATA      = getattr(const_mod, "OUTPUT_TYPE_DATA",      None)
            OUTPUT_TYPE_METADATA  = getattr(const_mod, "OUTPUT_TYPE_METADATA",  None)
            OUTPUT_TYPE_SIGNATURE = getattr(const_mod, "OUTPUT_TYPE_SIGNATURE", None)
            logger.info(f"Imported pipeline constants from {SECUREAI_PIPELINE_CONSTANTS_MODULE}")
        except ImportError as e:
            logger.error(f"Could not import pipeline constants: {e}")
    else:
        logger.warning(
            "SECUREAI_PIPELINE_CONSTANTS_MODULE not set; "
            "pipeline constants (DATA, METADATA, SIGNATURE) unavailable."
        )


class CradleDataLoadingStepBuilder(StepBuilderBase):
    """
    Builder for a Cradle Data Loading ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that executes the Cradle data loading script.
    """

    def __init__(
        self,
        config: CradleDataLoadConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the data loading step.

        Args:
            config: A CradleDataLoadConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, CradleDataLoadConfig):
            raise ValueError(
                "CradleDataLoadingStepBuilder requires a CradleDataLoadConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: CradleDataLoadConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating CradleDataLoadConfig…")
        
        # Validate required attributes
        required_attrs = [
            'processing_instance_type',
            'processing_instance_count',
            'processing_volume_size',
            'processing_entry_point',
            'processing_source_dir',
            'job_type'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"CradleDataLoadConfig missing required attribute: {attr}")
        
        # Validate job type
        if self.config.job_type not in ["training", "validation", "testing", "calibration"]:
            raise ValueError(
                f"job_type must be one of 'training', 'validation', 'testing', 'calibration', got '{self.config.job_type}'"
            )
        
        # Validate output names
        if not self.config.output_names:
            raise ValueError("output_names must be provided and non-empty")
        
        # Ensure required output channels are present
        required_outputs = ["data_output"]
        missing_outputs = [out for out in required_outputs if out not in self.config.output_names]
        if missing_outputs:
            raise ValueError(f"output_names missing required keys: {missing_outputs}")
        
        logger.info("CradleDataLoadConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=self.config.processing_instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('CradleDataLoading')}-{self.config.job_type}"
            ),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.
        These variables are used to control the behavior of the data loading script
        without needing to pass them as command-line arguments.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {
            "JOB_TYPE": self.config.job_type,
        }
        
        # Add optional environment variables if they exist
        if hasattr(self.config, "cradle_request") and self.config.cradle_request:
            env_vars["CRADLE_REQUEST"] = json.dumps(self.config.cradle_request)
            
        logger.info(f"Processing environment variables: {env_vars}")
        return env_vars

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        This defines the S3 location where the results of the processing job will be stored.

        Args:
            outputs: A dictionary mapping the logical output channel names to their S3 destination URIs.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        # Validate outputs
        if not outputs:
            raise ValueError("outputs must be provided and non-empty")
        
        # Ensure required output channels are present
        required_outputs = ["data_output"]
        missing_outputs = [out for out in required_outputs if out not in outputs]
        if missing_outputs:
            raise ValueError(f"outputs missing required keys: {missing_outputs}")
        
        # Define the outputs
        processing_outputs = []
        
        # Map output names to their destinations in the container
        output_destinations = {
            "data_output": "/opt/ml/processing/output/data",
            "metadata_output": "/opt/ml/processing/output/metadata",
            "signature_output": "/opt/ml/processing/output/signature"
        }
        
        # Create ProcessingOutput objects for each output channel
        for output_name, output_key in self.config.output_names.items():
            if output_key in outputs and output_name in output_destinations:
                processing_outputs.append(
                    ProcessingOutput(
                        output_name=output_key,
                        source=output_destinations[output_name],
                        destination=outputs[output_key]
                    )
                )
        
        return processing_outputs

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.
        This allows for parameterizing the script's execution at runtime.

        Returns:
            A list of strings representing the command-line arguments.
        """
        return ["--job_type", self.config.job_type]
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # This step doesn't require any inputs from previous steps
        input_reqs = {
            "outputs": f"Dictionary containing {', '.join([f'{v}' for v in (self.config.output_names or {}).values()])} S3 paths",
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
        output_props = {}
        
        # Map output names to their descriptions
        output_descriptions = {
            "data_output": "S3 URI of the loaded data",
            "metadata_output": "S3 URI of the metadata",
            "signature_output": "S3 URI of the signature"
        }
        
        # Create output properties dictionary
        for output_name, output_key in (self.config.output_names or {}).items():
            if output_name in output_descriptions:
                output_props[output_key] = output_descriptions[output_name]
            else:
                output_props[output_key] = f"S3 URI of the {output_name}"
        
        # Add constants if available
        if OUTPUT_TYPE_DATA:
            output_props[OUTPUT_TYPE_DATA] = "S3 URI of the loaded data"
        if OUTPUT_TYPE_METADATA:
            output_props[OUTPUT_TYPE_METADATA] = "S3 URI of the metadata"
        if OUTPUT_TYPE_SIGNATURE:
            output_props[OUTPUT_TYPE_SIGNATURE] = "S3 URI of the signature"
        
        return output_props
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to CradleDataLoading step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # No custom properties to match for this step
        return matched_inputs
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline.
        This method orchestrates the assembly of the processor, inputs, outputs, and
        script arguments into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - outputs: A dictionary mapping output channel names to their S3 destinations.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating CradleDataLoading ProcessingStep...")

        # Extract parameters
        outputs = self._extract_param(kwargs, 'outputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Validate required parameters
        if not outputs:
            raise ValueError("outputs must be provided")

        processor = self._create_processor()
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        step_name = f"{self._get_step_name('CradleDataLoading')}-{self.config.job_type.capitalize()}"
        
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=[],  # No inputs for data loading
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
