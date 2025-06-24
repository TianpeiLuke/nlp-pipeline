from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.sklearn import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, Step

from .config_currency_conversion_step import CurrencyConversionConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class CurrencyConversionStepBuilder(StepBuilderBase):
    """
    Builder for a Currency Conversion ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that executes the currency conversion script.
    """

    def __init__(
        self,
        config: CurrencyConversionConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the currency conversion step.

        Args:
            config: A CurrencyConversionConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, CurrencyConversionConfig):
            raise ValueError(
                "CurrencyConversionStepBuilder requires a CurrencyConversionConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: CurrencyConversionConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating CurrencyConversionConfig…")
        
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
                raise ValueError(f"CurrencyConversionConfig missing required attribute: {attr}")
        
        # Validate job type
        if self.config.job_type not in ["training", "validation", "testing"]:
            raise ValueError(
                f"job_type must be one of 'training', 'validation', 'testing', got '{self.config.job_type}'"
            )
        
        # Validate input and output names
        if "data_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'data_input'")
        
        if "converted_data" not in (self.config.output_names or {}):
            raise ValueError(
                "output_names must contain the key 'converted_data'"
            )
        
        logger.info("CurrencyConversionConfig validation succeeded.")

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
                f"{self._get_step_name('CurrencyConversion')}-{self.config.job_type}"
            ),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.
        These variables are used to control the behavior of the currency conversion script
        without needing to pass them as command-line arguments.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {
            "JOB_TYPE": self.config.job_type,
            "CURRENCY_FIELD": self.config.currency_field,
            "AMOUNT_FIELD": self.config.amount_field,
            "TARGET_CURRENCY": self.config.target_currency
        }
        
        # Add optional environment variables if they exist
        if hasattr(self.config, "date_field") and self.config.date_field:
            env_vars["DATE_FIELD"] = self.config.date_field
            
        if hasattr(self.config, "exchange_rate_source") and self.config.exchange_rate_source:
            env_vars["EXCHANGE_RATE_SOURCE"] = self.config.exchange_rate_source
            
        logger.info(f"Processing environment variables: {env_vars}")
        return env_vars

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'data_input')
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Get the data input key from config
        key_in = self.config.input_names["data_input"]
        
        # Check if inputs is empty or doesn't contain the required key
        if not inputs:
            raise ValueError(f"Inputs dictionary is empty. Must supply an S3 URI for '{key_in}'")
        
        if key_in not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{key_in}' in 'inputs'")

        # Define the primary data input channel.
        processing_inputs = [
            ProcessingInput(
                input_name=key_in,
                source=inputs[key_in],
                destination="/opt/ml/processing/input/data"
            )
        ]
        
        # Add optional exchange rates input if available
        if "exchange_rates_input" in self.config.input_names and "exchange_rates_input" in inputs:
            processing_inputs.append(
                ProcessingInput(
                    input_name=self.config.input_names["exchange_rates_input"],
                    source=inputs[self.config.input_names["exchange_rates_input"]],
                    destination="/opt/ml/processing/input/exchange_rates"
                )
            )
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        This defines the S3 location where the results of the processing job will be stored.

        Args:
            outputs: A dictionary mapping the logical output channel name ('converted_data')
                     to its S3 destination URI.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        key_out = self.config.output_names["converted_data"]
        if not outputs or key_out not in outputs:
            raise ValueError(f"Must supply an S3 URI for '{key_out}' in 'outputs'")
        
        # Define the output for converted data
        processing_outputs = [
            ProcessingOutput(
                output_name=key_out,
                source="/opt/ml/processing/output",
                destination=outputs[key_out]
            )
        ]
        
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
        Match custom properties specific to CurrencyConversion step.
        
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
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - outputs: A dictionary mapping output channel names to their S3 destinations.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating CurrencyConversion ProcessingStep...")

        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs')
        outputs = self._extract_param(kwargs, 'outputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Validate required parameters
        if not inputs:
            raise ValueError("inputs must be provided")
        if not outputs:
            raise ValueError("outputs must be provided")

        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        step_name = f"{self._get_step_name('CurrencyConversion')}-{self.config.job_type.capitalize()}"
        
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
