from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from .config_tabular_preprocessing_step import TabularPreprocessingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class TabularPreprocessingStepBuilder(StepBuilderBase):
    """
    Builder for a Tabular Preprocessing ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that executes the tabular preprocessing script.
    """

    def __init__(
        self,
        config: TabularPreprocessingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the preprocessing step.

        Args:
            config: A TabularPreprocessingConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, TabularPreprocessingConfig):
            raise ValueError(
                "TabularPreprocessingStepBuilder requires a TabularPreprocessingConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: TabularPreprocessingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating TabularPreprocessingConfig...")
        
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
                raise ValueError(f"TabularPreprocessingConfig missing required attribute: {attr}")
        
        # Validate job type
        if self.config.job_type not in ["training", "validation", "testing", "calibration"]:
            raise ValueError(
                f"job_type must be one of 'training', 'validation', 'testing', 'calibration', got '{self.config.job_type}'"
            )
        
        # Validate input and output names
        if "data_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'data_input'")
        
        if "preprocessed_data" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'preprocessed_data'")
        
        logger.info("TabularPreprocessingConfig validation succeeded.")

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
                f"{self._get_step_name('TabularPreprocessing')}-{self.config.job_type}"
            ),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.
        These variables are used to control the behavior of the preprocessing script
        without needing to pass them as command-line arguments.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {
            "JOB_TYPE": self.config.job_type,
        }
        
        # Add optional environment variables if they exist
        if hasattr(self.config, "target_column") and self.config.target_column:
            env_vars["TARGET_COLUMN"] = self.config.target_column
            
        if hasattr(self.config, "categorical_columns") and self.config.categorical_columns:
            env_vars["CATEGORICAL_COLUMNS"] = ",".join(self.config.categorical_columns)
            
        if hasattr(self.config, "numerical_columns") and self.config.numerical_columns:
            env_vars["NUMERICAL_COLUMNS"] = ",".join(self.config.numerical_columns)
            
        if hasattr(self.config, "text_columns") and self.config.text_columns:
            env_vars["TEXT_COLUMNS"] = ",".join(self.config.text_columns)
            
        if hasattr(self.config, "date_columns") and self.config.date_columns:
            env_vars["DATE_COLUMNS"] = ",".join(self.config.date_columns)
            
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

        # Define the primary data input channel
        processing_inputs = [
            ProcessingInput(
                input_name=key_in,
                source=inputs[key_in],
                destination="/opt/ml/processing/input/data"
            )
        ]
        
        # Add optional metadata input if available
        if "metadata_input" in self.config.input_names and "metadata_input" in inputs:
            processing_inputs.append(
                ProcessingInput(
                    input_name=self.config.input_names["metadata_input"],
                    source=inputs[self.config.input_names["metadata_input"]],
                    destination="/opt/ml/processing/input/metadata"
                )
            )
        
        # Add optional signature input if available
        if "signature_input" in self.config.input_names and "signature_input" in inputs:
            processing_inputs.append(
                ProcessingInput(
                    input_name=self.config.input_names["signature_input"],
                    source=inputs[self.config.input_names["signature_input"]],
                    destination="/opt/ml/processing/input/signature"
                )
            )
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        This defines the S3 location where the results of the processing job will be stored.

        Args:
            outputs: A dictionary mapping the logical output channel name ('preprocessed_data')
                     to its S3 destination URI.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        key_out = self.config.output_names["preprocessed_data"]
        if not outputs or key_out not in outputs:
            raise ValueError(f"Must supply an S3 URI for '{key_out}' in 'outputs'")
        
        # Define the output for preprocessed data
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
        Match custom properties specific to TabularPreprocessing step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Look for data output from a DataLoadingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our data_input
                data_key = self.config.input_names.get("data_input")
                if data_key:
                    # Look for an output with a name that contains 'data'
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and "data" in output.output_name.lower():
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if data_key not in inputs.get("inputs", {}):
                                inputs["inputs"][data_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found data from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract data from step: {e}")
                
        # Look for metadata output from a DataLoadingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our metadata_input
                metadata_key = self.config.input_names.get("metadata_input")
                if metadata_key:
                    # Look for an output with a name that contains 'metadata'
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and "metadata" in output.output_name.lower():
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if metadata_key not in inputs.get("inputs", {}):
                                inputs["inputs"][metadata_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found metadata from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract metadata from step: {e}")
                
        # Look for signature output from a DataLoadingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our signature_input
                signature_key = self.config.input_names.get("signature_input")
                if signature_key:
                    # Look for an output with a name that contains 'signature'
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and "signature" in output.output_name.lower():
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if signature_key not in inputs.get("inputs", {}):
                                inputs["inputs"][signature_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found signature from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract signature from step: {e}")
                
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
        logger.info("Creating TabularPreprocessing ProcessingStep...")

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

        step_name = f"{self._get_step_name('TabularPreprocessing')}-{self.config.job_type.capitalize()}"
        
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
