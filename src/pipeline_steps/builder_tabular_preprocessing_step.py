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
            'processing_instance_count',
            'processing_volume_size',
            'processing_entry_point',
            'processing_source_dir',
            'processing_framework_version',
            'job_type'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"TabularPreprocessingConfig missing required attribute: {attr}")
        
        # Validate instance type settings
        if not hasattr(self.config, 'processing_instance_type_large'):
            raise ValueError("Missing required attribute: processing_instance_type_large")
        if not hasattr(self.config, 'processing_instance_type_small'):
            raise ValueError("Missing required attribute: processing_instance_type_small")
        if not hasattr(self.config, 'use_large_processing_instance'):
            raise ValueError("Missing required attribute: use_large_processing_instance")
        
        # Validate job type
        if self.config.job_type not in ["training", "validation", "testing", "calibration"]:
            raise ValueError(
                f"job_type must be one of 'training', 'validation', 'testing', 'calibration', got '{self.config.job_type}'"
            )
        
        # Validate input and output names
        if "data_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'data_input'")
        
        if "processed_data" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'processed_data'")
        
        # full_data is now optional
        # if "full_data" not in (self.config.output_names or {}):
        #    raise ValueError("output_names must contain key 'full_data'")
        
        logger.info("TabularPreprocessingConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

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
        # Set the required environment variables for tabular_preprocess.py script:
        # - LABEL_FIELD: required by the script
        # - TRAIN_RATIO: used for splitting in training mode
        # - TEST_VAL_RATIO: used for splitting test vs validation
        env_vars = {
            "LABEL_FIELD": self.config.hyperparameters.label_name,
            "TRAIN_RATIO": str(self.config.train_ratio),
            "TEST_VAL_RATIO": str(self.config.test_val_ratio),
        }
        
        # Add optional environment variables if they exist
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
        This defines the S3 locations where the results of the processing job will be stored.

        The tabular_preprocess.py script creates processed data in split-specific subdirectories
        (train, test, val) under /opt/ml/processing/output. We map these outputs to the
        appropriate S3 destinations.

        Args:
            outputs: A dictionary mapping the logical output channel names to S3 destination URIs.
                    At minimum, must include the "processed_data" key.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        processed_data_key = self.config.output_names["processed_data"]
        full_data_key = self.config.output_names["full_data"]
        
        if not outputs:
            raise ValueError(f"Outputs dictionary is empty. Must supply an S3 URI for '{processed_data_key}'")
        
        if processed_data_key not in outputs:
            raise ValueError(f"Must supply an S3 URI for '{processed_data_key}' in 'outputs'")
        
        # Define the outputs for processed data
        # The script creates files in split-specific subfolders (train, test, val) 
        # under /opt/ml/processing/output
        processing_outputs = [
            ProcessingOutput(
                output_name=processed_data_key,
                source="/opt/ml/processing/output", 
                destination=outputs[processed_data_key]
            )
        ]
        
        # Only add the full_data output if it's provided
        if full_data_key in outputs:
            processing_outputs.append(
                ProcessingOutput(
                    output_name=full_data_key,
                    source="/opt/ml/processing/output",
                    destination=outputs[full_data_key]
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
        
        This method handles two types of dependency steps:
        1. CradleDataLoadingStep: Uses get_output_locations() to get output locations
        2. ProcessingStep: Looks for outputs with specific names
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        step_name = getattr(prev_step, 'name', str(prev_step))
        
        # First, try to handle CradleDataLoadingStep
        if self._match_cradle_data_loading_step(inputs, prev_step, matched_inputs):
            logger.info(f"Matched inputs from CradleDataLoadingStep: {step_name}")
            return matched_inputs
            
        # If not a CradleDataLoadingStep or matching failed, fall back to original logic
        self._match_processing_step_outputs(inputs, prev_step, matched_inputs)
        
        if matched_inputs:
            logger.info(f"Matched inputs from step: {step_name}")
            
        return matched_inputs
        
    def _match_cradle_data_loading_step(self, inputs: Dict[str, Any], prev_step: Step, 
                                       matched_inputs: Set[str]) -> bool:
        """
        Match outputs from a CradleDataLoadingStep.
        
        This method checks if the step is a CradleDataLoadingStep and if so,
        extracts output locations using get_output_locations().
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            matched_inputs: Set to add matched input names to
            
        Returns:
            True if the step is a CradleDataLoadingStep and matching was attempted,
            False otherwise
        """
        # Check if the step has get_output_locations method (CradleDataLoadingStep)
        if not hasattr(prev_step, "get_output_locations"):
            return False
            
        try:
            # Get output locations from CradleDataLoadingStep
            output_locations = prev_step.get_output_locations()
            if not output_locations:
                logger.warning(f"No output locations found in step: {getattr(prev_step, 'name', str(prev_step))}")
                return True  # Still return True as we identified it as a CradleDataLoadingStep
                
            # Import constants if available
            try:
                from secure_ai_sandbox_workflow_python_sdk.utils.constants import (
                    OUTPUT_TYPE_DATA,
                    OUTPUT_TYPE_METADATA,
                    OUTPUT_TYPE_SIGNATURE,
                )
            except ImportError:
                # Fallback to string constants if import fails
                OUTPUT_TYPE_DATA = "data"
                OUTPUT_TYPE_METADATA = "metadata"
                OUTPUT_TYPE_SIGNATURE = "signature"
                
            # Map output types to input keys
            output_type_to_input_key = {
                OUTPUT_TYPE_DATA: "data_input",
                OUTPUT_TYPE_METADATA: "metadata_input",
                OUTPUT_TYPE_SIGNATURE: "signature_input"
            }
            
            # Match each output type to corresponding input key
            for output_type, input_key_name in output_type_to_input_key.items():
                if output_type in output_locations and input_key_name in self.config.input_names:
                    input_key = self.config.input_names[input_key_name]
                    
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                        
                    # Add output location to inputs
                    if input_key not in inputs.get("inputs", {}):
                        inputs["inputs"][input_key] = output_locations[output_type]
                        matched_inputs.add("inputs")
                        logger.info(f"Found {output_type} from CradleDataLoadingStep")
                        
            return True
                
        except Exception as e:
            logger.warning(f"Error extracting output locations from CradleDataLoadingStep: {e}")
            return True  # Still return True as we identified it as a CradleDataLoadingStep
            
    def _match_processing_step_outputs(self, inputs: Dict[str, Any], prev_step: Step, 
                                      matched_inputs: Set[str]) -> None:
        """
        Match outputs from a ProcessingStep.
        
        This method looks for outputs with specific names in the dependency step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            matched_inputs: Set to add matched input names to
        """
        # Check if the step has outputs
        if not (hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0):
            return
            
        # Match data output
        self._match_output_by_name(
            inputs, prev_step, matched_inputs, 
            "data_input", "data"
        )
        
        # Match metadata output
        self._match_output_by_name(
            inputs, prev_step, matched_inputs, 
            "metadata_input", "metadata"
        )
        
        # Match signature output
        self._match_output_by_name(
            inputs, prev_step, matched_inputs, 
            "signature_input", "signature"
        )
        
    def _match_output_by_name(self, inputs: Dict[str, Any], prev_step: Step, 
                             matched_inputs: Set[str], input_key_name: str, 
                             output_name_pattern: str) -> None:
        """
        Match a specific output by name.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            matched_inputs: Set to add matched input names to
            input_key_name: Name of the input key in config.input_names
            output_name_pattern: Pattern to look for in output names
        """
        try:
            # Check if we have this input key in our config
            if input_key_name not in self.config.input_names:
                return
                
            input_key = self.config.input_names[input_key_name]
            
            # Look for an output with a name that matches the pattern
            for output in prev_step.outputs:
                if (hasattr(output, "output_name") and 
                    output_name_pattern in output.output_name.lower()):
                    
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    # Add output destination to inputs
                    if input_key not in inputs.get("inputs", {}):
                        inputs["inputs"][input_key] = output.destination
                        matched_inputs.add("inputs")
                        logger.info(f"Found {output_name_pattern} from ProcessingStep")
                        break
                        
        except Exception as e:
            logger.warning(f"Error matching {output_name_pattern} output: {e}")
    
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
