from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from .config_tabular_preprocessing_step import TabularPreprocessingConfig
from .builder_step_base import StepBuilderBase

# Register property path for Tabular Preprocessing outputs with template for output_descriptor
StepBuilderBase.register_property_path(
    "TabularPreprocessingStep",
    "processed_data",                                                           # Logical name
    "properties.ProcessingOutputConfig.Outputs['{output_descriptor}'].S3Output.S3Uri"  # Template path
)

# Register additional property paths for calibration data handling
StepBuilderBase.register_property_path(
    "TabularPreprocessingStep",
    "calibration_data",                                                         # Logical name
    "properties.ProcessingOutputConfig.Outputs['{output_descriptor}'].S3Output.S3Uri"  # Template path
)

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
        if "DATA" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'DATA'")
        
        if "processed_data" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'processed_data'")
        
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
        Constructs a list of ProcessingInput objects from the provided inputs dictionary
        using the standardized helper methods.

        Args:
            inputs: A dictionary mapping logical input channel names (keys from input_names)
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Validate required inputs using the standard helper method
        self._validate_inputs(inputs)
        
        processing_inputs = []
        
        # Process data input (required) using the standard helper method
        processing_inputs.append(
            self._create_standard_processing_input(
                "DATA", 
                inputs,
                "/opt/ml/processing/input/data"
            )
        )
        
        # Process metadata input (optional)
        metadata_logical_name = "METADATA"
        if (metadata_logical_name in self.config.input_names and 
            metadata_logical_name in inputs):
            processing_inputs.append(
                self._create_standard_processing_input(
                    metadata_logical_name,
                    inputs,
                    "/opt/ml/processing/input/metadata"
                )
            )
        
        # Process signature input (optional)
        signature_logical_name = "SIGNATURE"
        if (signature_logical_name in self.config.input_names and 
            signature_logical_name in inputs):
            processing_inputs.append(
                self._create_standard_processing_input(
                    signature_logical_name,
                    inputs,
                    "/opt/ml/processing/input/signature"
                )
            )
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs ProcessingOutput objects with enhanced resilience for Join objects.
        
        The tabular_preprocess.py script creates processed data in split-specific subdirectories
        (train, test, val) under /opt/ml/processing/output. We map these outputs to the
        appropriate S3 destinations.
        
        Args:
            outputs: A dictionary mapping the output VALUES from output_names to S3 destination URIs.
                    This may also be a Join object or other type with different access patterns.
        
        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        processing_outputs = []
        
        # Get the output descriptor name we need
        output_key = self._get_output_destination_name("processed_data")  # Should be "ProcessedTabularData"
        
        # Add detailed logging for debugging
        logger.debug(f"Building output for key '{output_key}', outputs type: {type(outputs)}")
        if isinstance(outputs, dict):
            logger.debug(f"Available output keys: {list(outputs.keys())}")
        
        try:
            # First try using our enhanced helper method
            processing_outputs.append(
                self._create_standard_processing_output(
                    "processed_data",  # Logical name in config.output_names
                    outputs,           # Outputs dictionary or Join object
                    "/opt/ml/processing/output"  # Standard source path
                )
            )
            logger.info(f"Successfully created standard processing output for '{output_key}'")
        except ValueError as e:
            # Enhanced helper failed - implement fallback mechanism
            self.log_warning("Standard output helper failed: %s, using fallback mechanism", e)
            
            # Follow pattern from mods_pipeline_xgboost_train_evaluate_e2e.py
            # Construct predictable S3 path based on config properties
            fallback_path = f"{self.config.pipeline_s3_loc}/tabular_preprocessing/{self.config.job_type}"
            
            # Create the processing output directly with our fallback path
            processing_outputs.append(
                ProcessingOutput(
                    output_name=output_key,  # "ProcessedTabularData"
                    source="/opt/ml/processing/output",
                    destination=fallback_path
                )
            )
            logger.info(f"Created fallback processing output for '{output_key}': {fallback_path}")
        
        # Handle full_data output if configured (same pattern as for processed_data)
        full_data_logical_name = "full_data"
        if full_data_logical_name in self.config.output_names:
            full_data_output_name = self._get_output_destination_name(full_data_logical_name)
            
            try:
                # Try to create full_data output using helper method
                full_data_output = self._create_standard_processing_output(
                    full_data_logical_name,
                    outputs,
                    "/opt/ml/processing/output"
                )
                processing_outputs.append(full_data_output)
                logger.info(f"Successfully created standard processing output for '{full_data_output_name}'")
            except ValueError as e:
                # Enhanced helper failed - implement fallback for full_data
                self.log_warning("Full data output helper failed: %s, using fallback mechanism", e)
                
                # Construct fallback path for full_data
                full_data_fallback_path = f"{self.config.pipeline_s3_loc}/tabular_preprocessing/{self.config.job_type}/full_data"
                processing_outputs.append(
                    ProcessingOutput(
                        output_name=full_data_output_name,
                        source="/opt/ml/processing/output",
                        destination=full_data_fallback_path
                    )
                )
                logger.info(f"Created fallback processing output for '{full_data_output_name}': {full_data_fallback_path}")
        
        # Ensure we created at least one output
        if not processing_outputs:
            raise ValueError("Failed to create any processing outputs")
        
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
        # Only include the actual input channel names from the config
        input_reqs = {}
        
        # Add all input channel names from config
        for k, v in (self.config.input_names or {}).items():
            input_reqs[k] = f"S3 path for {v}"
        
        # Add other required parameters
        input_reqs["outputs"] = f"Dictionary containing {', '.join([f'{k}' for k in (self.config.output_names or {}).keys()])} S3 paths"
        input_reqs["enable_caching"] = self.COMMON_PROPERTIES["enable_caching"]
        input_reqs["dependencies"] = "List of steps that this step depends on"
        
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
                OUTPUT_TYPE_DATA = "DATA"  # Upper Case, correct one
                OUTPUT_TYPE_METADATA = "METADATA"  # Upper Case, correct one
                OUTPUT_TYPE_SIGNATURE = "SIGNATURE"  # Upper Case, correct one
                
            # Map output types to input keys - use direct mapping with constants
            output_type_to_input_key = {
                OUTPUT_TYPE_DATA: OUTPUT_TYPE_DATA,  # Use constants directly for direct matching
                OUTPUT_TYPE_METADATA: OUTPUT_TYPE_METADATA,
                OUTPUT_TYPE_SIGNATURE: OUTPUT_TYPE_SIGNATURE
            }
            
            # Match each output type to corresponding input key
            for output_type, input_key_name in output_type_to_input_key.items():
                if output_type in output_locations and input_key_name in self.config.input_names:
                    input_key = self.config.input_names[input_key_name]
                    
                    # Add output location directly to inputs (no nesting)
                    inputs[input_key] = output_locations[output_type]
                    matched_inputs.add(input_key)
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
            
        # Match data output - now uses uppercase "DATA"
        self._match_output_by_name(
            inputs, prev_step, matched_inputs, 
            "DATA", "data"
        )
        
        # Match metadata output - now uses uppercase "METADATA"
        self._match_output_by_name(
            inputs, prev_step, matched_inputs, 
            "METADATA", "metadata"
        )
        
        # Match signature output - now uses uppercase "SIGNATURE"
        self._match_output_by_name(
            inputs, prev_step, matched_inputs, 
            "SIGNATURE", "signature"
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
            input_key_name: Name of the input key in config.input_names (now uses uppercase constants)
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
                    
                    # Add output destination directly to inputs (no nesting)
                    inputs[input_key] = output.destination
                    matched_inputs.add(input_key)
                    logger.info(f"Found {output_name_pattern} from ProcessingStep")
                    break
                        
        except Exception as e:
            logger.warning(f"Error matching {output_name_pattern} output: {e}")
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline
        using the standardized input handling approach.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - outputs: A dictionary mapping output channel names to their S3 destinations.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.
                - DATA, METADATA, SIGNATURE: Direct attribute references from pipeline template
                  representing Join objects (outputs from CradleDataLoadingStep)

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating TabularPreprocessing ProcessingStep...")

        # Extract parameters using standard methods
        inputs_raw = self._extract_param(kwargs, 'inputs', {})
        outputs = self._extract_param(kwargs, 'outputs', {})
        dependencies = self._extract_param(kwargs, 'dependencies', [])
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Normalize inputs using standard helper method
        inputs = self._normalize_inputs(inputs_raw)
        
        # Handle direct attribute references that may come from template's _instantiate_step
        # These would be top-level kwargs like DATA, METADATA, SIGNATURE
        for key in ["DATA", "METADATA", "SIGNATURE"]:
            if key in kwargs and key not in inputs:
                inputs[key] = kwargs[key]
                logger.info(f"Added {key} from direct attribute reference")
        
        # Validate required parameters
        if not outputs:
            raise ValueError("outputs must be provided")
            
        # Extra debug logging to help diagnose connection issues
        logger.info(f"Input keys after normalization: {list(inputs.keys())}")

        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        # Create step name
        step_name = f"{self._get_step_name('TabularPreprocessing')}-{self.config.job_type.capitalize()}"
        
        # Create and return the step
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
