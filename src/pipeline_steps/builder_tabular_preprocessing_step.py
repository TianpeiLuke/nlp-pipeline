from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import os, importlib

from sagemaker.sklearn import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from .config_tabular_preprocessing_step import TabularPreprocessingConfig
from .builder_step_base import StepBuilderBase

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

logger = logging.getLogger(__name__)


class TabularPreprocessingStepBuilder(StepBuilderBase):
    """
    Builder for a Tabular Preprocessing ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that executes the tabular data preprocessing script. It handles the setup of
    the processor, inputs, outputs, and script arguments.
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
            config: A TabularPreprocessingConfig instance containing all necessary settings,
                    such as instance types, script paths, and hyperparameters.
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
        logger.info("Validating TabularPreprocessingConfig…")
        if not self.config.hyperparameters.label_name:
            raise ValueError("hyperparameters.label_name must be provided and non‐empty")
        if not getattr(self.config, 'job_type', None):
            raise ValueError(
                "job_type must be provided (e.g. 'training','validation','testing')"
            )
        if "data_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'data_input'")
        
        # Ensures that the configuration specifies the name for the primary output channel.
        if "processed_data" not in (self.config.output_names or {}):
            raise ValueError(
                "output_names must contain the key 'processed_data'"
            )
        logger.info("TabularPreprocessingConfig validation succeeded.")

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.
        These variables are used to control the behavior of the preprocessing script
        without needing to pass them as command-line arguments.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {
            "LABEL_FIELD": str(self.config.hyperparameters.label_name),
            "TRAIN_RATIO": str(self.config.train_ratio),
            "TEST_VAL_RATIO": str(self.config.test_val_ratio)
        }
        logger.info(f"Processing environment variables: {env_vars}")
        return env_vars

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
            command=["python3"],
            role=self.role,
            instance_count=self.config.processing_instance_count,
            instance_type=self.config.get_instance_type(),
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('Processing')}-{self.config.job_type}"
            ),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'raw_data')
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Get the data input key from config
        key_in = self.config.input_names["data_input"]
        
        # Check if inputs is empty or doesn't contain the required key
        if not inputs:
            raise ValueError(f"Inputs dictionary is empty. Must supply an S3 URI for '{key_in}'")
        
        # Try to find the input using the standard key or the constant
        if key_in not in inputs:
            # If the standard key is not found, try to find it using the constant
            if OUTPUT_TYPE_DATA and OUTPUT_TYPE_DATA in inputs:
                # Use the constant key instead
                logger.info(f"Using {OUTPUT_TYPE_DATA} as input key instead of {key_in}")
                key_in = OUTPUT_TYPE_DATA
            else:
                raise ValueError(f"Must supply an S3 URI for '{key_in}' in 'inputs'. Available keys: {list(inputs.keys())}")

        # Define the primary data input channel.
        processing_inputs = [
            ProcessingInput(
                input_name=key_in,
                source=inputs[key_in],
                destination="/opt/ml/processing/input/data"
            )
        ]
        
        # Add optional metadata and signature inputs if available
        if "metadata_input" in self.config.input_names:
            metadata_key = self.config.input_names["metadata_input"]
            if metadata_key in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name=metadata_key,
                        source=inputs[metadata_key],
                        destination="/opt/ml/processing/input/metadata"
                    )
                )
            elif OUTPUT_TYPE_METADATA and OUTPUT_TYPE_METADATA in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name=OUTPUT_TYPE_METADATA,
                        source=inputs[OUTPUT_TYPE_METADATA],
                        destination="/opt/ml/processing/input/metadata"
                    )
                )
        
        if "signature_input" in self.config.input_names:
            signature_key = self.config.input_names["signature_input"]
            if signature_key in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name=signature_key,
                        source=inputs[signature_key],
                        destination="/opt/ml/processing/input/signature"
                    )
                )
            elif OUTPUT_TYPE_SIGNATURE and OUTPUT_TYPE_SIGNATURE in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name=OUTPUT_TYPE_SIGNATURE,
                        source=inputs[OUTPUT_TYPE_SIGNATURE],
                        destination="/opt/ml/processing/input/signature"
                    )
                )
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the single ProcessingOutput object needed for this step.
        This defines the S3 location where the results of the processing job will be stored.

        Args:
            outputs: A dictionary mapping the logical output channel name ('processed_data')
                     to its S3 destination URI.

        Returns:
            A list containing a single sagemaker.processing.ProcessingOutput object.
        """
        key_proc = self.config.output_names["processed_data"]
        if not outputs or key_proc not in outputs:
            raise ValueError(f"Must supply an S3 URI for '{key_proc}' in 'outputs'")
        
        # Define a single output. The script writes to subfolders (train, val, test)
        # inside the '/opt/ml/processing/output' directory. Setting this as the source
        # ensures all created subfolders are uploaded to the specified S3 destination.
        return [
            ProcessingOutput(
                output_name=key_proc,
                source="/opt/ml/processing/output",
                destination=outputs[key_proc]
            )
        ]

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.
        This allows for parameterizing the script's execution at runtime.

        Returns:
            A list of strings representing the command-line arguments (e.g., ['--job_type', 'training']).
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
        
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps.
        
        This method extracts the inputs required by the TabularPreprocessingStep from the dependency steps.
        Specifically, it looks for:
        1. data_input from a DataLoadStep or other processing steps
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            Dictionary of inputs extracted from dependency steps
        """
        inputs = {}
        outputs = {}
        
        # Look for data_input from a DataLoadStep or other processing steps
        for prev_step in dependency_steps:
            if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                try:
                    # Try to get the data output
                    if hasattr(prev_step.properties.ProcessingOutputConfig.Outputs, "__getitem__"):
                        # Try string keys (dict-like)
                        for key in prev_step.properties.ProcessingOutputConfig.Outputs:
                            output = prev_step.properties.ProcessingOutputConfig.Outputs[key]
                            if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                # If this is a data load step, use it as data_input
                                # Check for both the constants and fallback string values
                                if (OUTPUT_TYPE_DATA and key == OUTPUT_TYPE_DATA) or key == "DATA" or key == "RawData":
                                    if "data_input" in self.config.input_names:
                                        input_key = self.config.input_names["data_input"]
                                        if not inputs:
                                            inputs = {}
                                        inputs[input_key] = output.S3Output.S3Uri
                                        logger.info(f"Found data_input from step: {prev_step.name}")
                                        break
                                # Also check for metadata and signature if needed
                                elif (OUTPUT_TYPE_METADATA and key == OUTPUT_TYPE_METADATA) and "metadata_input" in self.config.input_names:
                                    input_key = self.config.input_names["metadata_input"]
                                    if not inputs:
                                        inputs = {}
                                    inputs[input_key] = output.S3Output.S3Uri
                                    logger.info(f"Found metadata_input from step: {prev_step.name}")
                                elif (OUTPUT_TYPE_SIGNATURE and key == OUTPUT_TYPE_SIGNATURE) and "signature_input" in self.config.input_names:
                                    input_key = self.config.input_names["signature_input"]
                                    if not inputs:
                                        inputs = {}
                                    inputs[input_key] = output.S3Output.S3Uri
                                    logger.info(f"Found signature_input from step: {prev_step.name}")
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Could not extract data output from step: {e}")
        
        # Set up outputs if not already set
        if not outputs and hasattr(self.config, "pipeline_s3_loc") and hasattr(self.config, "job_type"):
            output_path = f"{self.config.pipeline_s3_loc}/tabular_preprocessing/{self.config.job_type}"
            if "processed_data" in self.config.output_names:
                output_key = self.config.output_names["processed_data"]
                outputs = {output_key: output_path}
                logger.info(f"Set up output path: {output_path}")
        
        result = {}
        if inputs:
            result["inputs"] = inputs
        if outputs:
            result["outputs"] = outputs
        
        # Add enable_caching
        result["enable_caching"] = True
        
        return result

    def create_step(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline.
        This method orchestrates the assembly of the processor, inputs, outputs, and
        script arguments into a single, executable pipeline step.

        Args:
            inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
            outputs: A dictionary mapping output channel names to their S3 destinations.
            enable_caching: A boolean indicating whether to cache the results of this step
                            to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating TabularPreprocessing ProcessingStep...")

        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        step_name = f"{self._get_step_name('Processing')}-{self.config.job_type.capitalize()}"
        
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step

    def create_processing_step(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> ProcessingStep:
        """Backwards compatible method for creating processing step"""
        logger.warning("create_processing_step is deprecated, use create_step instead.")
        return self.create_step(inputs, outputs, enable_caching)
