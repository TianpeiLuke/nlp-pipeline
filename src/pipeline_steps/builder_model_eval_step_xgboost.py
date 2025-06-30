from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoostProcessor

from .config_model_eval_step_xgboost import XGBoostModelEvalConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class XGBoostModelEvalStepBuilder(StepBuilderBase):
    """
    Builder for an XGBoost Model Evaluation ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that evaluates an XGBoost model on a validation dataset.
    """

    def __init__(
        self,
        config: XGBoostModelEvalConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model evaluation step.

        Args:
            config: A XGBoostModelEvalConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, XGBoostModelEvalConfig):
            raise ValueError(
                "XGBoostModelEvalStepBuilder requires a XGBoostModelEvalConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: XGBoostModelEvalConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating XGBoostModelEvalConfig...")
        
        # Validate required attributes
        required_attrs = [
            'processing_entry_point',
            'processing_source_dir',
            'processing_instance_count', 
            'processing_volume_size',
            'pipeline_name',
            'job_type',
            'hyperparameters',
            'xgboost_framework_version'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostModelEvalConfig missing required attribute: {attr}")
        
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
        
        if "eval_data_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'eval_data_input'")
        
        if "eval_output" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'eval_output'")
        
        if "metrics_output" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'metrics_output'")
        
        logger.info("XGBoostModelEvalConfig validation succeeded.")

    def _create_processor(self) -> XGBoostProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Get the appropriate instance type based on use_large_processing_instance
        instance_type = self.config.processing_instance_type_large if self.config.use_large_processing_instance else self.config.processing_instance_type_small
        
        return XGBoostProcessor(
            framework_version=self.config.xgboost_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('XGBoostModelEval')}"
            ),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )
    

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the processing job.
        These variables are used to control the behavior of the evaluation script
        without needing to pass them as command-line arguments.

        Returns:
            A dictionary of environment variables.
        """        
        env_vars = {
            "ID_FIELD": str(self.config.hyperparameters.id_name),
            "LABEL_FIELD": str(self.config.hyperparameters.label_name),
        }
        logger.info(f"Evaluation environment variables: {env_vars}")
        return env_vars

    

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects using the standardized helper methods.
        
        Args:
            inputs: A dictionary mapping logical input channel names (keys from input_names)
                   to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Validate required inputs using the standard helper method
        self._validate_inputs(inputs)
        
        processing_inputs = []
        
        # Process model input (required) using standard helper
        processing_inputs.append(
            self._create_standard_processing_input(
                "model_input",
                inputs,
                "/opt/ml/processing/input/model"
            )
        )
        
        # Process evaluation data input (required) using standard helper
        processing_inputs.append(
            self._create_standard_processing_input(
                "eval_data_input",
                inputs,
                "/opt/ml/processing/input/eval_data"
            )
        )
        
        # Add optional hyperparameters input if available
        hyperparams_logical_name = "hyperparameters_input"
        if (hyperparams_logical_name in self.config.input_names and 
            hyperparams_logical_name in inputs):
            processing_inputs.append(
                self._create_standard_processing_input(
                    hyperparams_logical_name,
                    inputs,
                    "/opt/ml/processing/input/hyperparameters"
                )
            )
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step using standardized helper methods.
        Uses VALUES from output_names as keys in the outputs dictionary (standard pattern).

        Args:
            outputs: A dictionary mapping the output values from output_names to their S3 destination URIs.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        # Validate outputs using standard helper method
        self._validate_outputs(outputs)
        
        processing_outputs = []
        
        # Add evaluation results output using helper method
        processing_outputs.append(
            self._create_standard_processing_output(
                "eval_output",
                outputs,
                "/opt/ml/processing/output/eval"
            )
        )
        
        # Add metrics output using helper method
        processing_outputs.append(
            self._create_standard_processing_output(
                "metrics_output",
                outputs,
                "/opt/ml/processing/output/metrics"
            )
        )
        
        return processing_outputs

    def _get_job_arguments(self) -> List[str]:
        """
        Constructs the list of command-line arguments to be passed to the processing script.
        Passes the job_type from the configuration to the script, which requires this argument.

        Returns:
            A list of strings representing the command-line arguments.
        """
        # Pass the job_type from the configuration to satisfy the script requirement
        # and the SageMaker validation requirement for at least one command-line argument
        job_type = self.config.job_type
        logger.info(f"Setting job_type argument to: {job_type}")
        return ["--job_type", job_type]
        
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
        Match custom properties specific to XGBoostModelEval step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Look for model artifacts from a TrainingStep
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ModelArtifacts"):
            try:
                model_artifacts = prev_step.properties.ModelArtifacts.S3ModelArtifacts
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                
                # Use the hardcoded model_input key
                model_key = "model_input"
                if model_key not in inputs.get("inputs", {}):
                    inputs["inputs"][model_key] = model_artifacts
                    matched_inputs.add("inputs")
                    logger.info(f"Found model artifacts from TrainingStep: {getattr(prev_step, 'name', str(prev_step))}")
            except AttributeError as e:
                logger.warning(f"Could not extract model artifacts from step: {e}")
                
        # Look for validation data from a ProcessingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Use the hardcoded eval_data_input key
                eval_data_key = "eval_data_input"
                # Look for an output with a name that might contain evaluation data
                for output in prev_step.outputs:
                        if hasattr(output, "output_name") and any(term in output.output_name.lower() 
                                                                for term in ["valid", "test", "eval"]):
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if eval_data_key not in inputs.get("inputs", {}):
                                inputs["inputs"][eval_data_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found evaluation data from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract validation data from step: {e}")
                
        # Look for hyperparameters from a HyperparameterPrepStep
        if hasattr(prev_step, "hyperparameters_s3_uri"):
            try:
                hyperparameters_s3_uri = prev_step.hyperparameters_s3_uri
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                
                # Use the hardcoded hyperparameters_input key if available
                hyperparams_key = "hyperparameters_input"
                if hyperparams_key not in inputs.get("inputs", {}):
                    inputs["inputs"][hyperparams_key] = hyperparameters_s3_uri
                    matched_inputs.add("inputs")
                    logger.info(f"Found hyperparameters from step: {getattr(prev_step, 'name', str(prev_step))}")
            except AttributeError as e:
                logger.warning(f"Could not extract hyperparameters from step: {e}")
                
        return matched_inputs
    
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

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating XGBoostModelEval ProcessingStep...")

        # Extract parameters using standard methods
        inputs_raw = self._extract_param(kwargs, 'inputs', {})
        outputs = self._extract_param(kwargs, 'outputs', {})
        dependencies = self._extract_param(kwargs, 'dependencies', [])
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Normalize inputs using standard helper method
        inputs = self._normalize_inputs(inputs_raw)
        
        # Validate required parameters
        if not outputs:
            raise ValueError("outputs must be provided")
            
        # Log input keys for debugging
        logger.info(f"Input keys after normalization: {list(inputs.keys())}")

        # Create processor and get inputs/outputs
        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        # Create step name
        step_name = self._get_step_name('XGBoostModelEval')
        
        # Create step arguments
        step_args = processor.run(
            code=self.config.processing_entry_point,
            source_dir=self.config.processing_source_dir,
            inputs=proc_inputs,
            outputs=proc_outputs,
            arguments=job_args,
        )

        # Create and return the step
        processing_step = ProcessingStep(
            name=step_name,
            step_args=step_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
