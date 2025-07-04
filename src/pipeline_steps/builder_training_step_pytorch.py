from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.functions import Join

from .config_training_step_pytorch import PyTorchTrainingConfig
from .builder_step_base import StepBuilderBase
from .s3_utils import S3PathHandler

# Register property paths for PyTorch Training outputs
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep",
    "model_output",                                # Logical name in output_names
    "properties.ModelArtifacts.S3ModelArtifacts"   # Runtime property path
)

# Register path to training metrics
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep",
    "metrics_output",
    "properties.TrainingMetrics"
)

# Register path to training job name
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep", 
    "training_job_name",
    "properties.TrainingJobName"
)

# Register path to model data for compatibility with different naming patterns
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep", 
    "model_data",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Register path to output directory with both logical names
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep",
    "output_path",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Critical fix - Register ModelOutputPath specifically to match the descriptor used in pattern matching
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep",
    "ModelOutputPath",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Add more registrations for the model artifacts with names that might be used by different steps
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep",
    "ModelArtifacts",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Add mapping for common model input key names
StepBuilderBase.register_property_path(
    "PyTorchTrainingStep",
    "model_input",  # Common key name used by many step builders
    "properties.ModelArtifacts.S3ModelArtifacts"
)

logger = logging.getLogger(__name__)


class PyTorchTrainingStepBuilder(StepBuilderBase):
    """
    Builder for a PyTorch Training Step.
    This class is responsible for configuring and creating a SageMaker TrainingStep
    that trains a PyTorch model.
    """

    def __init__(
        self,
        config: PyTorchTrainingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the training step.

        Args:
            config: A PytorchTrainingConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Training Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, PyTorchTrainingConfig):
            raise ValueError(
                "PyTorchTrainingStepBuilder requires a PyTorchTrainingConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: PyTorchTrainingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating PyTorchTrainingConfig...")
        
        # Validate required attributes
        required_attrs = [
            'training_instance_type',
            'training_instance_count',
            'training_volume_size',
            'training_entry_point',
            'source_dir',
            'framework_version',
            'py_version'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PyTorchTrainingConfig missing required attribute: {attr}")
        
        # Validate input and output names
        if "input_path" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'input_path'")
        
        logger.info("PyTorchTrainingConfig validation succeeded.")

    def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
        """
        Normalizes an S3 URI to ensure it has no trailing slashes and is properly formatted.
        Uses S3PathHandler for consistent path handling.
        
        Args:
            uri: The S3 URI to normalize
            description: Description for logging purposes
            
        Returns:
            Normalized S3 URI
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
        
        # Handle Pipeline step references with Get key - return as is
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference during normalization: %s", uri)
            return uri
        
        return S3PathHandler.normalize(uri, description)
        
    def _validate_s3_uri(self, uri: str, description: str = "data") -> bool:
        """
        Validates that a string is a properly formatted S3 URI.
        Uses S3PathHandler for consistent path validation.
        
        Args:
            uri: The URI to validate
            description: Description of what the URI is for (used in error messages)
            
        Returns:
            True if valid, False otherwise
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            # For PipelineVariables, we trust they'll resolve to valid URIs at execution time
            return True
            
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            # For Get expressions, we also trust they'll resolve properly at execution time
            logger.info(f"Found Pipeline step reference: {uri}")
            return True
        
        if not isinstance(uri, str):
            logger.warning(f"Invalid {description} URI: type {type(uri).__name__}")
            return False
        
        # Use S3PathHandler for validation
        valid = S3PathHandler.is_valid(uri)
        if not valid:
            logger.warning(f"Invalid {description} URI format: {uri}")
        
        return valid

    def _create_estimator(self) -> PyTorch:
        """
        Creates and configures the PyTorch estimator for the SageMaker Training Job.
        This defines the execution environment for the training script, including the instance
        type, framework version, and hyperparameters.

        Returns:
            An instance of sagemaker.pytorch.PyTorch.
        """
        # Convert hyperparameters object to dict if available
        hyperparameters = {}
        if hasattr(self.config, "hyperparameters") and self.config.hyperparameters:
            # If the hyperparameters object has a to_dict method, use it
            if hasattr(self.config.hyperparameters, "to_dict"):
                hyperparameters.update(self.config.hyperparameters.to_dict())
            # Otherwise add all non-private attributes
            else:
                for key, value in vars(self.config.hyperparameters).items():
                    if not key.startswith('_'):
                        hyperparameters[key] = value
        
        return PyTorch(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('PyTorchTraining')}"
            ),
            hyperparameters=hyperparameters,
            sagemaker_session=self.session,
            output_path=self.config.output_path,
            checkpoint_s3_uri=self.config.get_checkpoint_uri(),
            environment=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the training job.
        These variables are used to control the behavior of the training script
        without needing to pass them as hyperparameters.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {}
        
        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)
            
        logger.info(f"Training environment variables: {env_vars}")
        return env_vars
        
    def _get_checkpoint_uri(self) -> str:
        """
        Gets the checkpoint URI for the training job.
        If the config has a checkpoint path, use that.
        Otherwise, construct a default checkpoint path based on the output path.

        Returns:
            The S3 URI for checkpoints.
        """
        if self.config.has_checkpoint():
            return self.config.get_checkpoint_uri()
        
        # Construct a default checkpoint path
        return f"{self.config.output_path}/checkpoints/{self.config.current_date}"
        
    def _get_metric_definitions(self) -> List[Dict[str, str]]:
        """
        Defines the metrics to be captured from the training logs.
        These metrics will be visible in the SageMaker console and can be used
        for monitoring and early stopping.

        Returns:
            A list of metric definitions.
        """
        return [
            {"Name": "Train Loss", "Regex": "Train Loss: ([0-9\\.]+)"},
            {"Name": "Validation Loss", "Regex": "Validation Loss: ([0-9\\.]+)"},
            {"Name": "Validation F1 Score", "Regex": "Validation F1 Score: ([0-9\\.]+)"},
            {"Name": "Validation AUC ROC", "Regex": "Validation AUC ROC: ([0-9\\.]+)"}
        ]
        
    def _create_profiler_config(self):
        """
        Creates a profiler configuration for the training job.
        This enables SageMaker to collect system metrics during training.

        Returns:
            A SageMaker profiler configuration object.
        """
        from sagemaker.debugger import ProfilerConfig, FrameworkProfile
        
        return ProfilerConfig(
            system_monitor_interval_millis=1000,
            framework_profile_params=FrameworkProfile(local_path="/opt/ml/output/profiler/")
        )
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
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
        # Define the output properties for the training step
        output_props = {
            "ModelArtifacts.S3ModelArtifacts": "S3 URI of the model artifacts"
        }
        return output_props
        
    def _get_training_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """
        Constructs a dictionary of TrainingInput objects from the provided inputs dictionary.
        This defines the data channels for the training job, mapping S3 locations
        to input channels for the training container.
        
        The training script expects a single input channel with train/val/test subdirectories.

        Args:
            inputs: A dictionary mapping logical input channel names to their S3 URIs or dynamic Step properties.

        Returns:
            A dictionary of channel names to sagemaker.inputs.TrainingInput objects.
        """
        training_inputs = {}
        
        # Get channel names from config
        input_path_key = next(iter(self.config.input_names.keys()), "input_path")
        data_key = "data"  # The SageMaker channel name for input data
        
        # Use the base class helper to normalize inputs
        normalized_inputs = self._normalize_inputs(inputs)
        
        # First check if input_path is in the normalized inputs (highest priority)
        if input_path_key in normalized_inputs:
            base_path = normalized_inputs[input_path_key]
            # Normalize the base path URI
            base_path = self._normalize_s3_uri(base_path, "base input path")
            
            if self._validate_s3_uri(base_path, "base input path"):
                # Create a single channel for the base path
                training_inputs[data_key] = TrainingInput(s3_data=base_path)
                logger.info(f"Using input_path from inputs: {base_path}")
                
        # Fallback to config's input_path if not in inputs
        elif hasattr(self.config, 'input_path') and self.config.input_path:
            input_base_path = self.config.input_path
            logger.info(f"Using input_path from config: {input_base_path}")
            
            # Create a single channel for the base path
            training_inputs[data_key] = TrainingInput(s3_data=input_base_path)
        else:
            logger.warning(f"No input path found for data channel")
        
        return training_inputs
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to PyTorchTraining step.
        This method dispatches to specialized handlers based on the type of step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        step_name = getattr(prev_step, 'name', str(prev_step))
        logger.info(f"Matching inputs from dependency step: {step_name}")
        
        # First check for TabularPreprocessingStep
        if hasattr(prev_step, 'name') and 'tabularpreprocessing' in prev_step.name.lower():
            matched_inputs = self._match_tabular_preprocessing_outputs(inputs, prev_step)
            if matched_inputs:
                logger.info(f"Matched inputs from TabularPreprocessingStep: {step_name}")
                return matched_inputs
        
        # Fall back to generic output matching
        matched_inputs = self._match_generic_outputs(inputs, prev_step)
        if matched_inputs:
            logger.info(f"Matched inputs from generic step: {step_name}")
                
        return matched_inputs
        
    def _match_tabular_preprocessing_outputs(self, inputs: Dict[str, Any], prev_step: Step) -> Set[str]:
        """
        Match outputs from a TabularPreprocessingStep.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Get the configured input path key from config
        input_path_key = next(iter(self.config.input_names.keys()), "input_path")
        
        # Check if this step has the expected output structure
        if not hasattr(prev_step, "outputs") or not prev_step.outputs:
            return matched_inputs
            
        try:
            # Find the processed_data output
            processed_data_output = None
            for output in prev_step.outputs:
                if (hasattr(output, "output_name") and 
                    "processed_data" in output.output_name.lower()):
                    processed_data_output = output
                    break
                    
            if not processed_data_output:
                return matched_inputs
                
            # TabularPreprocessingStep output is the base path that contains train/val/test subdirs
            base_path = processed_data_output.destination
            base_path = base_path.rstrip("/")
            
            # Initialize inputs dict if needed
            if "inputs" not in inputs:
                inputs["inputs"] = {}
                
            # Just use the base path directly - it contains all subdirectories
            # that the training script expects (train, val, test)
            if input_path_key not in inputs.get("inputs", {}):
                inputs["inputs"][input_path_key] = base_path
                matched_inputs.add("inputs")
                logger.info(f"Added input path: {base_path}")
                
        except Exception as e:
            logger.warning(f"Error matching TabularPreprocessingStep outputs: {e}")
            
        return matched_inputs
        
    def _match_generic_outputs(self, inputs: Dict[str, Any], prev_step: Step) -> Set[str]:
        """
        Match generic outputs from any step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Get input path key from config
        input_path_key = next(iter(self.config.input_names.keys()), "input_path")
        
        try:
            # Try to find a generic output path that might contain training data
            if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                outputs = prev_step.properties.ProcessingOutputConfig.Outputs
                
                # Log the type of outputs object to help with debugging
                logger.info(f"Processing outputs of type: {outputs.__class__.__name__ if hasattr(outputs, '__class__') else type(outputs)}")
                
                # Special handling for PropertiesList type
                if hasattr(outputs, "__class__") and outputs.__class__.__name__ == "PropertiesList":
                    logger.info("Detected PropertiesList object - using direct attribute access")
                    
                    # Try common output names that might contain processed data
                    common_names = ["ProcessedTabularData", "Data", "OutputData"]
                    for name in common_names:
                        if hasattr(outputs, name):
                            try:
                                output_uri = outputs[name].S3Output.S3Uri
                                
                                # Initialize inputs dict if needed
                                if "inputs" not in inputs:
                                    inputs["inputs"] = {}
                                    
                                # Add as input_path
                                if input_path_key not in inputs.get("inputs", {}):
                                    inputs["inputs"][input_path_key] = output_uri
                                    matched_inputs.add("inputs")
                                    logger.info(f"Added input path from PropertiesList attribute {name}: {output_uri}")
                                    return matched_inputs
                            except (AttributeError, KeyError) as e:
                                logger.debug(f"Error accessing PropertiesList attribute {name}: {e}")
                
                # Safe iteration approach instead of using len()
                try:
                    # Try to get the first item safely
                    output_name = next(iter(outputs), None)
                    if output_name is not None:
                        output_uri = outputs[output_name].S3Output.S3Uri
                        
                        # Initialize inputs dict if needed
                        if "inputs" not in inputs:
                            inputs["inputs"] = {}
                            
                        # Add as input_path
                        if input_path_key not in inputs.get("inputs", {}):
                            inputs["inputs"][input_path_key] = output_uri
                            matched_inputs.add("inputs")
                            logger.info(f"Added input path from generic step: {output_uri} (reference)")
                except (TypeError, StopIteration) as e:
                    logger.debug(f"Error iterating through outputs: {e}")
                    
        except (AttributeError, KeyError, IndexError) as e:
            logger.warning(f"Error matching generic outputs: {e}")
            
        return matched_inputs

    def create_step(self, **kwargs) -> TrainingStep:
        """
        Creates a SageMaker TrainingStep for the pipeline.
        
        This method creates the PyTorch estimator, sets up training inputs from the input data,
        and creates the SageMaker TrainingStep.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary mapping input channel names to their S3 locations
                - input_path: Direct parameter for training data input path (for backward compatibility)
                - output_path: Direct parameter for model output path (for backward compatibility)
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.
                
        Returns:
            A configured TrainingStep instance.
        """
        # Extract common parameters
        inputs_raw = self._extract_param(kwargs, 'inputs', {})
        input_path = self._extract_param(kwargs, 'input_path')
        output_path = self._extract_param(kwargs, 'output_path')
        dependencies = self._extract_param(kwargs, 'dependencies', [])
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        logger.info("Creating PyTorch TrainingStep...")
        
        # Get the step name
        step_name = self._get_step_name('PyTorchTraining')
        
        # Construct inputs dictionary - handle both nested and flat structures
        inputs = self._normalize_inputs(inputs_raw)
        
        # Add direct input_path parameter if provided
        if input_path is not None:
            inputs["input_path"] = input_path
            self.log_info("Using directly provided input_path: %s", input_path)
            
        # Look for inputs from dependencies if we don't have what we need
        if "input_path" not in inputs and dependencies:
            input_requirements = self.get_input_requirements()
            
            # Extract inputs from dependencies
            for dep_step in dependencies:
                # Temporary dictionary to collect inputs from matching
                temp_inputs = {}
                matched = self._match_custom_properties(temp_inputs, input_requirements, dep_step)
                
                if matched:
                    # Normalize any nested inputs from the matching
                    normalized_deps = self._normalize_inputs(temp_inputs)
                    
                    # Add to our main inputs dictionary
                    inputs.update(normalized_deps)
                    logger.info(f"Found inputs from dependency: {getattr(dep_step, 'name', None)}")
                    
        # Get training inputs (TrainingInput objects)
        training_inputs = self._get_training_inputs(inputs)
        
        # Make sure we have the inputs we need
        if len(training_inputs) == 0:
            raise ValueError("No training inputs available. Provide input_path or ensure dependencies supply necessary outputs.")
        
        logger.info(f"Final training inputs: {list(training_inputs.keys())}")
        
        # Create estimator with output path if provided
        estimator = self._create_estimator()
        
        # Create the training step
        try:
            training_step = TrainingStep(
                name=step_name,
                estimator=estimator,
                inputs=training_inputs,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching)
            )
            
            # Log successful creation
            logger.info(f"Created TrainingStep with name: {training_step.name}")
            
            return training_step
            
        except Exception as e:
            logger.error(f"Error creating PyTorch TrainingStep: {str(e)}")
            raise ValueError(f"Failed to create PyTorchTrainingStep: {str(e)}") from e
