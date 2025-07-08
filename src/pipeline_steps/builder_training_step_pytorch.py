from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.functions import Join

from .config_training_step_pytorch import PyTorchTrainingConfig
from .builder_step_base import StepBuilderBase
from .s3_utils import S3PathHandler

# Import PyTorch training specification
try:
    from ..pipeline_step_specs.pytorch_training_spec import PYTORCH_TRAINING_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    PYTORCH_TRAINING_SPEC = None
    SPEC_AVAILABLE = False

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
                         
        Raises:
            ValueError: If specification is not available or config is invalid
        """
        if not isinstance(config, PyTorchTrainingConfig):
            raise ValueError(
                "PyTorchTrainingStepBuilder requires a PyTorchTrainingConfig instance."
            )
            
        # Load PyTorch training specification
        if not SPEC_AVAILABLE or PYTORCH_TRAINING_SPEC is None:
            raise ValueError("PyTorch training specification not available")
            
        logger.info("Using PyTorch training specification")
        
        super().__init__(
            config=config,
            spec=PYTORCH_TRAINING_SPEC,  # Add specification
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
        
        # Input/output validation is now handled by specifications
        logger.info("Configuration validation relies on step specifications")
        
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
            output_path=None,  # Will be set by create_step method
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
        
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """
        Get inputs for the step using specification and contract.
        
        This method creates TrainingInput objects for each dependency defined in the specification.
        
        Args:
            inputs: Input data sources keyed by logical name
            
        Returns:
            Dictionary of TrainingInput objects keyed by channel name
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
            
        training_inputs = {}
        
        # Process each dependency in the specification
        for _, dependency_spec in self.spec.dependencies.items():
            logical_name = dependency_spec.logical_name
            
            # Skip if optional and not provided
            if not dependency_spec.required and logical_name not in inputs:
                continue
                
            # Make sure required inputs are present
            if dependency_spec.required and logical_name not in inputs:
                raise ValueError(f"Required input '{logical_name}' not provided")
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_input_paths:
                container_path = self.contract.expected_input_paths[logical_name]
            else:
                raise ValueError(f"No container path found for input: {logical_name}")
                
            # Handle different input types based on logical name
            if logical_name == "input_path":
                # Training data - single channel for PyTorch (contains train/val/test subdirs)
                base_path = inputs[logical_name]
                
                # Create a single data channel
                training_inputs["data"] = TrainingInput(s3_data=base_path)
                logger.info(f"Created data channel from input_path: {base_path}")
                
            elif logical_name == "config":
                # Configuration file - single file
                config_uri = inputs[logical_name]
                
                training_inputs["config"] = TrainingInput(s3_data=config_uri)
                logger.info(f"Created config channel from config: {config_uri}")
                
        return training_inputs

    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Get outputs for the step using specification and contract.
        
        For training steps, this returns the output path where model artifacts will be stored.
        
        Args:
            outputs: Output destinations keyed by logical name
            
        Returns:
            Output path for model artifacts
            
        Raises:
            ValueError: If no specification or contract is available
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
            
        # Process each output in the specification to find the primary model output
        primary_output_path = None
        
        for _, output_spec in self.spec.outputs.items():
            logical_name = output_spec.logical_name
            
            # Get container path from contract
            container_path = None
            if logical_name in self.contract.expected_output_paths:
                container_path = self.contract.expected_output_paths[logical_name]
            else:
                raise ValueError(f"No container path found for output: {logical_name}")
                
            # For training steps, look for the primary model output
            if logical_name == "model_output" or "model" in logical_name.lower():
                # Try to find destination in outputs
                if logical_name in outputs:
                    primary_output_path = outputs[logical_name]
                else:
                    # Generate default output path using base config
                    bucket = getattr(self.config, 'bucket', 'default-bucket')
                    pipeline_name = getattr(self.config, 'pipeline_name', 'pytorch-model')
                    current_date = getattr(self.config, 'current_date', '2025-07-07')
                    primary_output_path = f"s3://{bucket}/{pipeline_name}/training_output/{current_date}/model"
                break
                
        # If no model output found in spec, generate default output path
        if primary_output_path is None:
            bucket = getattr(self.config, 'bucket', 'default-bucket')
            pipeline_name = getattr(self.config, 'pipeline_name', 'pytorch-model')
            current_date = getattr(self.config, 'current_date', '2025-07-07')
            primary_output_path = f"s3://{bucket}/{pipeline_name}/training_output/{current_date}/model"
            
        return primary_output_path
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
        inputs_raw = kwargs.get('inputs', {})
        input_path = kwargs.get('input_path')
        output_path = kwargs.get('output_path')
        dependencies = kwargs.get('dependencies', [])
        enable_caching = kwargs.get('enable_caching', True)
        
        logger.info("Creating PyTorch TrainingStep...")
        
        # Get the step name
        step_name = self._get_step_name('PyTorchTraining')
        
        # Handle inputs
        inputs = {}
        
        # If dependencies are provided, extract inputs from them using the resolver
        if dependencies:
            try:
                extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                inputs.update(extracted_inputs)
            except Exception as e:
                logger.warning(f"Failed to extract inputs from dependencies: {e}")
                
        # Add explicitly provided inputs (overriding any extracted ones)
        inputs.update(inputs_raw)
        
        # Add direct parameters if provided
        if input_path is not None:
            inputs["input_path"] = input_path
            
        # Get training inputs using specification-driven method
        training_inputs = self._get_inputs(inputs)
        
        # Make sure we have the inputs we need
        if len(training_inputs) == 0:
            raise ValueError("No training inputs available. Provide input_path or ensure dependencies supply necessary outputs.")
        
        logger.info(f"Final training inputs: {list(training_inputs.keys())}")
        
        # Get output path using specification-driven method
        output_path = self._get_outputs({})
        
        # Create estimator
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
            
            # Attach specification to the step for future reference
            setattr(training_step, '_spec', self.spec)
            
            # Log successful creation
            logger.info(f"Created TrainingStep with name: {training_step.name}")
            
            return training_step
            
        except Exception as e:
            logger.error(f"Error creating PyTorch TrainingStep: {str(e)}")
            raise ValueError(f"Failed to create PyTorchTrainingStep: {str(e)}") from e
