from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.pytorch import PyTorch

from .config_training_step_pytorch import PyTorchTrainingConfig
from .builder_step_base import StepBuilderBase

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
            config: A PyTorchTrainingConfig instance containing all necessary settings.
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
            'instance_type',
            'instance_count',
            'volume_size',
            'entry_point',
            'source_dir',
            'framework_version',
            'py_version',
            'job_name'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PyTorchTrainingConfig missing required attribute: {attr}")
        
        # Validate input and output names
        if "training_data" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'training_data'")
        
        logger.info("PyTorchTrainingConfig validation succeeded.")

    def _create_estimator(self) -> PyTorch:
        """
        Creates and configures the PyTorch estimator for the SageMaker Training Job.
        This defines the execution environment for the training script, including the instance
        type, framework version, and hyperparameters.

        Returns:
            An instance of sagemaker.pytorch.PyTorch.
        """
        hyperparameters = {}
        if hasattr(self.config, "hyperparameters") and self.config.hyperparameters:
            hyperparameters.update(self.config.hyperparameters)
        
        return PyTorch(
            entry_point=self.config.entry_point,
            source_dir=self.config.source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.instance_type,
            instance_count=self.config.instance_count,
            volume_size=self.config.volume_size,
            max_run=self.config.max_run,
            keep_alive_period_in_seconds=self.config.keep_alive_period,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('PyTorchTraining')}"
            ),
            hyperparameters=hyperparameters,
            sagemaker_session=self.session,
            enable_sagemaker_metrics=self.config.enable_sagemaker_metrics,
            debugger_hook_config=self.config.debugger_hook_config,
            tensorboard_output_config=self.config.tensorboard_output_config,
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
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to PyTorchTraining step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Look for preprocessed data from a ProcessingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our training_data input
                training_key = self.config.input_names.get("training_data")
                if training_key:
                    # Look for an output with a name that might contain training data
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and any(term in output.output_name.lower() 
                                                                for term in ["train", "preprocess"]):
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if training_key not in inputs.get("inputs", {}):
                                inputs["inputs"][training_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found training data from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract training data from step: {e}")
                
        # Look for validation data from a ProcessingStep
        if hasattr(prev_step, "outputs") and len(prev_step.outputs) > 0:
            try:
                # Check if the step has an output that matches our validation_data input
                validation_key = self.config.input_names.get("validation_data")
                if validation_key:
                    # Look for an output with a name that might contain validation data
                    for output in prev_step.outputs:
                        if hasattr(output, "output_name") and any(term in output.output_name.lower() 
                                                                for term in ["valid", "val", "test"]):
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                            
                            if validation_key not in inputs.get("inputs", {}):
                                inputs["inputs"][validation_key] = output.destination
                                matched_inputs.add("inputs")
                                logger.info(f"Found validation data from step: {getattr(prev_step, 'name', str(prev_step))}")
                                break
            except AttributeError as e:
                logger.warning(f"Could not extract validation data from step: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> TrainingStep:
        """
        Creates the final, fully configured SageMaker TrainingStep for the pipeline.
        This method orchestrates the assembly of the estimator and its inputs
        into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.TrainingStep instance.
        """
        logger.info("Creating PyTorch TrainingStep...")

        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Validate required parameters
        if not inputs:
            raise ValueError("inputs must be provided")

        estimator = self._create_estimator()

        # Prepare the inputs for the estimator
        estimator_inputs = {}
        for logical_name, s3_uri in inputs.items():
            # Map the logical name to the actual channel name expected by the estimator
            channel_name = logical_name
            if logical_name in self.config.input_names:
                channel_name = self.config.input_names[logical_name]
            estimator_inputs[channel_name] = s3_uri

        step_name = self._get_step_name('PyTorchTraining')
        
        training_step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=estimator_inputs,
            job_name=self.config.job_name,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created TrainingStep with name: {training_step.name}")
        return training_step
