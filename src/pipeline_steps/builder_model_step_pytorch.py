from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import CreateModelStep, Step
from sagemaker.pytorch import PyTorchModel
from sagemaker.model import Model

from .config_model_step_pytorch import PyTorchModelStepConfig
from .builder_step_base import StepBuilderBase

# Register property paths for PyTorch Model outputs
StepBuilderBase.register_property_path(
    "PyTorchModelStep",
    "model",                                # Logical name in output_names
    "properties.ModelName"                  # Runtime property path
)

# Register path to model name for compatibility with different naming patterns
StepBuilderBase.register_property_path(
    "PyTorchModelStep", 
    "ModelName",
    "properties.ModelName"
)

logger = logging.getLogger(__name__)


class PyTorchModelStepBuilder(StepBuilderBase):
    """
    Builder for a PyTorch Model Step.
    This class is responsible for configuring and creating a SageMaker ModelStep
    that creates a PyTorch model from a trained model artifact.
    """

    def __init__(
        self,
        config: PyTorchModelStepConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model step.

        Args:
            config: A PyTorchModelStepConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Model.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, PyTorchModelStepConfig):
            raise ValueError(
                "PyTorchModelStepBuilder requires a PyTorchModelStepConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: PyTorchModelStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating PyTorchModelStepConfig...")
        
        # Validate required attributes
        required_attrs = [
            'instance_type',
            'entry_point',
            'source_dir'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PyTorchModelStepConfig missing required attribute: {attr}")
        
        # If not using PyTorch framework, image_uri is required
        if hasattr(self.config, 'use_pytorch_framework') and not self.config.use_pytorch_framework:
            if not hasattr(self.config, "image_uri") or not self.config.image_uri:
                raise ValueError("image_uri must be provided when use_pytorch_framework is False")
        
        logger.info("PyTorchModelStepConfig validation succeeded.")

    def _create_pytorch_model(self, model_data: str) -> PyTorchModel:
        """
        Creates and configures the PyTorchModel.
        This defines the model that will be deployed, including the model artifacts,
        inference code, and environment.

        Args:
            model_data: The S3 URI of the model artifacts.

        Returns:
            An instance of sagemaker.pytorch.PyTorchModel.
        """
        return PyTorchModel(
            model_data=model_data,
            role=self.role,
            entry_point=self.config.entry_point,
            source_dir=self.config.source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            image_uri=getattr(self.config, "image_uri", None),
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _create_model(self, model_data: str) -> Model:
        """
        Creates and configures a generic SageMaker Model.
        This is used when a custom image URI is provided instead of using the PyTorch framework.

        Args:
            model_data: The S3 URI of the model artifacts.

        Returns:
            An instance of sagemaker.model.Model.
        """
        return Model(
            image_uri=self.config.image_uri,
            model_data=model_data,
            role=self.role,
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the model.
        These variables are used to control the behavior of the inference code.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {}
        
        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)
            
        logger.info(f"Model environment variables: {env_vars}")
        return env_vars
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        input_reqs = {}
        
        # Add all input channel names from config
        for k, v in (self.config.input_names or {}).items():
            input_reqs[k] = f"S3 path for {v}"
        
        # Add other required parameters
        input_reqs["dependencies"] = self.COMMON_PROPERTIES["dependencies"]
        input_reqs["enable_caching"] = self.COMMON_PROPERTIES["enable_caching"]
        
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
        Match custom properties specific to PyTorchModel step.
        
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
                
                # Get the input key from config
                model_data_key = next(iter(self.config.input_names.keys()), "model_data")
                
                # Initialize inputs dict if needed
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                    
                # Add model artifacts to inputs
                if model_data_key not in inputs.get("inputs", {}):
                    inputs["inputs"][model_data_key] = model_artifacts
                    matched_inputs.add("inputs")
                    logger.info(f"Found model artifacts from TrainingStep: {getattr(prev_step, 'name', str(prev_step))}")
            except AttributeError as e:
                logger.warning(f"Could not extract model artifacts from step: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> CreateModelStep:
        """
        Creates the final, fully configured SageMaker ModelStep for the pipeline.
        This method orchestrates the assembly of the model and its configuration
        into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary mapping input channel names to their S3 locations
                - model_data: Direct parameter for model artifacts S3 URI (for backward compatibility)
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.
                
        Returns:
            A configured ModelStep instance.
        """
        logger.info("Creating PyTorch ModelStep...")

        # Extract parameters
        inputs_raw = self._extract_param(kwargs, 'inputs', {})
        model_data = self._extract_param(kwargs, 'model_data')
        dependencies = self._extract_param(kwargs, 'dependencies', [])
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Normalize inputs
        inputs = self._normalize_inputs(inputs_raw)
        
        # Add direct model_data parameter if provided
        if model_data is not None:
            inputs["model_data"] = model_data
            self.log_info("Using directly provided model_data: %s", model_data)
            
        # Look for inputs from dependencies if we don't have what we need
        if "model_data" not in inputs and dependencies:
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
                    
        # Get model_data from inputs
        model_data_key = next(iter(self.config.input_names.keys()), "model_data")
        model_data_value = inputs.get(model_data_key)
        
        # Validate required parameters
        if not model_data_value:
            raise ValueError(f"{model_data_key} must be provided")

        # Create the model
        if hasattr(self.config, 'use_pytorch_framework') and self.config.use_pytorch_framework:
            model = self._create_pytorch_model(model_data_value)
        else:
            model = self._create_model(model_data_value)

        step_name = self._get_step_name('PyTorchModel')
        
        model_step = CreateModelStep(
            name=step_name,
            step_args=model.create(
                instance_type=self.config.instance_type,
                accelerator_type=getattr(self.config, 'accelerator_type', None),
                tags=getattr(self.config, 'tags', None),
                model_name=self.config.get_model_name() if hasattr(self.config, 'get_model_name') else None
            ),
            depends_on=dependencies or []
        )
        logger.info(f"Created ModelStep with name: {model_step.name}")
        return model_step
