from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import ModelStep, Step
from sagemaker.pytorch import PyTorchModel
from sagemaker.model import Model

from .config_model_step_pytorch import ModelStepPyTorchConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class PyTorchModelStepBuilder(StepBuilderBase):
    """
    Builder for a PyTorch Model Step.
    This class is responsible for configuring and creating a SageMaker ModelStep
    that creates a PyTorch model from a trained model artifact.
    """

    def __init__(
        self,
        config: ModelStepPyTorchConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model step.

        Args:
            config: A ModelStepPyTorchConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Model.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, ModelStepPyTorchConfig):
            raise ValueError(
                "PyTorchModelStepBuilder requires a ModelStepPyTorchConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: ModelStepPyTorchConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating ModelStepPyTorchConfig...")
        
        # Validate required attributes
        required_attrs = [
            'model_name',
            'image_uri',
            'instance_type',
            'entry_point',
            'source_dir'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelStepPyTorchConfig missing required attribute: {attr}")
        
        logger.info("ModelStepPyTorchConfig validation succeeded.")

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
            image_uri=self.config.image_uri,
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
        # Get input requirements
        input_reqs = {
            "model_data": "S3 URI of the model artifacts",
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
        # Define the output properties for the model step
        output_props = {
            "ModelName": "Name of the created SageMaker model"
        }
        return output_props
        
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
                if "model_data" in input_requirements:
                    inputs["model_data"] = model_artifacts
                    matched_inputs.add("model_data")
                    logger.info(f"Found model artifacts from TrainingStep: {getattr(prev_step, 'name', str(prev_step))}")
            except AttributeError as e:
                logger.warning(f"Could not extract model artifacts from step: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> ModelStep:
        """
        Creates the final, fully configured SageMaker ModelStep for the pipeline.
        This method orchestrates the assembly of the model and its configuration
        into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - model_data: The S3 URI of the model artifacts.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.ModelStep instance.
        """
        logger.info("Creating PyTorch ModelStep...")

        # Extract parameters
        model_data = self._extract_param(kwargs, 'model_data')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Validate required parameters
        if not model_data:
            raise ValueError("model_data must be provided")

        # Create the model
        if self.config.use_pytorch_framework:
            model = self._create_pytorch_model(model_data)
        else:
            model = self._create_model(model_data)

        step_name = self._get_step_name('PyTorchModel')
        
        model_step = ModelStep(
            name=step_name,
            step_args=model.create(
                instance_type=self.config.instance_type,
                accelerator_type=self.config.accelerator_type,
                tags=self.config.tags,
                model_name=self.config.model_name
            ),
            depends_on=dependencies or []
        )
        logger.info(f"Created ModelStep with name: {model_step.name}")
        return model_step
