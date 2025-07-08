from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import CreateModelStep, Step
from sagemaker.pytorch import PyTorchModel
from sagemaker.model import Model

from .config_model_step_pytorch import PyTorchModelStepConfig
from .builder_step_base import StepBuilderBase
from ..pipeline_step_specs.pytorch_model_spec import PYTORCH_MODEL_SPEC

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
        
        # Validate specification availability
        if PYTORCH_MODEL_SPEC is None:
            raise ValueError("PyTorch model specification not available")
            
        super().__init__(
            config=config,
            spec=PYTORCH_MODEL_SPEC,  # Add specification
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

    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use specification dependencies to get model_data.
        
        Args:
            inputs: Dictionary of available inputs
            
        Returns:
            Dictionary containing processed inputs for model creation
        """
        # Spec defines: model_data dependency from PyTorchTraining, ProcessingStep, ModelArtifactsStep
        model_data_key = "model_data"  # From spec.dependencies
        
        if model_data_key not in inputs:
            raise ValueError(f"Required input '{model_data_key}' not found")
            
        return {model_data_key: inputs[model_data_key]}
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> str:
        """
        Use specification outputs - returns model name.
        
        Args:
            outputs: Dictionary to store outputs (not used for CreateModelStep)
            
        Returns:
            None - CreateModelStep handles outputs automatically
        """
        # Spec defines: model output with property_path="properties.ModelName"
        # For CreateModelStep, we don't need to return specific outputs
        # The step automatically provides ModelName property
        return None
    
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
        dependencies = self._extract_param(kwargs, 'dependencies', [])
        
        # Use dependency resolver to extract inputs
        if dependencies:
            extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        else:
            # Handle direct parameters for backward compatibility
            extracted_inputs = self._normalize_inputs(kwargs.get('inputs', {}))
            model_data = self._extract_param(kwargs, 'model_data')
            if model_data:
                extracted_inputs['model_data'] = model_data
        
        # Use specification-driven input processing
        model_inputs = self._get_inputs(extracted_inputs)
        model_data_value = model_inputs['model_data']

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
