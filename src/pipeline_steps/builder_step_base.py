from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from pathlib import Path
import logging
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import Step
from sagemaker.workflow.steps import CacheConfig


from .config_base import BasePipelineConfig

logger = logging.getLogger(__name__)


STEP_NAMES = {
    'Base':                 'BaseStep',
    'Processing':           'ProcessingStep',
    'PytorchTraining':      'PytorchTrainingStep',
    'XGBoostTraining':      'XGBoostTrainingStep',
    'PytorchModel':         'CreatePytorchModelStep',
    'XGBoostModel':         'CreateXGBoostModelStep',
    'Package':              'PackagingStep',
    'Payload':              'PayloadTestStep',
    'Registration':         'RegistrationStep',
    'TabularPreprocessing': 'TabularPreprocessingStep',
    'CurrencyConversion':   'CurrencyConversionStep',
    'CradleDataLoading':    'CradleDataLoadingStep',
    'BatchTransform':       'BatchTransformStep',
    'XGBoostModelEval':     'XGBoostModelEvaluationStep',
    'PytorchModelEval':     'PytorchModelEvaluationStep',
    'HyperparameterPrep':   'HyperparameterPrepStep',
    }


class StepBuilderBase(ABC):
    """Base class for all step builders"""

    REGION_MAPPING: Dict[str, str] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }

    # Define standard step names
    STEP_NAMES = STEP_NAMES

    # Common properties that all steps might need
    COMMON_PROPERTIES = {
        "dependencies": "Optional list of dependent steps",
        "enable_caching": "Whether to enable caching for this step (default: True)"
    }
    
    # Standard output properties for training steps
    TRAINING_OUTPUT_PROPERTIES = {
        "training_job_name": "Name of the training job",
        "model_data": "S3 path to the model artifacts",
        "model_data_url": "S3 URL to the model artifacts"
    }
    
    # Standard output properties for model steps
    MODEL_OUTPUT_PROPERTIES = {
        "model_artifacts_path": "S3 path to model artifacts",
        "model": "SageMaker model object"
    }

    def __init__(
        self,
        config: BasePipelineConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize base step builder.
        
        Args:
            config: Model configuration
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
        """
        self.config = config
        self.session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()

        # Validate and set AWS region
        self.aws_region = self.REGION_MAPPING.get(self.config.region)
        if not self.aws_region:
            raise ValueError(
                f"Invalid region code: {self.config.region}. "
                f"Must be one of: {', '.join(self.REGION_MAPPING.keys())}"
            )

        logger.info(f"Initializing {self.__class__.__name__} with region: {self.config.region}")
        self.validate_configuration()

    def _sanitize_name_for_sagemaker(self, name: str, max_length: int = 63) -> str:
        """
        Sanitize a string to be a valid SageMaker resource name component.
        
        Args:
            name: Name to sanitize
            max_length: Maximum length of sanitized name
            
        Returns:
            Sanitized name
        """
        if not name:
            return "default-name"
        sanitized = "".join(c if c.isalnum() else '-' for c in str(name))
        sanitized = '-'.join(filter(None, sanitized.split('-')))
        return sanitized[:max_length].rstrip('-')

    def _get_step_name(self, step_type: str) -> str:
        """
        Get standard step name.
        
        Args:
            step_type: Type of step (e.g., 'Training', 'Model', 'Package')
            
        Returns:
            Standard step name
        """
        if step_type not in self.STEP_NAMES:
            logger.warning(f"Unknown step type: {step_type}. Using default name.")
            return f"Default{step_type}Step"
        return self.STEP_NAMES[step_type]

    def _get_cache_config(self, enable_caching: bool = True) -> Dict[str, Any]:
        """
        Get cache configuration for step.         
        ProcessingStep.to_request() can call .config safely.
        
        Args:
            enable_caching: Whether to enable caching
            
        Returns:
            Cache configuration dictionary
        """
        return CacheConfig(
            enable_caching=enable_caching,
            expire_after="P30D"
        )

    @abstractmethod
    def validate_configuration(self) -> None:
        """
        Validate configuration requirements.
        
        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        This method should return a dictionary mapping input parameter names to
        descriptions of what they represent. This helps the pipeline builder
        understand what inputs this step expects.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Base implementation returns input_names from config
        # Subclasses can override this to add additional input requirements
        return {k: v for k, v in (self.config.input_names or {}).items()}
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        This method should return a dictionary mapping output property names to
        descriptions of what they represent. This helps the pipeline builder
        understand what outputs this step provides to downstream steps.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Base implementation returns output_names from config
        # Subclasses can override this to add additional output properties
        return {k: v for k, v in (self.config.output_names or {}).items()}
    
    @abstractmethod
    def create_step(self, *args, **kwargs) -> Step:
        """
        Create pipeline step.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            SageMaker pipeline step
        """
        pass
