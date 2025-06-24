from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable, Tuple, Set
from pathlib import Path
import logging
from inspect import signature
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
    
    # Common patterns for matching inputs to outputs
    # This can be extended by derived classes
    INPUT_PATTERNS = {
        "model": ["model", "model_data", "model_artifacts", "model_path"],
        "data": ["data", "dataset", "input_data", "training_data"],
        "output": ["output", "result", "artifacts", "s3_uri"]
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
    
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps.
        
        This method extracts the inputs required by this step from the dependency steps.
        It can be overridden by step builders that need to extract specific inputs.
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            Dictionary of inputs extracted from dependency steps
        """
        # Validate input
        if dependency_steps is None:
            logger.warning("No dependency steps provided to extract_inputs_from_dependencies")
            return {}
            
        # Base implementation looks for common patterns in dependency steps' outputs
        inputs = {}
        matched_inputs = set()
        
        # Get input requirements for this step
        input_requirements = self.get_input_requirements()
        
        if not input_requirements:
            logger.info(f"No input requirements defined for {self.__class__.__name__}")
            return inputs
            
        logger.info(f"Extracting inputs for {self.__class__.__name__} from {len(dependency_steps)} dependency steps")
        logger.debug(f"Input requirements: {list(input_requirements.keys())}")
        
        # Look for common patterns in dependency steps' outputs
        for i, prev_step in enumerate(dependency_steps):
            # Try to match inputs to outputs based on common patterns
            step_name = getattr(prev_step, 'name', f"Step_{i}")
            logger.debug(f"Attempting to match inputs from step: {step_name}")
            
            new_matches = self._match_inputs_to_outputs(inputs, input_requirements, prev_step)
            matched_inputs.update(new_matches)
            
        # Log which inputs were matched and which are still missing
        if matched_inputs:
            logger.info(f"Successfully matched inputs: {sorted(matched_inputs)}")
            
        missing_inputs = set(input_requirements.keys()) - matched_inputs
        if missing_inputs:
            # Filter out optional inputs
            required_missing = [name for name in missing_inputs 
                               if "optional" not in input_requirements[name].lower()]
            if required_missing:
                logger.warning(f"Could not match required inputs: {sorted(required_missing)}")
            else:
                logger.debug(f"Could not match optional inputs: {sorted(missing_inputs)}")
        
        return inputs
    
    
    def _match_inputs_to_outputs(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match input requirements with outputs from a dependency step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        if not input_requirements:
            return set()
            
        matched_inputs = set()
        
        # Get step name for better logging
        step_name = getattr(prev_step, 'name', str(prev_step))
        
        # Try different matching strategies
        matched_from_model = self._match_model_artifacts(inputs, input_requirements, prev_step)
        matched_inputs.update(matched_from_model)
        
        matched_from_processing = self._match_processing_outputs(inputs, input_requirements, prev_step)
        matched_inputs.update(matched_from_processing)
        
        # Try to match any custom properties
        matched_from_custom = self._match_custom_properties(inputs, input_requirements, prev_step)
        matched_inputs.update(matched_from_custom)
        
        if matched_inputs:
            logger.debug(f"Matched inputs from step {step_name}: {sorted(matched_inputs)}")
            
        return matched_inputs
        
    def _match_model_artifacts(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                              prev_step: Step) -> Set[str]:
        """
        Match model artifacts from a step to input requirements.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Check for model artifacts path (common in model steps)
        if hasattr(prev_step, "model_artifacts_path"):
            model_path = prev_step.model_artifacts_path
            logger.debug(f"Found model_artifacts_path: {model_path}")
            
            for input_name in input_requirements:
                if any(kw in input_name.lower() for kw in self.INPUT_PATTERNS["model"]):
                    inputs[input_name] = model_path
                    matched_inputs.add(input_name)
                    logger.debug(f"Matched input '{input_name}' to model_artifacts_path")
        
        return matched_inputs
        
    def _match_processing_outputs(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                 prev_step: Step) -> Set[str]:
        """
        Match processing outputs from a step to input requirements.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Check for processing output (common in processing steps)
        if not (hasattr(prev_step, "properties") and 
                hasattr(prev_step.properties, "ProcessingOutputConfig")):
            return matched_inputs
            
        try:
            # Check if outputs are accessible
            if not hasattr(prev_step.properties.ProcessingOutputConfig, "Outputs"):
                return matched_inputs
                
            outputs = prev_step.properties.ProcessingOutputConfig.Outputs
            if not hasattr(outputs, "__getitem__"):
                return matched_inputs
                
            # Try to match list-like outputs
            matched_from_list = self._match_list_outputs(inputs, input_requirements, outputs)
            matched_inputs.update(matched_from_list)
            
            # Try to match dict-like outputs
            matched_from_dict = self._match_dict_outputs(inputs, input_requirements, outputs)
            matched_inputs.update(matched_from_dict)
                
        except (AttributeError, IndexError) as e:
            logger.warning(f"Error matching processing outputs: {e}")
            
        return matched_inputs
        
    def _match_list_outputs(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                           outputs) -> Set[str]:
        """
        Match list-like outputs to input requirements.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            outputs: List-like outputs object
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        try:
            # Try numeric index (list-like)
            s3_uri = outputs[0].S3Output.S3Uri
            logger.debug(f"Found list output S3Uri: {s3_uri}")
            
            # Match to appropriate input based on patterns
            for input_name in input_requirements:
                if any(kw in input_name.lower() for kw in self.INPUT_PATTERNS["output"]):
                    inputs[input_name] = s3_uri
                    matched_inputs.add(input_name)
                    logger.debug(f"Matched input '{input_name}' to list output")
        except (IndexError, TypeError, AttributeError):
            # Not a list or no S3Output
            pass
            
        return matched_inputs
        
    def _match_dict_outputs(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                           outputs) -> Set[str]:
        """
        Match dictionary-like outputs to input requirements.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            outputs: Dictionary-like outputs object
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        try:
            # Try string keys (dict-like)
            for key in outputs:
                output = outputs[key]
                if not (hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri")):
                    continue
                    
                s3_uri = output.S3Output.S3Uri
                logger.debug(f"Found dict output '{key}' S3Uri: {s3_uri}")
                
                # Match to appropriate input based on key and patterns
                for input_name in input_requirements:
                    # Direct key match
                    if key.lower() in input_name.lower():
                        inputs[input_name] = s3_uri
                        matched_inputs.add(input_name)
                        logger.debug(f"Matched input '{input_name}' to output key '{key}'")
                        continue
                        
                    # Pattern-based match
                    for pattern_type, keywords in self.INPUT_PATTERNS.items():
                        if any(kw in input_name.lower() and kw in key.lower() for kw in keywords):
                            inputs[input_name] = s3_uri
                            matched_inputs.add(input_name)
                            logger.debug(f"Matched input '{input_name}' to output key '{key}' via pattern '{pattern_type}'")
                            break
        except (TypeError, AttributeError):
            # Not a dict or iteration error
            pass
            
        return matched_inputs
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties from a step to input requirements.
        This is a hook for derived classes to implement custom matching logic.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        # Base implementation doesn't match any custom properties
        # Derived classes can override this to implement custom matching logic
        return set()
    
    @abstractmethod
    def create_step(self, **kwargs) -> Step:
        """
        Create pipeline step.
        
        This method should be implemented by all step builders to create a SageMaker pipeline step.
        It accepts a dictionary of keyword arguments that can be used to configure the step.
        
        Common parameters that all step builders should handle:
        - dependencies: Optional list of steps that this step depends on
        - enable_caching: Whether to enable caching for this step (default: True)
        
        Step-specific parameters should be extracted from kwargs as needed.
        
        Args:
            **kwargs: Keyword arguments for configuring the step
            
        Returns:
            SageMaker pipeline step
        """
        pass
    
    def _filter_kwargs(self, func: Callable, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter kwargs to only include parameters accepted by the function.
        
        Args:
            func: Function to filter kwargs for
            kwargs: Dictionary of keyword arguments
            
        Returns:
            Filtered dictionary of keyword arguments
        """
        # Validate inputs
        if func is None:
            logger.warning("Cannot filter kwargs: function is None")
            return {}
            
        if kwargs is None:
            logger.warning("Cannot filter kwargs: kwargs is None")
            return {}
            
        try:
            params = signature(func).parameters
            
            # Create filtered dict and track which keys were filtered out
            filtered_kwargs = {}
            filtered_out = []
            
            for k, v in kwargs.items():
                if k in params:
                    filtered_kwargs[k] = v
                else:
                    filtered_out.append(k)
                    
            if filtered_out:
                logger.debug(f"Filtered out kwargs: {filtered_out}")
                
            return filtered_kwargs
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Error filtering kwargs: {e}")
            return {}
    
    def build(self, dependency_steps: List[Step]) -> Step:
        """
        Build a pipeline step with appropriate inputs from dependencies.
        
        This method combines the extract_inputs_from_dependencies and create_step methods
        to create a pipeline step with the appropriate inputs from dependency steps.
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            SageMaker pipeline step
        """
        # Validate input
        if dependency_steps is None:
            dependency_steps = []
            logger.warning("No dependency steps provided to build method, using empty list")
            
        logger.info(f"Building step for {self.__class__.__name__} with {len(dependency_steps)} dependencies")
        
        # Extract inputs from dependency steps
        kwargs = self.extract_inputs_from_dependencies(dependency_steps)
        
        # Auto-inject the dependencies list and default cache flag
        kwargs.setdefault("dependencies", dependency_steps)
        kwargs.setdefault("enable_caching", True)
        
        # Filter kwargs to only include parameters accepted by create_step
        filtered_kwargs = self._filter_kwargs(self.create_step, kwargs)
        
        # Check for missing required inputs
        missing_inputs = self._check_missing_inputs(filtered_kwargs)
        if missing_inputs:
            error_msg = f"Missing required inputs for {self.__class__.__name__}: {missing_inputs}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Create the step with filtered kwargs
        logger.info(f"Creating step for {self.__class__.__name__} with parameters: {list(filtered_kwargs.keys())}")
        return self.create_step(**filtered_kwargs)
    
    def _check_missing_inputs(self, kwargs: Dict[str, Any]) -> List[str]:
        """
        Check for missing required inputs.
        
        Args:
            kwargs: Dictionary of keyword arguments
            
        Returns:
            List of missing required inputs
        """
        # Validate input
        if kwargs is None:
            logger.warning("Cannot check missing inputs: kwargs is None")
            return []
            
        # Get input requirements for this step
        input_requirements = self.get_input_requirements()
        if not input_requirements:
            return []
            
        # Check for missing required inputs
        missing_inputs = []
        for input_name, input_desc in input_requirements.items():
            # Skip optional inputs (those with "optional" in the description)
            if input_desc and "optional" in input_desc.lower():
                continue
            
            # Check if input is missing
            if input_name not in kwargs:
                missing_inputs.append(input_name)
                
        return missing_inputs
    
    def _extract_param(self, kwargs: Dict[str, Any], param_name: str, default=None):
        """
        Extract a parameter from kwargs with a default value if not present.
        
        Args:
            kwargs: Dictionary of keyword arguments
            param_name: Name of the parameter to extract
            default: Default value to use if parameter is not present
            
        Returns:
            Value of the parameter or default value
        """
        # Validate input
        if kwargs is None:
            logger.debug(f"Cannot extract parameter '{param_name}': kwargs is None, using default")
            return default
            
        value = kwargs.get(param_name, default)
        if value is None and default is not None:
            logger.debug(f"Parameter '{param_name}' not found in kwargs, using default")
            
        return value
