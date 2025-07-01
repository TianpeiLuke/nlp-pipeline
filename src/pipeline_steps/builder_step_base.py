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


def safe_value_for_logging(value):
    """
    Safely format a value for logging, handling Pipeline variables appropriately.
    
    Args:
        value: Any value that might be a Pipeline variable
        
    Returns:
        A string representation safe for logging
    """
    # Check if it's a Pipeline variable or has the expr attribute
    if hasattr(value, 'expr'):
        return f"[Pipeline Variable: {value.__class__.__name__}]"
    
    # Handle collections containing Pipeline variables
    if isinstance(value, dict):
        return "{...}"  # Avoid iterating through dict values which might contain Pipeline variables
    if isinstance(value, (list, tuple, set)):
        return f"[{type(value).__name__} with {len(value)} items]" 
    
    # For simple values, return the string representation
    try:
        return str(value)
    except Exception:
        return f"[Object of type: {type(value).__name__}]"


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
    """
    Base class for all step builders
    
    ## Safe Logging Methods
    
    To handle Pipeline variables safely in logs, use these methods:
    
    ```python
    # Instead of:
    logger.info(f"Using input path: {input_path}")  # May raise TypeError for Pipeline variables
    
    # Use:
    self.log_info("Using input path: %s", input_path)  # Handles Pipeline variables safely
    ```
    
    Standard Pattern for `input_names` and `output_names`:
    
    1. In **config classes**:
       ```python
       output_names = {"logical_name": "DescriptiveValue"}  # VALUE used as key in outputs dict
       input_names = {"logical_name": "ScriptInputName"}    # KEY used as key in inputs dict
       ```
    
    2. In **pipeline code**:
       ```python
       # Get output using VALUE from output_names
       output_value = step_a.config.output_names["logical_name"]
       output_uri = step_a.properties.ProcessingOutputConfig.Outputs[output_value].S3Output.S3Uri
       
       # Set input using KEY from input_names
       inputs = {"logical_name": output_uri}
       ```
    
    3. In **step builders**:
       ```python
       # For outputs - validate using VALUES
       value = self.config.output_names["logical_name"]
       if value not in outputs:
           raise ValueError(f"Must supply an S3 URI for '{value}'")
           
       # For inputs - validate using KEYS
       for logical_name in self.config.input_names.keys():
           if logical_name not in inputs:
               raise ValueError(f"Must supply an S3 URI for '{logical_name}'")
       ```
    
    Developers should follow this standard pattern when creating new step builders.
    The base class provides helper methods to enforce and simplify this pattern:
    
    - `_validate_inputs()`: Validates inputs using KEYS from input_names
    - `_validate_outputs()`: Validates outputs using VALUES from output_names
    - `_get_script_input_name()`: Maps logical name to script input name
    - `_get_output_destination_name()`: Maps logical name to output destination name
    - `_create_standard_processing_input()`: Creates standardized ProcessingInput
    - `_create_standard_processing_output()`: Creates standardized ProcessingOutput
    
    Property Path Registry:
    
    To bridge the gap between definition-time and runtime, step builders can register
    property paths that define how to access their outputs at runtime. This solves the
    issue where outputs are defined statically but only accessible via specific runtime paths.
    
    - `register_property_path()`: Registers a property path for a logical output name
    - `get_property_paths()`: Gets all registered property paths for this step
    """

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
    
    # Class-level property path registry
    # Maps step types to dictionaries of {logical_name: property_path}
    _PROPERTY_PATH_REGISTRY = {}

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

        # Initialize instance-specific property paths
        self._instance_property_paths = {}

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
        
    @classmethod
    def register_property_path(cls, step_type: str, logical_name: str, property_path: str):
        """
        Register a runtime property path for a step type and logical name.
        
        This classmethod registers how to access a specific output at runtime
        by mapping a step type and logical output name to a property path.
        
        Args:
            step_type (str): The type of step (e.g., 'XGBoostTrainingStep')
            logical_name (str): Logical name of the output (KEY in output_names)
            property_path (str): Runtime property path to access this output
                               Can include placeholders like {output_descriptor}
        
        Example:
            ```python
            # Register how to access model artifacts from XGBoostTrainingStep
            StepBuilderBase.register_property_path(
                "XGBoostTrainingStep",
                "model_output",
                "properties.ModelArtifacts.S3ModelArtifacts"
            )
            
            # Register how to access processing outputs with placeholders
            StepBuilderBase.register_property_path(
                "ProcessingStep",
                "data_output",
                "properties.ProcessingOutputConfig.Outputs['{output_descriptor}'].S3Output.S3Uri"
            )
            ```
        """
        if step_type not in cls._PROPERTY_PATH_REGISTRY:
            cls._PROPERTY_PATH_REGISTRY[step_type] = {}
        
        cls._PROPERTY_PATH_REGISTRY[step_type][logical_name] = property_path
        logger.debug(f"Registered property path for {step_type}.{logical_name}: {property_path}")
        
    def register_instance_property_path(self, logical_name: str, property_path: str):
        """
        Register a property path specific to this instance.
        
        This instance method registers how to access a specific output at runtime
        for this specific instance of a step builder. This is useful for dynamic paths
        that depend on instance configuration.
        
        Args:
            logical_name (str): Logical name of the output (KEY in output_names)
            property_path (str): Runtime property path to access this output
        
        Example:
            ```python
            # In __init__ method of a custom step builder
            self.register_instance_property_path(
                "model_output",
                f"properties.{self.config.custom_output_property}"
            )
            ```
        """
        self._instance_property_paths[logical_name] = property_path
        logger.debug(f"Registered instance property path for {logical_name}: {property_path}")
        
    def get_property_paths(self) -> Dict[str, str]:
        """
        Get the runtime property paths registered for this step type.
        
        Returns:
            dict: Mapping from logical output names to runtime property paths
        """
        # Get the step type from the class name
        step_type = self.__class__.__name__.replace("Builder", "Step")
        return self._PROPERTY_PATH_REGISTRY.get(step_type, {})
        
    def get_all_property_paths(self) -> Dict[str, str]:
        """
        Get all property paths for this step, combining class-level and instance-level.
        
        Returns:
            dict: Combined mapping from logical output names to runtime property paths
        """
        # Start with class-level paths
        paths = self.get_property_paths().copy()
        
        # Override with instance-specific paths
        paths.update(self._instance_property_paths)
        
        return paths
        
    def log_info(self, message, *args, **kwargs):
        """
        Safely log info messages, handling Pipeline variables.
        
        Args:
            message: The log message
            *args, **kwargs: Values to format into the message
        """
        try:
            # Convert args and kwargs to safe strings
            safe_args = [safe_value_for_logging(arg) for arg in args]
            safe_kwargs = {k: safe_value_for_logging(v) for k, v in kwargs.items()}
            
            # Log with safe values
            logger.info(message, *safe_args, **safe_kwargs)
        except Exception as e:
            logger.info(f"Original logging failed ({e}), logging raw message: {message}")
    
    def log_debug(self, message, *args, **kwargs):
        """Debug version of safe logging"""
        try:
            safe_args = [safe_value_for_logging(arg) for arg in args]
            safe_kwargs = {k: safe_value_for_logging(v) for k, v in kwargs.items()}
            logger.debug(message, *safe_args, **safe_kwargs)
        except Exception as e:
            logger.debug(f"Original logging failed ({e}), logging raw message: {message}")
    
    def log_warning(self, message, *args, **kwargs):
        """Warning version of safe logging"""
        try:
            safe_args = [safe_value_for_logging(arg) for arg in args]
            safe_kwargs = {k: safe_value_for_logging(v) for k, v in kwargs.items()}
            logger.warning(message, *safe_args, **safe_kwargs)
        except Exception as e:
            logger.warning(f"Original logging failed ({e}), logging raw message: {message}")

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
        else:
            # Log the parameter type to help with debugging
            param_type = type(value).__name__
            if isinstance(value, dict):
                logger.debug(f"Extracted '{param_name}': {param_type} with keys: {list(value.keys())}")
            elif isinstance(value, list):
                logger.debug(f"Extracted '{param_name}': {param_type} with length: {len(value)}")
            else:
                if param_name not in ['model_input', 'inference_scripts_input']:  # Don't log possibly sensitive data
                    logger.debug(f"Extracted '{param_name}': {param_type}")
                else:
                    logger.debug(f"Extracted '{param_name}': {param_type} (value redacted)")
            
        return value
        
    def _normalize_inputs(self, inputs_raw: Any) -> Dict[str, Any]:
        """
        Normalize inputs to a flat dictionary format.
        
        This method handles different input structures:
        1. None -> Empty dictionary
        2. Flat dictionary -> Used as is
        3. Dictionary with nested 'inputs' field -> Merged with outer keys
        
        Args:
            inputs_raw: Raw inputs object, typically from kwargs
            
        Returns:
            Normalized flat dictionary of inputs
        """
        inputs = {}
        
        if not inputs_raw:
            return inputs
            
        if isinstance(inputs_raw, dict):
            # Copy direct key-value pairs (excluding 'inputs' field)
            inputs.update({k: v for k, v in inputs_raw.items() if k != "inputs"})
            
            # Handle nested "inputs" field if present
            if "inputs" in inputs_raw and isinstance(inputs_raw["inputs"], dict):
                for k, v in inputs_raw["inputs"].items():
                    if k not in inputs:
                        inputs[k] = v
        
        return inputs
    
    def _check_missing_inputs(self, inputs: Dict[str, Any], check_input_names: bool = False) -> List[str]:
        """
        Check for missing required inputs.
        
        This method can check inputs against either:
        1. input_requirements from get_input_requirements() (default)
        2. input_names keys from config.input_names (when check_input_names=True)
        
        Args:
            inputs: Dictionary of input values
            check_input_names: If True, check against config.input_names keys
                              If False, check against get_input_requirements()
        
        Returns:
            List of missing required input names
        """
        # Validate input
        if inputs is None:
            logger.warning("Cannot check missing inputs: inputs is None")
            return []
            
        missing_inputs = []
        
        if check_input_names:
            # Check against config.input_names keys
            if hasattr(self.config, 'input_names') and self.config.input_names:
                for input_name in self.config.input_names.keys():
                    if input_name not in inputs:
                        missing_inputs.append(input_name)
            else:
                logger.debug("config.input_names is not defined or empty")
        else:
            # Check against input_requirements (original behavior)
            input_requirements = self.get_input_requirements()
            if not input_requirements:
                return []
                
            for input_name, input_desc in input_requirements.items():
                # Skip optional inputs (those with "optional" in the description)
                if input_desc and "optional" in input_desc.lower():
                    continue
                
                # Check if input is missing
                if input_name not in inputs:
                    missing_inputs.append(input_name)
                    
        return missing_inputs
    
    def _validate_inputs(self, inputs: Dict[str, Any], raise_error: bool = True) -> bool:
        """
        Validate that all required inputs are present.
        
        This method checks that all keys in config.input_names are present in inputs.
        
        Args:
            inputs: Dictionary of inputs to validate
            raise_error: Whether to raise an error if inputs are missing
            
        Returns:
            True if all required inputs are present, False otherwise
            
        Raises:
            ValueError: If required inputs are missing and raise_error=True
        """
        missing = self._check_missing_inputs(inputs, check_input_names=True)
        
        if missing and raise_error:
            raise ValueError(f"Missing required inputs: {', '.join(missing)}")
            
        return len(missing) == 0
    
    def _get_script_input_name(self, logical_name: str) -> str:
        """
        Get the script input name for a logical input name.
        
        Args:
            logical_name: Logical input name (key in input_names)
            
        Returns:
            Script input name (value in input_names) or logical_name if not found
            
        Example:
            If config.input_names = {"data": "InputData"},
            _get_script_input_name("data") returns "InputData"
        """
        if not hasattr(self.config, 'input_names') or not self.config.input_names:
            return logical_name
            
        return self.config.input_names.get(logical_name, logical_name)
    
    def _get_output_destination_name(self, logical_name: str) -> str:
        """
        Get the output destination name (VALUE) for a logical output name.
        
        Args:
            logical_name: Logical output name (key in output_names)
            
        Returns:
            Output destination name (value in output_names) or logical_name if not found
            
        Example:
            If config.output_names = {"results": "ProcessedData"},
            _get_output_destination_name("results") returns "ProcessedData"
        """
        if not hasattr(self.config, 'output_names') or not self.config.output_names:
            return logical_name
            
        return self.config.output_names.get(logical_name, logical_name)
        
    def _validate_outputs(self, outputs: Dict[str, Any], raise_error: bool = True) -> bool:
        """
        Validate that all required outputs are present, using VALUES from output_names.
        
        This enhanced method is more resilient to different output types and structures,
        including Join objects and other non-standard dictionary-like structures.
        
        Args:
            outputs: Dictionary of outputs to validate (may be a Join object or other type)
            raise_error: Whether to raise an error if outputs are missing
            
        Returns:
            True if all required outputs are present, False otherwise
            
        Raises:
            ValueError: If required outputs are missing and raise_error is True
        """
        if not outputs:
            if raise_error:
                raise ValueError("outputs must not be empty")
            return False
        
        # Track missing outputs
        missing = []
        output_keys = set()
        
        # Collect all keys from outputs, handling potential Join objects
        try:
            # First try to get keys directly
            if hasattr(outputs, 'keys'):
                output_keys = set(outputs.keys())
                logger.debug(f"Extracted keys from outputs: {output_keys}")
            # If outputs is a string (single key), add it
            elif isinstance(outputs, str):
                output_keys.add(outputs)
                logger.debug(f"Added string output as key: {outputs}")
            # If output is an iterable with items() method, extract keys
            elif hasattr(outputs, 'items'):
                try:
                    for k, _ in outputs.items():
                        output_keys.add(k)
                    logger.debug(f"Extracted keys from items(): {output_keys}")
                except (AttributeError, TypeError):
                    logger.debug("Could not extract keys using items()")
            # Try converting to string and search for output values
            else:
                outputs_str = str(outputs)
                logger.debug(f"Using string representation of outputs: {outputs_str[:100]}...")
        except Exception as e:
            logger.debug(f"Exception when extracting keys: {e}")
        
        # If we have output_names, validate that all required outputs are present
        if hasattr(self.config, 'output_names') and self.config.output_names:
            for logical_name, output_value in self.config.output_names.items():
                # Check if the output value exists as a key in outputs
                if output_value not in output_keys:
                    # Try additional matching strategies
                    found = False
                    
                    # Check for string representation match
                    try:
                        outputs_str = str(outputs)
                        if output_value in outputs_str:
                            logger.info(f"Found output '{output_value}' in string representation of outputs")
                            found = True
                    except:
                        pass
                        
                    # Check if outputs is the value we're looking for
                    if not found and outputs == output_value:
                        logger.info(f"outputs object equals '{output_value}'")
                        found = True
                        
                    # Check if the outputs object has an attribute with the output name
                    if not found and hasattr(outputs, output_value):
                        logger.info(f"outputs object has attribute '{output_value}'")
                        found = True
                    
                    # If still not found, consider it missing
                    if not found:
                        missing.append(output_value)
                        logger.warning(f"Missing required output: '{output_value}' (logical name: '{logical_name}')")
        
        # If we're missing any required outputs, log or raise an error
        if missing and raise_error:
            error_msg = f"Missing required outputs: {', '.join(missing)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        return len(missing) == 0
        
    def _create_standard_processing_input(self, logical_name: str, inputs: Dict[str, Any], 
                                          destination: str = None, **kwargs) -> Any:
        """
        Create a standard ProcessingInput for the given logical name.
        
        Args:
            logical_name: Logical input name (key in input_names)
            inputs: Dictionary of inputs containing logical names as keys
            destination: Optional destination path (if None, uses standard path)
            **kwargs: Additional keyword arguments to pass to ProcessingInput
                     (e.g., s3_data_distribution_type, s3_input_mode, etc.)
            
        Returns:
            A ProcessingInput object
            
        Raises:
            ValueError: If the logical name is not in inputs
            
        Example:
            _create_standard_processing_input("data", inputs, "/opt/ml/processing/input/data")
            
        Advanced Example:
            _create_standard_processing_input(
                "model", inputs, "/opt/ml/processing/input/model",
                s3_data_distribution_type="FullyReplicated",
                s3_input_mode="File"
            )
        """
        # Import ProcessingInput here to avoid circular imports
        from sagemaker.processing import ProcessingInput
        
        if logical_name not in inputs:
            raise ValueError(f"Input '{logical_name}' not found in inputs dictionary")
        
        script_input_name = self._get_script_input_name(logical_name)
        
        # If no destination specified, create a standard path
        if destination is None:
            destination = f"/opt/ml/processing/input/{logical_name.lower()}"
            
        return ProcessingInput(
            input_name=script_input_name,
            source=inputs[logical_name],
            destination=destination,
            **kwargs  # Forward additional arguments to ProcessingInput
        )
    
    def _create_standard_processing_output(self, logical_name: str, outputs: Dict[str, Any],
                                           source: str = None) -> Any:
        """
        Create a standard ProcessingOutput for the given logical name.
        
        This enhanced method is more resilient to different output types and structures,
        including Join objects and other non-standard dictionary-like structures.
        
        Args:
            logical_name: Logical output name (key in output_names)
            outputs: Dictionary of outputs containing destination values as keys 
            source: Optional source path (if None, uses standard path)
            
        Returns:
            A ProcessingOutput object
            
        Raises:
            ValueError: If the output destination name is not in outputs
            
        Example:
            _create_standard_processing_output("results", outputs, "/opt/ml/processing/output")
        """
        # Import ProcessingOutput here to avoid circular imports
        from sagemaker.processing import ProcessingOutput
        
        # Get the output destination name (VALUE) for this logical name
        output_dest_name = self._get_output_destination_name(logical_name)
        
        # Initialize destination
        destination = None
        
        # Try different strategies to find the destination
        try:
            # First, try direct dictionary access
            if hasattr(outputs, 'get') or hasattr(outputs, '__getitem__'):
                if output_dest_name in outputs:
                    destination = outputs[output_dest_name]
                    logger.debug(f"Found destination for '{output_dest_name}' using direct access")
            
            # If destination is still None, try string representation searching
            if destination is None:
                # Check if the outputs contains this value as a string
                outputs_str = str(outputs)
                if output_dest_name in outputs_str:
                    # As a fallback, construct a path similar to _generate_outputs
                    step_type = BasePipelineConfig.get_step_name(type(self.config).__name__)
                    job_type = getattr(self.config, "job_type", "")
                    
                    # Similar pattern to _generate_outputs
                    if hasattr(self.config, 'pipeline_s3_loc'):
                        base_s3_loc = self.config.pipeline_s3_loc
                        path_parts = [base_s3_loc, step_type.lower()]
                        if job_type:
                            path_parts.append(job_type)
                        base_path = "/".join([p for p in path_parts if p])
                        
                        # Use logical_name for the path since it maps to output_dest_name
                        destination = f"{base_path}/{logical_name}"
                        logger.info(f"Auto-generated destination for '{output_dest_name}': {destination}")
        except Exception as e:
            logger.warning(f"Error trying to find destination for '{output_dest_name}': {e}")
        
        # If still not found, raise an error
        if destination is None:
            raise ValueError(f"Output destination '{output_dest_name}' not found in outputs and couldn't be generated")
        
        # If no source specified, create a standard path
        if source is None:
            source = f"/opt/ml/processing/output/{logical_name.lower()}"
            
        return ProcessingOutput(
            output_name=output_dest_name, 
            source=source,
            destination=destination
        )

    @classmethod
    def validate_builder_pattern_compliance(cls, builder_cls) -> Dict[str, Any]:
        """
        Validate that a step builder class follows the standard input/output naming pattern.
        
        Args:
            builder_cls: The step builder class to validate
            
        Returns:
            Dictionary with validation results
            
        Example:
            ```python
            from src.pipeline_steps.builder_step_base import StepBuilderBase
            from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
            
            results = StepBuilderBase.validate_builder_pattern_compliance(MIMSPackagingStepBuilder)
            if results['compliant']:
                print("Builder is compliant")
            else:
                print(f"Builder is not compliant: {results['issues']}")
            ```
        """
        import inspect
        
        results = {
            "compliant": True,
            "issues": [],
            "checked_methods": []
        }
        
        # Check if the class inherits from StepBuilderBase
        if not issubclass(builder_cls, cls):
            results["issues"].append(f"Class {builder_cls.__name__} does not inherit from StepBuilderBase")
            results["compliant"] = False
            return results
        
        methods_to_check = [
            ("_validate_inputs", "Validate inputs using KEYS"),
            ("_validate_outputs", "Validate outputs using VALUES"),
            ("_get_script_input_name", "Maps logical name to script input name"),
            ("_get_output_destination_name", "Maps logical name to output destination name")
        ]
        
        # Check each required method
        for method_name, description in methods_to_check:
            results["checked_methods"].append(method_name)
            
            # Check if method exists
            if not hasattr(builder_cls, method_name):
                results["issues"].append(f"Missing method: {method_name} ({description})")
                results["compliant"] = False
                continue
                
            # Get the method implementation
            method = getattr(builder_cls, method_name)
            
            # Check if the method is inherited or overridden
            if method.__qualname__ == f"StepBuilderBase.{method_name}":
                # Method is inherited from StepBuilderBase (good - using standard implementation)
                continue
            
            # Method is overridden - check if it follows the standard pattern
            source = inspect.getsource(method)
            
            if method_name == "_validate_inputs":
                # Should use logical names (KEYS) for input validation
                if "input_names" in source and "keys()" not in source and "self._check_missing_inputs" not in source:
                    results["issues"].append(
                        f"Method {method_name} may not validate using KEYS from input_names")
                    results["compliant"] = False
                    
            elif method_name == "_validate_outputs":
                # Should use values from output_names for output validation
                if "output_names" in source and "values()" in source:
                    results["issues"].append(
                        f"Method {method_name} may be using VALUES with values() instead of direct VALUES")
                    results["compliant"] = False
        
        # All standard validation methods passed
        return results
