"""
Legacy compatibility module for deprecated step builder methods.

This module contains deprecated methods that were previously part of StepBuilderBase.
These methods are maintained here for backward compatibility and will be removed
in a future version. New code should use the specification-driven approach instead.

## Deprecation Notice

All methods in this module are deprecated and will be removed in a future version.
Use the specification-driven approach instead:

- Use OutputSpec.property_path in specifications instead of property path registration
- Use UnifiedDependencyResolver for dependency resolution
- Use direct specification access methods for input/output information
"""

import warnings
import logging
from typing import Dict, List, Any, Optional, Set
from sagemaker.workflow.steps import Step

logger = logging.getLogger(__name__)


class PropertyPathRegistryMixin:
    """
    Mixin for legacy property path registration methods.
    
    This mixin provides backward compatibility for step builders that use the
    property path registration methods. These methods are deprecated and will
    be removed in a future version.
    
    Migration guide:
    
    Instead of:
    ```python
    StepBuilderBase.register_property_path(
        "XGBoostTrainingStep", 
        "model_output", 
        "properties.ModelArtifacts.S3ModelArtifacts"
    )
    ```
    
    Use OutputSpec in step specification:
    ```python
    OutputSpec(
        logical_name="model_output",
        property_path="properties.ModelArtifacts.S3ModelArtifacts",
        ...
    )
    ```
    """
    
    # Class-level property path registry
    # Maps step types to dictionaries of {logical_name: property_path}
    _PROPERTY_PATH_REGISTRY: Dict[str, Dict[str, str]] = {}
    
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
                               
        Deprecated:
            This method is deprecated. Use OutputSpec.property_path in step
            specifications instead.
        """
        warnings.warn(
            "register_property_path() is deprecated and will be removed in a future version. "
            "Use OutputSpec.property_path in step specifications instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            
        Deprecated:
            This method is deprecated. Use OutputSpec.property_path in step
            specifications instead.
        """
        warnings.warn(
            "register_instance_property_path() is deprecated and will be removed in a future version. "
            "Use OutputSpec.property_path in step specifications instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not hasattr(self, "_instance_property_paths"):
            self._instance_property_paths = {}
            
        self._instance_property_paths[logical_name] = property_path
        logger.debug(f"Registered instance property path for {logical_name}: {property_path}")
    
    def get_property_paths(self) -> Dict[str, str]:
        """
        Get the runtime property paths registered for this step type.
        
        Returns:
            dict: Mapping from logical output names to runtime property paths
            
        Deprecated:
            This method is deprecated. Use get_property_path() or access
            specification.outputs directly.
        """
        warnings.warn(
            "get_property_paths() is deprecated and will be removed in a future version. "
            "Use get_property_path() or access specification.outputs directly.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # First try to use specification outputs if available
        if hasattr(self, "spec") and self.spec and hasattr(self.spec, 'outputs'):
            paths = {}
            for _, output_spec in self.spec.outputs.items():
                if output_spec.property_path:
                    paths[output_spec.logical_name] = output_spec.property_path
            
            # If we found paths in the specification, return them
            if paths:
                return paths
                
        # Fall back to registry
        step_type = self.__class__.__name__.replace("Builder", "Step")
        registry_paths = self._PROPERTY_PATH_REGISTRY.get(step_type, {})
        
        # Combine with instance-specific paths if available
        result = registry_paths.copy()
        if hasattr(self, "_instance_property_paths"):
            result.update(self._instance_property_paths)
            
        return result


class DependencyMatchingMixin:
    """
    Mixin for legacy dependency matching methods.
    
    This mixin provides backward compatibility for step builders that use the
    traditional dependency matching methods. These methods are deprecated and will
    be removed in a future version.
    
    Migration guide:
    
    Instead of using these matching methods, use the UnifiedDependencyResolver:
    
    ```python
    from ..pipeline_deps.dependency_resolver import UnifiedDependencyResolver
    
    resolver = UnifiedDependencyResolver()
    resolver.register_specification(step_name, self.spec)
    
    # Register dependencies and enhance them with metadata
    available_steps = []
    self._enhance_dependency_steps_with_specs(resolver, dependency_steps, available_steps)
    
    # One method call handles what used to require multiple matching methods
    resolved = resolver.resolve_step_dependencies(step_name, available_steps)
    ```
    """
    
    # Common patterns for matching inputs to outputs
    # This can be extended by derived classes
    INPUT_PATTERNS = {
        "model": ["model", "model_data", "model_artifacts", "model_path"],
        "data": ["data", "dataset", "input_data", "training_data"],
        "output": ["output", "result", "artifacts", "s3_uri"]
    }
    
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
            
        Deprecated:
            This method is deprecated. Use UnifiedDependencyResolver instead.
        """
        warnings.warn(
            "_match_inputs_to_outputs() is deprecated and will be removed in a future version. "
            "Use UnifiedDependencyResolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            
        Deprecated:
            This method is deprecated. Use UnifiedDependencyResolver instead.
        """
        warnings.warn(
            "_match_model_artifacts() is deprecated and will be removed in a future version. "
            "Use UnifiedDependencyResolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            
        Deprecated:
            This method is deprecated. Use UnifiedDependencyResolver instead.
        """
        warnings.warn(
            "_match_processing_outputs() is deprecated and will be removed in a future version. "
            "Use UnifiedDependencyResolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            
        Deprecated:
            This method is deprecated. Use UnifiedDependencyResolver instead.
        """
        warnings.warn(
            "_match_list_outputs() is deprecated and will be removed in a future version. "
            "Use UnifiedDependencyResolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            
        Deprecated:
            This method is deprecated. Use UnifiedDependencyResolver instead.
        """
        warnings.warn(
            "_match_dict_outputs() is deprecated and will be removed in a future version. "
            "Use UnifiedDependencyResolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
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
            
        Deprecated:
            This method is deprecated. Use UnifiedDependencyResolver instead.
        """
        warnings.warn(
            "_match_custom_properties() is deprecated and will be removed in a future version. "
            "Use UnifiedDependencyResolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Base implementation doesn't match any custom properties
        # Derived classes can override this to implement custom matching logic
        return set()


class InputOutputInformationMixin:
    """
    Mixin for legacy input/output information methods.
    
    This mixin provides backward compatibility for step builders that use the
    traditional input/output information methods. These methods are deprecated
    and will be removed in a future version.
    
    Migration guide:
    
    Instead of:
    ```python
    input_requirements = self.get_input_requirements()
    output_properties = self.get_output_properties()
    ```
    
    Use direct specification access:
    ```python
    # For required dependencies
    required_deps = self.get_required_dependencies()
    
    # For optional dependencies
    optional_deps = self.get_optional_dependencies()
    
    # For outputs
    outputs = self.get_outputs()
    ```
    """
    
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        This method should return a dictionary mapping input parameter names to
        descriptions of what they represent. This helps the pipeline builder
        understand what inputs this step expects.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
            
        Deprecated:
            This method is deprecated. Access specification.dependencies directly
            or use get_required_dependencies() and get_optional_dependencies().
        """
        warnings.warn(
            "get_input_requirements() is deprecated and will be removed in a future version. "
            "Access specification.dependencies directly or use get_required_dependencies() "
            "and get_optional_dependencies().",
            DeprecationWarning,
            stacklevel=2
        )
        
        # First try to use spec.dependencies if available
        if hasattr(self, "spec") and self.spec and hasattr(self.spec, 'dependencies'):
            return {
                d.logical_name: (d.description or d.logical_name) 
                for _, d in self.spec.dependencies.items()
            }
            
        # Fall back to input_names from config
        if hasattr(self, "config") and hasattr(self.config, "input_names"):
            return {k: v for k, v in (self.config.input_names or {}).items()}
            
        return {}
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        This method should return a dictionary mapping output property names to
        descriptions of what they represent. This helps the pipeline builder
        understand what outputs this step provides to downstream steps.
        
        Returns:
            Dictionary mapping output property names to descriptions
            
        Deprecated:
            This method is deprecated. Use specification.outputs directly
            or get_outputs() method instead.
        """
        warnings.warn(
            "get_output_properties() is deprecated and will be removed in a future version. "
            "Use get_outputs() method instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # First try to use spec.outputs if available
        if hasattr(self, "spec") and self.spec and hasattr(self.spec, 'outputs'):
            return {
                o.logical_name: (o.description or o.logical_name) 
                for _, o in self.spec.outputs.items()
            }
            
        # Fall back to output_names from config
        if hasattr(self, "config") and hasattr(self.config, "output_names"):
            return {k: v for k, v in (self.config.output_names or {}).items()}
            
        return {}


class LegacyDependencyResolutionMixin:
    """
    Mixin for legacy dependency resolution methods.
    
    This mixin provides backward compatibility for step builders that use the
    traditional dependency resolution methods. These methods are deprecated and
    will be removed in a future version.
    
    Migration guide:
    
    Instead of using extract_inputs_from_dependencies with the traditional approach,
    use the specification-driven approach with UnifiedDependencyResolver:
    
    ```python
    def extract_inputs_using_resolver(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        resolver = UnifiedDependencyResolver()
        resolver.register_specification(step_name, self.spec)
        
        available_steps = []
        self._enhance_dependency_steps_with_specs(resolver, dependency_steps, available_steps)
        
        resolved = resolver.resolve_step_dependencies(step_name, available_steps)
        
        return {name: prop_ref.to_sagemaker_property() for name, prop_ref in resolved.items()}
    ```
    """
    
    def extract_inputs_from_dependencies_traditional(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps using the traditional approach.
        
        This method is a fallback for backward compatibility. New code should use
        extract_inputs_using_resolver instead.
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            Dictionary of inputs extracted from dependency steps
            
        Deprecated:
            This method is deprecated. Use extract_inputs_using_resolver instead.
        """
        warnings.warn(
            "extract_inputs_from_dependencies_traditional() is deprecated and will be removed "
            "in a future version. Use extract_inputs_using_resolver instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Validate input
        if dependency_steps is None:
            logger.warning("No dependency steps provided to extract_inputs_from_dependencies_traditional")
            return {}
            
        # Base implementation looks for common patterns in dependency steps' outputs
        inputs = {}
        matched_inputs = set()
        
        # Get input requirements for this step
        if hasattr(self, "get_input_requirements"):
            input_requirements = self.get_input_requirements()
        else:
            logger.warning("get_input_requirements method not found, using empty dict")
            input_requirements = {}
        
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
            
            if hasattr(self, "_match_inputs_to_outputs"):
                new_matches = self._match_inputs_to_outputs(inputs, input_requirements, prev_step)
                matched_inputs.update(new_matches)
            
        # Log which inputs were matched and which are still missing
        if matched_inputs:
            logger.info(f"Successfully matched inputs: {sorted(matched_inputs)}")
            
        missing_inputs = set(input_requirements.keys()) - matched_inputs
        if missing_inputs:
            # Filter out optional inputs
            required_missing = [name for name in missing_inputs 
                              if "optional" not in input_requirements.get(name, "").lower()]
            if required_missing:
                logger.warning(f"Could not match required inputs: {sorted(required_missing)}")
            else:
                logger.debug(f"Could not match optional inputs: {sorted(missing_inputs)}")
        
        return inputs


class LegacyCompatibility(PropertyPathRegistryMixin, DependencyMatchingMixin, 
                         InputOutputInformationMixin, LegacyDependencyResolutionMixin):
    """
    Combined compatibility class for all legacy methods.
    
    This class combines all the legacy mixins into a single class that can be
    used as a base class for step builders that need backward compatibility.
    
    Example:
    ```python
    from ..pipeline_steps.legacy_compatibility import LegacyCompatibility
    
    class MyStepBuilder(StepBuilderBase, LegacyCompatibility):
        # This class has access to both new and legacy methods
        pass
    ```
    """
    pass
