"""
Enhanced version of StepBuilderBase with deprecation warnings for legacy methods.

This module shows how to implement deprecation warnings for legacy methods
in the StepBuilderBase class. These changes can be applied to the main
StepBuilderBase class once reviewed.
"""

import warnings
import functools
from typing import Dict, List, Any, Set, Optional

# Import from the base class for demonstration
from src.v2.pipeline_steps.builder_step_base import StepBuilderBase
from sagemaker.workflow.steps import Step


def deprecated(func):
    """
    Decorator to mark functions as deprecated.
    
    Args:
        func: The function to mark as deprecated
        
    Returns:
        Wrapped function that issues a deprecation warning
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{func.__name__} is deprecated and will be removed in a future version. "
            f"Use UnifiedDependencyResolver for dependency resolution instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    return wrapper


class EnhancedStepBuilderBase(StepBuilderBase):
    """Enhanced version of StepBuilderBase with deprecation warnings for legacy methods."""
    
    @deprecated
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
            This method is deprecated and will be removed in a future version.
            Use UnifiedDependencyResolver for dependency resolution instead.
        """
        return super()._match_inputs_to_outputs(inputs, input_requirements, prev_step)
        
    @deprecated
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
            This method is deprecated and will be removed in a future version.
            Use UnifiedDependencyResolver for dependency resolution instead.
        """
        return super()._match_model_artifacts(inputs, input_requirements, prev_step)
        
    @deprecated
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
            This method is deprecated and will be removed in a future version.
            Use UnifiedDependencyResolver for dependency resolution instead.
        """
        return super()._match_processing_outputs(inputs, input_requirements, prev_step)
        
    @deprecated
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
            This method is deprecated and will be removed in a future version.
            Use UnifiedDependencyResolver for dependency resolution instead.
        """
        return super()._match_list_outputs(inputs, input_requirements, outputs)
        
    @deprecated
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
            This method is deprecated and will be removed in a future version.
            Use UnifiedDependencyResolver for dependency resolution instead.
        """
        return super()._match_dict_outputs(inputs, input_requirements, outputs)


# Example of how to implement enhanced logging for dependency resolution
def extract_inputs_using_resolver_with_enhanced_logging(self, dependency_steps: List[Step]) -> Dict[str, Any]:
    """
    Extract inputs from dependency steps using the UnifiedDependencyResolver with enhanced logging.
    
    This method uses the sophisticated matching algorithms of the dependency resolver
    to find compatible outputs from previous steps that satisfy this step's dependencies.
    
    Args:
        dependency_steps: List of dependency steps
        
    Returns:
        Dictionary of inputs extracted from dependency steps
    """
    if not hasattr(self, "DEPENDENCY_RESOLVER_AVAILABLE") or not self.DEPENDENCY_RESOLVER_AVAILABLE:
        self.log_warning("Dependency resolver not available, using traditional methods")
        return {}
        
    if not self.spec:
        self.log_warning("No specification available for dependency resolution")
        return {}
        
    try:
        # Create resolver
        self.log_info("Creating UnifiedDependencyResolver instance")
        resolver = self.UnifiedDependencyResolver()
        
        # Register this step's specification
        step_name = self.__class__.__name__.replace("Builder", "Step")
        self.log_info("Registering specification for %s", step_name)
        resolver.register_specification(step_name, self.spec)
        
        # Register specifications for dependency steps
        available_steps = []
        for i, dep_step in enumerate(dependency_steps):
            dep_name = getattr(dep_step, 'name', f"Step_{i}")
            available_steps.append(dep_name)
            
            # Try to get specification from step
            dep_spec = None
            if hasattr(dep_step, '_spec'):
                dep_spec = getattr(dep_step, '_spec')
            elif hasattr(dep_step, 'spec'):
                dep_spec = getattr(dep_step, 'spec')
                
            if dep_spec:
                self.log_debug("Registering specification for dependency step '%s'", dep_name)
                resolver.register_specification(dep_name, dep_spec)
            else:
                self.log_debug("No specification found for dependency step '%s'", dep_name)
        
        # Resolve dependencies
        self.log_info("Resolving dependencies for %s with %d available steps", step_name, len(available_steps))
        resolved = resolver.resolve_step_dependencies(step_name, available_steps)
        if not resolved:
            self.log_warning("Resolver returned empty result")
            return {}
        
        # Convert PropertyReferences to actual values
        inputs = {}
        for dep_name, prop_ref in resolved.items():
            from_type = type(prop_ref).__name__
            
            if hasattr(prop_ref, "to_sagemaker_property"):
                inputs[dep_name] = prop_ref.to_sagemaker_property()
                to_type = "SageMaker property"
            else:
                inputs[dep_name] = prop_ref
                to_type = type(prop_ref).__name__
                
            self.log_info("Resolved dependency '%s': %s → %s", dep_name, from_type, to_type)
            
        self.log_info("Successfully resolved %d dependencies", len(inputs))
        return inputs
    except Exception as e:
        self.log_warning("Error using dependency resolver: %s", e, exc_info=True)
        return {}
