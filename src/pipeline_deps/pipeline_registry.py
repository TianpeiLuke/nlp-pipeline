"""
Pipeline-scoped registry management for declarative dependency management.

This module provides a solution for pipeline-scoped registries, where each pipeline
has its own isolated registry instance, but steps within a pipeline share the same registry.
This prevents cross-pipeline interference while maintaining intra-pipeline dependency resolution.
"""

from typing import Dict, List, Optional, Any, Set, Union
import logging
from .base_specifications import SpecificationRegistry, StepSpecification, DependencySpec, OutputSpec

logger = logging.getLogger(__name__)


class PipelineRegistry(SpecificationRegistry):
    """
    Pipeline-scoped registry for managing step specifications within a single pipeline.
    
    Extends the base SpecificationRegistry with pipeline-specific functionality.
    """
    
    def __init__(self, pipeline_name: str):
        """
        Initialize a pipeline-scoped registry.
        
        Args:
            pipeline_name: Name of the pipeline this registry belongs to
        """
        super().__init__()
        self.pipeline_name = pipeline_name
        logger.info(f"Created pipeline-scoped registry for pipeline '{pipeline_name}'")
    
    def register(self, step_name: str, specification: StepSpecification):
        """
        Register a step specification in this pipeline's registry.
        
        Args:
            step_name: Name of the step
            specification: Step specification to register
        """
        # Add pipeline name to logging for clarity
        super().register(step_name, specification)
        logger.info(f"Registered specification for step '{step_name}' in pipeline '{self.pipeline_name}'")
    
    def __repr__(self) -> str:
        """String representation of the pipeline registry."""
        return f"PipelineRegistry(pipeline='{self.pipeline_name}', steps={len(self._specifications)})"


class RegistryManager:
    """
    Manager for pipeline-scoped registries.
    
    This class manages multiple pipeline registries, allowing each pipeline to have
    its own isolated registry while providing a centralized access point.
    """
    
    def __init__(self):
        """Initialize the registry manager."""
        self._pipeline_registries: Dict[str, PipelineRegistry] = {}
        self._default_registry = SpecificationRegistry()
        logger.info("Initialized registry manager")
    
    def get_pipeline_registry(self, pipeline_name: str, create_if_missing: bool = True) -> PipelineRegistry:
        """
        Get the registry for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            create_if_missing: Whether to create a new registry if one doesn't exist
            
        Returns:
            Pipeline-specific registry
        """
        if pipeline_name not in self._pipeline_registries and create_if_missing:
            self._pipeline_registries[pipeline_name] = PipelineRegistry(pipeline_name)
            logger.info(f"Created new registry for pipeline '{pipeline_name}'")
        
        return self._pipeline_registries.get(pipeline_name)
    
    def get_default_registry(self) -> SpecificationRegistry:
        """
        Get the default registry.
        
        This registry can be used for shared specifications or when a pipeline-specific
        registry is not available.
        
        Returns:
            Default registry
        """
        return self._default_registry
    
    def list_pipeline_registries(self) -> List[str]:
        """
        Get list of all registered pipeline names.
        
        Returns:
            List of pipeline names with registries
        """
        return list(self._pipeline_registries.keys())
    
    def clear_pipeline_registry(self, pipeline_name: str) -> bool:
        """
        Clear the registry for a specific pipeline.
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            True if the registry was cleared, False if it didn't exist
        """
        if pipeline_name in self._pipeline_registries:
            del self._pipeline_registries[pipeline_name]
            logger.info(f"Cleared registry for pipeline '{pipeline_name}'")
            return True
        return False
    
    def __repr__(self) -> str:
        """String representation of the registry manager."""
        return f"RegistryManager(pipelines={len(self._pipeline_registries)})"


# Global registry manager instance
registry_manager = RegistryManager()


def get_pipeline_registry(pipeline_name: str) -> PipelineRegistry:
    """
    Get the registry for a specific pipeline.
    
    This is a convenience function that uses the global registry manager.
    
    Args:
        pipeline_name: Name of the pipeline
        
    Returns:
        Pipeline-specific registry
    """
    return registry_manager.get_pipeline_registry(pipeline_name)


def get_default_registry() -> SpecificationRegistry:
    """
    Get the default registry.
    
    This is a convenience function that uses the global registry manager.
    
    Returns:
        Default registry
    """
    return registry_manager.get_default_registry()


# Integration with PipelineBuilderTemplate
def integrate_with_pipeline_builder(pipeline_builder_cls):
    """
    Decorator to integrate pipeline-scoped registries with a pipeline builder class.
    
    This decorator modifies a pipeline builder class to use pipeline-scoped registries.
    
    Args:
        pipeline_builder_cls: Pipeline builder class to modify
        
    Returns:
        Modified pipeline builder class
    """
    original_init = pipeline_builder_cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Call original __init__
        original_init(self, *args, **kwargs)
        
        # Get pipeline name from base_config
        pipeline_name = 'default_pipeline'
        if hasattr(self, 'base_config'):
            try:
                if hasattr(self.base_config, 'pipeline_name') and self.base_config.pipeline_name:
                    pipeline_name = self.base_config.pipeline_name
            except (AttributeError, TypeError):
                pass
        
        # Create pipeline-specific registry
        self.registry = get_pipeline_registry(pipeline_name)
        logger.info(f"Pipeline builder using registry for pipeline '{pipeline_name}'")
    
    # Replace __init__ method
    pipeline_builder_cls.__init__ = new_init
    
    return pipeline_builder_cls
