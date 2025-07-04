"""
Pipeline Dependencies module - Declarative dependency management for SageMaker pipelines.

This module provides declarative specifications and intelligent resolution
for pipeline step dependencies.
"""

from .base_specifications import (
    DependencyType, DependencySpec, OutputSpec, PropertyReference, 
    StepSpecification, SpecificationRegistry
)
from .dependency_resolver import UnifiedDependencyResolver, DependencyResolutionError, global_resolver
from .semantic_matcher import SemanticMatcher, semantic_matcher
from .pipeline_registry import (
    PipelineRegistry, RegistryManager, registry_manager,
    get_pipeline_registry, get_default_registry, integrate_with_pipeline_builder
)

__all__ = [
    # Core specification classes
    'DependencyType',
    'DependencySpec', 
    'OutputSpec',
    'PropertyReference',
    'StepSpecification',
    'SpecificationRegistry',
    
    # Dependency resolution
    'UnifiedDependencyResolver',
    'DependencyResolutionError',
    
    # Semantic matching
    'SemanticMatcher',
    
    # Pipeline registry
    'PipelineRegistry',
    'RegistryManager',
    'get_pipeline_registry',
    'get_default_registry',
    'integrate_with_pipeline_builder',
    
    # Global instances
    'global_resolver',
    'semantic_matcher',
    'registry_manager',
]
