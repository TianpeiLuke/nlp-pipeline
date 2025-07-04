"""
Pipeline Dependencies module - Declarative dependency management for SageMaker pipelines.

This module provides declarative specifications and intelligent resolution
for pipeline step dependencies.
"""

from .base_specifications import (
    DependencyType, DependencySpec, OutputSpec, PropertyReference, 
    StepSpecification
)
from .specification_registry import SpecificationRegistry
from .registry_manager import (
    RegistryManager, registry_manager,
    get_registry, get_pipeline_registry, get_default_registry, 
    integrate_with_pipeline_builder, list_contexts, clear_context, get_context_stats
)
from .dependency_resolver import UnifiedDependencyResolver, DependencyResolutionError, global_resolver
from .semantic_matcher import SemanticMatcher, semantic_matcher

__all__ = [
    # Core specification classes
    'DependencyType',
    'DependencySpec', 
    'OutputSpec',
    'PropertyReference',
    'StepSpecification',
    
    # Registry management
    'SpecificationRegistry',
    'RegistryManager',
    'get_registry',
    'get_pipeline_registry',
    'get_default_registry',
    'integrate_with_pipeline_builder',
    'list_contexts',
    'clear_context',
    'get_context_stats',
    
    # Dependency resolution
    'UnifiedDependencyResolver',
    'DependencyResolutionError',
    
    # Semantic matching
    'SemanticMatcher',
    
    # Global instances
    'global_resolver',
    'semantic_matcher',
    'registry_manager',
]
