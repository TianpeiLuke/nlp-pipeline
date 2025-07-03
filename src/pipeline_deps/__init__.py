"""
Pipeline Dependencies module - Declarative dependency management for SageMaker pipelines.

This module provides declarative specifications and intelligent resolution
for pipeline step dependencies.
"""

from .base_specifications import (
    DependencyType, DependencySpec, OutputSpec, PropertyReference, 
    StepSpecification, SpecificationRegistry, global_registry
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
    'SpecificationRegistry',
    
    # Dependency resolution
    'UnifiedDependencyResolver',
    'DependencyResolutionError',
    
    # Semantic matching
    'SemanticMatcher',
    
    # Global instances
    'global_registry',
    'global_resolver',
    'semantic_matcher',
]
