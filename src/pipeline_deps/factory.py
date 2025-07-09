"""
Factory functions for creating pipeline dependency components.

This module provides convenience functions for instantiating the core
components of the dependency resolution system with proper wiring.
"""

from .semantic_matcher import SemanticMatcher
from .specification_registry import SpecificationRegistry
from .registry_manager import RegistryManager
from .dependency_resolver import UnifiedDependencyResolver, create_dependency_resolver

def create_pipeline_components(context_name=None):
    """Create all necessary pipeline components with proper dependencies."""
    semantic_matcher = SemanticMatcher()
    registry_manager = RegistryManager()
    registry = registry_manager.get_registry(context_name or "default")
    resolver = UnifiedDependencyResolver(registry, semantic_matcher)
    
    return {
        "semantic_matcher": semantic_matcher,
        "registry_manager": registry_manager,
        "registry": registry,
        "resolver": resolver
    }

__all__ = ["create_pipeline_components", "create_dependency_resolver"]
