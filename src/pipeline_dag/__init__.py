"""
Pipeline DAG module - Comprehensive DAG management for SageMaker pipelines.

This module consolidates all DAG-related functionality:
- Base DAG operations (from base_dag.py)
- Enhanced DAG with port-based dependencies (from enhanced_dag.py)
- Dependency resolution algorithms
- DAG validation and optimization
"""

# Core DAG classes
from .base_dag import PipelineDAG

# Enhanced DAG classes
from .enhanced_dag import EnhancedPipelineDAG
from .edge_types import DependencyEdge, EdgeCollection, EdgeType, ConditionalEdge, ParallelEdge

# Backward compatibility alias
BasePipelineDAG = PipelineDAG

__all__ = [
    'PipelineDAG',           # Base DAG class (moved from pipeline_builder)
    'BasePipelineDAG',       # Explicit base reference for compatibility
    'EnhancedPipelineDAG',   # Enhanced DAG with ports and dependencies
    'DependencyEdge',        # Typed dependency edges
    'EdgeCollection',        # Collection of edges with utilities
    'EdgeType',              # Edge type enumeration
    'ConditionalEdge',       # Conditional dependency edges
    'ParallelEdge',          # Parallel execution hint edges
]
