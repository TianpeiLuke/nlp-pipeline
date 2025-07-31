"""
Pipeline API - High-level interface for pipeline construction.

This module provides user-friendly APIs for compiling PipelineDAG structures
directly into executable SageMaker pipelines without requiring custom template classes.
"""

from .dag_compiler import compile_dag_to_pipeline, PipelineDAGCompiler
from .validation import ValidationResult, ResolutionPreview, ConversionReport
from .exceptions import (
    PipelineAPIError,
    ConfigurationError,
    AmbiguityError,
    ValidationError,
    ResolutionError
)
from ..pipeline_registry.exceptions import RegistryError

__all__ = [
    # Main API functions
    'compile_dag_to_pipeline',
    'PipelineDAGCompiler',
    
    # Validation and preview
    'ValidationResult',
    'ResolutionPreview',
    
    # Exceptions
    'PipelineAPIError',
    'ConfigurationError',
    'RegistryError',  # Re-exported from pipeline_registry
    'AmbiguityError',
    'ValidationError',
    'ResolutionError',
]

__version__ = "1.0.0"
