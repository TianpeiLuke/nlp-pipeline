"""
Pipeline API - High-level interface for pipeline construction.

This module provides user-friendly APIs for converting PipelineDAG structures
directly into executable SageMaker pipelines without requiring custom template classes.
"""

from .dag_converter import dag_to_pipeline_template, PipelineDAGConverter
from .validation import ValidationResult, ResolutionPreview, ConversionReport
from .exceptions import (
    PipelineAPIError,
    ConfigurationError,
    RegistryError,
    AmbiguityError,
    ValidationError,
    ResolutionError
)

__all__ = [
    # Main API functions
    'dag_to_pipeline_template',
    'PipelineDAGConverter',
    
    # Validation and preview
    'ValidationResult',
    'ResolutionPreview',
    
    # Exceptions
    'PipelineAPIError',
    'ConfigurationError',
    'RegistryError',
    'AmbiguityError',
    'ValidationError',
]

__version__ = "1.0.0"
