"""
Step specifications for pipeline dependency resolution.

This module contains declarative specifications for different types of pipeline steps,
defining their input dependencies and output properties.
"""

from .preprocessing_spec import PREPROCESSING_SPEC
from .xgboost_training_spec import XGBOOST_TRAINING_SPEC
from .registration_spec import REGISTRATION_SPEC
from .data_loading_spec import DATA_LOADING_SPEC
from .packaging_spec import PACKAGING_SPEC
from .payload_spec import PAYLOAD_SPEC

__all__ = [
    "PREPROCESSING_SPEC",
    "XGBOOST_TRAINING_SPEC", 
    "REGISTRATION_SPEC",
    "DATA_LOADING_SPEC",
    "PACKAGING_SPEC",
    "PAYLOAD_SPEC"
]
