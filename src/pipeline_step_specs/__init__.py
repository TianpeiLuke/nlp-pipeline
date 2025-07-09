"""
Step specifications for pipeline dependency resolution.

This module contains declarative specifications for different types of pipeline steps,
defining their input dependencies and output properties.
"""

# Original generic specifications
from .preprocessing_spec import PREPROCESSING_SPEC
from .xgboost_training_spec import XGBOOST_TRAINING_SPEC
from .registration_spec import REGISTRATION_SPEC
from .data_loading_spec import DATA_LOADING_SPEC
from .packaging_spec import PACKAGING_SPEC
from .payload_spec import PAYLOAD_SPEC

# Job type-specific data loading specifications
from .data_loading_training_spec import DATA_LOADING_TRAINING_SPEC
from .data_loading_validation_spec import DATA_LOADING_VALIDATION_SPEC
from .data_loading_testing_spec import DATA_LOADING_TESTING_SPEC
from .data_loading_calibration_spec import DATA_LOADING_CALIBRATION_SPEC

# Job type-specific preprocessing specifications
from .preprocessing_training_spec import PREPROCESSING_TRAINING_SPEC
from .preprocessing_validation_spec import PREPROCESSING_VALIDATION_SPEC
from .preprocessing_testing_spec import PREPROCESSING_TESTING_SPEC
from .preprocessing_calibration_spec import PREPROCESSING_CALIBRATION_SPEC
from .model_eval_spec import MODEL_EVAL_SPEC

__all__ = [
    "MODEL_EVAL_SPEC",
    # Original generic specifications
    "PREPROCESSING_SPEC",
    "XGBOOST_TRAINING_SPEC", 
    "REGISTRATION_SPEC",
    "DATA_LOADING_SPEC",
    "PACKAGING_SPEC",
    "PAYLOAD_SPEC",
    
    # Job type-specific data loading specifications
    "DATA_LOADING_TRAINING_SPEC",
    "DATA_LOADING_VALIDATION_SPEC",
    "DATA_LOADING_TESTING_SPEC",
    "DATA_LOADING_CALIBRATION_SPEC",
    
    # Job type-specific preprocessing specifications
    "PREPROCESSING_TRAINING_SPEC",
    "PREPROCESSING_VALIDATION_SPEC",
    "PREPROCESSING_TESTING_SPEC",
    "PREPROCESSING_CALIBRATION_SPEC"
]
