"""
Script Contracts Module.

This module contains script contracts that define the expected input and output
paths for scripts used in pipeline steps, as well as required environment variables.
These contracts are used by step specifications to map logical names to container paths.
"""

# Base contract classes
from .base_script_contract import ScriptContract, ValidationResult, ScriptAnalyzer

# Processing script contracts
from .currency_conversion_contract import CURRENCY_CONVERSION_CONTRACT
from .hyperparameter_prep_contract import HYPERPARAMETER_PREP_CONTRACT
from .mims_package_contract import MIMS_PACKAGE_CONTRACT
from .mims_payload_contract import MIMS_PAYLOAD_CONTRACT
from .mims_registration_contract import MIMS_REGISTRATION_CONTRACT
from .model_evaluation_contract import MODEL_EVALUATION_CONTRACT
from .risk_table_mapping_contract import RISK_TABLE_MAPPING_CONTRACT
from .tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT

# Training script contracts
from .training_script_contract import TrainingScriptContract
from .pytorch_train_contract import PYTORCH_TRAIN_CONTRACT
from .xgboost_train_contract import XGBOOST_TRAIN_CONTRACT
