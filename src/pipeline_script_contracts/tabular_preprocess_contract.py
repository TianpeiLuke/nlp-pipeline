"""
Tabular Preprocessing Script Contract

Defines the contract for the tabular preprocessing script that handles data loading,
cleaning, and splitting for training/validation/testing.
"""

from .base_script_contract import ScriptContract

TABULAR_PREPROCESS_CONTRACT = ScriptContract(
    entry_point="tabular_preprocess.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "METADATA": "/opt/ml/processing/input/metadata",
        "SIGNATURE": "/opt/ml/processing/input/signature"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output",
        "full_data": "/opt/ml/processing/output/full"
    },
    required_env_vars=[
        "LABEL_FIELD",
        "TRAIN_RATIO", 
        "TEST_VAL_RATIO"
    ],
    optional_env_vars={
        "CATEGORICAL_COLUMNS": "",
        "NUMERICAL_COLUMNS": "",
        "TEXT_COLUMNS": "",
        "DATE_COLUMNS": ""
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0"
    },
    description="""
    Tabular preprocessing script that:
    1. Combines data shards from input directory
    2. Optionally uses metadata and signature for validation
    3. Cleans and processes label field
    4. Splits data into train/test/val for training jobs
    5. Outputs processed CSV files by split
    6. Optionally outputs full dataset without splits
    
    Contract aligned with step specification:
    - Inputs: DATA (required), METADATA (optional), SIGNATURE (optional)
    - Outputs: processed_data (primary), full_data (optional)
    """
)
