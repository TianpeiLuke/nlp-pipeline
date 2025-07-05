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
        # Note: Current script only uses data input, but specification expects metadata/signature
        # This represents the gap identified in the analysis
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output"
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
    2. Cleans and processes label field
    3. Splits data into train/test/val for training jobs
    4. Outputs processed CSV files by split
    
    Current Gap: Script only uses /opt/ml/processing/input/data but step specification
    expects DATA, METADATA, and SIGNATURE inputs. This represents a misalignment
    that should be addressed.
    """
)
