"""
Risk Table Mapping Script Contract

Defines the contract for the risk table mapping script that creates risk tables
for categorical features and handles missing value imputation for numeric features.
"""

from .base_script_contract import ScriptContract

RISK_TABLE_MAPPING_CONTRACT = ScriptContract(
    entry_point="risk_table_mapping.py",
    expected_input_paths={
        "data_input": "/opt/ml/processing/input/data",
        "config_input": "/opt/ml/processing/input/config"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output"
    },
    required_env_vars=[
        # No strictly required environment variables - script has defaults
    ],
    optional_env_vars={
        "TRAIN_RATIO": "0.7",
        "TEST_VAL_RATIO": "0.5"
    },
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0"
    },
    description="""
    Risk table mapping script that:
    1. Creates risk tables for categorical features based on target variable correlation
    2. Handles missing value imputation for numeric features
    3. Supports both training mode (fit and transform) and inference mode (transform only)
    4. Applies smoothing and count thresholds for robust risk estimation
    5. Saves fitted artifacts for reuse in inference
    
    Input Structure:
    - /opt/ml/processing/input/data: Data files
      - Training mode: data.csv (single unsplit dataset)
      - Inference mode: train/, validation/, test/ subdirectories with processed data
    - /opt/ml/processing/input/config: Configuration files
      - config.json: Model configuration including category risk parameters
      - metadata.csv: Variable metadata with types and imputation strategies
    
    Output Structure:
    - /opt/ml/processing/output/{split}/{split}_processed_data.csv: Transformed data by split
    - /opt/ml/processing/output/bin_mapping.pkl: Risk table mappings for categorical features
    - /opt/ml/processing/output/missing_value_imputation.pkl: Imputation values for numeric features
    - /opt/ml/processing/output/config.pkl: Serialized configuration with metadata
    
    Environment Variables:
    - TRAIN_RATIO: Training data ratio for splitting (default: 0.7)
    - TEST_VAL_RATIO: Test/validation split ratio (default: 0.5)
    
    Command Line Arguments:
    - --job_type: Type of job (training, validation, testing)
    
    Training Mode:
    - Fits risk tables and imputers on entire dataset
    - Transforms data and splits into train/test/val
    
    Inference Mode:
    - Fits on training data only
    - Transforms all splits using fitted artifacts
    """
)
