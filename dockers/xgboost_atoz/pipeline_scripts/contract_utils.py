"""
Contract Validation Utilities for XGBoost Pipeline Scripts

Provides contract-aware helper functions for SageMaker XGBoost pipeline scripts to ensure
alignment between step specifications, script contracts, and actual implementations.

This is a standalone version for use within XGBoost Docker containers.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_contract_environment(contract) -> None:
    """
    Validate SageMaker environment matches contract expectations.
    
    Args:
        contract: ScriptContract instance defining expected environment
        
    Raises:
        RuntimeError: If contract validation fails
    """
    errors = []
    
    # Check required environment variables
    for var in contract.required_env_vars:
        if var not in os.environ:
            errors.append(f"Missing required environment variable: {var}")
        else:
            logger.info(f"✓ Environment variable found: {var}")
    
    # Check optional environment variables and set defaults
    for var, default_value in contract.optional_env_vars.items():
        if var not in os.environ:
            os.environ[var] = default_value
            logger.info(f"Set default environment variable: {var}={default_value}")
        else:
            logger.info(f"✓ Optional environment variable found: {var}")
    
    # Check input paths exist (SageMaker mounts these)
    for logical_name, path in contract.expected_input_paths.items():
        if not os.path.exists(path):
            errors.append(f"Input path not found: {path} ({logical_name})")
        else:
            logger.info(f"✓ Input path exists: {path} ({logical_name})")
    
    # Ensure output directories exist
    for logical_name, path in contract.expected_output_paths.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"✓ Ensured output directory exists: {path} ({logical_name})")
    
    if errors:
        error_msg = f"Contract validation failed: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ Contract environment validation passed")


def get_contract_paths(contract) -> Dict[str, Dict[str, str]]:
    """
    Get input/output paths from contract for easy access.
    
    Args:
        contract: ScriptContract instance
        
    Returns:
        Dictionary with 'inputs' and 'outputs' keys containing path mappings
    """
    paths = {
        'inputs': contract.expected_input_paths.copy(),
        'outputs': contract.expected_output_paths.copy()
    }
    
    logger.info(f"Contract paths loaded - Inputs: {len(paths['inputs'])}, Outputs: {len(paths['outputs'])}")
    return paths


def get_input_path(contract, logical_name: str) -> str:
    """
    Get input path by logical name from contract.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the input
        
    Returns:
        Absolute path to the input
        
    Raises:
        ValueError: If logical name not found in contract
    """
    if logical_name not in contract.expected_input_paths:
        available_inputs = list(contract.expected_input_paths.keys())
        raise ValueError(f"Unknown input '{logical_name}'. Available inputs: {available_inputs}")
    
    path = contract.expected_input_paths[logical_name]
    logger.info(f"Retrieved input path: {logical_name} -> {path}")
    return path


def get_output_path(contract, logical_name: str) -> str:
    """
    Get output path by logical name from contract.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the output
        
    Returns:
        Absolute path to the output directory
        
    Raises:
        ValueError: If logical name not found in contract
    """
    if logical_name not in contract.expected_output_paths:
        available_outputs = list(contract.expected_output_paths.keys())
        raise ValueError(f"Unknown output '{logical_name}'. Available outputs: {available_outputs}")
    
    path = contract.expected_output_paths[logical_name]
    os.makedirs(path, exist_ok=True)
    logger.info(f"Retrieved output path: {logical_name} -> {path}")
    return path


def validate_xgboost_model_files(model_dir: str) -> None:
    """
    Validate that required XGBoost model files exist.
    
    Args:
        model_dir: Path to model directory
        
    Raises:
        RuntimeError: If required model files are missing
    """
    required_files = [
        "xgboost_model.bst",
        "risk_table_map.pkl", 
        "impute_dict.pkl",
        "feature_columns.txt",
        "hyperparameters.json"
    ]
    
    errors = []
    for filename in required_files:
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            errors.append(f"Required model file not found: {file_path}")
        else:
            logger.info(f"✓ Model file found: {filename}")
    
    if errors:
        error_msg = f"XGBoost model validation failed: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ XGBoost model files validation passed")


def validate_data_files(data_dir: str, expected_extensions: List[str] = None) -> List[str]:
    """
    Validate and find data files in a directory.
    
    Args:
        data_dir: Path to data directory
        expected_extensions: List of expected file extensions (default: ['.csv', '.parquet'])
        
    Returns:
        List of found data files
        
    Raises:
        RuntimeError: If no data files are found
    """
    if expected_extensions is None:
        expected_extensions = ['.csv', '.parquet']
    
    data_files = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise RuntimeError(f"Data directory does not exist: {data_dir}")
    
    for ext in expected_extensions:
        files = list(data_path.glob(f"**/*{ext}"))
        data_files.extend([str(f) for f in files if f.is_file()])
    
    if not data_files:
        raise RuntimeError(f"No data files found in {data_dir} with extensions {expected_extensions}")
    
    logger.info(f"✓ Found {len(data_files)} data files: {data_files}")
    return data_files


def log_contract_summary(contract) -> None:
    """
    Log a summary of the contract for debugging purposes.
    
    Args:
        contract: ScriptContract instance
    """
    logger.info("=== XGBoost Contract Summary ===")
    logger.info(f"Entry Point: {contract.entry_point}")
    logger.info(f"Description: {contract.description}")
    
    logger.info("Input Paths:")
    for logical_name, path in contract.expected_input_paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        logger.info(f"  {logical_name}: {path} {exists}")
    
    logger.info("Output Paths:")
    for logical_name, path in contract.expected_output_paths.items():
        logger.info(f"  {logical_name}: {path}")
    
    logger.info("Required Environment Variables:")
    for var in contract.required_env_vars:
        value = os.environ.get(var, "NOT SET")
        logger.info(f"  {var}: {value}")
    
    logger.info("Optional Environment Variables:")
    for var, default in contract.optional_env_vars.items():
        value = os.environ.get(var, default)
        logger.info(f"  {var}: {value} (default: {default})")
    
    logger.info("Framework Requirements:")
    for framework, version in contract.framework_requirements.items():
        logger.info(f"  {framework}: {version}")
    
    logger.info("================================")


def validate_xgboost_environment() -> None:
    """
    Validate XGBoost-specific environment requirements.
    
    Raises:
        RuntimeError: If XGBoost environment validation fails
    """
    errors = []
    
    # Check XGBoost availability
    try:
        import xgboost as xgb
        logger.info(f"✓ XGBoost version: {xgb.__version__}")
    except ImportError:
        errors.append("XGBoost not available")
    
    # Check other required packages for XGBoost workflows
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib'
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            logger.info(f"✓ {package_name} available")
        except ImportError:
            errors.append(f"{package_name} not available")
    
    if errors:
        error_msg = f"XGBoost environment validation failed: {errors}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    logger.info("✅ XGBoost environment validation passed")


def create_output_file_path(contract, logical_name: str, filename: str) -> str:
    """
    Create a full file path within an output directory.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the output
        filename: Name of the file to create
        
    Returns:
        Full path to the output file
    """
    output_dir = get_output_path(contract, logical_name)
    file_path = os.path.join(output_dir, filename)
    
    logger.info(f"Created output file path: {logical_name}/{filename} -> {file_path}")
    return file_path


def find_files_in_input(contract, logical_name: str, pattern: str = "*") -> List[str]:
    """
    Find files matching a pattern in an input directory.
    
    Args:
        contract: ScriptContract instance
        logical_name: Logical name of the input
        pattern: Glob pattern to match files (default: "*")
        
    Returns:
        List of file paths matching the pattern
        
    Raises:
        ValueError: If logical name not found in contract
    """
    input_path = get_input_path(contract, logical_name)
    input_dir = Path(input_path)
    
    if not input_dir.exists():
        logger.warning(f"Input directory does not exist: {input_path}")
        return []
    
    files = list(input_dir.glob(pattern))
    file_paths = [str(f) for f in files if f.is_file()]
    
    logger.info(f"Found {len(file_paths)} files matching '{pattern}' in {logical_name}: {file_paths}")
    return file_paths


class XGBoostContractEnforcer:
    """
    Context manager for XGBoost contract enforcement in SageMaker scripts.
    
    Usage:
        with XGBoostContractEnforcer(contract) as enforcer:
            # XGBoost script logic here
            model_dir = enforcer.get_input_path('model_input')
            eval_data_dir = enforcer.get_input_path('eval_data_input')
            output_dir = enforcer.get_output_path('eval_output')
    """
    
    def __init__(self, contract, validate_model_files: bool = True, validate_data_files: bool = True):
        self.contract = contract
        self.validate_model_files = validate_model_files
        self.validate_data_files = validate_data_files
        
    def __enter__(self):
        """Enter the XGBoost contract enforcement context"""
        logger.info("🔒 Entering XGBoost contract enforcement context")
        
        # Log contract summary
        log_contract_summary(self.contract)
        
        # Validate XGBoost environment
        validate_xgboost_environment()
        
        # Validate contract environment
        validate_contract_environment(self.contract)
        
        # Validate XGBoost model files if requested
        if self.validate_model_files and 'model_input' in self.contract.expected_input_paths:
            model_dir = self.contract.expected_input_paths['model_input']
            validate_xgboost_model_files(model_dir)
        
        # Validate data files if requested
        if self.validate_data_files:
            for logical_name, path in self.contract.expected_input_paths.items():
                if 'data' in logical_name.lower():
                    validate_data_files(path)
        
        logger.info("✅ XGBoost contract enforcement validation complete")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the XGBoost contract enforcement context"""
        if exc_type is None:
            logger.info("✅ XGBoost script completed successfully within contract")
        else:
            logger.error(f"❌ XGBoost script failed with exception: {exc_type.__name__}: {exc_val}")
        
        logger.info("🔓 Exiting XGBoost contract enforcement context")
        
    def get_input_path(self, logical_name: str) -> str:
        """Get input path by logical name"""
        return get_input_path(self.contract, logical_name)
        
    def get_output_path(self, logical_name: str) -> str:
        """Get output path by logical name"""
        return get_output_path(self.contract, logical_name)
        
    def create_output_file_path(self, logical_name: str, filename: str) -> str:
        """Create output file path"""
        return create_output_file_path(self.contract, logical_name, filename)
        
    def find_files_in_input(self, logical_name: str, pattern: str = "*") -> List[str]:
        """Find files in input directory"""
        return find_files_in_input(self.contract, logical_name, pattern)


# XGBoost-specific contract definitions for standalone use
class SimpleScriptContract:
    """
    Simplified contract class for use in Docker containers without full pipeline dependencies.
    """
    
    def __init__(self, entry_point: str, expected_input_paths: Dict[str, str], 
                 expected_output_paths: Dict[str, str], required_env_vars: List[str],
                 optional_env_vars: Dict[str, str] = None, 
                 framework_requirements: Dict[str, str] = None,
                 description: str = ""):
        self.entry_point = entry_point
        self.expected_input_paths = expected_input_paths
        self.expected_output_paths = expected_output_paths
        self.required_env_vars = required_env_vars
        self.optional_env_vars = optional_env_vars or {}
        self.framework_requirements = framework_requirements or {}
        self.description = description


# Pre-defined contracts for common XGBoost scripts
XGBOOST_TRAINING_CONTRACT = SimpleScriptContract(
    entry_point="train_xgb.py",
    expected_input_paths={
        "train_data": "/opt/ml/input/data/train",
        "val_data": "/opt/ml/input/data/val",
        "test_data": "/opt/ml/input/data/test",
        "config": "/opt/ml/input/data/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "data_output": "/opt/ml/output/data"
    },
    required_env_vars=[],
    framework_requirements={
        "xgboost": ">=1.6.0",
        "scikit-learn": ">=1.0.0",
        "pandas": ">=1.3.0"
    },
    description="XGBoost training script contract"
)

XGBOOST_EVALUATION_CONTRACT = SimpleScriptContract(
    entry_point="model_eval_xgb.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "eval_data_input": "/opt/ml/processing/input/eval_data"
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
        "metrics_output": "/opt/ml/processing/output/metrics"
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    framework_requirements={
        "xgboost": ">=1.6.0",
        "scikit-learn": ">=1.0.0",
        "pandas": ">=1.3.0",
        "matplotlib": ">=3.5.0"
    },
    description="XGBoost model evaluation script contract"
)
