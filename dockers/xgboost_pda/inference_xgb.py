#!/usr/bin/env python3

# Standard library imports
import os
import sys
import json
import logging
import pickle as pkl  # Add this line
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List, Optional
from io import StringIO, BytesIO

# Third-party imports
import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Response

# Local imports
from processing.risk_table_processor import RiskTableMappingProcessor
from processing.numerical_imputation_processor import NumericalVariableImputationProcessor

# Constants
__version__ = "1.0.0"

# File names
MODEL_FILE = "xgboost_model.bst"
RISK_TABLE_FILE = "risk_table_map.pkl"
IMPUTE_DICT_FILE = "impute_dict.pkl"
FEATURE_IMPORTANCE_FILE = "feature_importance.json"
FEATURE_COLUMNS_FILE = "feature_columns.txt"
HYPERPARAMETERS_FILE = "hyperparameters.json"

# Calibration model files
CALIBRATION_DIR = "calibration"
CALIBRATION_MODEL_FILE = "calibration_model.pkl"
CALIBRATION_SUMMARY_FILE = "calibration_summary.json"
CALIBRATION_MODELS_DIR = "calibration_models"  # For multiclass calibration models

# Content types
CONTENT_TYPE_CSV = 'text/csv'
CONTENT_TYPE_JSON = 'application/json'
CONTENT_TYPE_PARQUET = 'application/x-parquet'


# Simple Response class for type hints
class InferenceResponse:
    """Simple response class for type hints."""
    def __init__(self, response: str, status: int = 200, mimetype: str = 'text/plain'):
        self.response = response
        self.status = status
        self.mimetype = mimetype


# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


#--------------------------------------------------------------------------------
#                           Model BLOCK
#--------------------------------------------------------------------------------

def validate_model_files(model_dir: str) -> None:
    """
    Validate that all required model files exist.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Raises:
        FileNotFoundError: If any required file is missing
    """
    required_files = [MODEL_FILE, RISK_TABLE_FILE, IMPUTE_DICT_FILE, FEATURE_COLUMNS_FILE]
    for file in required_files:
        file_path = os.path.join(model_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file {file} not found in {model_dir}")
        logger.info(f"Found required file: {file}")


def read_feature_columns(model_dir: str) -> List[str]:
    """
    Read feature columns in correct order from feature_columns.txt
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        List[str]: Ordered list of feature column names
        
    Raises:
        FileNotFoundError: If feature_columns.txt is not found
        ValueError: If file format is invalid
    """
    feature_file = os.path.join(model_dir, FEATURE_COLUMNS_FILE)
    ordered_features = []
    
    try:
        with open(feature_file, 'r') as f:
            for line in f:
                # Skip comments
                if line.startswith('#'):
                    continue
                # Parse "<index>,<column_name>" format
                try:
                    idx, column = line.strip().split(',')
                    ordered_features.append(column)
                except ValueError:
                    continue
        
        if not ordered_features:
            raise ValueError(f"No valid feature columns found in {feature_file}")
        
        logger.info(f"Loaded {len(ordered_features)} ordered feature columns")
        return ordered_features
    except Exception as e:
        logger.error(f"Error reading feature columns file: {e}", exc_info=True)
        raise

        
def load_xgboost_model(model_dir: str) -> xgb.Booster:
    """Load XGBoost model from file."""
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, MODEL_FILE))
    return model


def load_risk_tables(model_dir: str) -> Dict[str, Any]:
    """Load risk tables from pickle file."""
    with open(os.path.join(model_dir, RISK_TABLE_FILE), "rb") as f:
        return pkl.load(f)

    
def create_risk_processors(risk_tables: Dict[str, Any]) -> Dict[str, RiskTableMappingProcessor]:
    """Create risk table processors for each categorical feature."""
    risk_processors = {}
    for feature, risk_table in risk_tables.items():
        processor = RiskTableMappingProcessor(
            column_name=feature,
            label_name="label",  # Not used during inference
            risk_tables=risk_table
        )
        risk_processors[feature] = processor
    return risk_processors


def load_imputation_dict(model_dir: str) -> Dict[str, Any]:
    """Load imputation dictionary from pickle file."""
    with open(os.path.join(model_dir, IMPUTE_DICT_FILE), "rb") as f:
        return pkl.load(f)

    
def create_numerical_processor(impute_dict: Dict[str, Any]) -> NumericalVariableImputationProcessor:
    """Create numerical imputation processor."""
    return NumericalVariableImputationProcessor(imputation_dict=impute_dict)


def load_feature_importance(model_dir: str) -> Dict[str, Any]:
    """Load feature importance from JSON file."""
    try:
        with open(os.path.join(model_dir, FEATURE_IMPORTANCE_FILE), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning(f"{FEATURE_IMPORTANCE_FILE} not found, skipping feature importance")
        return {}


def load_hyperparameters(model_dir: str) -> Dict[str, Any]:
    """Load hyperparameters from JSON file."""
    try:
        with open(os.path.join(model_dir, HYPERPARAMETERS_FILE), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load {HYPERPARAMETERS_FILE}: {e}")
        return {}


def load_calibration_model(model_dir: str) -> Optional[Any]:
    """
    Load calibration model if it exists.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Calibration model if found, None otherwise
    """
    # Check for binary calibration model
    calibration_path = os.path.join(model_dir, CALIBRATION_DIR, CALIBRATION_MODEL_FILE)
    if os.path.exists(calibration_path):
        logger.info(f"Loading binary calibration model from {calibration_path}")
        try:
            with open(calibration_path, 'rb') as f:
                return pkl.load(f)
        except Exception as e:
            logger.warning(f"Failed to load binary calibration model: {e}")
    
    # Check for multiclass calibration models
    multiclass_dir = os.path.join(model_dir, CALIBRATION_DIR, CALIBRATION_MODELS_DIR)
    if os.path.exists(multiclass_dir) and os.path.isdir(multiclass_dir):
        logger.info(f"Loading multiclass calibration models from {multiclass_dir}")
        try:
            calibrators = {}
            for file in os.listdir(multiclass_dir):
                if file.endswith('.pkl'):
                    class_name = file.replace('calibration_model_class_', '').replace('.pkl', '')
                    with open(os.path.join(multiclass_dir, file), 'rb') as f:
                        calibrators[class_name] = pkl.load(f)
            return calibrators if calibrators else None
        except Exception as e:
            logger.warning(f"Failed to load multiclass calibration models: {e}")
    
    logger.info("No calibration model found")
    return None


def apply_calibration(scores: np.ndarray, calibrator: Any, is_multiclass: bool) -> np.ndarray:
    """
    Apply calibration to raw model scores.
    
    Args:
        scores: Raw model prediction scores
        calibrator: Loaded calibration model(s)
        is_multiclass: Whether this is a multiclass model
        
    Returns:
        Calibrated scores
    """
    if calibrator is None:
        return scores
    
    try:
        if is_multiclass:
            # Handle multiclass calibration - apply each calibrator to corresponding class
            calibrated = np.zeros_like(scores)
            for i in range(scores.shape[1]):
                class_name = str(i)
                if class_name in calibrator:
                    if hasattr(calibrator[class_name], 'transform'):
                        calibrated[:, i] = calibrator[class_name].transform(scores[:, i])
                    elif hasattr(calibrator[class_name], 'predict_proba'):
                        calibrated[:, i] = calibrator[class_name].predict_proba(scores[:, i].reshape(-1, 1))
                    else:
                        calibrated[:, i] = scores[:, i]  # Fallback to raw scores
                else:
                    calibrated[:, i] = scores[:, i]  # No calibrator for this class
            
            # Normalize if needed
            row_sums = calibrated.sum(axis=1)
            calibrated = calibrated / row_sums[:, np.newaxis]
            return calibrated
        else:
            # Binary calibration - single model
            if hasattr(calibrator, 'transform'):
                # Reshape for isotonic regression which expects 1D array
                calibrated = np.zeros_like(scores)
                calibrated[:, 0] = 1 - calibrator.transform(scores[:, 1])  # class 0 probability
                calibrated[:, 1] = calibrator.transform(scores[:, 1])      # class 1 probability
                return calibrated
            elif hasattr(calibrator, 'predict_proba'):
                # Reshape for GAM or Platt scaling which expects 2D array
                calibrated = np.zeros_like(scores)
                probas = calibrator.predict_proba(scores[:, 1].reshape(-1, 1))
                calibrated[:, 1] = probas  # class 1 probability
                calibrated[:, 0] = 1 - probas  # class 0 probability
                return calibrated
            else:
                logger.warning(f"Unknown calibrator type: {type(calibrator)}")
                return scores  # Fallback
    except Exception as e:
        logger.error(f"Error applying calibration: {str(e)}", exc_info=True)
        return scores


def create_model_config(
    model: xgb.Booster,
    feature_columns: List[str],
    hyperparameters: Dict[str, Any]
) -> Dict[str, Any]:
    """Create model configuration dictionary."""
    return {
        "is_multiclass": True if hasattr(model, 'num_class') and model.num_class() > 2 else False,
        "num_classes": model.num_class() if hasattr(model, 'num_class') else 2,
        "feature_columns": feature_columns,
        "hyperparameters": hyperparameters
    }


def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and preprocessing artifacts from model_dir.
    
    Args:
        model_dir (str): Directory containing model artifacts
        
    Returns:
        Dict[str, Any]: Dictionary containing model, processors, and configuration
        
    Raises:
        FileNotFoundError: If required model files are missing
        Exception: For other loading errors
    """
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # Validate all required files exist
        validate_model_files(model_dir)
        
        # Load model and artifacts
        model = load_xgboost_model(model_dir)
        risk_tables = load_risk_tables(model_dir)
        risk_processors = create_risk_processors(risk_tables)
        
        impute_dict = load_imputation_dict(model_dir)
        numerical_processor = create_numerical_processor(impute_dict)
        
        feature_importance = load_feature_importance(model_dir)
        feature_columns = read_feature_columns(model_dir)
        hyperparameters = load_hyperparameters(model_dir)
        
        # Create configuration
        config = create_model_config(model, feature_columns, hyperparameters)
        
        # Load calibration model if available
        calibrator = load_calibration_model(model_dir)
        if calibrator:
            logger.info("Calibration model loaded successfully")
        
        return {
            "model": model,
            "risk_processors": risk_processors,
            "numerical_processor": numerical_processor,
            "feature_importance": feature_importance,
            "config": config,
            "version": __version__,
            "calibrator": calibrator
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

#--------------------------------------------------------------------------------
#                           INPUT BLOCK
#--------------------------------------------------------------------------------

def input_fn(
    request_body: Union[str, bytes], 
    request_content_type: str, 
    context: Optional[Any] = None
) -> Union[pd.DataFrame, InferenceResponse]:
    """
    Deserialize the Invoke request body into an object we can perform prediction on.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        context: Additional context (optional)
        
    Returns:
        Union[pd.DataFrame, Response]: Parsed DataFrame or error Response
    """
    logger.info(f"Received request with Content-Type: {request_content_type}")
    try:
        if request_content_type == CONTENT_TYPE_CSV:
            logger.info("Processing content type: text/csv")
            decoded = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
            logger.debug(f"Decoded CSV data:\n{decoded[:500]}...")
            try:
                df = pd.read_csv(StringIO(decoded), header=None, index_col=None)
                if df.empty:
                    raise ValueError("Empty CSV input provided")
                logger.info(f"Successfully parsed CSV into DataFrame. Shape: {df.shape}")
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse CSV data: {parse_error}")
                raise

        elif request_content_type == CONTENT_TYPE_JSON:
            logger.info("Processing content type: application/json")
            decoded = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
            try:
                if "\n" in decoded:
                    # Multi-record JSON (NDJSON) handling
                    records = [json.loads(line) for line in decoded.strip().splitlines() if line.strip()]
                    df = pd.DataFrame(records)
                else:
                    json_obj = json.loads(decoded)
                    if isinstance(json_obj, dict):
                        df = pd.DataFrame([json_obj])
                    elif isinstance(json_obj, list):
                        df = pd.DataFrame(json_obj)
                    else:
                        raise ValueError("Unsupported JSON structure")
                
                if df.empty:
                    raise ValueError("Empty JSON input provided")
                logger.info(f"Successfully parsed JSON into DataFrame. Shape: {df.shape}")
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse JSON data: {parse_error}")
                raise

        elif request_content_type == CONTENT_TYPE_PARQUET:
            logger.info("Processing content type: application/x-parquet")
            df = pd.read_parquet(BytesIO(request_body))
            if df.empty:
                raise ValueError("Empty Parquet input provided")
            logger.info(f"Successfully parsed Parquet into DataFrame. Shape: {df.shape}")
            return df

        else:
            logger.warning(f"Unsupported content type: {request_content_type}")
            return Response(
                response=f'This predictor only supports CSV, JSON, or Parquet data. Received: {request_content_type}',
                status=415,
                mimetype='text/plain'
            )
    except Exception as e:
        logger.error(f"Failed to parse input ({request_content_type}). Error: {e}", exc_info=True)
        return Response(
            response=f'Invalid input format or corrupted data. Error during parsing: {e}',
            status=400,
            mimetype='text/plain'
        )

#--------------------------------------------------------------------------------
#                           PREDICT BLOCK
#--------------------------------------------------------------------------------

def validate_input_data(input_data: pd.DataFrame, feature_columns: List[str]) -> None:
    """
    Validate input data meets requirements.
    
    Args:
        input_data: Input DataFrame
        feature_columns: Expected feature columns
        
    Raises:
        ValueError: If validation fails
    """
    if input_data.empty:
        raise ValueError("Input DataFrame is empty")
    
    # If input is headerless CSV, validate column count
    if all(isinstance(col, int) for col in input_data.columns):
        if len(input_data.columns) != len(feature_columns):
            raise ValueError(
                f"Input data has {len(input_data.columns)} columns but model expects {len(feature_columns)} features"
            )
    else:
        # Validate required features present
        missing_features = set(feature_columns) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


def assign_column_names(input_data: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Assign column names to headerless input data.
    
    Args:
        input_data: Input DataFrame
        feature_columns: Feature column names to assign
        
    Returns:
        DataFrame with assigned column names
    """
    df = input_data.copy()
    if all(isinstance(col, int) for col in df.columns):
        df.columns = feature_columns
    return df


def apply_preprocessing(
    df: pd.DataFrame,
    feature_columns: List[str],
    risk_processors: Dict[str, Any],
    numerical_processor: Any
) -> pd.DataFrame:
    """
    Apply preprocessing steps to input data.
    
    Args:
        df: Input DataFrame
        feature_columns: List of feature columns
        risk_processors: Dictionary of risk table processors
        numerical_processor: Numerical imputation processor
        
    Returns:
        Preprocessed DataFrame
    """
    # Log initial state
    logger.debug("Initial data types and unique values:")
    for col in feature_columns:
        logger.debug(f"{col}: dtype={df[col].dtype}, unique values={df[col].unique()}")
    
    # Apply risk table mapping
    for feature, processor in risk_processors.items():
        if feature in df.columns:
            logger.debug(f"Applying risk table mapping for feature: {feature}")
            df[feature] = processor.transform(df[feature])
    
    # Apply numerical imputation
    df = numerical_processor.transform(df)
    
    return df


def safe_numeric_conversion(series: pd.Series, default_value: float = 0.0) -> pd.Series:
    """
    Safely convert a series to numeric values.
    
    Args:
        series: Input pandas Series
        default_value: Value to use for non-numeric entries
        
    Returns:
        Converted numeric series
    """
    # If series is already numeric, return as is
    if pd.api.types.is_numeric_dtype(series):
        return series
    
    # Replace string 'Default' with default_value
    series = series.replace('Default', str(default_value))
    
    # Try converting to numeric, forcing errors to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')
    
    # Fill NaN with default_value
    numeric_series = numeric_series.fillna(default_value)
    
    return numeric_series


def convert_to_numeric(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Convert all columns to numeric type.
    
    Args:
        df: Input DataFrame
        feature_columns: Columns to convert
        
    Returns:
        DataFrame with numeric columns
        
    Raises:
        ValueError: If conversion fails
    """
    for col in feature_columns:
        logger.debug(f"Converting {col} to numeric. Current values: {df[col].unique()}")
        df[col] = safe_numeric_conversion(df[col])
        logger.debug(f"After conversion {col}: unique values={df[col].unique()}, dtype={df[col].dtype}")
    
    # Verify numeric conversion
    non_numeric_cols = df[feature_columns].select_dtypes(exclude=['int64', 'float64']).columns
    if not non_numeric_cols.empty:
        logger.error("Non-numeric columns found after preprocessing:")
        for col in non_numeric_cols:
            logger.error(f"{col}: dtype={df[col].dtype}, unique values={df[col].unique()}")
        raise ValueError(f"Following columns contain non-numeric values after preprocessing: {list(non_numeric_cols)}")
    
    # Convert to float type
    df[feature_columns] = df[feature_columns].astype(float)
    return df


def generate_predictions(
    model: xgb.Booster,
    df: pd.DataFrame,
    feature_columns: List[str],
    is_multiclass: bool
) -> np.ndarray:
    """
    Generate predictions using the XGBoost model.
    
    Args:
        model: XGBoost model
        df: Preprocessed DataFrame
        feature_columns: Feature columns to use
        is_multiclass: Whether this is a multiclass model
        
    Returns:
        numpy array of predictions
    """
    dtest = xgb.DMatrix(df[feature_columns].values)
    predictions = model.predict(dtest)
    
    if not is_multiclass and len(predictions.shape) == 1:
        predictions = np.column_stack([1 - predictions, predictions])
    
    return predictions


def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Generate predictions from preprocessed input data.
    
    Args:
        input_data: DataFrame containing the preprocessed input
        model_artifacts: Dictionary containing model and preprocessing objects
        
    Returns:
        Dict[str, np.ndarray]: Dictionary with raw and calibrated predictions
        
    Raises:
        ValueError: If input data is invalid or missing required features
    """
    try:
        # Extract configuration
        model = model_artifacts["model"]
        risk_processors = model_artifacts["risk_processors"]
        numerical_processor = model_artifacts["numerical_processor"]
        config = model_artifacts["config"]
        feature_columns = config["feature_columns"]
        is_multiclass = config["is_multiclass"]
        calibrator = model_artifacts.get("calibrator")
        
        # Validate input
        validate_input_data(input_data, feature_columns)
        
        # Assign column names if needed
        df = assign_column_names(input_data, feature_columns)
        
        # Apply preprocessing
        df = apply_preprocessing(df, feature_columns, risk_processors, numerical_processor)
        
        # Convert to numeric
        df = convert_to_numeric(df, feature_columns)
        
        # Generate raw predictions
        raw_predictions = generate_predictions(
            model=model,
            df=df,
            feature_columns=feature_columns,
            is_multiclass=is_multiclass
        )
        
        # Apply calibration if available, otherwise use raw predictions
        if calibrator is not None:
            try:
                calibrated_predictions = apply_calibration(raw_predictions, calibrator, is_multiclass)
                logger.info("Applied calibration to predictions")
            except Exception as e:
                logger.warning(f"Failed to apply calibration, using raw predictions: {e}")
                calibrated_predictions = raw_predictions.copy()
        else:
            # No calibrator available, use raw predictions
            logger.info("No calibration model found, using raw predictions for calibrated output")
            calibrated_predictions = raw_predictions.copy()
        
        return {
            "raw_predictions": raw_predictions,
            "calibrated_predictions": calibrated_predictions
        }
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        logger.error("Input data types and unique values:")
        for col in feature_columns:
            if col in input_data.columns:
                logger.error(f"{col}: dtype={input_data[col].dtype}, unique values={input_data[col].unique()}")
        raise


#--------------------------------------------------------------------------------
#                           OUTPUT BLOCK
#--------------------------------------------------------------------------------

def normalize_predictions(
    prediction_output: Union[np.ndarray, List, Dict[str, np.ndarray]]
) -> Tuple[List[List[float]], List[List[float]], bool]:
    """
    Normalize prediction output into a consistent format.
    
    Args:
        prediction_output: Raw prediction output from model or dict with raw and calibrated predictions
        
    Returns:
        Tuple of (raw scores list, calibrated scores list, is_multiclass flag)
        
    Raises:
        ValueError: If prediction format is invalid
    """
    # Handle the new dictionary output format
    if isinstance(prediction_output, dict):
        raw_predictions = prediction_output.get("raw_predictions")
        calibrated_predictions = prediction_output.get("calibrated_predictions")
        
        if raw_predictions is None:
            raise ValueError("Missing raw predictions in output dictionary")
            
        # Convert raw predictions to list format
        if isinstance(raw_predictions, np.ndarray):
            raw_scores_list = raw_predictions.tolist()
        elif isinstance(raw_predictions, list):
            raw_scores_list = raw_predictions
        else:
            msg = f"Unsupported raw prediction type: {type(raw_predictions)}"
            logger.error(msg)
            raise ValueError(msg)
            
        # Convert calibrated predictions to list format
        if calibrated_predictions is not None:
            if isinstance(calibrated_predictions, np.ndarray):
                calibrated_scores_list = calibrated_predictions.tolist()
            elif isinstance(calibrated_predictions, list):
                calibrated_scores_list = calibrated_predictions
            else:
                msg = f"Unsupported calibrated prediction type: {type(calibrated_predictions)}"
                logger.error(msg)
                calibrated_scores_list = raw_scores_list  # Fallback to raw scores
        else:
            # If no calibrated predictions, use raw scores
            calibrated_scores_list = raw_scores_list
    else:
        # Legacy code path for direct numpy array or list input
        if isinstance(prediction_output, np.ndarray):
            logger.info(f"Prediction output numpy array shape: {prediction_output.shape}")
            raw_scores_list = prediction_output.tolist()
        elif isinstance(prediction_output, list):
            raw_scores_list = prediction_output
        else:
            msg = f"Unsupported prediction output type: {type(prediction_output)}"
            logger.error(msg)
            raise ValueError(msg)
            
        # In legacy mode, calibrated scores are same as raw scores
        calibrated_scores_list = raw_scores_list
        
    if not raw_scores_list:
        raise ValueError("Empty prediction output")
    
    # Check if the predictions are already in list format
    if not isinstance(raw_scores_list[0], list):
        # Single probability output, convert to list of lists
        raw_scores_list = [[score] for score in raw_scores_list]
        if calibrated_scores_list == raw_scores_list:
            calibrated_scores_list = [[score] for score in calibrated_scores_list]
    
    # Check number of classes (length of probability vector)
    num_classes = len(raw_scores_list[0])
    is_multiclass = num_classes > 2
    
    logger.debug(f"Number of classes: {num_classes}, is_multiclass: {is_multiclass}")
    return raw_scores_list, calibrated_scores_list, is_multiclass


def format_json_record(raw_probs: List[float], calibrated_probs: List[float], is_multiclass: bool) -> Dict[str, Any]:
    """
    Format a single prediction record for JSON output with both raw and calibrated scores.
    
    Args:
        raw_probs: List of raw probability scores
        calibrated_probs: List of calibrated probability scores
        is_multiclass: Whether this is a multiclass prediction
        
    Returns:
        Dictionary containing formatted prediction record
        
    Notes:
        Binary classification (2 classes):
            - legacy-score: raw class-1 probability
            - calibrated-score: calibrated class-1 probability
            - custom-output-label: predicted class
        Multiclass (>2 classes):
            - prob_01, calibrated_prob_01, prob_02, calibrated_prob_02, etc.
            - custom-output-label: predicted class
    """
    if not raw_probs:
        raise ValueError("Empty probability list")
    
    # Ensure calibrated_probs exists, use raw_probs as fallback
    if calibrated_probs is None or len(calibrated_probs) != len(raw_probs):
        calibrated_probs = raw_probs
    
    # Use raw scores for prediction decision
    max_idx = raw_probs.index(max(raw_probs))
    
    if not is_multiclass:
        # Binary classification
        if len(raw_probs) != 2:
            raise ValueError(f"Binary classification expects 2 probabilities, got {len(raw_probs)}")
            
        # Order: legacy-score, calibrated-score, custom-output-label
        record = {
            "legacy-score": str(raw_probs[1]),           # Raw class-1 probability
            "calibrated-score": str(calibrated_probs[1]),  # Calibrated class-1 probability
            "custom-output-label": f"class-{max_idx}"      # Prediction based on raw scores
        }
    else:
        # Multiclass: include all probabilities in interleaved format
        record = {}
        
        # Interleaved raw and calibrated probabilities
        for i in range(len(raw_probs)):
            class_prefix = str(i+1).zfill(2)
            record[f"prob_{class_prefix}"] = str(raw_probs[i])
            record[f"calibrated_prob_{class_prefix}"] = str(calibrated_probs[i])
            
        # Add the predicted class at the end
        record["custom-output-label"] = f"class-{max_idx}"
    
    return record


def format_json_response(
    raw_scores_list: List[List[float]],
    calibrated_scores_list: List[List[float]],
    is_multiclass: bool
) -> Tuple[str, str]:
    """
    Format predictions as JSON response with both raw and calibrated scores.
    
    Args:
        raw_scores_list: List of raw prediction scores
        calibrated_scores_list: List of calibrated prediction scores
        is_multiclass: Whether this is a multiclass prediction
        
    Returns:
        Tuple of (JSON response string, content type)
        
    Example outputs:
        Binary: {
            "predictions": [
                {
                    "legacy-score": "0.7",
                    "calibrated-score": "0.75",
                    "custom-output-label": "class-1"
                },
                ...
            ]
        }
        
        Multiclass: {
            "predictions": [
                {
                    "prob_01": "0.2",
                    "calibrated_prob_01": "0.18",
                    "prob_02": "0.3", 
                    "calibrated_prob_02": "0.32",
                    "prob_03": "0.5",
                    "calibrated_prob_03": "0.5",
                    "custom-output-label": "class-2"
                },
                ...
            ]
        }
    """
    output_records = [
        format_json_record(raw_probs, cal_probs, is_multiclass) 
        for raw_probs, cal_probs in zip(raw_scores_list, calibrated_scores_list)
    ]
    
    # Simple response format without metadata
    response = json.dumps({"predictions": output_records})
    return response, CONTENT_TYPE_JSON


def format_csv_response(
    raw_scores_list: List[List[float]],
    calibrated_scores_list: List[List[float]],
    is_multiclass: bool
) -> Tuple[str, str]:
    """
    Format predictions as CSV response without headers.
    
    Args:
        raw_scores_list: List of raw prediction scores
        calibrated_scores_list: List of calibrated prediction scores
        is_multiclass: Whether this is a multiclass prediction
        
    Returns:
        Tuple of (CSV response string, content type)
        
    Notes:
        Binary classification ordering: legacy-score, calibrated-score, custom-output-label
        Multiclass ordering: prob_01, calibrated_prob_01, prob_02, calibrated_prob_02, ..., custom-output-label
    """
    csv_lines = []
    
    # Ensure calibrated scores exist, use raw scores as fallback
    if calibrated_scores_list is None or len(calibrated_scores_list) != len(raw_scores_list):
        calibrated_scores_list = raw_scores_list
    
    if not is_multiclass:
        # Binary classification - no header
        for i, raw_probs in enumerate(raw_scores_list):
            if len(raw_probs) != 2:
                raise ValueError(f"Binary classification expects 2 probabilities, got {len(raw_probs)}")
            
            # Raw score (legacy-score)
            raw_score = round(float(raw_probs[1]), 4)  # class-1 probability
            
            # Calibrated score (calibrated-score)
            calibrated_score = round(float(calibrated_scores_list[i][1]), 4)
            
            # Output label (using raw scores for prediction)
            prediction = "class-1" if raw_probs[1] > raw_probs[0] else "class-0"
            
            # Create line with exactly this order: legacy-score, calibrated-score, custom-output-label
            line = [f"{raw_score:.4f}", f"{calibrated_score:.4f}", prediction]
            csv_lines.append(",".join(map(str, line)))
    else:
        # Multiclass - no header
        for i, raw_probs in enumerate(raw_scores_list):
            calibrated_probs = calibrated_scores_list[i]
            num_classes = len(raw_probs)
            
            # Create interleaved raw and calibrated probabilities
            line = []
            for class_idx in range(num_classes):
                # Raw probability
                raw_prob = round(float(raw_probs[class_idx]), 4)
                line.append(f"{raw_prob:.4f}")
                
                # Calibrated probability
                cal_prob = round(float(calibrated_probs[class_idx]), 4)
                line.append(f"{cal_prob:.4f}")
            
            # Add prediction (using raw scores for prediction)
            max_idx = raw_probs.index(max(raw_probs))
            line.append(f"class-{max_idx}")
            
            csv_lines.append(",".join(map(str, line)))

    response_body = "\n".join(csv_lines) + "\n"
    return response_body, CONTENT_TYPE_CSV


def output_fn(
    prediction_output: Union[np.ndarray, List, Dict[str, np.ndarray]], 
    accept: str = CONTENT_TYPE_JSON
) -> Tuple[str, str]:
    """
    Serializes the prediction output.

    Args:
        prediction_output: Model predictions (raw and calibrated)
        accept: The requested response MIME type

    Returns:
        Tuple[str, str]: (response_body, content_type)
        
    Raises:
        ValueError: If prediction output format is invalid or content type is unsupported
    """
    logger.info(f"Received prediction output of type: {type(prediction_output)} for accept type: {accept}")

    try:
        # Normalize prediction format
        raw_scores_list, calibrated_scores_list, is_multiclass = normalize_predictions(prediction_output)
        
        # Format response based on accept type
        if accept.lower() == CONTENT_TYPE_JSON:
            return format_json_response(raw_scores_list, calibrated_scores_list, is_multiclass)
            
        elif accept.lower() == CONTENT_TYPE_CSV:
            return format_csv_response(raw_scores_list, calibrated_scores_list, is_multiclass)
            
        else:
            logger.error(f"Unsupported accept type: {accept}")
            error_msg = (
                f"Unsupported accept type: {accept}. "
                f"Supported types are {CONTENT_TYPE_JSON} and {CONTENT_TYPE_CSV}"
            )
            raise ValueError(error_msg)

    except Exception as e:
        logger.error(f"Error during output formatting: {e}", exc_info=True)
        error_response = json.dumps({
            'error': f'Failed to format output: {e}',
            'version': __version__
        })
        return error_response, CONTENT_TYPE_JSON
