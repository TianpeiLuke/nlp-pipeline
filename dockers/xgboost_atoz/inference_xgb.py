#!/usr/bin/env python3

# Standard library imports
import os
import json
import logging
import pickle as pkl  # Add this line
from pathlib import Path
from typing import Dict, Any, Union, Tuple, List, Optional
from io import StringIO, BytesIO

# Third-party imports
import boto3
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
CALIBRATION_MODEL_FILE = "calibration_model.joblib"

# Content types
CONTENT_TYPE_CSV = 'text/csv'
CONTENT_TYPE_JSON = 'application/json'
CONTENT_TYPE_PARQUET = 'application/x-parquet'

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
    Load calibration model if it exists in model.tar.gz
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Calibration model or None if not found
    """
    calibration_path = os.path.join(model_dir, CALIBRATION_MODEL_FILE)
    if os.path.exists(calibration_path):
        try:
            import joblib
            calibrator = joblib.load(calibration_path)
            logger.info(f"Loaded calibration model: {type(calibrator).__name__}")
            return calibrator
        except Exception as e:
            logger.warning(f"Failed to load calibration model: {e}")
            return None
    else:
        logger.info("No calibration model found - using raw scores")
        return None


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
        
        # Load calibration model (optional)
        calibrator = load_calibration_model(model_dir)
        
        # Create configuration
        config = create_model_config(model, feature_columns, hyperparameters)
        
        return {
            "model": model,
            "risk_processors": risk_processors,
            "numerical_processor": numerical_processor,
            "feature_importance": feature_importance,
            "config": config,
            "calibrator": calibrator,
            "version": __version__
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
) -> Union[pd.DataFrame, Response]:
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


def apply_calibration(raw_scores: np.ndarray, calibrator: Any, is_multiclass: bool) -> np.ndarray:
    """
    Apply calibration to raw scores if calibrator is available.
    
    Args:
        raw_scores: Raw model scores (probabilities)
        calibrator: Trained calibration model
        is_multiclass: Whether this is a multiclass model
        
    Returns:
        Calibrated scores for the legacy-score field
    """
    if calibrator is None:
        return None
    
    try:
        # Handle different calibration model types
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        
        # Extract probabilities for calibration
        if is_multiclass:
            # For multiclass, calibrate the class-0 probability (legacy-score)
            if len(raw_scores.shape) == 2 and raw_scores.shape[1] > 2:
                target_probs = raw_scores[:, 0]  # Use class-0 probabilities
            else:
                target_probs = raw_scores.flatten()
        else:
            # For binary classification, calibrate class-1 probability
            if len(raw_scores.shape) == 2 and raw_scores.shape[1] == 2:
                target_probs = raw_scores[:, 1]  # Use class-1 probabilities
            else:
                target_probs = raw_scores.flatten()
        
        # Apply calibration based on model type
        if isinstance(calibrator, IsotonicRegression):
            calibrated_probs = calibrator.transform(target_probs)
        elif isinstance(calibrator, LogisticRegression):
            calibrated_probs = calibrator.predict_proba(target_probs.reshape(-1, 1))[:, 1]
        else:
            # Assume GAM or other model with predict_proba method
            calibrated_probs = calibrator.predict_proba(target_probs.reshape(-1, 1))
            if len(calibrated_probs.shape) == 2:
                calibrated_probs = calibrated_probs[:, 1]  # Extract class-1 probabilities
        
        logger.info(f"Applied calibration: {type(calibrator).__name__} for {'multiclass' if is_multiclass else 'binary'} model")
        return calibrated_probs
        
    except Exception as e:
        logger.warning(f"Calibration failed, using raw scores: {e}")
        return None


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


def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate predictions from preprocessed input data.
    
    Args:
        input_data: DataFrame containing the preprocessed input
        model_artifacts: Dictionary containing model and preprocessing objects
        
    Returns:
        Dict containing both raw and calibrated predictions
        
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
            is_multiclass=config["is_multiclass"]
        )
        
        # Apply calibration if available (for both binary and multiclass)
        calibrated_scores = None
        if calibrator is not None:
            calibrated_scores = apply_calibration(raw_predictions, calibrator, config["is_multiclass"])
        
        return {
            "raw_predictions": raw_predictions,
            "calibrated_scores": calibrated_scores
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
    prediction_output: Union[np.ndarray, List]
) -> Tuple[List[List[float]], bool]:
    """
    Normalize prediction output into a consistent format.
    
    Args:
        prediction_output: Raw prediction output from model
        
    Returns:
        Tuple of (normalized scores list, is_multiclass flag)
        
    Raises:
        ValueError: If prediction format is invalid
    """
    if isinstance(prediction_output, np.ndarray):
        logger.info(f"Prediction output numpy array shape: {prediction_output.shape}")
        scores_list = prediction_output.tolist()
    elif isinstance(prediction_output, list):
        scores_list = prediction_output
    else:
        msg = f"Unsupported prediction output type: {type(prediction_output)}"
        logger.error(msg)
        raise ValueError(msg)
        
    if not scores_list:
        raise ValueError("Empty prediction output")
    
    # Check if the predictions are already in list format
    if not isinstance(scores_list[0], list):
        # Single probability output, convert to list of lists
        scores_list = [[score] for score in scores_list]
    
    # Check number of classes (length of probability vector)
    num_classes = len(scores_list[0])
    is_multiclass = num_classes > 2
    
    logger.debug(f"Number of classes: {num_classes}, is_multiclass: {is_multiclass}")
    return scores_list, is_multiclass


def format_json_record(probs: List[float], is_multiclass: bool, calibrated_score: Optional[float] = None) -> Dict[str, Any]:
    """
    Format a single prediction record for JSON output.
    
    Args:
        probs: List of probability scores
        is_multiclass: Whether this is a multiclass prediction
        calibrated_score: Optional calibrated score for binary classification
        
    Returns:
        Dictionary containing formatted prediction record
        
    Notes:
        Binary classification (2 classes):
            - legacy-score: class-1 probability
            - calibrated-score: calibrated class-1 probability (if available)
            - no additional probability fields
        Multiclass (>2 classes):
            - legacy-score: class-0 probability
            - prob_02 onwards: remaining class probabilities
    """
    if not probs:
        raise ValueError("Empty probability list")
    
    max_idx = probs.index(max(probs))
    record = {}
    
    if not is_multiclass:
        # Binary classification: only include class-1 probability as legacy-score
        if len(probs) != 2:
            raise ValueError(f"Binary classification expects 2 probabilities, got {len(probs)}")
        record["legacy-score"] = str(probs[1])  # class-1 probability
        
        # Add calibrated score if available
        if calibrated_score is not None:
            record["calibrated-score"] = str(calibrated_score)
    else:
        # Multiclass: include all probabilities
        record["legacy-score"] = str(probs[0])  # class-0 probability
        
        # Add calibrated score if available (for multiclass, calibrate legacy-score)
        if calibrated_score is not None:
            record["calibrated-score"] = str(calibrated_score)
        
        # Add remaining probabilities (starting from prob_02)
        record.update({
            f"prob_{str(i+1).zfill(2)}": str(p) 
            for i, p in enumerate(probs[1:])  # Start from second probability
        })
    
    # Add the predicted class
    record["output-label"] = f"class-{max_idx}"
    
    return record


def format_json_response(
    scores_list: List[List[float]], 
    is_multiclass: bool
) -> Tuple[str, str]:
    """
    Format predictions as JSON response.
    
    Args:
        scores_list: List of prediction scores
        is_multiclass: Whether this is a multiclass prediction
        
    Returns:
        Tuple of (JSON response string, content type)
        
    Example outputs:
        Binary: {
            "predictions": [
                {
                    "legacy-score": 0.7,
                    "output-label": "class-1"
                },
                ...
            ]
        }
        
        Multiclass: {
            "predictions": [
                {
                    "legacy-score": 0.2,
                    "prob_02": 0.3,
                    "prob_03": 0.5,
                    "output-label": "class-2"
                },
                ...
            ]
        }
    """
    output_records = [
        format_json_record(probs, is_multiclass) 
        for probs in scores_list
    ]
    
    # Simple response format without metadata
    response = json.dumps({"predictions": output_records})
    return response, CONTENT_TYPE_JSON


def format_csv_response(
    scores_list: List[List[float]],
    is_multiclass: bool
) -> Tuple[str, str]:
    """
    Format predictions as CSV response.
    
    Args:
        scores_list: List of prediction scores
        is_multiclass: Whether this is a multiclass prediction
        
    Returns:
        Tuple of (CSV response string, content type)
        
    Notes:
        For binary classification, first column is class-1 probability (legacy-score)
        For multiclass, probability columns are in original order
    """
    csv_lines = []
    
    if not is_multiclass:
        # Binary classification
        header = ["class_1_prob", "prediction"]  # class-1 probability first
        csv_lines.append(",".join(header))
        
        for probs in scores_list:
            if len(probs) != 2:
                raise ValueError(f"Binary classification expects 2 probabilities, got {len(probs)}")
            
            # Use class-1 probability (second value)
            class_1_prob = round(float(probs[1]), 4)
            prediction = "class-1" if probs[1] > probs[0] else "class-0"
            
            line = [f"{class_1_prob:.4f}", prediction]
            csv_lines.append(",".join(map(str, line)))
    else:
        # Multiclass
        num_classes = len(scores_list[0])
        header = [f"class_{i}_prob" for i in range(num_classes)]
        header.append("prediction")
        csv_lines.append(",".join(header))
        
        for probs in scores_list:
            # Format all probabilities
            formatted_probs = [f"{round(float(p), 4):.4f}" for p in probs]
            max_idx = probs.index(max(probs))
            
            line = formatted_probs + [f"class-{max_idx}"]
            csv_lines.append(",".join(map(str, line)))

    response_body = "\n".join(csv_lines) + "\n"
    return response_body, CONTENT_TYPE_CSV


def format_csv_response_with_calibration(
    scores_list: List[List[float]],
    is_multiclass: bool,
    calibrated_scores: Optional[np.ndarray] = None
) -> Tuple[str, str]:
    """
    Format predictions as CSV response with optional calibrated scores.
    
    Args:
        scores_list: List of prediction scores
        is_multiclass: Whether this is a multiclass prediction
        calibrated_scores: Optional calibrated scores
        
    Returns:
        Tuple of (CSV response string, content type)
        
    Notes:
        Column order: legacy-score, calibrated-score (if available), other probs, prediction
        For binary: legacy-score (class-1), calibrated-score (if available), prediction
        For multiclass: legacy-score (class-0), calibrated-score (if available), prob_02, prob_03, ..., prediction
    """
    csv_lines = []
    
    if not is_multiclass:
        # Binary classification
        header = ["legacy_score"]
        if calibrated_scores is not None:
            header.append("calibrated_score")
        header.append("prediction")
        csv_lines.append(",".join(header))
        
        for i, probs in enumerate(scores_list):
            if len(probs) != 2:
                raise ValueError(f"Binary classification expects 2 probabilities, got {len(probs)}")
            
            # Legacy score (class-1 probability)
            legacy_score = round(float(probs[1]), 4)
            line = [f"{legacy_score:.4f}"]
            
            # Add calibrated score if available
            if calibrated_scores is not None and i < len(calibrated_scores):
                calibrated_score = round(float(calibrated_scores[i]), 4)
                line.append(f"{calibrated_score:.4f}")
            
            # Add prediction
            prediction = "class-1" if probs[1] > probs[0] else "class-0"
            line.append(prediction)
            
            csv_lines.append(",".join(map(str, line)))
    else:
        # Multiclass
        header = ["legacy_score"]  # class-0 probability
        if calibrated_scores is not None:
            header.append("calibrated_score")
        
        # Add remaining probability columns
        num_classes = len(scores_list[0])
        for i in range(1, num_classes):  # Start from class-1 onwards
            header.append(f"prob_{str(i+1).zfill(2)}")
        header.append("prediction")
        csv_lines.append(",".join(header))
        
        for i, probs in enumerate(scores_list):
            # Legacy score (class-0 probability)
            legacy_score = round(float(probs[0]), 4)
            line = [f"{legacy_score:.4f}"]
            
            # Add calibrated score if available
            if calibrated_scores is not None and i < len(calibrated_scores):
                calibrated_score = round(float(calibrated_scores[i]), 4)
                line.append(f"{calibrated_score:.4f}")
            
            # Add remaining probabilities (class-1 onwards)
            for j in range(1, len(probs)):
                prob = round(float(probs[j]), 4)
                line.append(f"{prob:.4f}")
            
            # Add prediction
            max_idx = probs.index(max(probs))
            line.append(f"class-{max_idx}")
            
            csv_lines.append(",".join(map(str, line)))

    response_body = "\n".join(csv_lines) + "\n"
    return response_body, CONTENT_TYPE_CSV


def format_json_response_with_calibration(
    scores_list: List[List[float]], 
    is_multiclass: bool,
    calibrated_scores: Optional[np.ndarray] = None
) -> Tuple[str, str]:
    """
    Format predictions as JSON response with optional calibrated scores.
    
    Args:
        scores_list: List of prediction scores
        is_multiclass: Whether this is a multiclass prediction
        calibrated_scores: Optional calibrated scores
        
    Returns:
        Tuple of (JSON response string, content type)
    """
    output_records = []
    
    for i, probs in enumerate(scores_list):
        calibrated_score = None
        if calibrated_scores is not None and i < len(calibrated_scores):
            calibrated_score = float(calibrated_scores[i])
        
        record = format_json_record(probs, is_multiclass, calibrated_score)
        output_records.append(record)
    
    response = json.dumps({"predictions": output_records})
    return response, CONTENT_TYPE_JSON


def output_fn(
    prediction_output: Union[np.ndarray, List, Dict], 
    accept: str = CONTENT_TYPE_JSON
) -> Tuple[str, str]:
    """
    Serializes the prediction output including calibrated scores.

    Args:
        prediction_output: Model predictions as numpy array, list, or dict with calibrated scores
        accept: The requested response MIME type

    Returns:
        Tuple[str, str]: (response_body, content_type)
        
    Raises:
        ValueError: If prediction output format is invalid or content type is unsupported
    """
    logger.info(f"Received prediction output of type: {type(prediction_output)} for accept type: {accept}")

    try:
        # Handle new dictionary format with calibrated scores
        if isinstance(prediction_output, dict):
            raw_predictions = prediction_output["raw_predictions"]
            calibrated_scores = prediction_output.get("calibrated_scores")
        else:
            # Backward compatibility
            raw_predictions = prediction_output
            calibrated_scores = None
        
        # Normalize prediction format
        scores_list, is_multiclass = normalize_predictions(raw_predictions)
        
        # Format response based on accept type
        if accept.lower() == CONTENT_TYPE_JSON:
            return format_json_response_with_calibration(scores_list, is_multiclass, calibrated_scores)
            
        elif accept.lower() == CONTENT_TYPE_CSV:
            # Use calibration-aware CSV formatting
            return format_csv_response_with_calibration(scores_list, is_multiclass, calibrated_scores)
            
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
