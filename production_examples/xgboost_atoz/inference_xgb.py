#!/usr/bin/env python3
import os
import sys

from subprocess import check_call
import boto3

os.environ["AWS_STS_REGIONAL_ENDPOINTS"] = "regional"
sts = boto3.client("sts", region_name="us-east-1")
caller_identity = sts.get_caller_identity()
assumed_role_object = sts.assume_role(
    RoleArn="arn:aws:iam::675292366480:role/SecurePyPIReadRole_" + caller_identity["Account"],
    RoleSessionName="SecurePypiReadRole",
)
credentials = assumed_role_object["Credentials"]
code_artifact_client = boto3.client(
    "codeartifact",
    aws_access_key_id=credentials["AccessKeyId"],
    aws_secret_access_key=credentials["SecretAccessKey"],
    aws_session_token=credentials["SessionToken"],
    region_name="us-west-2",
)
token = code_artifact_client.get_authorization_token(domain="amazon", domainOwner="149122183214")[
    "authorizationToken"
]

check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        f"https://aws:{token}@amazon-149122183214.d.codeartifact.us-west-2.amazonaws.com/pypi/secure-pypi/simple/",
        "scikit-learn>=0.23.2,<1.0.0",
        "pandas>=1.2.0,<2.0.0",
        "beautifulsoup4>=4.9.3",
        "pyarrow>=4.0.0,<6.0.0",
        "pydantic>=2.0.0,<3.0.0",
        "typing-extensions>=4.2.0",
        "flask>=2.0.0,<3.0.0"
    ]
)
print("***********************Package Installed*********************")


import json
import logging
import pickle as pkl
from typing import Dict, Any, Union, Tuple
from pathlib import Path
from io import StringIO, BytesIO
from flask import Response


import pandas as pd
import numpy as np
import xgboost as xgb

from processing.risk_table_processor import RiskTableMappingProcessor
from processing.numerical_imputation_processor import NumericalVariableImputationProcessor


__version__ = "1.0.0"

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

def model_fn(model_dir: str) -> Dict[str, Any]:
    """
    Load the model and preprocessing artifacts from model_dir.
    
    Args:
        model_dir (str): Directory containing model artifacts
        
    Returns:
        Dict containing model, processors, and configuration
    """
    logger.info(f"Loading model from {model_dir}")
    
    try:
        # Load XGBoost model
        model = xgb.Booster()
        model.load_model(os.path.join(model_dir, "xgboost_model.bst"))
        
        # Load risk tables and create processors
        with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
            risk_tables = pkl.load(f)
            
        # Initialize risk table processors for each categorical feature
        risk_processors = {}
        for feature, risk_table in risk_tables.items():
            processor = RiskTableMappingProcessor(
                column_name=feature,
                label_name="label",  # Not used during inference
                risk_tables=risk_table
            )
            risk_processors[feature] = processor
            
        # Load imputation dictionary and create processor
        with open(os.path.join(model_dir, "impute_dict.pkl"), "rb") as f:
            impute_dict = pkl.load(f)
        
        numerical_processor = NumericalVariableImputationProcessor(
            imputation_dict=impute_dict
        )
            
        # Load feature importance (optional)
        with open(os.path.join(model_dir, "feature_importance.json"), "r") as f:
            feature_importance = json.load(f)
            
        # Determine model configuration
        config = {
            "is_multiclass": True if hasattr(model, 'num_class') and model.num_class() > 2 else False,
            "num_classes": model.num_class() if hasattr(model, 'num_class') else 2,
            "feature_columns": list(impute_dict.keys()) + list(risk_processors.keys())
        }
            
        return {
            "model": model,
            "risk_processors": risk_processors,
            "numerical_processor": numerical_processor,
            "feature_importance": feature_importance,
            "config": config,
            "version": __version__
        }
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        raise

def input_fn(request_body, request_content_type, context=None):
    """
    Deserialize the Invoke request body into an object we can perform prediction on.
    """
    logger.info(f"Received request with Content-Type: {request_content_type}") # Log content type
    try:
        if request_content_type == 'text/csv':
            logger.info("Processing content type: text/csv")
            decoded = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
            logger.debug(f"Decoded CSV data:\n{decoded[:500]}...") # Optional: Log decoded data
            try:
                df = pd.read_csv(StringIO(decoded), header=None, index_col=None)
                logger.info(f"Successfully parsed CSV into DataFrame. Shape: {df.shape}, Type: {type(df)}")
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse CSV data: {parse_error}")
                raise

        elif request_content_type == 'application/json':
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
                logger.info(f"Successfully parsed JSON into DataFrame. Shape: {df.shape}")
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse JSON data: {parse_error}")
                raise

        elif request_content_type == 'application/x-parquet':
            logger.info("Processing content type: application/x-parquet")
            df = pd.read_parquet(BytesIO(request_body))
            logger.info(f"Successfully parsed Parquet into DataFrame. Shape: {df.shape}, Type: {type(df)}")
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

def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Generate predictions from preprocessed input data.
    
    Args:
        input_data: DataFrame containing the preprocessed input
        model_artifacts: Dictionary containing model and preprocessing objects
        
    Returns:
        numpy array of predictions
    """
    try:
        # Get artifacts and config
        model = model_artifacts["model"]
        risk_processors = model_artifacts["risk_processors"]
        numerical_processor = model_artifacts["numerical_processor"]
        config = model_artifacts["config"]
        
        # Validate required features
        required_features = set(config["feature_columns"])
        missing_features = required_features - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Apply preprocessing
        df = input_data.copy()
        
        # Apply numerical imputation
        df = numerical_processor.transform(df)
        
        # Apply risk table mapping to categorical features
        for feature, processor in risk_processors.items():
            if feature in df.columns:
                logger.debug(f"Applying risk table mapping for feature: {feature}")
                df[feature] = processor.transform(df[feature])
                
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(df[config["feature_columns"]].values)
        
        # Get predictions
        predictions = model.predict(dtest)
        
        # Handle binary vs multiclass output format
        if not config["is_multiclass"] and len(predictions.shape) == 1:
            predictions = np.column_stack([1 - predictions, predictions])
            
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise


def output_fn(prediction_output, accept='application/json'):
    """
    Serializes the multi-class prediction output.

    Args:
        prediction_output: The output from predict_fn, expected to be a
                           numpy array of shape (N, num_classes) or list of lists.
        accept: The requested response MIME type (e.g., 'application/json').

    Returns:
        tuple: (response_body, content_type)
    """
    logger.info(f"Received prediction output of type: {type(prediction_output)} for accept type: {accept}")

    scores_list = None

    # Step 1: Normalize input format into a list of lists
    if isinstance(prediction_output, np.ndarray):
        logger.info(f"Prediction output numpy array shape: {prediction_output.shape}")
        scores_list = prediction_output.tolist()
    elif isinstance(prediction_output, list):
        scores_list = prediction_output
    else:
        msg = f"Unsupported prediction output type: {type(prediction_output)}"
        logger.error(msg)
        raise ValueError(msg)
        
    try:
        is_multiclass = isinstance(scores_list[0], list)

        # Step 2: JSON output formatting
        if accept.lower() == 'application/json':
            output_records = []
            for probs in scores_list:
                probs = probs if isinstance(probs, list) else [probs]
                max_idx = probs.index(max(probs)) if probs else -1

                # Create the base record with legacy-score for the first probability
                record = {"legacy-score": probs[0]} if probs else {"legacy-score": None}
        
                # Add the rest of the probabilities starting from prob_02
                record.update({
                    f"prob_{str(i+1).zfill(2)}": p 
                    for i, p in enumerate(probs[1:])
                })
                
                # Add the output label
                record["output-label"] = f"class-{max_idx}" if max_idx >= 0 else "unknown"
                
                output_records.append(record)

            response = json.dumps({"predictions": output_records})
            return response, 'application/json'

        # Step 3: CSV output formatting
        elif accept.lower() == 'text/csv':
            csv_lines = []
            for probs in scores_list:
                probs = probs if isinstance(probs, list) else [probs]
                max_idx = probs.index(max(probs)) if probs else -1
                formatted_probs = [round(float(p), 4) for p in probs]
                list_str = ",".join(f"{p:.4f}" for p in formatted_probs)
                
                line = [list_str] + [f"class-{max_idx}" if max_idx >= 0 else "unknown"]
                csv_lines.append(",".join(map(str, line)))

            response_body = "\n".join(csv_lines) + "\n"
            return response_body, 'text/csv'

        # Step 4: Unsupported content type
        else:
            logger.error(f"Unsupported accept type: {accept}")
            raise ValueError(f"Unsupported accept type: {accept}")

    except Exception as e:
        logger.error(f"Error during DataFrame creation or serialization in output_fn: {e}", exc_info=True)
        error_response = json.dumps({'error': f'Failed to serialize output: {e}'})
        return error_response, 'application/json'
