#!/usr/bin/env python3
import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np
import pickle as pkl
import xgboost as xgb


# -------------------------------------------------------------------------
# Assuming the processor is in a directory that can be imported
# -------------------------------------------------------------------------
from processing.risk_table_processor import RiskTableMappingProcessor
from processing.numerical_imputation_processor import NumericalVariableImputationProcessor


# -------------------------------------------------------------------------
# Logging setup - Updated for CloudWatch compatibility
# -------------------------------------------------------------------------
def setup_logging():
    """Configure logging for CloudWatch compatibility"""
    # Remove any existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True,
        handlers=[
            # StreamHandler with stdout for CloudWatch
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure our module's logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = True  # Allow propagation to root logger
    
    # Force flush stdout
    sys.stdout.flush()
    
    return logger

# Initialize logger
logger = setup_logging()

# -------------------------------------------------------------------------
# Pydantic V2 model for all hyperparameters
# -------------------------------------------------------------------------
from pydantic import BaseModel, Field, model_validator
from hyperparameters.hyperparameters_xgboost import XGBoostModelHyperparameters

class XGBoostConfig(XGBoostModelHyperparameters):
    """
    Load everything from your pipeline’s XGBoostModelHyperparameters,
    plus the two risk-table params this script needs.
    """
    smooth_factor: float = Field(
        default=0.0, description="Smoothing factor for risk table"
    )
    count_threshold: int = Field(
        default=0, description="Minimum count threshold for risk table"
    )

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
def load_and_validate_config(hparam_path: str) -> dict:
    """Loads and validates the hyperparameters JSON file."""
    try:
        with open(hparam_path, "r") as f:
            config = json.load(f)
        
        required_keys = ["tab_field_list", "cat_field_list", "label_name", "is_binary", "num_classes"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config: {key}")
        
        # Validate class_weights if present
        if "class_weights" in config:
            if len(config["class_weights"]) != config["num_classes"]:
                raise ValueError(f"Number of class weights ({len(config['class_weights'])}) "
                                 f"does not match number of classes ({config['num_classes']})")
        
        return config
    except Exception as err:
        logger.error(f"Failed to load/validate hyperparameters: {err}")
        raise

def find_first_data_file(data_dir: str) -> str:
    """Finds the first supported data file in a directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return os.path.join(data_dir, fname)
    return None

def load_datasets(input_path: str) -> tuple:
    """Loads the training, validation, and test datasets."""
    train_file = find_first_data_file(os.path.join(input_path, "train"))
    val_file = find_first_data_file(os.path.join(input_path, "val"))
    test_file = find_first_data_file(os.path.join(input_path, "test"))

    if not train_file or not val_file or not test_file:
        raise FileNotFoundError("Training, validation, or test data file not found in the expected subfolders.")

    train_df = pd.read_parquet(train_file) if train_file.endswith('.parquet') else pd.read_csv(train_file)
    val_df = pd.read_parquet(val_file) if val_file.endswith('.parquet') else pd.read_csv(val_file)
    test_df = pd.read_parquet(test_file) if test_file.endswith('.parquet') else pd.read_csv(test_file)
    
    logger.info(f"Loaded data -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    return train_df, val_df, test_df

def apply_numerical_imputation(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Applies numerical imputation to the datasets."""
    imputer = NumericalVariableImputationProcessor(variables=config['tab_field_list'], strategy='mean')
    imputer.fit(train_df)
    
    train_df_imputed = imputer.transform(train_df)
    val_df_imputed = imputer.transform(val_df)
    test_df_imputed = imputer.transform(test_df)
    
    return train_df_imputed, val_df_imputed, test_df_imputed, imputer.get_params()['imputation_dict']

def fit_and_apply_risk_tables(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """Fits risk tables on training data and applies them to all splits."""
    risk_processors = {}
    train_df_transformed = train_df.copy()
    val_df_transformed = val_df.copy()
    test_df_transformed = test_df.copy()

    for var in config['cat_field_list']:
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=config['label_name'],
            smooth_factor=config.get('smooth_factor', 0.0),
            count_threshold=config.get('count_threshold', 0),
        )
        proc.fit(train_df)
        risk_processors[var] = proc
        
        train_df_transformed[var] = proc.transform(train_df_transformed[var])
        val_df_transformed[var] = proc.transform(val_df_transformed[var])
        test_df_transformed[var] = proc.transform(test_df_transformed[var])
        
    consolidated_risk_tables = {var: proc.get_risk_tables() for var, proc in risk_processors.items()}
    return train_df_transformed, val_df_transformed, test_df_transformed, consolidated_risk_tables

def prepare_dmatrices(config: dict, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[xgb.DMatrix, xgb.DMatrix, List[str]]:
    """
    Prepares XGBoost DMatrix objects from dataframes.
    
    Returns:
        Tuple containing:
        - Training DMatrix
        - Validation DMatrix
        - List of feature columns in the exact order used for the model
    """
    # Maintain exact ordering of features as they'll be used in the model
    feature_columns = config['tab_field_list'] + config['cat_field_list']
    
    # Check for any remaining NaN/inf values
    X_train = train_df[feature_columns].astype(float)
    X_val = val_df[feature_columns].astype(float)
    
    if X_train.isna().any().any() or np.isinf(X_train).any().any():
        raise ValueError("Training data contains NaN or inf values after preprocessing")
    if X_val.isna().any().any() or np.isinf(X_val).any().any():
        raise ValueError("Validation data contains NaN or inf values after preprocessing")
        
    dtrain = xgb.DMatrix(X_train.values, label=train_df[config['label_name']].astype(int).values)
    dval = xgb.DMatrix(X_val.values, label=val_df[config['label_name']].astype(int).values)
    
    # Set feature names in DMatrix to ensure they're preserved
    dtrain.feature_names = feature_columns
    dval.feature_names = feature_columns
    
    return dtrain, dval, feature_columns

def train_model(config: dict, dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> xgb.Booster:
    """
    Trains the XGBoost model.
    
    Args:
        config: Configuration dictionary containing model parameters
        dtrain: Training data as XGBoost DMatrix
        dval: Validation data as XGBoost DMatrix
        
    Returns:
        Trained XGBoost model
    """
    # Base parameters
    xgb_params = {
        "eta": config.get("eta", 0.1),
        "gamma": config.get("gamma", 0),
        "max_depth": config.get("max_depth", 6),
        "subsample": config.get("subsample", 1),
        "colsample_bytree": config.get("colsample_bytree", 1),
        "lambda": config.get("lambda_xgb", 1),
        "alpha": config.get("alpha_xgb", 0)
    }
    
    # Set objective and num_class based on hyperparameters
    # Handle class weights
    if config.get("is_binary", True):
        xgb_params["objective"] = "binary:logistic"
        if "class_weights" in config and len(config["class_weights"]) == 2:
            # For binary classification, use scale_pos_weight
            xgb_params["scale_pos_weight"] = config["class_weights"][1] / config["class_weights"][0]
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = config["num_classes"]
    
    logger.info(f"Starting XGBoost training with params: {xgb_params}")
    logger.info(f"Number of classes from config: {config.get('num_classes', 2)}")
    
    # Print label distribution for debugging
    y_train = dtrain.get_label()
    y_val = dval.get_label()
    logger.info(f"Label distribution in training data: {pd.Series(y_train).value_counts().sort_index()}")
    logger.info(f"Label distribution in validation data: {pd.Series(y_val).value_counts().sort_index()}")
    
    # Handle class weights for multiclass
    if not config.get("is_binary", True) and "class_weights" in config:
        sample_weights = np.ones(len(y_train))
        for i, weight in enumerate(config["class_weights"]):
            sample_weights[y_train == i] = weight
        dtrain.set_weight(sample_weights)
    
    return xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=config.get("num_round", 100),
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.get("early_stopping_rounds", 10),
        verbose_eval=True
    )

def save_artifacts(model: xgb.Booster, risk_tables: dict, impute_dict: dict, model_path: str, feature_columns: List[str], config: dict):
    """
    Saves the trained model and preprocessing artifacts.
    
    Args:
        model: Trained XGBoost model
        risk_tables: Dictionary of risk tables
        impute_dict: Dictionary of imputation values
        model_path: Path to save model artifacts
        feature_columns: List of feature column names
        config: Configuration dictionary containing hyperparameters
    """
    os.makedirs(model_path, exist_ok=True)
    
    # Save XGBoost model
    model_file = os.path.join(model_path, "xgboost_model.bst")
    model.save_model(model_file)
    logger.info(f"Saved XGBoost model to {model_file}")

    # Save risk tables
    risk_map_file = os.path.join(model_path, "risk_table_map.pkl")
    with open(risk_map_file, "wb") as f:
        pkl.dump(risk_tables, f)
    logger.info(f"Saved consolidated risk table map to {risk_map_file}")
    
    # Save imputation dictionary
    impute_file = os.path.join(model_path, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(impute_dict, f)
    logger.info(f"Saved imputation dictionary to {impute_file}")

    # Save feature importance
    fmap_json = os.path.join(model_path, "feature_importance.json")
    with open(fmap_json, "w") as f:
        json.dump(model.get_fscore(), f, indent=2)
    logger.info(f"Saved feature importance to {fmap_json}")
    
    # Save feature columns with ordering information
    feature_columns_file = os.path.join(model_path, "feature_columns.txt")
    with open(feature_columns_file, "w") as f:
        # Add a header comment to document the importance of ordering
        f.write("# Feature columns in exact order required for XGBoost model inference\n")
        f.write("# DO NOT MODIFY THE ORDER OF THESE COLUMNS\n")
        f.write("# Each line contains: <column_index>,<column_name>\n")
        for idx, column in enumerate(feature_columns):
            f.write(f"{idx},{column}\n")
    logger.info(f"Saved ordered feature columns to {feature_columns_file}")

    # Save hyperparameters configuration
    hyperparameters_file = os.path.join(model_path, "hyperparameters.json")
    with open(hyperparameters_file, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    logger.info(f"Saved hyperparameters configuration to {hyperparameters_file}")
    
# -------------------------------------------------------------------------
# Main Orchestrator
# -------------------------------------------------------------------------
def main(hparam_path: str, input_path: str, model_path: str):
    """Main function to execute the XGBoost training logic."""
    logger.info("Starting XGBoost training process...")
    logger.info(f"Loading configuration from {hparam_path}")
    config = load_and_validate_config(hparam_path)
    logger.info("Configuration loaded successfully")
    
    logger.info("Loading datasets...")
    train_df, val_df, test_df = load_datasets(input_path)
    logger.info("Datasets loaded successfully")
    
    # Apply numerical imputation
    logger.info("Starting numerical imputation...")
    train_df, val_df, test_df, impute_dict = apply_numerical_imputation(config, train_df, val_df, test_df)
    logger.info("Numerical imputation completed")
    
    # Apply risk table mapping
    logger.info("Starting risk table mapping...")
    train_df, val_df, test_df, risk_tables = fit_and_apply_risk_tables(config, train_df, val_df, test_df)
    logger.info("Risk table mapping completed")
    
    logger.info("Preparing DMatrices for XGBoost...")
    dtrain, dval, feature_columns = prepare_dmatrices(config, train_df, val_df)
    logger.info("DMatrices prepared successfully")
    logger.info(f"Using {len(feature_columns)} features in order: {feature_columns}")
    
    logger.info("Starting model training...")
    model = train_model(config, dtrain, dval)
    logger.info("Model training completed")
    
    logger.info("Saving model artifacts...")
    save_artifacts(
        model=model,
        risk_tables=risk_tables,
        impute_dict=impute_dict,
        model_path=model_path,
        feature_columns=feature_columns,
        config=config
    )
    
    logger.info("Training script finished successfully.")

# -------------------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Script starting...")
    prefix = "/opt/ml"
    
    # The data from the previous step is on the main 'data' channel
    input_path = os.path.join(prefix, "input", "data")
    
    # The model artifacts are saved to the standard model directory
    model_path = os.path.join(prefix, "model")
    
    # FIX: The path to hyperparameters is now determined by the 'config' input channel,
    # which is the recommended way to pass large configuration files.
    config_channel_path = os.path.join(input_path, "config")
    hparam_path = os.path.join(config_channel_path, "hyperparameters.json")

    try:
        logger.info(f"Starting main process with paths: input={input_path}, model={model_path}")
        main(hparam_path, input_path, model_path)
    except Exception:
        logger.error(f"Exception during training:\n{traceback.format_exc()}")
        sys.exit(1)
