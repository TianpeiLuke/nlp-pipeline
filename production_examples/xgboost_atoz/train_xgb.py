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

from pydantic import BaseModel, Field, model_validator

# -------------------------------------------------------------------------
# Assuming the processor is in a directory that can be imported
# -------------------------------------------------------------------------
from processing.risk_table_processor import RiskTableMappingProcessor

# -------------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)

# -------------------------------------------------------------------------
# Pydantic V2 model for all hyperparameters
# -------------------------------------------------------------------------
class XGBoostConfig(BaseModel):
    """Pydantic V2 model for all hyperparameters expected from hyperparameters.json."""
    # ----- Data and general fields -----
    full_field_list: List[str] = Field(..., description="All field names.")
    cat_field_list: List[str] = Field(..., description="List of categorical column names.")
    tab_field_list: List[str] = Field(..., description="List of numeric column names.")
    id_name: str = Field(..., description="Name of the ID column.")
    label_name: str = Field(..., description="Name of the label column.")
    is_binary: bool = Field(default=True, description="Binary classification flag.")
    num_classes: int = Field(default=2, description="Number of classes.")
    
    # ----- XGBoost-specific parameters -----
    booster: str = Field(default="gbtree", description="Booster to use.")
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Learning rate.")
    gamma: float = Field(default=0.0, ge=0.0, description="Minimum loss reduction to split.")
    max_depth: int = Field(default=6, ge=0, description="Maximum tree depth.")
    min_child_weight: float = Field(default=1.0, ge=0.0, description="Minimum sum of instance weight.")
    subsample: float = Field(default=1.0, gt=0.0, le=1.0, description="Subsample ratio of rows.")
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0, description="Subsample ratio of columns per tree.")
    lambda_xgb: float = Field(default=1.0, ge=0.0, description="L2 regularization term.")
    alpha_xgb: float = Field(default=0.0, ge=0.0, description="L1 regularization term.")
    objective: str = Field(default="binary:logistic", description="Learning objective.")
    eval_metric: Optional[Union[str, List[str]]] = Field(default=None, description="Evaluation metric(s).")
    num_round: int = Field(default=100, ge=1, description="Number of boosting rounds.")
    early_stopping_rounds: Optional[int] = Field(default=None, ge=1, description="Enable early stopping.")
    
    # ----- Risk-table smoothing parameters -----
    smooth_factor: float = Field(default=0.0, ge=0.0, le=1.0, description="Smoothing factor for risk table.")
    count_threshold: int = Field(default=0, ge=0, description="Minimum count threshold for risk table.")

    @model_validator(mode="after")
    def _validate_all(cls, state: "XGBoostConfig") -> "XGBoostConfig":
        if state.early_stopping_rounds is not None and not state.eval_metric:
            raise ValueError("`early_stopping_rounds` requires `eval_metric` to be set.")
        return state

    class Config:
        extra = "forbid"

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def load_and_validate_config(hparam_path: str) -> XGBoostConfig:
    """Loads and validates the hyperparameters JSON file."""
    try:
        with open(hparam_path, "r") as f:
            raw_hparams = json.load(f)
        config = XGBoostConfig(**raw_hparams)
        logger.info("Successfully loaded and validated hyperparameters.")
        return config
    except Exception as err:
        logger.error(f"Failed to load/validate hyperparameters: {err}")
        raise

def find_first_data_file(data_dir: str) -> Optional[str]:
    """Finds the first supported data file in a directory."""
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return fname
    return None

def _read_any_data(path: str) -> pd.DataFrame:
    """Reads a data file into a pandas DataFrame, handling various formats."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype=object)
    elif path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".json"):
        return pd.read_json(path, lines=True, dtype=object)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def load_datasets(input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads the training, validation, and test datasets."""
    train_data_dir = os.path.join(input_path, "train")
    val_data_dir = os.path.join(input_path, "val")
    test_data_dir = os.path.join(input_path, "test")
    
    train_file = find_first_data_file(train_data_dir)
    val_file = find_first_data_file(val_data_dir)
    test_file = find_first_data_file(test_data_dir)

    if not train_file or not val_file or not test_file:
        raise FileNotFoundError("Training, validation, or test data file not found in the expected subfolders.")

    train_df = _read_any_data(os.path.join(train_data_dir, train_file))
    val_df = _read_any_data(os.path.join(val_data_dir, val_file))
    test_df = _read_any_data(os.path.join(test_data_dir, test_file))
    
    logger.info(f"Loaded data -> train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")
    return train_df, val_df, test_df

def fit_and_apply_risk_tables(config: XGBoostConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Fits risk tables on training data and applies them to all splits."""
    risk_processors: Dict[str, RiskTableMappingProcessor] = {}
    train_df_transformed = train_df.copy()
    val_df_transformed = val_df.copy()
    test_df_transformed = test_df.copy()

    for var in config.cat_field_list:
        logger.info(f"Fitting and transforming risk table for '{var}'...")
        proc = RiskTableMappingProcessor(
            column_name=var,
            label_name=config.label_name,
            smooth_factor=config.smooth_factor,
            count_threshold=config.count_threshold,
        )
        proc.fit(train_df)
        risk_processors[var] = proc
        
        train_df_transformed[var] = proc.transform(train_df_transformed[var])
        val_df_transformed[var] = proc.transform(val_df_transformed[var])
        test_df_transformed[var] = proc.transform(test_df_transformed[var])
        
    consolidated_risk_tables = {var: proc.get_risk_tables() for var, proc in risk_processors.items()}
    return train_df_transformed, val_df_transformed, test_df_transformed, consolidated_risk_tables

def fit_and_apply_imputation(config: XGBoostConfig, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Calculates imputation values from training data and applies them to all splits."""
    impute_dict: Dict[str, float] = {}
    train_df_imputed = train_df.copy()
    val_df_imputed = val_df.copy()
    test_df_imputed = test_df.copy()

    for var in config.tab_field_list:
        median_val = train_df[var].median(skipna=True)
        impute_dict[var] = float(median_val) if pd.notna(median_val) else 0.0

    for df in (train_df_imputed, val_df_imputed, test_df_imputed):
        df.fillna(impute_dict, inplace=True)
        
    return train_df_imputed, val_df_imputed, test_df_imputed, impute_dict

def prepare_dmatrices(config: XGBoostConfig, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
    """Prepares XGBoost DMatrix objects from dataframes."""
    feature_columns = config.tab_field_list + config.cat_field_list
    
    X_train = train_df[feature_columns].astype(float).values
    y_train = train_df[config.label_name].astype(int).values
    X_val = val_df[feature_columns].astype(float).values
    y_val = val_df[config.label_name].astype(int).values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    return dtrain, dval

def train_model(config: XGBoostConfig, dtrain: xgb.DMatrix, dval: xgb.DMatrix) -> xgb.Booster:
    """Trains the XGBoost model."""
    xgb_params = config.model_dump(include={
        "objective", "eta", "gamma", "max_depth", "subsample", "colsample_bytree"
    })
    xgb_params["lambda"] = config.lambda_xgb
    xgb_params["alpha"] = config.alpha_xgb
    
    logger.info(f"Starting XGBoost training with params: {xgb_params}")
    
    return xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=config.num_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=config.early_stopping_rounds,
        verbose_eval=True
    )

def save_artifacts(model: xgb.Booster, risk_tables: Dict, impute_dict: Dict, model_path: str):
    """Saves the trained model and preprocessing artifacts."""
    os.makedirs(model_path, exist_ok=True)
    
    model_file = os.path.join(model_path, "xgboost_model.bst")
    model.save_model(model_file)
    logger.info(f"Saved XGBoost model to {model_file}")

    risk_map_file = os.path.join(model_path, "risk_table_map.pkl")
    with open(risk_map_file, "wb") as f:
        pkl.dump(risk_tables, f)
    logger.info(f"Saved consolidated risk table map to {risk_map_file}")
    
    impute_file = os.path.join(model_path, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(impute_dict, f)
    logger.info(f"Saved imputation dictionary to {impute_file}")

    fmap_json = os.path.join(model_path, "feature_importance.json")
    with open(fmap_json, "w") as f:
        json.dump(model.get_fscore(), f, indent=2)
    logger.info(f"Saved feature importance to {fmap_json}")

# -------------------------------------------------------------------------
# Main Orchestrator
# -------------------------------------------------------------------------
def main(hparam_path: str, input_path: str, model_path: str):
    """Main function to execute the XGBoost training logic."""
    config = load_and_validate_config(hparam_path)
    train_df, val_df, test_df = load_datasets(input_path)
    
    train_df, val_df, test_df, risk_tables = fit_and_apply_risk_tables(config, train_df, val_df, test_df)
    train_df, val_df, test_df, impute_dict = fit_and_apply_imputation(config, train_df, val_df, test_df)
    
    dtrain, dval = prepare_dmatrices(config, train_df, val_df)
    
    model = train_model(config, dtrain, dval)
    
    save_artifacts(model, risk_tables, impute_dict, model_path)
    
    logger.info("Training script finished successfully.")

# -------------------------------------------------------------------------
# Script Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    prefix = "/opt/ml"
    
    # The data from the previous step is on the main 'data' channel
    input_path = os.path.join(prefix, "input", "data")
    
    # The model artifacts are saved to the standard model directory
    model_path = os.path.join(prefix, "model")
    
    # FIX: The path to hyperparameters is now determined by the 'config' input channel,
    # which is the recommended way to pass large configuration files.
    config_channel_path = os.path.join(prefix, "input", "data", "config")
    hparam_path = os.path.join(config_channel_path, "hyperparameters.json")

    try:
        main(hparam_path, input_path, model_path)
    except Exception:
        logger.error(f"Exception during training:\n{traceback.format_exc()}")
        sys.exit(1)
