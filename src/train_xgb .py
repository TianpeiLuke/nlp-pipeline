#!/usr/bin/env python3
import os
import json
import sys
import traceback
import ast
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import pickle as pkl
import argparse
import subprocess # For parts of original preprocessing if ever re-introduced
import psutil # For memory logging
from multiprocessing import Pool, cpu_count
# from sklearn.model_selection import train_test_split # Not used if loading pre-split data
import xgboost as xgb
from typing import List, Tuple, Dict, Any, Union, Optional

# Pydantic is not strictly needed in this script anymore if we directly use parsed HPs
# However, if you had a shared Pydantic model for HPs, you might import it for type hints.
# from pydantic import BaseModel, Field, validator # Removed as inline Config is removed

# ================== SageMaker Environment Paths =================
PREFIX = "/opt/ml/"
INPUT_PATH = os.path.join(PREFIX, "input/data")
OUTPUT_PATH = os.path.join(PREFIX, "output/data") 
MODEL_PATH = os.path.join(PREFIX, "model") 
HPARAM_PATH = os.path.join(PREFIX, "input/config/hyperparameters.json")
ARTIFACT_INPUT_PATH = os.path.join(INPUT_PATH, "artifacts") 

# Paths to pre-processed data from a SageMaker Processing job (or similar)
PROCESSED_TRAIN_PATH = os.path.join(INPUT_PATH, "train") 
PROCESSED_VAL_PATH = os.path.join(INPUT_PATH, "validation") 

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# =================== Preprocessing Functions (from user's original script) ===================
# These functions are kept if there's any residual preprocessing on the already processed data,
# or if this script were to be adapted to also handle earlier stages.

def load_config_pkl(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a Pickle configuration file."""
    with open(config_path, "rb") as file:
        config_dict = pkl.load(file)
    logger.info(f"Loaded Pickle config from: {config_path}")
    return config_dict

# currency_conversion_single_variable, parallel_currency_conversion,
# map_single_variable, parallel_mapping,
# impute_single_variable, parallel_imputation, combine_two_cols
# These functions would be used if the data loaded from PROCESSED_TRAIN_PATH
# still requires these steps. For this simplified version, we assume these were
# done in a prior SageMaker Processing job, and this script consumes its output.
# If they are needed, ensure the necessary columns and config (e.g., bin_map, imputation_dict)
# are correctly passed and loaded.

# For brevity in this simplified script, I will comment out the parallel processing functions
# as they depend on the full preprocessing context which is assumed to be done prior to this script.
# If these are still needed, they can be uncommented and their inputs managed.

# def currency_conversion_single_variable(...): ...
# def parallel_currency_conversion(...): ...
# def map_single_variable(...): ...
# def parallel_mapping(...): ...
# def impute_single_variable(...): ...
# def parallel_imputation(...): ...
# def combine_two_cols(...): ...


# =================== Hyperparameter Parsing ===================
def safe_cast(val: Any) -> Any:
    """Safely casts string values from hyperparameters.json to Python types."""
    if isinstance(val, str):
        val_stripped = val.strip()
        if val_stripped.lower() == "true": return True
        if val_stripped.lower() == "false": return False
        if (val_stripped.startswith("[") and val_stripped.endswith("]")) or \
           (val_stripped.startswith("{") and val_stripped.endswith("}")):
            try: return json.loads(val_stripped)
            except json.JSONDecodeError: pass 
        try: return ast.literal_eval(val_stripped)
        except (ValueError, SyntaxError, TypeError): pass 
    return val

def sanitize_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Casts hyperparameter values and handles kebab-case to snake_case for keys."""
    sanitized = {}
    for key, val in config.items():
        sanitized_key = key.replace('-', '_') 
        sanitized[sanitized_key] = safe_cast(val)
    return sanitized

def load_and_parse_hyperparameters(hparam_path: str) -> Dict[str, Any]:
    """Loads hyperparameters from JSON, sanitizes, and casts them."""
    logger.info(f"Loading hyperparameters from: {hparam_path}")
    if not os.path.exists(hparam_path):
        logger.error(f"Hyperparameter file not found at {hparam_path}. This is critical.")
        raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}")
    with open(hparam_path, "r") as f:
        raw_hyperparameters = json.load(f)
    logger.info(f"Raw hyperparameters: {raw_hyperparameters}")
    
    parsed_hyperparameters = sanitize_hyperparameters(raw_hyperparameters)
    logger.info(f"Parsed and cast hyperparameters: {parsed_hyperparameters}")
    return parsed_hyperparameters


# =================== Main Training Logic ===================
def main(cmd_args): # Renamed from args to cmd_args to avoid conflict with hyperparameters dict
    logger.info("Starting XGBoost training script.")
    logger.info(f"Received command-line arguments: {cmd_args}")
    
    # --- 1. Load Hyperparameters ---
    # These hyperparameters should conform to the structure of XGBoostModelHyperparameters
    # (including fields like 'tag', 'final_model_var_list', and all XGBoost HPs)
    hyperparameters = load_and_parse_hyperparameters(HPARAM_PATH)
    
    # Extract key configurations from hyperparameters
    # These are expected to be defined in your hyperparameters.json based on XGBoostModelHyperparameters
    tag_col_name = hyperparameters.get("tag")
    final_model_var_list = hyperparameters.get("final_model_var_list")

    if not tag_col_name:
        raise ValueError("Hyperparameter 'tag' (target column name) must be provided.")
    if not final_model_var_list or not isinstance(final_model_var_list, list):
        raise ValueError("Hyperparameter 'final_model_var_list' (list of feature names) must be provided.")

    logger.info(f"Target column (tag): {tag_col_name}")
    logger.info(f"Final model variable list: {final_model_var_list}")

    # --- 2. Load External Configurations (if any are still needed after pre-processing) ---
    # Example: if metadata_df is still needed for some reason.
    # For this simplified script, we assume most complex preprocessing artifacts (bin_map, etc.)
    # were used in a prior job, and this script consumes data ready for DMatrix.
    # If general_config.pkl is still needed for some parameters not in HPs:
    # general_config_path = os.path.join(ARTIFACT_INPUT_PATH, "config.pkl")
    # if os.path.exists(general_config_path):
    #     general_config = load_config_pkl(general_config_path)
    #     logger.info("Loaded general_config.pkl")
    # else:
    #     logger.warning("general_config.pkl not found in artifacts. Proceeding without it.")
    #     general_config = {}


    # --- 3. Load Pre-processed Data ---
    logger.info(f"Memory usage before loading data: {dict(psutil.virtual_memory()._asdict())}")

    train_data_path = os.path.join(PROCESSED_TRAIN_PATH, "processed_data.csv")
    validation_data_path = os.path.join(PROCESSED_VAL_PATH, "processed_data.csv")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Processed training data not found: {train_data_path}.")
        
    logger.info(f"Loading processed training data from: {train_data_path}")
    expected_cols = [tag_col_name] + final_model_var_list
    df_train = pd.read_csv(train_data_path, header=None, names=expected_cols, dtype=np.float32)
    df_train[tag_col_name] = df_train[tag_col_name].astype(int)
    logger.info(f"Training data shape: {df_train.shape}")

    df_val = None
    watchlist = []
    if os.path.exists(validation_data_path):
        logger.info(f"Loading processed validation data from: {validation_data_path}")
        df_val = pd.read_csv(validation_data_path, header=None, names=expected_cols, dtype=np.float32)
        df_val[tag_col_name] = df_val[tag_col_name].astype(int)
        logger.info(f"Validation data shape: {df_val.shape}")
    else:
        logger.warning(f"Processed validation data not found: {validation_data_path}. Training without a validation set for early stopping.")
        
    logger.info(f"Memory usage after loading data: {dict(psutil.virtual_memory()._asdict())}")

    X_train = df_train[final_model_var_list]
    y_train = df_train[tag_col_name]
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=final_model_var_list)
    logger.info(f"Created DMatrix for training. Num rows: {dtrain.num_row()}, Num cols: {dtrain.num_col()}")
    watchlist.append((dtrain, 'train'))

    if df_val is not None:
        X_val = df_val[final_model_var_list]
        y_val = df_val[tag_col_name]
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=final_model_var_list)
        watchlist.append((dval, 'validation'))
        logger.info(f"Created DMatrix for validation. Num rows: {dval.num_row()}, Num cols: {dval.num_col()}")


    # --- 4. XGBoost Model Training ---
    logger.info("Starting XGBoost model training.")
    
    # Prepare XGBoost parameters directly from the loaded hyperparameters dictionary.
    # The external XGBoostModelHyperparameters Pydantic class defines the expected structure.
    # We need to ensure keys match what xgb.train expects (e.g., 'lambda', 'alpha').
    # The sanitize_hyperparameters function should handle kebab-case to snake_case.
    # Pydantic's by_alias=True would handle 'reg_lambda' -> 'lambda' if we were using the model.
    # Here, we directly use the parsed dictionary.
    
    xgb_params = {}
    # Explicitly map known Pydantic field names (and their aliases) to XGBoost param names
    # This assumes your XGBoostModelHyperparameters class uses these field names or aliases.
    param_mapping = {
        "reg_lambda": "lambda", # Pydantic field 'reg_lambda' maps to xgb 'lambda'
        "reg_alpha": "alpha",   # Pydantic field 'reg_alpha' maps to xgb 'alpha'
        # Add other direct mappings if names differ significantly
        # For most, Pydantic field name will match XGBoost param name after sanitization
    }

    for key, value in hyperparameters.items():
        xgb_key = param_mapping.get(key, key) # Use mapped key if exists, else original key
        # Exclude keys that are not XGBoost booster/learning task parameters
        if key not in ['tag', 'final_model_var_list', 'num_round', 'early_stopping_rounds']:
             if value is not None: # XGBoost doesn't like None for most params
                xgb_params[xgb_key] = value

    num_boost_round = hyperparameters.get('num_round', 100)
    early_stopping_rounds_val = hyperparameters.get('early_stopping_rounds')

    # Objective specific params like 'num_class'
    if hyperparameters.get('objective', '').startswith("multi:") and hyperparameters.get('num_class') is not None:
        xgb_params['num_class'] = hyperparameters['num_class']
    
    logger.info(f"XGBoost training parameters: {xgb_params}")
    logger.info(f"Number of boosting rounds: {num_boost_round}")
    if early_stopping_rounds_val:
        logger.info(f"Early stopping rounds: {early_stopping_rounds_val}")

    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds_val,
        verbose_eval=10 
    )
    
    logger.info("XGBoost model training complete.")
    if early_stopping_rounds_val and bst.best_iteration != num_boost_round -1 :
         logger.info(f"Best iteration: {bst.best_iteration + 1} (due to early stopping)")

    # --- 5. Save Model and Artifacts ---
    # SageMaker expects the model in a specific structure if using its hosting,
    # often as model.tar.gz containing the model file and optionally an inference script.
    # For XGBoost, saving the raw model file is common.
    model_filename = "xgboost-model.json" # Or .xgb for binary
    model_save_path = os.path.join(MODEL_PATH, model_filename)
    
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True) # Ensure model directory exists
    logger.info(f"Saving trained XGBoost model to: {model_save_path}")
    bst.save_model(model_save_path) 
    
    features_filename = "features.json"
    features_save_path = os.path.join(MODEL_PATH, features_filename)
    with open(features_save_path, 'w') as f:
        json.dump(final_model_var_list, f)
    logger.info(f"Saved feature list to: {features_save_path}")

    # Save the raw hyperparameters used for this training run
    hparams_save_filename = "hyperparameters_used.json"
    hparams_save_path = os.path.join(MODEL_PATH, hparams_save_filename)
    with open(hparams_save_path, 'w') as f:
        json.dump(hyperparameters, f, indent=2) # Save the initially loaded & parsed HPs
    logger.info(f"Saved raw hyperparameters used to: {hparams_save_path}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # n_workers is kept for potential use by any remaining parallelizable helper functions,
    # though the main parallel preprocessing steps are assumed to be done prior to this script.
    parser.add_argument("--n_workers", type=int, default=max(1, cpu_count() // 2), 
                        help="Number of worker processes for any parallel tasks within this script.")
    
    # data_type is not used in this simplified training script as it loads pre-split data.
    # parser.add_argument("--data_type", type=str, default="training") 

    args, unknown = parser.parse_known_args() # Parse known args, ignore others (like SageMaker's)
    if unknown:
        logger.info(f"Ignoring unknown arguments: {unknown}")

    try:
        main(args) 
    except Exception as e:
        trc = traceback.format_exc()
        error_message = f"Exception during training: {str(e)}\n{trc}"
        logger.error(error_message)
        
        error_file_path = os.path.join(OUTPUT_PATH, "failure") 
        try:
            Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
            with open(error_file_path, "w") as f:
                f.write(error_message)
            logger.info(f"Failure reason written to {error_file_path}")
        except Exception as fe:
            logger.error(f"Failed to write failure file: {fe}")
            
        sys.exit(255)