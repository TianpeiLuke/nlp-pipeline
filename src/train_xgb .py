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
import subprocess # For process_downloaded_data
import psutil # For memory logging
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split
import xgboost as xgb
from typing import List, Tuple, Dict, Any, Union, Optional

# --- Pydantic for Hyperparameter Validation ---
from pydantic import BaseModel, Field, validator

# ================== SageMaker Environment Paths =================
PREFIX = "/opt/ml/"
INPUT_PATH = os.path.join(PREFIX, "input/data")
OUTPUT_PATH = os.path.join(PREFIX, "output/data") # For processed data, if any intermediate are saved by script
MODEL_PATH = os.path.join(PREFIX, "model") # Standard path for final model artifacts
HPARAM_PATH = os.path.join(PREFIX, "input/config/hyperparameters.json")
# ARTIFACT_INPUT_PATH is where pre-computed things like bin_mapping.pkl would be if provided as a channel
# For this script, it's assumed to be part of the source_dir or another input channel.
# The provided preprocessing script uses /opt/ml/processing/input/artifacts,
# for training, this might be /opt/ml/input/data/artifacts or part of source_dir.
# Let's assume artifacts are provided in a channel named 'artifacts'
ARTIFACT_INPUT_PATH = os.path.join(INPUT_PATH, "artifacts") 

# Define input channels based on SageMaker conventions for training
# The preprocessing script uses 'input_data_dir' directly.
# For SageMaker training, data comes in channels.
# We'll assume 'train' and 'validation' channels are provided.
# The 'data_type' argument will distinguish if we are processing the 'train' or 'validation' set
# if the script is called separately for each.
# However, the preprocessing logic splits training data into train/validation itself.
# Let's assume the script receives one primary dataset (e.g., 'training_data_channel')
# and then splits it or uses a pre-split validation set if provided.

# For simplicity, let's assume the main dataset comes through a channel named 'dataset'
# and the script handles splitting if 'data_type' is 'training'.
# If separate train/val channels are used, the script would need to adapt.
# The example preprocessing script saves processed train/validation to output.
# This training script will READ those processed files.

PROCESSED_TRAIN_PATH = os.path.join(INPUT_PATH, "train") # Expects processed_data.csv here
PROCESSED_VAL_PATH = os.path.join(INPUT_PATH, "validation") # Expects processed_data.csv here

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# =================== Preprocessing Functions (from user's script) ===================
def load_config_json(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a JSON configuration file."""
    with open(config_path, "r") as file:
        config_dict = json.load(file)
    logger.info(f"Loaded JSON config from: {config_path}")
    return config_dict

def load_config_pkl(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a Pickle configuration file."""
    with open(config_path, "rb") as file:
        config_dict = pkl.load(file)
    logger.info(f"Loaded Pickle config from: {config_path}")
    return config_dict

def process_downloaded_data(data_dir: str, data_type: str) -> Tuple[str, str]:
    """
    Unzip and process downloaded data files.
    This function unzips the data files, combines them into a single CSV file, and creates a signature file.
    It also replaces '__DOT__' with '.' in the signature file. After processing, the original part files are removed.
    """
    logger.info(f"Processing downloaded data in {data_dir} for data_type: {data_type}")
    # Check if part files exist before attempting to unzip
    part_files_gz = list(Path(data_dir).glob('part*.gz'))
    if part_files_gz:
        logger.info(f"Found .gz part files, unzipping: {part_files_gz}")
        os.system(f"gzip -d {data_dir}/part*.gz")
    else:
        logger.info("No .gz part files found to unzip.")

    part_files_csv = list(Path(data_dir).glob('part*.csv'))
    if not part_files_csv:
        logger.warning(f"No part*.csv files found in {data_dir}. Cannot process.")
        # Depending on requirements, either raise error or return empty paths
        # For now, let's assume this might be okay if data is already processed.
        # However, the original script implies these part files are the raw input.
        raise FileNotFoundError(f"No part*.csv files found in {data_dir} to process.")


    data_file_name = f"{data_type}_data_combined.csv" # Ensure unique name
    data_path = os.path.join(data_dir, data_file_name)
    signature_file_name = f"{data_type}_signature.csv"
    signature_path = os.path.join(data_dir, signature_file_name)

    # Ensure there's at least one part file to get the header from
    first_part_file = sorted(part_files_csv)[0].name

    # Command to combine CSVs and extract header
    # Using bash -c for complex commands with pipes and loops
    combine_cmd = (
        f"cd {data_dir} && "
        f"head -n1 {first_part_file} > {signature_file_name} && " # Get header from the first part file
        f"for file in part*.csv; do sed '1d' \"$file\"; done > {data_file_name} && " # Concatenate data (skip headers)
        f"sed -i 's/__DOT__/./g' {signature_file_name}" # Fix signature
    )
    logger.info(f"Executing data combination command: {combine_cmd}")
    subprocess.run(combine_cmd, shell=True, check=True, executable='/bin/bash')

    logger.info(f"Removing original part files from {data_dir}")
    os.system(f"rm {data_dir}/part*.csv")
    return data_path, signature_path


def currency_conversion_single_variable(
    args: Tuple[pd.DataFrame, str, pd.Series] # Modified type hint for exchange_rate_series
) -> pd.Series:
    df, variable, exchange_rate_series = args
    # Ensure alignment if df is a slice. For safety, use .loc with original index if needed.
    # However, exchange_rate_series is already aligned with the original df.
    return df[variable].astype(float) / exchange_rate_series.values


def parallel_currency_conversion(
    df: pd.DataFrame,
    currency_col: str,
    currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, float],
    n_workers: int,
) -> pd.DataFrame:
    logger.info(f"Starting parallel currency conversion for {len(currency_conversion_vars)} variables.")
    exchange_rate_series = df[currency_col].apply(lambda x: currency_conversion_dict.get(x, 1.0))
    
    # Filter for variables that actually exist in the DataFrame to avoid errors
    valid_conversion_vars = [var for var in currency_conversion_vars if var in df.columns]
    if not valid_conversion_vars:
        logger.warning("No valid currency conversion variables found in DataFrame.")
        return df

    processes = min(cpu_count(), len(valid_conversion_vars), n_workers)
    logger.info(f"Using {processes} workers for currency conversion.")
    
    tasks = [(df[[var]].copy(), var, exchange_rate_series) for var in valid_conversion_vars]

    with Pool(processes=processes) as pool:
        results = pool.map(currency_conversion_single_variable, tasks)
    
    # Concatenate results and assign back to the DataFrame
    if results:
        converted_df = pd.concat(results, axis=1)
        for var in valid_conversion_vars:
            if var in converted_df.columns:
                df[var] = converted_df[var]
    logger.info("Parallel currency conversion finished.")
    return df


def map_single_variable(args: Tuple[pd.DataFrame, str, Dict[str, Any]]) -> pd.Series:
    df, variable, mapping_dict_for_var_family = args # mapping_dict_for_var_family is the global bin_map
    
    if variable not in mapping_dict_for_var_family:
        logger.warning(f"No mapping found for variable '{variable}' in mapping_dict. Returning original.")
        return df[variable]
        
    mapping = mapping_dict_for_var_family[variable] # This should be the specific map for 'variable'
                                                 # e.g., {"bins": {...}, "default_bin": ...}
    
    if not isinstance(mapping, dict) or "bins" not in mapping:
        logger.warning(f"Mapping for variable '{variable}' is not in the expected format. Returning original.")
        return df[variable]

    category_to_value_map = mapping["bins"]
    default_val = mapping.get("default_bin")
    
    # Ensure input is string for mapping, as keys in 'bins' are usually strings
    return df[variable].astype(str).map(category_to_value_map).fillna(default_val)


def parallel_mapping(
    df: pd.DataFrame,
    cat_vars: List[str],
    mapping_dict: Dict[str, Dict[str, Any]], # This is the global bin_map
    n_workers: int,
) -> pd.DataFrame:
    logger.info(f"Starting parallel mapping for {len(cat_vars)} categorical variables.")
    valid_cat_vars = [var for var in cat_vars if var in df.columns]
    if not valid_cat_vars:
        logger.warning("No valid categorical variables found in DataFrame for mapping.")
        return df

    processes = min(cpu_count(), len(valid_cat_vars), n_workers)
    logger.info(f"Using {processes} workers for mapping.")
    
    tasks = [(df[[var]].copy(), var, mapping_dict) for var in valid_cat_vars]
    
    with Pool(processes=processes) as pool:
        results = pool.map(map_single_variable, tasks)
        
    if results:
        mapped_df = pd.concat(results, axis=1)
        for var in valid_cat_vars:
            if var in mapped_df.columns:
                df[var] = mapped_df[var]
    logger.info("Parallel mapping finished.")
    return df


def impute_single_variable(args: Tuple[pd.DataFrame, str, Dict[str, Any]]) -> pd.Series:
    df, variable, imputation_dict_for_var_family = args # imputation_dict_for_var_family is the global one
    
    if variable not in imputation_dict_for_var_family:
        logger.warning(f"No imputation value found for variable '{variable}'. Skipping imputation for this variable.")
        return df[variable]
        
    impute_value = imputation_dict_for_var_family[variable]
    return df[variable].fillna(impute_value)


def parallel_imputation(
    df: pd.DataFrame, num_vars: List[str], imputation_dict: Dict[str, Any], n_workers: int
) -> pd.DataFrame:
    logger.info(f"Starting parallel imputation for {len(num_vars)} numeric variables.")
    valid_num_vars = [var for var in num_vars if var in df.columns]
    if not valid_num_vars:
        logger.warning("No valid numeric variables found in DataFrame for imputation.")
        return df
        
    processes = min(cpu_count(), len(valid_num_vars), n_workers)
    logger.info(f"Using {processes} workers for imputation.")

    tasks = [(df[[var]].copy(), var, imputation_dict) for var in valid_num_vars]
    
    with Pool(processes=processes) as pool:
        results = pool.map(impute_single_variable, tasks)

    if results:
        imputed_df = pd.concat(results, axis=1)
        for var in valid_num_vars:
            if var in imputed_df.columns:
                df[var] = imputed_df[var]
    logger.info("Parallel imputation finished.")
    return df


def combine_two_cols(df, col_0, col_1, keep_col_preference: int): # Renamed keep_col for clarity
    """
    Combines two columns. If preferred column (based on keep_col_preference) is NaN, uses the other.
    keep_col_preference = 0 means prefer col_0.
    keep_col_preference = 1 means prefer col_1.
    """
    if col_0 not in df.columns or col_1 not in df.columns:
        logger.warning(f"One or both columns for combination not found: {col_0}, {col_1}. Skipping.")
        # Decide how to handle this: return one, or raise error, or return an empty series.
        # For now, if one is missing, just return the other if it exists, or original df if both missing.
        if col_0 in df.columns and col_1 not in df.columns: return df[col_0]
        if col_1 in df.columns and col_0 not in df.columns: return df[col_1]
        return pd.Series(index=df.index, dtype=object) # Or raise error

    if keep_col_preference == 0: # Prefer col_0
        return df[col_0].combine_first(df[col_1])
    else: # Prefer col_1
        return df[col_1].combine_first(df[col_0])

# =================== XGBoost Pydantic Config ===================
class XGBoostTrainConfig(BaseModel):
    # Data related (usually not hyperparameters but part of script config)
    tag: str = Field(default="tag", description="Name of the target variable column.")
    final_model_var_list: List[str] = Field(default_factory=list, description="List of feature names to be used for training.")
    
    # XGBoost specific hyperparameters
    # General Parameters
    objective: str = Field(default="binary:logistic", description="Learning objective.")
    booster: str = Field(default="gbtree", description="Booster type: gbtree, gblinear or dart.")
    nthread: Optional[int] = Field(default=None, description="Number of parallel threads.")
    
    # Booster Parameters (most common ones)
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Learning rate.")
    gamma: float = Field(default=0.0, ge=0.0, description="Minimum loss reduction. (min_split_loss)")
    max_depth: int = Field(default=6, ge=0, description="Maximum depth of a tree.")
    min_child_weight: float = Field(default=1.0, ge=0.0, description="Minimum sum of instance weight.")
    max_delta_step: float = Field(default=0.0, description="Maximum delta step.")
    subsample: float = Field(default=1.0, gt=0.0, le=1.0, description="Subsample ratio of training instance.")
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0, description="Subsample ratio of columns per tree.")
    colsample_bylevel: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Subsample ratio of columns per level.")
    colsample_bynode: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Subsample ratio of columns per node.")
    reg_lambda: float = Field(default=1.0, ge=0.0, alias="lambda", description="L2 regularization term.") # XGB uses 'lambda'
    reg_alpha: float = Field(default=0.0, ge=0.0, alias="alpha", description="L1 regularization term.") # XGB uses 'alpha'
    tree_method: Optional[str] = Field(default=None, description="Tree construction algorithm.") # e.g., 'hist', 'gpu_hist'
    scale_pos_weight: float = Field(default=1.0, description="Control balance of positive and negative weights.")
    
    # Learning Task Parameters
    eval_metric: Optional[Union[str, List[str]]] = Field(default=None, description="Evaluation metric(s).")
    seed: Optional[int] = Field(default=None, description="Random seed.")
    
    # Training control
    num_round: int = Field(default=100, ge=1, description="Number of boosting rounds.")
    early_stopping_rounds: Optional[int] = Field(default=None, ge=1, description="Activates early stopping.")
    
    # For multiclass classification
    num_class: Optional[int] = Field(default=None, description="Number of classes for multi-class objective.")

    # Allow other XGBoost parameters to be passed through
    class Config:
        extra = "allow"
        populate_by_name = True # To allow aliases like 'lambda' and 'alpha'

    @validator('objective')
    def objective_must_be_valid(cls, v):
        # Add more valid objectives as needed
        valid_objectives = ["binary:logistic", "reg:squarederror", "multi:softmax", "multi:softprob"]
        if v not in valid_objectives:
            raise ValueError(f"Objective '{v}' is not a recognized XGBoost objective.")
        return v

    @validator('num_class')
    def num_class_required_for_multiclass(cls, v, values):
        objective = values.get('objective')
        if objective and objective.startswith("multi:") and (v is None or v < 2):
            raise ValueError("num_class must be provided and >= 2 for multiclass objectives.")
        return v

# =================== Hyperparameter Parsing (from PyTorch example) ===================
def safe_cast(val: Any) -> Any:
    """Safely casts string values from hyperparameters.json to Python types."""
    if isinstance(val, str):
        val_stripped = val.strip()
        if val_stripped.lower() == "true": return True
        if val_stripped.lower() == "false": return False
        if (val_stripped.startswith("[") and val_stripped.endswith("]")) or \
           (val_stripped.startswith("{") and val_stripped.endswith("}")):
            try: return json.loads(val_stripped) # More robust for lists/dicts
            except json.JSONDecodeError: pass # Fall through to ast.literal_eval or original
        try: return ast.literal_eval(val_stripped) # For numbers, simple lists/tuples
        except (ValueError, SyntaxError, TypeError): pass # If not a literal, return original string
    return val

def sanitize_hyperparameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitizes and casts hyperparameter values."""
    sanitized = {}
    for key, val in config.items():
        # SageMaker often passes HPs as strings, even if they are numbers/bools in JSON
        # The safe_cast function will handle conversion.
        # XGBoost Python API typically expects native types, not strings for numbers/bools.
        sanitized_key = key.replace('-', '_') # Convert kebab-case to snake_case if needed
        sanitized[sanitized_key] = safe_cast(val)
    return sanitized

def load_and_parse_hyperparameters(hparam_path: str) -> Dict[str, Any]:
    """Loads hyperparameters from JSON, sanitizes, and casts them."""
    logger.info(f"Loading hyperparameters from: {hparam_path}")
    if not os.path.exists(hparam_path):
        logger.warning(f"Hyperparameter file not found at {hparam_path}. Using empty dict.")
        return {}
    with open(hparam_path, "r") as f:
        raw_hyperparameters = json.load(f)
    logger.info(f"Raw hyperparameters: {raw_hyperparameters}")
    
    # Sanitize (e.g. for XGBoost, it expects eta not learning_rate, alpha not reg_alpha if passed directly)
    # The Pydantic model with aliases handles this for known HPs.
    # For other HPs passed via extra='allow', direct casting is important.
    parsed_hyperparameters = sanitize_hyperparameters(raw_hyperparameters)
    logger.info(f"Parsed and cast hyperparameters: {parsed_hyperparameters}")
    return parsed_hyperparameters


# =================== Main Training Logic ===================
def main(args):
    logger.info("Starting main training and preprocessing script.")
    logger.info(f"Received arguments: {args}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Listing /opt/ml/: {os.listdir('/opt/ml/')}")
    logger.info(f"Listing /opt/ml/input/: {os.listdir('/opt/ml/input/')}")
    logger.info(f"Listing /opt/ml/input/data/: {os.listdir('/opt/ml/input/data/')}")


    # --- 1. Load Hyperparameters and Validate with Pydantic ---
    hyperparameters = load_and_parse_hyperparameters(HPARAM_PATH)
    try:
        # Pydantic model will use the parsed hyperparameters.
        # It will also pick up any extra HPs if config.extra = "allow"
        script_config = XGBoostTrainConfig(**hyperparameters)
    except Exception as e: # Catch Pydantic ValidationError specifically if possible
        logger.error(f"Pydantic Config Validation Error: {e}")
        raise
    
    logger.info(f"Validated XGBoost Script Config: {script_config.model_dump_json(indent=2)}")

    # --- 2. Load External Configurations and Artifacts (bin maps, imputation dicts) ---
    # These paths need to point to where SageMaker makes these files available.
    # Typically, these would be in an 'artifacts' channel or part of the source_dir.
    # Adjust ARTIFACT_INPUT_PATH if these are in a different input channel.
    # If these are not found, the script should handle it gracefully or error.

    # Check if ARTIFACT_INPUT_PATH exists
    if not os.path.exists(ARTIFACT_INPUT_PATH):
        logger.error(f"Artifact input path does not exist: {ARTIFACT_INPUT_PATH}")
        # Depending on whether these artifacts are critical, either raise error or proceed with caution
        # For this example, let's assume they are critical.
        raise FileNotFoundError(f"Artifact directory not found: {ARTIFACT_INPUT_PATH}. Ensure it's provided as an input channel.")

    bin_mapping_file = "bin_mapping.pkl"
    bin_map_path = os.path.join(ARTIFACT_INPUT_PATH, bin_mapping_file)
    if not os.path.exists(bin_map_path): raise FileNotFoundError(f"Bin mapping file not found: {bin_map_path}")
    bin_map = pkl.load(open(bin_map_path, "rb"))
    logger.info(f"Loaded bin_mapping from {bin_map_path}")

    missing_value_imputation_file = "missing_value_imputation.pkl"
    imputation_dict_path = os.path.join(ARTIFACT_INPUT_PATH, missing_value_imputation_file)
    if not os.path.exists(imputation_dict_path): raise FileNotFoundError(f"Imputation dict file not found: {imputation_dict_path}")
    missing_value_impute_dict = pkl.load(open(imputation_dict_path, "rb"))
    logger.info(f"Loaded missing_value_imputation_dict from {imputation_dict_path}")

    # General config (containing metadata, currency info, etc.)
    # This config was loaded by the preprocessing script. For training, some parts might be needed.
    # It might be passed as another artifact or its relevant parts included in hyperparameters.json
    general_config_file = "config.pkl" # Assuming this was also an output of preprocessing step
    general_config_path = os.path.join(ARTIFACT_INPUT_PATH, general_config_file)
    if not os.path.exists(general_config_path): raise FileNotFoundError(f"General config file not found: {general_config_path}")
    general_config = load_config_pkl(general_config_path) # Using pkl loader
    logger.info(f"Loaded general_config from {general_config_path}")

    data_config = general_config.get("data_config", {})
    marketplace_info = data_config.get("marketplace_info", {})
    data_processing_info = data_config.get("data_processing_info", {})
    preprocessing_combine_var_name_pair_list = data_processing_info.get("preprocessing_combine_var_name_pair_list", [])
    
    # Extract necessary info from general_config or script_config (hyperparameters)
    # Prioritize from script_config (hyperparameters) if available
    tag_col_name = script_config.tag
    final_model_var_list = script_config.final_model_var_list

    currency_col = general_config.get("currency_col", "currency_code") # Default from general config
    enable_currency_conversion = general_config.get("enable_currency_conversion", False)
    currency_conversion_var_list = general_config.get("currency_conversion_var_list", [])
    currency_conversion_dict = general_config.get("currency_conversion_dict", {})
    metadata_df = general_config.get("metadata") # Expecting a DataFrame
    if not isinstance(metadata_df, pd.DataFrame) and metadata_df is not None:
        logger.warning("Metadata is not a DataFrame. Trying to load if it's a path.")
        # Potentially load metadata if it's a path string, or handle error
        # For now, assume it's loaded as DataFrame or is None.
        metadata_df = None # Or raise error if critical
    
    marketplace_id_col = general_config.get("marketplace_id_col", "marketplace_id")


    # --- 3. Load and Preprocess Data ---
    # The preprocessing script saves processed 'training' and 'validation' data.
    # This training script will load those.
    # `args.data_type` from command line helps select which dataset to load if script is generic.
    # For a SageMaker training job, we typically get 'train' and 'validation' channels.

    logger.info(f"Memory usage before loading data: {dict(psutil.virtual_memory()._asdict())}")

    # Determine if we are processing combined raw data or reading pre-split processed files
    # For SageMaker training, we usually get distinct 'train' and 'validation' channels
    # containing ALREADY PROCESSED data from a SageMaker Processing job.

    train_data_path = os.path.join(PROCESSED_TRAIN_PATH, "processed_data.csv")
    validation_data_path = os.path.join(PROCESSED_VAL_PATH, "processed_data.csv")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Processed training data not found: {train_data_path}. Ensure preprocessing job ran and outputted correctly.")
    if not os.path.exists(validation_data_path):
        logger.warning(f"Processed validation data not found: {validation_data_path}. Proceeding without validation set for XGBoost if not critical.")
        # XGBoost can train without a validation set, but early stopping won't work.
        
    logger.info(f"Loading processed training data from: {train_data_path}")
    # The processed_data.csv is saved headerless. Columns are [tag] + final_model_var_list
    # The tag is the first column.
    expected_cols = [tag_col_name] + final_model_var_list
    df_train = pd.read_csv(train_data_path, header=None, names=expected_cols, dtype=np.float32)
    df_train[tag_col_name] = df_train[tag_col_name].astype(int)
    logger.info(f"Training data shape: {df_train.shape}")

    df_val = None
    if os.path.exists(validation_data_path):
        logger.info(f"Loading processed validation data from: {validation_data_path}")
        df_val = pd.read_csv(validation_data_path, header=None, names=expected_cols, dtype=np.float32)
        df_val[tag_col_name] = df_val[tag_col_name].astype(int)
        logger.info(f"Validation data shape: {df_val.shape}")

    logger.info(f"Memory usage after loading data: {dict(psutil.virtual_memory()._asdict())}")

    # Separate features and labels
    X_train = df_train[final_model_var_list]
    y_train = df_train[tag_col_name]
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=final_model_var_list)
    logger.info(f"Created DMatrix for training. Num rows: {dtrain.num_row()}, Num cols: {dtrain.num_col()}")

    watchlist = [(dtrain, 'train')]
    if df_val is not None:
        X_val = df_val[final_model_var_list]
        y_val = df_val[tag_col_name]
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=final_model_var_list)
        watchlist.append((dval, 'validation'))
        logger.info(f"Created DMatrix for validation. Num rows: {dval.num_row()}, Num cols: {dval.num_col()}")


    # --- 4. XGBoost Model Training ---
    logger.info("Starting XGBoost model training.")
    
    # Prepare XGBoost parameters from the validated Pydantic config
    # Pydantic model uses aliases 'lambda' and 'alpha' for reg_lambda, reg_alpha
    # XGBoost Python API expects 'lambda' and 'alpha' directly.
    # model_dump(by_alias=True) ensures correct names are used.
    xgb_params = script_config.model_dump(by_alias=True, exclude_none=True, 
                                          exclude={'tag', 'final_model_var_list'}) # Exclude non-XGB HPs
    
    # Remove num_round and early_stopping_rounds as they are xgb.train specific, not booster params
    num_boost_round = xgb_params.pop('num_round', 100) # Default from Pydantic if not in HPs
    early_stopping_rounds_val = xgb_params.pop('early_stopping_rounds', None)

    # Ensure objective specific params are set (e.g. num_class for multi:*)
    if script_config.objective.startswith("multi:") and script_config.num_class:
        xgb_params['num_class'] = script_config.num_class
    
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
        verbose_eval=10 # Log every 10 rounds
    )
    
    logger.info("XGBoost model training complete.")
    if early_stopping_rounds_val and bst.best_iteration != num_boost_round -1 : # 0-indexed vs 1-indexed
         logger.info(f"Best iteration: {bst.best_iteration + 1} (due to early stopping)")


    # --- 5. Save Model and Artifacts ---
    model_filename = "xgboost-model.json" # SageMaker typically expects model.tar.gz, but can also load raw model files.
                                      # Using .json for human readability if model is saved in JSON format.
                                      # Or .xgb / .model for binary format.
    model_save_path = os.path.join(MODEL_PATH, model_filename)
    
    logger.info(f"Saving trained XGBoost model to: {model_save_path}")
    bst.save_model(model_save_path) # Saves in XGBoost internal binary format or JSON if specified
    
    # Save the list of features used for training (important for inference)
    features_filename = "features.json"
    features_save_path = os.path.join(MODEL_PATH, features_filename)
    with open(features_save_path, 'w') as f:
        json.dump(final_model_var_list, f)
    logger.info(f"Saved feature list to: {features_save_path}")

    # Optionally, save the Pydantic config used for this training run
    config_save_filename = "training_script_config.json"
    config_save_path = os.path.join(MODEL_PATH, config_save_filename)
    with open(config_save_path, 'w') as f:
        f.write(script_config.model_dump_json(indent=2))
    logger.info(f"Saved training script Pydantic config to: {config_save_path}")

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add any script-specific arguments here, if different from hyperparameters
    # For example, the preprocessing script took n_workers and data_type.
    # This training script might not need them if it consumes already processed data.
    # However, n_workers for preprocessing functions if they were to be run here would be useful.
    # The current design assumes preprocessing is done, and this script loads processed data.
    
    # These arguments are from the original preprocessing script's main block.
    # They are not directly used in this version of main() as it loads pre-processed data.
    # If this script were to also run the initial raw data processing, these would be relevant.
    parser.add_argument("--n_workers", type=int, default=max(1, cpu_count() // 2), help="Number of worker processes for parallel tasks.")
    # parser.add_argument("--data_type", type=str, default="training", help="Type of data being processed (e.g., training, testing).")

    args, _ = parser.parse_known_args() # Parse them even if not all are used by main() directly

    try:
        main(args) # Pass parsed args to main
    except Exception as e:
        # Write out an error file. This is important for SageMaker Processing Jobs.
        trc = traceback.format_exc()
        error_message = f"Exception during training: {str(e)}\n{trc}"
        logger.error(error_message)
        
        # Standard SageMaker output error file
        error_file_path = os.path.join(OUTPUT_PATH, "failure") 
        try:
            Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True) # Ensure output dir exists
            with open(error_file_path, "w") as f:
                f.write(error_message)
            logger.info(f"Failure reason written to {error_file_path}")
        except Exception as fe:
            logger.error(f"Failed to write failure file: {fe}")
            
        sys.exit(255) # Non-zero exit code for SageMaker to mark job as failed