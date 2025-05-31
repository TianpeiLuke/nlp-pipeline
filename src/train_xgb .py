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
import subprocess
import psutil
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import train_test_split # For internal train/val split if needed
import xgboost as xgb
from typing import List, Tuple, Dict, Any, Union, Optional

# ================== SageMaker Environment Paths =================
PREFIX = "/opt/ml/"
INPUT_DATA_BASE_PATH = os.path.join(PREFIX, "input/data") # Base for channels
OUTPUT_PATH = os.path.join(PREFIX, "output/data") 
MODEL_PATH = os.path.join(PREFIX, "model") 
HPARAM_PATH = os.path.join(PREFIX, "input/config/hyperparameters.json")
ARTIFACT_INPUT_PATH = os.path.join(INPUT_DATA_BASE_PATH, "artifacts") # Default channel for artifacts

# Input data channels (SageMaker convention)
# This script will primarily use 'train_data_channel' and 'validation_data_channel'
# if provided. If only 'train_data_channel' is given, it might perform its own split.
DEFAULT_TRAIN_CHANNEL = "train" # Default name for the main training data channel
DEFAULT_VALIDATION_CHANNEL = "validation" # Default name for the validation data channel


# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout) # Log to stdout for CloudWatch
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers(): # Add handler only if it doesn't have one to avoid duplicates
    logger.addHandler(handler)


# =================== Preprocessing Functions ===================
def load_config_pkl(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a Pickle configuration file."""
    with open(config_path, "rb") as file:
        config_dict = pkl.load(file)
    logger.info(f"Loaded Pickle config from: {config_path}")
    return config_dict

def process_channel_data(channel_dir: str, data_type_name: str) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Processes data within a given SageMaker input channel directory.
    If part files exist, they are unzipped and combined. Otherwise, tries to load a single CSV.
    Returns DataFrame and signature (column names).
    """
    logger.info(f"Processing data in channel directory: {channel_dir} for: {data_type_name}")
    
    part_files_gz = list(Path(channel_dir).glob('part*.gz'))
    part_files_csv = list(Path(channel_dir).glob('part*.csv'))

    data_df = None
    signature = None

    if part_files_gz:
        logger.info(f"Found .gz part files in {channel_dir}, unzipping.")
        subprocess.run(f"gzip -d {channel_dir}/part*.gz", shell=True, check=False) # Allow to fail if no .gz
        part_files_csv = list(Path(channel_dir).glob('part*.csv')) # Re-check for CSVs

    if part_files_csv:
        logger.info(f"Found {len(part_files_csv)} part*.csv files in {channel_dir}. Combining them.")
        combined_data_file = Path(channel_dir) / f"{data_type_name}_combined_data.csv"
        signature_file = Path(channel_dir) / f"{data_type_name}_signature.csv"
        
        first_part_file = sorted(part_files_csv)[0]
        
        # Get header for signature
        with open(first_part_file, 'r') as f_in, open(signature_file, 'w') as f_out:
            header = f_in.readline().strip()
            f_out.write(header.replace("__DOT__", ".")) # Write corrected header
        signature = pd.read_csv(signature_file).columns # Read corrected header as columns
        
        # Combine data parts, skipping headers
        with open(combined_data_file, 'wb') as f_out: # Open in binary write to handle bytes from subprocess
            for i, part_file in enumerate(sorted(part_files_csv)):
                # Write header only from the first file (but signature already captured it)
                # So, skip header for all part files when combining actual data
                cmd = f"sed '1d' \"{part_file}\"" # Skip header
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, executable='/bin/bash')
                stdout, stderr = process.communicate()
                if process.returncode == 0:
                    f_out.write(stdout)
                else:
                    logger.error(f"Error processing {part_file}: {stderr.decode() if stderr else 'Unknown error'}")
                    raise RuntimeError(f"Failed to process part file {part_file}")

        logger.info(f"Combined data into {combined_data_file}, signature in {signature_file}")
        data_df = pd.read_csv(combined_data_file, names=signature, header=None) # Read with corrected signature
        logger.info(f"Removing original part files from {channel_dir}")
        for pf_csv in part_files_csv:
            try:
                pf_csv.unlink()
            except OSError as e:
                logger.warning(f"Could not remove part file {pf_csv}: {e}")

    else: # No part files, look for a single CSV or Parquet (as per preprocessing output)
        single_csv_path = Path(channel_dir) / "processed_data.csv" # Typical output from prior SM Processing
        single_parquet_path = Path(channel_dir) / "processed_data.parquet" # Alternative

        if single_csv_path.exists():
            logger.info(f"Found {single_csv_path}. Loading it.")
            # This file is expected to be headerless with specific columns
            # The column names (signature) must be known from config (tag + final_model_var_list)
            data_df = pd.read_csv(single_csv_path, header=None) 
            # Signature will be set later based on config
        elif single_parquet_path.exists():
            logger.info(f"Found {single_parquet_path}. Loading it.")
            data_df = pd.read_parquet(single_parquet_path)
            signature = data_df.columns # Parquet usually has embedded schema
        else:
            raise FileNotFoundError(f"No part files or 'processed_data.csv/parquet' found in {channel_dir}.")

    if data_df is not None:
        logger.info(f"Loaded {data_type_name} data, shape: {data_df.shape}")
    return data_df, signature


def currency_conversion_single_variable(args: Tuple[pd.DataFrame, str, pd.Series]) -> pd.Series:
    df_slice, variable, exchange_rate_series_aligned = args
    # It's crucial that df_slice[variable] and exchange_rate_series_aligned are aligned.
    # If df_slice is truly a slice, its index might not match original.
    # Assuming exchange_rate_series_aligned is passed correctly matched to the df_slice.
    return df_slice[variable].astype(float) / exchange_rate_series_aligned.values

def parallel_currency_conversion(
    df: pd.DataFrame, currency_col: str, currency_conversion_vars: List[str],
    currency_conversion_dict: Dict[str, float], n_workers: int,
) -> pd.DataFrame:
    logger.info(f"Starting parallel currency conversion for {len(currency_conversion_vars)} variables.")
    if not currency_conversion_vars or currency_col not in df.columns:
        logger.warning("Currency column or conversion variables missing. Skipping currency conversion.")
        return df

    exchange_rate_series = df[currency_col].apply(lambda x: currency_conversion_dict.get(str(x), 1.0)) # Ensure x is str for dict lookup
    
    valid_conversion_vars = [var for var in currency_conversion_vars if var in df.columns]
    if not valid_conversion_vars:
        logger.warning("No valid currency conversion variables found in DataFrame columns.")
        return df

    processes = min(cpu_count(), len(valid_conversion_vars), n_workers)
    logger.info(f"Using {processes} workers for currency conversion.")
    
    # Pass only necessary data to workers for efficiency
    tasks = [(df[[var]].copy(), var, exchange_rate_series.loc[df.index]) for var in valid_conversion_vars]

    with Pool(processes=processes) as pool:
        results = pool.map(currency_conversion_single_variable, tasks)
    
    if results:
        for i, var in enumerate(valid_conversion_vars):
            df[var] = results[i] # Assign series back
    logger.info("Parallel currency conversion finished.")
    return df

def map_single_variable(args: Tuple[pd.DataFrame, str, Dict[str, Any]]) -> pd.Series:
    df_slice, variable, mapping_dict_global = args 
    
    if variable not in mapping_dict_global:
        logger.warning(f"No mapping found for variable '{variable}'. Returning original values.")
        return df_slice[variable]
        
    variable_map_config = mapping_dict_global[variable]
    if not isinstance(variable_map_config, dict) or "bins" not in variable_map_config:
        logger.warning(f"Mapping for variable '{variable}' is not in the expected format {{'bins': {{...}}, 'default_bin': ...}}. Returning original.")
        return df_slice[variable]

    category_to_value_map = variable_map_config["bins"]
    default_val = variable_map_config.get("default_bin")
    
    return df_slice[variable].astype(str).map(category_to_value_map).fillna(default_val)

def parallel_mapping(
    df: pd.DataFrame, cat_vars: List[str], mapping_dict: Dict[str, Dict[str, Any]], n_workers: int,
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
        for i, var in enumerate(valid_cat_vars):
            df[var] = results[i]
    logger.info("Parallel mapping finished.")
    return df

def impute_single_variable(args: Tuple[pd.DataFrame, str, Dict[str, Any]]) -> pd.Series:
    df_slice, variable, imputation_dict_global = args
    if variable not in imputation_dict_global:
        logger.warning(f"No imputation value for variable '{variable}'. Skipping.")
        return df_slice[variable]
    impute_value = imputation_dict_global[variable]
    return df_slice[variable].fillna(impute_value)

def parallel_imputation(
    df: pd.DataFrame, num_vars: List[str], imputation_dict: Dict[str, Any], n_workers: int
) -> pd.DataFrame:
    logger.info(f"Starting parallel imputation for {len(num_vars)} numeric variables.")
    valid_num_vars = [var for var in num_vars if var in df.columns]
    if not valid_num_vars:
        logger.warning("No valid numeric variables found for imputation.")
        return df
        
    processes = min(cpu_count(), len(valid_num_vars), n_workers)
    logger.info(f"Using {processes} workers for imputation.")
    tasks = [(df[[var]].copy(), var, imputation_dict) for var in valid_num_vars]
    
    with Pool(processes=processes) as pool:
        results = pool.map(impute_single_variable, tasks)

    if results:
        for i, var in enumerate(valid_num_vars):
            df[var] = results[i]
    logger.info("Parallel imputation finished.")
    return df

def combine_two_cols(df, col_0, col_1, keep_col_preference: int):
    if col_0 not in df.columns:
        logger.warning(f"Column {col_0} not in DataFrame for combine_two_cols. Using {col_1} if present.")
        return df[col_1] if col_1 in df.columns else pd.Series(index=df.index)
    if col_1 not in df.columns:
        logger.warning(f"Column {col_1} not in DataFrame for combine_two_cols. Using {col_0}.")
        return df[col_0]

    if keep_col_preference == 0: 
        return df[col_0].combine_first(df[col_1])
    else: 
        return df[col_1].combine_first(df[col_0])

# =================== Hyperparameter Parsing ===================
def safe_cast(val: Any) -> Any:
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
    sanitized = {}
    for key, val in config.items():
        sanitized_key = key.replace('-', '_') 
        sanitized[sanitized_key] = safe_cast(val)
    return sanitized

def load_and_parse_hyperparameters(hparam_path: str) -> Dict[str, Any]:
    logger.info(f"Loading hyperparameters from: {hparam_path}")
    if not os.path.exists(hparam_path):
        logger.error(f"Hyperparameter file not found at {hparam_path}.")
        raise FileNotFoundError(f"Hyperparameter file not found: {hparam_path}")
    with open(hparam_path, "r") as f:
        raw_hyperparameters = json.load(f)
    logger.info(f"Raw hyperparameters: {raw_hyperparameters}")
    
    parsed_hyperparameters = sanitize_hyperparameters(raw_hyperparameters)
    logger.info(f"Parsed and cast hyperparameters: {parsed_hyperparameters}")
    return parsed_hyperparameters

# =================== Main Data Processing and Training Logic ===================
def run_preprocessing(df: pd.DataFrame, hyperparameters: Dict[str, Any], general_config: Dict[str, Any], n_workers: int) -> pd.DataFrame:
    """Runs the sequence of preprocessing steps on the given DataFrame."""
    logger.info(f"Starting run_preprocessing on data with shape: {df.shape}")

    data_config = general_config.get("data_config", {})
    marketplace_info = data_config.get("marketplace_info", {})
    data_processing_info = data_config.get("data_processing_info", {})
    preprocessing_combine_var_name_pair_list = data_processing_info.get("preprocessing_combine_var_name_pair_list", [])
    
    tag_col_name = hyperparameters.get("tag")
    final_model_var_list = hyperparameters.get("final_model_var_list")
    currency_col_config = general_config.get("currency_col", "currency_code") # From general config
    enable_currency_conversion = general_config.get("enable_currency_conversion", False)
    currency_conversion_var_list = general_config.get("currency_conversion_var_list", [])
    currency_conversion_dict = general_config.get("currency_conversion_dict", {})
    metadata_df = general_config.get("metadata") 
    if not isinstance(metadata_df, pd.DataFrame) and metadata_df is not None:
        logger.warning("Metadata is not a DataFrame as expected from general_config.")
        metadata_df = None 
    marketplace_id_col = general_config.get("marketplace_id_col", "marketplace_id")

    # Clean column names (if they came with __DOT__)
    df.columns = [str(sub).replace("__DOT__", ".") for sub in df.columns]
    logger.info(f"Cleaned data columns: {df.columns.tolist()}")

    logger.info("================")
    logger.info(f"First 5 rows of incoming data for preprocessing:\n{df.head()}")
    logger.info("================")

    if marketplace_id_col in df.columns:
        df = df.dropna(subset=[marketplace_id_col]).reset_index(drop=True)
    else:
        logger.warning(f"Marketplace ID column '{marketplace_id_col}' not found. Skipping dropna.")

    # Currency Handling
    if enable_currency_conversion:
        logger.info("Processing currency conversion.")
        # Ensure currency_col_config is defined and exists, or derive it
        current_currency_col_for_df = currency_col_config # This is the name of the column holding currency codes
        
        if marketplace_id_col in df.columns and str(int(df[marketplace_id_col].iloc[0])) in marketplace_info : # Check if marketplace_info can be used
            def get_currency_code(marketplace_id):
                try:
                    mp_id_str = str(int(float(marketplace_id))) # Handle potential float then int conversion
                    if mp_id_str in marketplace_info:
                        return marketplace_info[mp_id_str].get("currency_code")
                except ValueError: # Handle non-numeric marketplace_id
                    pass
                return np.nan
            df["currency_code_from_marketplace_id"] = df[marketplace_id_col].apply(get_currency_code)

            if current_currency_col_for_df in df.columns: # If a currency column already exists
                logger.info(f"Combining 'currency_code_from_marketplace_id' and existing '{current_currency_col_for_df}' (preferring latter).")
                df[current_currency_col_for_df] = combine_two_cols(df, "currency_code_from_marketplace_id", current_currency_col_for_df, 1)
            else: # No existing currency column, use the derived one
                logger.info(f"Using 'currency_code_from_marketplace_id' as the currency column.")
                df[current_currency_col_for_df] = df["currency_code_from_marketplace_id"]
            
            if "currency_code_from_marketplace_id" in df.columns: # Drop the temporary column
                df = df.drop(columns=["currency_code_from_marketplace_id"])

            if current_currency_col_for_df in df.columns:
                 df = df.dropna(subset=[current_currency_col_for_df]).reset_index(drop=True)
            else:
                logger.warning(f"Final currency column '{current_currency_col_for_df}' not found after processing. Currency conversion might fail or be skipped.")
        else:
            logger.warning(f"Cannot derive currency from marketplace_id or '{marketplace_id_col}' not found. Currency conversion may rely on existing '{current_currency_col_for_df}'.")

        # Actual currency conversion call
        if current_currency_col_for_df in df.columns and currency_conversion_vars:
             df = parallel_currency_conversion(df, current_currency_col_for_df, currency_conversion_vars, currency_conversion_dict, n_workers)
        else:
            logger.warning("Skipping parallel currency conversion due to missing currency column or conversion vars.")
    else:
        logger.info("Currency conversion disabled.")


    # Variable Type Classification
    cat_vars = []
    num_vars = []
    if metadata_df is not None:
        metadata_cat_vars = set(metadata_df.loc[metadata_df["iscategory"].astype(bool), "varname"])
        metadata_num_vars = set(metadata_df.loc[~metadata_df["iscategory"].astype(bool), "varname"])
        logger.info(f"Metadata defined cat_vars: {metadata_cat_vars}")
        logger.info(f"Metadata defined num_vars: {metadata_num_vars}")
        
        for var in df.columns:
            if var in metadata_cat_vars and var in final_model_var_list:
                cat_vars.append(var)
            elif var in metadata_num_vars and var in final_model_var_list:
                num_vars.append(var)
                # Ensure numeric, coercing errors. Preprocessing script filled with -1, but good to be sure.
                df[var] = pd.to_numeric(df[var], errors='coerce') 
    else:
        logger.warning("Metadata not available. Cannot classify cat_vars and num_vars automatically. Ensure data types are correct.")
        # Fallback: attempt to guess based on final_model_var_list or assume they are already processed
        # For simplicity, if no metadata, assume final_model_var_list are all numeric and require imputation if needed.
        # This part would need more robust logic if metadata can be missing.
        num_vars = [var for var in final_model_var_list if var != tag_col_name and var in df.columns]


    logger.info(f"Identified cat_vars for processing: {cat_vars}")
    logger.info(f"Identified num_vars for processing: {num_vars}")

    # Tag Filtering (assuming tag is binary 0/1)
    if tag_col_name in df.columns:
        df[tag_col_name] = pd.to_numeric(df[tag_col_name], errors='coerce')
        df = df[df[tag_col_name].isin([0, 1])].reset_index(drop=True)
        logger.info(f"Data shape after filtering tag for 0/1: {df.shape}")
    else:
        logger.warning(f"Tag column '{tag_col_name}' not found for filtering.")


    # Combine Column Pairs
    if preprocessing_combine_var_name_pair_list:
        logger.info("Starting combining column pairs.")
        for pair in preprocessing_combine_var_name_pair_list:
            col_0, col_1 = pair
            if col_1 in df.columns: # Ensure the target column for combined result exists
                df[col_1] = combine_two_cols(df, col_0, col_1, 1) # Prefer col_1
                logger.info(f"Combined {col_0} into {col_1} (preferring {col_1}).")
            else:
                logger.warning(f"Target column {col_1} for combination not found. Skipping pair: {pair}")
    
    # Categorical Variable Mapping
    if cat_vars:
        # Load bin_mapping.pkl (expected to be in ARTIFACT_INPUT_PATH)
        bin_map_path = os.path.join(ARTIFACT_INPUT_PATH, "bin_mapping.pkl")
        if os.path.exists(bin_map_path):
            bin_map = load_config_pkl(bin_map_path)
            logger.info(f"Applying parallel mapping to: {cat_vars}")
            df = parallel_mapping(df, cat_vars, bin_map, n_workers)
        else:
            logger.warning(f"bin_mapping.pkl not found at {bin_map_path}. Skipping categorical mapping.")
    else:
        logger.info("No categorical variables to map.")
        
    # Numerical Variable Imputation
    if num_vars:
        imputation_dict_path = os.path.join(ARTIFACT_INPUT_PATH, "missing_value_imputation.pkl")
        if os.path.exists(imputation_dict_path):
            missing_value_impute_dict = load_config_pkl(imputation_dict_path)
            logger.info(f"Applying parallel imputation to: {num_vars}")
            df = parallel_imputation(df, num_vars, missing_value_impute_dict, n_workers)
        else:
            logger.warning(f"missing_value_imputation.pkl not found at {imputation_dict_path}. Skipping numerical imputation.")
    else:
        logger.info("No numeric variables to impute.")

    # Final selection and dropna
    output_vars_for_model = [tag_col_name] + final_model_var_list
    # Ensure all selected columns exist
    output_vars_for_model = [col for col in output_vars_for_model if col in df.columns]
    if tag_col_name not in output_vars_for_model and tag_col_name in df.columns: # ensure tag is there
        output_vars_for_model = [tag_col_name] + [col for col in final_model_var_list if col in df.columns and col != tag_col_name]


    if not output_vars_for_model or (tag_col_name not in output_vars_for_model and len(output_vars_for_model) == 0) :
        raise ValueError("No output variables selected for the model or tag column missing. Check final_model_var_list and tag in HPs.")

    logger.info(f"Selecting final output variables for model: {output_vars_for_model}")
    df_processed = df[output_vars_for_model].copy()
    
    rows_before_dropna = len(df_processed)
    df_processed = df_processed.dropna(subset=output_vars_for_model, how="any")
    rows_after_dropna = len(df_processed)
    logger.info(f"Dropped {rows_before_dropna - rows_after_dropna} rows due to NaNs in final selected features.")
    logger.info(f"Final processed data shape: {df_processed.shape}")
    
    return df_processed


def main(cmd_args):
    logger.info("Starting main training script execution.")
    
    # --- 1. Load Hyperparameters ---
    hyperparameters = load_and_parse_hyperparameters(HPARAM_PATH)
    
    tag_col_name = hyperparameters.get("tag")
    final_model_var_list = hyperparameters.get("final_model_var_list")
    if not tag_col_name or not final_model_var_list:
        raise ValueError("'tag' and 'final_model_var_list' must be defined in hyperparameters.")

    # --- 2. Load External General Configuration (for preprocessing params not in HPs) ---
    general_config_path = os.path.join(ARTIFACT_INPUT_PATH, "config.pkl")
    if not os.path.exists(general_config_path):
        raise FileNotFoundError(f"General config file (config.pkl) not found at {general_config_path}. "
                                "This file is expected to contain metadata, currency info, etc.")
    general_config = load_config_pkl(general_config_path)
    logger.info("Loaded general_config.pkl")

    # --- 3. Load and Preprocess Training Data ---
    # SageMaker training jobs usually have 'train' and 'validation' channels.
    # This script will process the data from the 'train' channel.
    # If a 'validation' channel is also provided, it would be processed similarly.
    
    train_channel_path = os.path.join(INPUT_DATA_BASE_PATH, DEFAULT_TRAIN_CHANNEL)
    if not os.path.isdir(train_channel_path):
        raise FileNotFoundError(f"Training data channel directory not found: {train_channel_path}")

    logger.info(f"Attempting to load and process data from training channel: {train_channel_path}")
    # The process_channel_data will look for part files or a single processed_data.csv/parquet
    # If it's already a single processed_data.csv from a prior job, it needs correct column names.
    df_train_raw, train_signature = process_channel_data(train_channel_path, "training")
    
    if df_train_raw is None:
        raise ValueError("Failed to load training data.")

    # If process_channel_data loaded a raw combined CSV, set column names from signature.
    # If it loaded a 'processed_data.csv', it was headerless, so names need to be set.
    # The original preprocessing script saved 'processed_data.csv' as [tag] + final_model_var_list.
    # If we are re-running the full preprocessing, we need the raw columns from signature.
    if train_signature is not None and len(train_signature) == len(df_train_raw.columns):
        logger.info("Applying signature to raw loaded training data.")
        df_train_raw.columns = train_signature
    elif len([tag_col_name] + final_model_var_list) == len(df_train_raw.columns) and train_signature is None:
        logger.info("Assuming raw loaded training data columns are [tag] + final_model_var_list (headerless).")
        df_train_raw.columns = [tag_col_name] + final_model_var_list
    else:
        logger.warning(f"Mismatch in column count ({len(df_train_raw.columns)}) vs expected signature. Columns might be incorrect.")
        # Proceeding with existing columns, ensure they are correct or preprocessing handles it.

    # Run the full preprocessing sequence on the training data
    df_train_processed = run_preprocessing(df_train_raw, hyperparameters, general_config, cmd_args.n_workers)

    # --- Optional: Process Validation Data ---
    df_val_processed = None
    validation_channel_path = os.path.join(INPUT_DATA_BASE_PATH, DEFAULT_VALIDATION_CHANNEL)
    if os.path.isdir(validation_channel_path) and any(os.scandir(validation_channel_path)): # Check if dir exists and is not empty
        logger.info(f"Attempting to load and process data from validation channel: {validation_channel_path}")
        df_val_raw, val_signature = process_channel_data(validation_channel_path, "validation")
        if df_val_raw is not None:
            if val_signature is not None and len(val_signature) == len(df_val_raw.columns):
                df_val_raw.columns = val_signature
            elif len([tag_col_name] + final_model_var_list) == len(df_val_raw.columns) and val_signature is None:
                df_val_raw.columns = [tag_col_name] + final_model_var_list

            df_val_processed = run_preprocessing(df_val_raw, hyperparameters, general_config, cmd_args.n_workers)
    else:
        logger.warning(f"Validation data channel directory not found or empty: {validation_channel_path}. Training without a separate validation set for XGBoost watchlist.")


    # --- 4. Prepare DMatrix and Train XGBoost Model ---
    X_train = df_train_processed[final_model_var_list]
    y_train = df_train_processed[tag_col_name]
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=final_model_var_list)
    logger.info(f"Created DMatrix for training. Shape: ({dtrain.num_row()}, {dtrain.num_col()})")

    watchlist = [(dtrain, 'train')]
    if df_val_processed is not None and not df_val_processed.empty:
        X_val = df_val_processed[final_model_var_list]
        y_val = df_val_processed[tag_col_name]
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=final_model_var_list)
        watchlist.append((dval, 'validation'))
        logger.info(f"Created DMatrix for validation. Shape: ({dval.num_row()}, {dval.num_col()})")
    else: # No validation data, or it became empty after processing
        logger.info("No validation DMatrix created.")
        # If no validation data, early stopping won't work unless an eval set is passed to xgb.train from training data itself
        # For simplicity, if no dval, early_stopping_rounds might cause issues if not handled by xgb.train
        if hyperparameters.get('early_stopping_rounds') is not None:
             logger.warning("Early stopping is configured but no validation data is available. XGBoost might error or ignore early stopping.")


    xgb_params = {}
    param_mapping = {"reg_lambda": "lambda", "reg_alpha": "alpha"}
    excluded_hps_from_xgb_train = ['tag', 'final_model_var_list', 'num_round', 'early_stopping_rounds', 
                                   'enable_currency_conversion', 'downsample_good_orders', 'target_positive_rate'] # Add other non-xgb HPs

    for key, value in hyperparameters.items():
        xgb_key = param_mapping.get(key, key)
        if key not in excluded_hps_from_xgb_train:
            if value is not None:
                xgb_params[xgb_key] = value
    
    num_boost_round = int(hyperparameters.get('num_round', 100)) # Ensure int
    early_stopping_rounds_val = hyperparameters.get('early_stopping_rounds')
    if early_stopping_rounds_val is not None:
        early_stopping_rounds_val = int(early_stopping_rounds_val)
        if not watchlist or len(watchlist) < 2 : # Need validation set in watchlist for early stopping
            logger.warning("Early stopping rounds specified, but no validation set in watchlist. Early stopping may not work.")
            # early_stopping_rounds_val = None # Optionally disable it


    if hyperparameters.get('objective', '').startswith("multi:") and hyperparameters.get('num_class') is not None:
        xgb_params['num_class'] = int(hyperparameters['num_class'])
    
    logger.info(f"XGBoost training parameters: {xgb_params}")
    logger.info(f"Number of boosting rounds: {num_boost_round}")

    bst = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=early_stopping_rounds_val,
        verbose_eval=10
    )
    logger.info("XGBoost model training complete.")

    # --- 5. Save Model and Artifacts ---
    model_filename = "xgboost-model.json" 
    model_save_path = os.path.join(MODEL_PATH, model_filename)
    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving trained XGBoost model to: {model_save_path}")
    bst.save_model(model_save_path)
    
    features_filename = "features.json"
    features_save_path = os.path.join(MODEL_PATH, features_filename)
    with open(features_save_path, 'w') as f:
        json.dump(final_model_var_list, f)
    logger.info(f"Saved feature list to: {features_save_path}")

    hparams_save_filename = "hyperparameters_used.json"
    hparams_save_path = os.path.join(MODEL_PATH, hparams_save_filename)
    with open(hparams_save_path, 'w') as f:
        # Save the initial hyperparameters dictionary for record
        json.dump(hyperparameters, f, indent=2)
    logger.info(f"Saved hyperparameters used to: {hparams_save_path}")

    logger.info("Script finished successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=max(1, cpu_count() -1), 
                        help="Number of worker processes for parallel tasks.")
    # data_type argument from original preprocessing, less relevant if train/val channels are distinct
    # parser.add_argument("--data_type", type=str, default="training") 
    
    # SageMaker passes hyperparameters as command-line arguments or in hyperparameters.json
    # We are reading from hyperparameters.json, so we don't need to parse them here.
    # However, other script-specific args can be added.

    cmd_args, unknown = parser.parse_known_args()
    if unknown:
        logger.info(f"Ignoring unknown command-line arguments: {unknown}")

    try:
        main(cmd_args) 
    except Exception as e:
        trc = traceback.format_exc()
        error_message = f"Exception during script execution: {str(e)}\n{trc}"
        logger.error(error_message)
        
        error_file_path = os.path.join(PREFIX, "failure") # Standard SageMaker failure path
        try:
            # Output path might not be standard /opt/ml/output/data in training, but /opt/ml/output/
            Path(os.path.dirname(error_file_path)).mkdir(parents=True, exist_ok=True)
            with open(error_file_path, "w") as f:
                f.write(error_message)
            logger.info(f"Failure reason written to {error_file_path}")
        except Exception as fe:
            logger.error(f"Failed to write failure file to {error_file_path}: {fe}")
            # Also print to stdout as a last resort
            print(f"FAILURE_TRACE:\n{error_message}", file=sys.stderr)
            
        sys.exit(255)