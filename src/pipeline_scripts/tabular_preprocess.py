#!/usr/bin/env python
import os
import sys
import argparse
import gzip
import shutil
import glob
import tempfile
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle as pkl
from sklearn.impute import SimpleImputer

# --------------------------------------------------------------------------------
# Helper Functions for File Combination
# --------------------------------------------------------------------------------

def _is_gzipped(path: str) -> bool:
    return path.lower().endswith(".gz")

def _decompress_gzip(source_path: str, target_path: str) -> None:
    """Decompress a .gz file to the given target path."""
    with gzip.open(source_path, "rb") as f_in, open(target_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def _combine_csv_tsv_shards(input_dir: str, data_type: str, delimiter: str) -> pd.DataFrame:
    """
    Combine multiple CSV/TSV shards into a single DataFrame.
    Assumes:
      - There is exactly one “header” file pattern, e.g. part-00000*.csv (or .tsv).
      - All other shards begin with the same header; we skip their first row.
    Returns:
      A single pandas DataFrame.
    """
    temp_dir = tempfile.mkdtemp()
    header_file = None
    shard_files = []

    # Step 1: Collect and decompress if needed
    for ext in ["csv", "tsv"]:
        pattern = os.path.join(input_dir, f"part-*.{ext}*")
        for path in glob.glob(pattern):
            if _is_gzipped(path):
                # Decompress to temp_dir
                fn = os.path.basename(path).replace(".gz", "")
                decompressed = os.path.join(temp_dir, fn)
                _decompress_gzip(path, decompressed)
                shard_files.append(decompressed)
            else:
                shard_files.append(path)

    if not shard_files:
        raise RuntimeError(f"No CSV/TSV shards found under {input_dir}")

    # Identify the “first” shard (by lex order) as header source
    shard_files.sort()
    header_file = shard_files[0]
    all_data_paths = shard_files

    # Read header separately
    header_df = pd.read_csv(header_file, sep=delimiter, nrows=0)
    columns = header_df.columns.tolist()

    # Read each shard by skipping first row, then concatenate
    dfs = []
    for shard in all_data_paths:
        df_shard = pd.read_csv(shard, sep=delimiter, header=0)
        dfs.append(df_shard)
    combined = pd.concat(dfs, axis=0, ignore_index=True)
    # Ensure column names are as in header
    combined.columns = columns
    shutil.rmtree(temp_dir, ignore_errors=True)
    return combined

def _combine_json_shards(input_dir: str) -> pd.DataFrame:
    """
    Read and concatenate multiple JSON shards (one JSON object per line or array form).
    """
    json_paths = glob.glob(os.path.join(input_dir, "part-*.json*"))
    if not json_paths:
        raise RuntimeError(f"No JSON shards found under {input_dir}")

    dfs = []
    for path in json_paths:
        if _is_gzipped(path):
            with gzip.open(path, "rt") as f_in:
                dfs.append(pd.read_json(f_in, lines=True))
        else:
            dfs.append(pd.read_json(path, lines=True))
    return pd.concat(dfs, axis=0, ignore_index=True)

def _combine_parquet_shards(input_dir: str) -> pd.DataFrame:
    """
    Read and concatenate multiple Parquet shards.
    """
    pq_paths = glob.glob(os.path.join(input_dir, "part-*.parquet"))
    if not pq_paths:
        raise RuntimeError(f"No Parquet shards found under {input_dir}")
    dfs = [pd.read_parquet(p) for p in pq_paths]
    return pd.concat(dfs, axis=0, ignore_index=True)

def combine_shards(input_dir: str) -> pd.DataFrame:
    """
    Detect file type in input_dir and combine accordingly:
      - CSV (.csv or .csv.gz) or TSV (.tsv or .tsv.gz) → use _combine_csv_tsv_shards
      - JSON (.json or .json.gz)                → use _combine_json_shards
      - Parquet (.parquet)                     → use _combine_parquet_shards
    Returns a single pandas DataFrame.
    """
    # Check for any CSV/TSV shards
    if glob.glob(os.path.join(input_dir, "part-*.csv")) or glob.glob(os.path.join(input_dir, "part-*.csv.gz")):
        return _combine_csv_tsv_shards(input_dir, data_type="", delimiter=",")
    if glob.glob(os.path.join(input_dir, "part-*.tsv")) or glob.glob(os.path.join(input_dir, "part-*.tsv.gz")):
        return _combine_csv_tsv_shards(input_dir, data_type="", delimiter="\t")
    # Check for JSON
    if glob.glob(os.path.join(input_dir, "part-*.json")) or glob.glob(os.path.join(input_dir, "part-*.json.gz")):
        return _combine_json_shards(input_dir)
    # Check for Parquet
    if glob.glob(os.path.join(input_dir, "part-*.parquet")):
        return _combine_parquet_shards(input_dir)

    raise RuntimeError(f"No recognizable shards found in {input_dir}")


# --------------------------------------------------------------------------------
# Parallel Imputation & Mapping Helpers
# --------------------------------------------------------------------------------

def impute_single_variable(args):
    """Helper for parallel numeric imputation."""
    df_chunk, var, impute_dict = args
    fill_val = impute_dict.get(var, 0.0)
    return df_chunk[var].fillna(fill_val)

def parallel_imputation(df: pd.DataFrame, num_vars: list, impute_dict: dict, n_workers: int) -> pd.DataFrame:
    """
    Impute missing numeric values in parallel across `num_vars`.
    Each var is replaced by df[var].fillna(impute_value).
    """
    nproc = min(cpu_count(), len(num_vars), n_workers)
    with Pool(nproc) as pool:
        tasks = [(df[[var]], var, impute_dict) for var in num_vars]
        results = pool.map(impute_single_variable, tasks)
    # results is a list of pd.Series in the same order as num_vars
    df[num_vars] = pd.concat(results, axis=1)
    return df

def map_single_variable(args):
    """Helper for parallel risk‐table mapping of a single categorical var."""
    df_chunk, var, mapping_dict = args
    var_map = mapping_dict[var]
    default_val = var_map.get("default_bin", 0.0)
    return df_chunk[var].map(lambda x: var_map.get(x, default_val))

def parallel_risk_mapping(df: pd.DataFrame, cat_vars: list, bin_map: dict, n_workers: int) -> pd.DataFrame:
    """
    Replace each categorical column in `cat_vars` by its “risk” value
    according to `bin_map[var]`, in parallel.
    """
    nproc = min(cpu_count(), len(cat_vars), n_workers)
    with Pool(nproc) as pool:
        tasks = [(df[[var]], var, bin_map) for var in cat_vars]
        results = pool.map(map_single_variable, tasks)
    df[cat_vars] = pd.concat(results, axis=1)
    return df


# --------------------------------------------------------------------------------
# Main Preprocessing Logic
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="One of ['training','validation','testing','calibration']",
    )
    args, _ = parser.parse_known_args()
    data_type = args.data_type

    # 1) Environment variable inputs
    NUMERIC_FIELDS = os.environ.get("NUMERIC_FIELDS", "")
    CATEGORICAL_FIELDS = os.environ.get("CATEGORICAL_FIELDS", "")
    LABEL_FIELD = os.environ.get("LABEL_FIELD", "")
    N_WORKERS = int(os.environ.get("N_WORKERS", "50"))
    # e.g. "age,income,price"
    num_vars = [v.strip() for v in NUMERIC_FIELDS.split(",") if v.strip()]
    cat_vars = [v.strip() for v in CATEGORICAL_FIELDS.split(",") if v.strip()]
    label_field = LABEL_FIELD.strip()
    if not label_field:
        raise RuntimeError("LABEL_FIELD must be set as an environment variable")

    # 2) Directories inside the Processing container
    input_data_dir = "/opt/ml/processing/input/data"
    config_dir = "/opt/ml/processing/input/config"
    output_dir = "/opt/ml/processing/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 3) Combine shards into a single DataFrame
    print(f"[INFO] Combining shards under {input_data_dir} …")
    df = combine_shards(input_data_dir)
    print(f"[INFO] Combined data shape: {df.shape}")

    # 4) Rename columns: replace "__DOT__" with "."
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # 5) Verify and convert the label field
    if label_field not in df.columns:
        raise RuntimeError(f"Label field '{label_field}' not found among columns: {df.columns.tolist()}")

    # If label is not already numeric, map text labels to integer codes
    if not pd.api.types.is_integer_dtype(df[label_field]) and not pd.api.types.is_float_dtype(df[label_field]):
        unique_labels = df[label_field].dropna().unique().tolist()
        unique_labels.sort()
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[label_field] = df[label_field].map(lambda x: label_map.get(x, np.nan))
    # Now ensure that labels are integers from 0..k-1
    df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype("Int64")

    # 6) Load missing‐value imputation dictionary (if available)
    impute_dict = {}
    impute_path = os.path.join(config_dir, "missing_value_imputation.pkl")
    if os.path.exists(impute_path):
        with open(impute_path, "rb") as f:
            impute_dict = pkl.load(f)

    # 7) Load binning‐mapping dictionary for categorical risk mapping (if available)
    bin_map = {}
    bin_map_path = os.path.join(config_dir, "bin_mapping.pkl")
    if os.path.exists(bin_map_path):
        with open(bin_map_path, "rb") as f:
            bin_map = pkl.load(f)

    # 8) Impute numeric variables in parallel
    present_num_vars = [v for v in num_vars if v in df.columns]
    if present_num_vars and impute_dict:
        print(f"[INFO] Imputing numeric variables: {present_num_vars}")
        df = parallel_imputation(df, present_num_vars, impute_dict, N_WORKERS)
    else:
        print("[INFO] Skipping numeric imputation (no variables or no impute dict)")

    # 9) Risk‐table mapping for categorical variables
    present_cat_vars = [v for v in cat_vars if v in df.columns]
    if present_cat_vars and bin_map:
        print(f"[INFO] Applying risk‐table mapping on: {present_cat_vars}")
        df = parallel_risk_mapping(df, present_cat_vars, bin_map, N_WORKERS)
    else:
        print("[INFO] Skipping categorical risk mapping (no variables or no bin_map)")

    # 10) Drop any rows with missing label or missing any of the requested vars
    required_cols = [label_field] + present_num_vars + present_cat_vars
    df_before = df.shape[0]
    df = df.dropna(subset=required_cols, how="any")
    print(f"[INFO] Dropped {df_before - df.shape[0]} rows due to missing required columns")

    # 11) Save the “processed” subset (only required_cols) and “full” DataFrame
    processed_path = os.path.join(output_dir, f"{data_type}_processed_data.csv")
    full_path = os.path.join(output_dir, f"{data_type}_full_data.csv")

    # Save processed: cast label to int, numeric/categorical to float
    df_proc = df[required_cols].copy()
    df_proc[label_field] = df_proc[label_field].astype(int)
    for col in present_num_vars + present_cat_vars:
        df_proc[col] = df_proc[col].astype(float)
    df_proc.to_csv(processed_path, index=False)
    print(f"[INFO] Saved processed data to {processed_path} (shape={df_proc.shape})")

    # Save full: including all columns (label as int where possible)
    df_full = df.copy()
    df_full[label_field] = df_full[label_field].astype(int)
    df_full.to_csv(full_path, index=False)
    print(f"[INFO] Saved full data to {full_path} (shape={df_full.shape})")

    print("[INFO] Preprocessing complete.")
