#!/usr/bin/env python
import os
import gzip
import glob
import tempfile
import shutil
import csv
import json
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------------
# Helper Functions for File Combination
# --------------------------------------------------------------------------------

def _is_gzipped(path: str) -> bool:
    return path.lower().endswith(".gz")


def _detect_separator_from_sample(sample_lines: str) -> str:
    """
    Use csv.Sniffer to detect a delimiter from a short sample of text.
    If detection fails, default to comma.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_lines)
        return dialect.delimiter
    except Exception:
        return ","


def peek_json_format(file_path: Path, open_func=open) -> str:
    """
    Check if the JSON file is in JSON Lines format (one object per line)
    or regular JSON format (single object or array).
    Returns 'lines' for newline-delimited JSON, or 'regular' otherwise.
    """
    try:
        with open_func(str(file_path), "rt") as f:
            first_char = f.read(1)
            if not first_char:
                raise ValueError("Empty file")

            f.seek(0)
            first_line = f.readline().strip()

            # Try to parse first line as JSON
            try:
                json.loads(first_line)
                # If successful and first char isn't '[', it's likely JSON Lines
                return "lines" if first_char != "[" else "regular"
            except json.JSONDecodeError:
                # If can't parse first line, try to parse entire file
                f.seek(0)
                whole = f.read()
                try:
                    json.loads(whole)
                    return "regular"
                except json.JSONDecodeError:
                    raise ValueError("Invalid JSON format")
    except Exception as e:
        raise RuntimeError(f"Error checking JSON format for {file_path}: {e}")


def _read_json_file(file_path: Path) -> pd.DataFrame:
    """
    Read a JSON file, handling both JSON Lines and regular JSON formats,
    and return a pandas DataFrame.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".gz":
        open_func = gzip.open
        inner_suffix = Path(file_path.stem).suffix.lower()
    else:
        open_func = open
        inner_suffix = suffix

    fmt = peek_json_format(file_path, open_func)
    if fmt == "lines":
        return pd.read_json(str(file_path), lines=True, compression="infer")
    else:
        with open_func(str(file_path), "rt") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        return pd.json_normalize(data)


def _read_file_to_df(file_path: Path) -> pd.DataFrame:
    """
    Read a single file (CSV/TSV/JSON/Parquet), decompressing if needed,
    and return a pandas DataFrame. Supports:
      - gzipped or plain CSV (*.csv, *.csv.gz)
      - gzipped or plain TSV (*.tsv, *.tsv.gz)
      - gzipped or plain JSON (*.json, *.json.gz)
      - Parquet (*.parquet, *.snappy.parquet, or gzipped Parquet *.parquet.gz)
    """
    suffix = file_path.suffix.lower()
    stem_extension = Path(file_path.stem).suffix.lower()

    # Case A: Gzipped file (".gz" suffix)
    if suffix == ".gz":
        inner_ext = stem_extension
        if inner_ext == ".csv" or inner_ext == ".tsv":
            with gzip.open(str(file_path), "rt") as f:
                sample = f.readline() + f.readline()
            sep = _detect_separator_from_sample(sample)
            return pd.read_csv(str(file_path), sep=sep, compression="gzip")

        elif inner_ext == ".json":
            return _read_json_file(file_path)

        elif inner_ext.endswith(".parquet"):
            # Decompress to a temporary file, then read
            tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
            with gzip.open(str(file_path), "rb") as f_in, open(tmp.name, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            df = pd.read_parquet(tmp.name)
            tmp.close()
            os.unlink(tmp.name)
            return df

        else:
            raise ValueError(f"Unsupported gzipped file type: {file_path}")

    # Case B: Plaintext or Parquet file (no ".gz")
    else:
        if suffix == ".csv" or suffix == ".tsv":
            with open(str(file_path), "rt") as f:
                sample = f.readline() + f.readline()
            sep = _detect_separator_from_sample(sample)
            return pd.read_csv(str(file_path), sep=sep)

        elif suffix == ".json":
            return _read_json_file(file_path)

        elif suffix.endswith(".parquet"):
            return pd.read_parquet(str(file_path))

        else:
            raise ValueError(f"Unsupported file type: {file_path}")


def combine_shards(input_dir: str) -> pd.DataFrame:
    """
    Detect and combine all shards in `input_dir`. Supports:
      - CSV (.csv, .csv.gz)
      - JSON (.json, .json.gz)
      - Parquet (.parquet, .snappy.parquet, .parquet.gz)

    Returns a single pandas DataFrame containing all rows from every shard.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    # 1) Find all relevant shard files
    patterns = [
        "part-*.csv",
        "part-*.csv.gz",
        "part-*.json",
        "part-*.json.gz",
        "part-*.parquet",
        "part-*.snappy.parquet",
        "part-*.parquet.gz",
    ]

    all_shards = []
    for pat in patterns:
        all_shards.extend(input_path.glob(pat))

    if not all_shards:
        raise RuntimeError(f"No CSV/JSON/Parquet shards found under {input_dir}")

    # 2) Sort lexicographically so that reading order is deterministic
    all_shards = sorted(all_shards)

    # 3) Read each shard into a DataFrame
    dfs = []
    for shard in all_shards:
        try:
            df_shard = _read_file_to_df(shard)
            dfs.append(df_shard)
        except Exception as e:
            raise RuntimeError(f"Failed to read shard '{shard}': {e}")

    # 4) Concatenate all DataFrames
    try:
        combined = pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate shards: {e}")

    return combined


# --------------------------------------------------------------------------------
# Parallel Imputation Helper (unchanged)
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
    df[num_vars] = pd.concat(results, axis=1)
    return df


# --------------------------------------------------------------------------------
# Main Preprocessing Logic with Train/Test/Val Split
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_type",
        type=str,
        required=True,
        help="One of ['training','validation','testing','calibration']"
    )
    args, _ = parser.parse_known_args()
    job_type = args.job_type

    # 1) Required environment variable inputs
    LABEL_FIELD = os.environ.get("LABEL_FIELD", "").strip()
    TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.7"))
    TEST_VAL_RATIO = float(os.environ.get("TEST_VAL_RATIO", "0.5"))

    if not LABEL_FIELD:
        raise RuntimeError("LABEL_FIELD must be set as an environment variable")
    if not (0.0 < TRAIN_RATIO < 1.0):
        raise RuntimeError(f"TRAIN_RATIO must be in (0,1), got {TRAIN_RATIO}")
    if not (0.0 < TEST_VAL_RATIO < 1.0):
        raise RuntimeError(f"TEST_VAL_RATIO must be in (0,1), got {TEST_VAL_RATIO}")

    # 2) Directories inside the Processing container
    input_base_dir = "/opt/ml/processing/input"
    input_data_dir = f"{input_base_dir}/data"
    input_metadata_dir = f"{input_base_dir}/metadata"  # Optional
    input_signature_dir = f"{input_base_dir}/signature"  # Optional
    output_dir = "/opt/ml/processing/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 3) Combine main data shards into a single DataFrame
    print(f"[INFO] Combining data shards under {input_data_dir} …")
    df = combine_shards(input_data_dir)
    print(f"[INFO] Combined data shape: {df.shape}")

    # Optional: Load metadata if available
    metadata = None
    if Path(input_metadata_dir).exists():
        print(f"[INFO] Loading metadata from {input_metadata_dir} ...")
        try:
            metadata_files = list(Path(input_metadata_dir).glob("*"))
            if metadata_files:
                metadata = _read_file_to_df(metadata_files[0])
                print(f"[INFO] Loaded metadata shape: {metadata.shape}")
        except Exception as e:
            print(f"[WARN] Failed to load metadata: {e}")

    # Optional: Load signature if available
    signature = None
    if Path(input_signature_dir).exists():
        print(f"[INFO] Loading signature from {input_signature_dir} ...")
        try:
            signature_files = list(Path(input_signature_dir).glob("*"))
            if signature_files:
                signature = _read_file_to_df(signature_files[0])
                print(f"[INFO] Loaded signature shape: {signature.shape}")
        except Exception as e:
            print(f"[WARN] Failed to load signature: {e}")

    # 4) Rename columns: replace "__DOT__" with "."
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # 5) Verify and convert the label field
    if LABEL_FIELD not in df.columns:
        raise RuntimeError(f"Label field '{LABEL_FIELD}' not found among columns: {df.columns.tolist()}")

    # Map non-numeric labels to integer codes
    if not pd.api.types.is_integer_dtype(df[LABEL_FIELD]) and not pd.api.types.is_float_dtype(df[LABEL_FIELD]):
        unique_labels = df[LABEL_FIELD].dropna().unique().tolist()
        unique_labels.sort()
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[LABEL_FIELD] = df[LABEL_FIELD].map(lambda x: label_map.get(x, np.nan))
    df[LABEL_FIELD] = pd.to_numeric(df[LABEL_FIELD], errors="coerce").astype("Int64")

    # 6) Drop any rows with missing label
    df_before = df.shape[0]
    df = df.dropna(subset=[LABEL_FIELD], how="any")
    print(f"[INFO] Dropped {df_before - df.shape[0]} rows due to missing label")

    # --------------------------------------------------------------------------------
    # 7) If job_type == "training", do a three‐way train/test/val split
    #    Otherwise, just write a single folder named after job_type.
    # --------------------------------------------------------------------------------

    if job_type == "training":
        # (a) First split: train vs holdout
        train_df, holdout_df = train_test_split(
            df,
            train_size=TRAIN_RATIO,
            random_state=42,
            stratify=df[LABEL_FIELD]
        )

        # (b) Split holdout into test vs val
        test_df, val_df = train_test_split(
            holdout_df,
            test_size=TEST_VAL_RATIO,
            random_state=42,
            stratify=holdout_df[LABEL_FIELD]
        )

        splits = [("train", train_df), ("test", test_df), ("val", val_df)]
    else:
        # Single split for validation/testing/calibration
        splits = [(job_type, df)]

    for split_name, split_df in splits:
        subfolder = Path(output_dir) / split_name
        subfolder.mkdir(parents=True, exist_ok=True)

        # Required columns = [LABEL_FIELD] + (all other columns)
        required_cols = [LABEL_FIELD] + [c for c in split_df.columns if c != LABEL_FIELD]

        # (i) Processed subset
        df_proc = split_df[required_cols].copy()
        df_proc[LABEL_FIELD] = df_proc[LABEL_FIELD].astype(int)
        for col in required_cols:
            if col != LABEL_FIELD:
                df_proc[col] = df_proc[col].astype(float, errors="ignore")
        proc_path = subfolder / f"{split_name}_processed_data.csv"
        df_proc.to_csv(proc_path, index=False)
        print(f"[INFO] Saved {split_name}_processed_data.csv to {proc_path} (shape={df_proc.shape})")

        # (ii) Full
        df_full = split_df.copy()
        df_full[LABEL_FIELD] = df_full[LABEL_FIELD].astype(int)
        full_path = subfolder / f"{split_name}_full_data.csv"
        df_full.to_csv(full_path, index=False)
        print(f"[INFO] Saved {split_name}_full_data.csv to {full_path} (shape={df_full.shape})")

    print("[INFO] Preprocessing complete.")