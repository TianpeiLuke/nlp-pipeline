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
    else:
        open_func = open

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
    and return a pandas DataFrame.
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
    Detect and combine all shards in `input_dir`. Supports various formats.
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    patterns = [
        "part-*.csv", "part-*.csv.gz", "part-*.json", "part-*.json.gz",
        "part-*.parquet", "part-*.snappy.parquet", "part-*.parquet.gz",
    ]

    all_shards = [shard for pat in patterns for shard in input_path.glob(pat)]

    if not all_shards:
        raise RuntimeError(f"No CSV/JSON/Parquet shards found under {input_dir}")

    all_shards.sort()

    dfs = [_read_file_to_df(shard) for shard in all_shards]

    try:
        return pd.concat(dfs, axis=0, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"Failed to concatenate shards: {e}")


def parallel_imputation(df: pd.DataFrame, num_vars: list, impute_dict: dict, n_workers: int) -> pd.DataFrame:
    """Impute missing numeric values in parallel."""
    def impute_single_variable(args):
        df_chunk, var, impute_val = args
        return df_chunk[var].fillna(impute_val)

    nproc = min(cpu_count(), len(num_vars), n_workers)
    with Pool(nproc) as pool:
        tasks = [(df[[var]], var, impute_dict.get(var, 0.0)) for var in num_vars]
        results = pool.map(impute_single_variable, tasks)
    df[num_vars] = pd.concat(results, axis=1)
    return df

# --------------------------------------------------------------------------------
# Main Preprocessing Logic - Refactored into a testable function
# --------------------------------------------------------------------------------
def main(data_type: str, label_field: str, train_ratio: float, test_val_ratio: float, input_dir: str, output_dir: str):
    """
    Core logic for preprocessing, splitting, and saving data.
    This function is now testable by passing parameters directly.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) Combine shards into a single DataFrame
    print(f"[INFO] Combining shards under {input_dir}…")
    df = combine_shards(input_dir)
    print(f"[INFO] Combined data shape: {df.shape}")

    # 2) Rename columns: replace "__DOT__" with "."
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # 3) Verify and convert the label field
    if label_field not in df.columns:
        raise RuntimeError(f"Label field '{label_field}' not found among columns: {df.columns.tolist()}")

    # If label is not already numeric, map text labels to integer codes 0..k-1
    if not pd.api.types.is_numeric_dtype(df[label_field]):
        unique_labels = sorted(df[label_field].dropna().unique())
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[label_field] = df[label_field].map(label_map)

    df[label_field] = pd.to_numeric(df[label_field], errors="coerce").astype("Int64")

    # 4) Drop any rows with missing label
    df_before = df.shape[0]
    df = df.dropna(subset=[label_field], how="any")
    print(f"[INFO] Dropped {df_before - df.shape[0]} rows due to missing label")

    # 5) Split data if in 'training' mode, otherwise save as a single set
    if data_type == "training":
        train_df, holdout_df = train_test_split(
            df, train_size=train_ratio, random_state=42, stratify=df[label_field]
        )
        test_df, val_df = train_test_split(
            holdout_df, test_size=test_val_ratio, random_state=42, stratify=holdout_df[label_field]
        )
        splits = {"train": train_df, "test": test_df, "val": val_df}
    else:
        splits = {data_type: df}

    # 6) Save the output files for each split
    for split_name, split_df in splits.items():
        subfolder = output_path / split_name
        subfolder.mkdir(exist_ok=True)

        required_cols = [label_field] + [c for c in split_df.columns if c != label_field]
        df_proc = split_df[required_cols].copy()
        df_proc[label_field] = df_proc[label_field].astype(int)

        for col in required_cols:
            if col != label_field:
                df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce')

        proc_path = subfolder / f"{split_name}_processed_data.csv"
        df_proc.to_csv(proc_path, index=False)
        print(f"[INFO] Saved {split_name}_processed_data.csv to {proc_path} (shape={df_proc.shape})")

        full_path = subfolder / f"{split_name}_full_data.csv"
        split_df.to_csv(full_path, index=False)
        print(f"[INFO] Saved {split_name}_full_data.csv to {full_path} (shape={split_df.shape})")

    print("[INFO] Preprocessing complete.")

# --------------------------------------------------------------------------------
# Script Entry Point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type", type=str, required=True,
        help="One of ['training','validation','testing','calibration']",
    )
    args, _ = parser.parse_known_args()

    # Read configuration from environment variables
    LABEL_FIELD = os.environ.get("LABEL_FIELD", "").strip()
    TRAIN_RATIO = float(os.environ.get("TRAIN_RATIO", "0.7"))
    TEST_VAL_RATIO = float(os.environ.get("TEST_VAL_RATIO", "0.5"))

    if not LABEL_FIELD:
        raise RuntimeError("LABEL_FIELD must be set as an environment variable")

    # Define standard SageMaker paths
    input_data_dir = "/opt/ml/processing/input/data"
    output_dir = "/opt/ml/processing/output"

    # Execute the main processing logic
    main(
        data_type=args.data_type,
        label_field=LABEL_FIELD,
        train_ratio=TRAIN_RATIO,
        test_val_ratio=TEST_VAL_RATIO,
        input_dir=input_data_dir,
        output_dir=output_dir
    )
