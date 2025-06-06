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
import pandas as pd
import numpy as np

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
    # Determine if gzipped
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
      - gzipped or plain CSV  (*.csv, *.csv.gz)
      - gzipped or plain TSV  (*.tsv, *.tsv.gz)
      - gzipped or plain JSON (*.json, *.json.gz)
      - Parquet (*.parquet, *.snappy.parquet, .parquet.gz)
    """
    suffix = file_path.suffix.lower()
    stem_extension = Path(file_path.stem).suffix.lower()

    # Case A: Gzipped file (".gz" suffix)
    if suffix == ".gz":
        inner_ext = stem_extension
        if inner_ext in (".csv", ".tsv"):
            delimiter = "," if inner_ext == ".csv" else "\t"
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
        if suffix in (".csv", ".tsv"):
            delimiter = "," if suffix == ".csv" else "\t"
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
# Main Preprocessing Logic (no imputation / no risk mapping)
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        help="One of ['training','validation','testing','calibration']"
    )
    args, _ = parser.parse_known_args()
    data_type = args.data_type

    # 1) Required environment‐variable inputs
    LABEL_FIELD = os.environ.get("LABEL_FIELD", "").strip()
    if not LABEL_FIELD:
        raise RuntimeError("LABEL_FIELD must be set as an environment variable")

    # 2) Directories inside the Processing container
    input_data_dir = "/opt/ml/processing/input/data"
    output_dir     = "/opt/ml/processing/output"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 3) Combine shards into a single DataFrame
    print(f"[INFO] Combining shards under {input_data_dir} …")
    df = combine_shards(input_data_dir)
    print(f"[INFO] Combined data shape: {df.shape}")

    # 4) Rename columns: replace "__DOT__" with "."
    df.columns = [col.replace("__DOT__", ".") for col in df.columns]

    # 5) Verify and convert the label field
    if LABEL_FIELD not in df.columns:
        raise RuntimeError(f"Label field '{LABEL_FIELD}' not found. "
                           f"Columns: {df.columns.tolist()}")

    # If label is not numeric, map each unique value → [0..k-1]
    if not pd.api.types.is_integer_dtype(df[LABEL_FIELD]) and not pd.api.types.is_float_dtype(df[LABEL_FIELD]):
        unique_labels = df[LABEL_FIELD].dropna().unique().tolist()
        unique_labels.sort()
        label_map = {val: idx for idx, val in enumerate(unique_labels)}
        df[LABEL_FIELD] = df[LABEL_FIELD].map(lambda x: label_map.get(x, np.nan))

    # Now ensure labels are integer 0..k-1 (with NA → <NA>)
    df[LABEL_FIELD] = pd.to_numeric(df[LABEL_FIELD], errors="coerce").astype("Int64")

    # 6) Drop any rows with a missing label
    before_count = df.shape[0]
    df = df.dropna(subset=[LABEL_FIELD], how="any")
    dropped = before_count - df.shape[0]
    print(f"[INFO] Dropped {dropped} rows due to missing label")

    # 7) Save “processed” (only LABEL_FIELD + any columns you wish downstream) 
    #    and “full” DataFrame (all columns)
    #    – Here, we choose “processed” to include LABEL_FIELD + everything
    #      else (downstream can select features later).
    processed_path = os.path.join(output_dir, f"{data_type}_processed_data.csv")
    full_path      = os.path.join(output_dir, f"{data_type}_full_data.csv")

    # Cast label to native int
    df[LABEL_FIELD] = df[LABEL_FIELD].astype(int)

    # Save "processed": write out exactly the same columns (could subset if desired)
    df.to_csv(processed_path, index=False)
    print(f"[INFO] Saved processed data to {processed_path} (shape={df.shape})")

    # For “full,” we simply write the same DF again (could be identical to processed in this version)
    df.to_csv(full_path, index=False)
    print(f"[INFO] Saved full data to      {full_path} (shape={df.shape})")

    print("[INFO] Preprocessing complete.")

