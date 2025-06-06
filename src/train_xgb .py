#!/usr/bin/env python3
import os
import sys
import json
import pickle as pkl
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import xgboost as xgb

# ================== Model, Data and Hyperparameter Folder =================
prefix = "/opt/ml/"
input_path = os.path.join(prefix, "input/data")
output_path = os.path.join(prefix, "output/data")
model_path = os.path.join(prefix, "model")
hparam_path = os.path.join(prefix, "input/config/hyperparameters.json")
checkpoint_path = os.environ.get("SM_CHECKPOINT_DIR", os.path.join(prefix, "checkpoints"))

train_channel = "train"
train_path = os.path.join(input_path, train_channel)
val_channel = "val"
val_path = os.path.join(input_path, val_channel)
test_channel = "test"
test_path = os.path.join(input_path, test_channel)

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def load_hyperparameters(hparam_file: str) -> Dict[str, Any]:
    """
    Load hyperparameters JSON into a plain dictionary.
    """
    if not os.path.isfile(hparam_file):
        raise FileNotFoundError(f"Hyperparameter file not found: {hparam_file}")

    with open(hparam_file, "r") as f:
        hparams = json.load(f)

    logger.info("Loaded hyperparameters:")
    for k, v in hparams.items():
        logger.info("  %s = %s", k, v)
    return hparams


def build_risk_mapping(
    df: pd.DataFrame,
    cat_fields: List[str],
    label_col: str
) -> Dict[str, Dict[Any, float]]:
    """
    For each categorical field in cat_fields, compute risk = P(label=1 | category).
    Returns a dict: { field_name: { category_value: risk, ..., "default_bin": default_risk } }
    """
    risk_map: Dict[str, Dict[Any, float]] = {}
    default_risk = df[label_col].mean()

    for var in cat_fields:
        if df[var].isna().all():
            risk_map[var] = {"default_bin": default_risk}
            continue

        grouped = df.groupby(var)[label_col].mean().reset_index()
        bins = dict(zip(grouped[var].tolist(), grouped[label_col].tolist()))

        mapping = bins.copy()
        mapping["default_bin"] = default_risk
        risk_map[var] = mapping

    return risk_map


def apply_risk_mapping(
    df: pd.DataFrame,
    cat_fields: List[str],
    risk_map: Dict[str, Dict[Any, float]]
) -> pd.DataFrame:
    """
    Replace each categorical column in cat_fields by its risk value, using risk_map.
    Missing or unseen categories map to default_bin.
    """
    for var in cat_fields:
        mapping = risk_map.get(var, {})
        default_bin = mapping.get("default_bin", 0.0)
        df[var] = df[var].map(lambda x: mapping.get(x, default_bin))
    return df


def compute_imputation_dict(
    df: pd.DataFrame,
    num_fields: List[str]
) -> Dict[str, float]:
    """
    Compute median-based imputation values for each numeric field in num_fields.
    Returns { field_name: median_value }.
    """
    impute_dict: Dict[str, float] = {}
    for var in num_fields:
        if df[var].isna().all():
            impute_dict[var] = 0.0
        else:
            impute_dict[var] = float(df[var].median())
    return impute_dict


def apply_imputation(
    df: pd.DataFrame,
    num_fields: List[str],
    impute_dict: Dict[str, float]
) -> pd.DataFrame:
    """
    Fill NaN in each numeric column with its corresponding value from impute_dict.
    """
    for var in num_fields:
        fill_val = impute_dict.get(var, 0.0)
        df[var] = df[var].fillna(fill_val)
    return df


def load_csv_file(
    directory: str,
    filename: str
) -> pd.DataFrame:
    """
    Given a directory (e.g., "/opt/ml/input/data/train") and a filename
    (e.g., "train_processed_data.csv"), load the CSV into a DataFrame.
    """
    csv_path = Path(directory) / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected file not found: {csv_path}")
    return pd.read_csv(csv_path)


def train_xgboost(
    raw_hparams: Dict[str, Any],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str
) -> xgb.Booster:
    """
    Train an XGBoost model using the provided DataFrames and hyperparameters dict.
    Returns the trained Booster.
    """
    # Extract parameters from raw_hparams, with defaults
    params: Dict[str, Any] = {
        "booster":          raw_hparams.get("booster", "gbtree"),
        "eta":              raw_hparams.get("eta", 0.3),
        "gamma":            raw_hparams.get("gamma", 0.0),
        "max_depth":        raw_hparams.get("max_depth", 6),
        "min_child_weight": raw_hparams.get("min_child_weight", 1.0),
        "max_delta_step":   raw_hparams.get("max_delta_step", 0.0),
        "subsample":        raw_hparams.get("subsample", 1.0),
        "colsample_bytree": raw_hparams.get("colsample_bytree", 1.0),
        "colsample_bylevel":raw_hparams.get("colsample_bylevel", 1.0),
        "colsample_bynode": raw_hparams.get("colsample_bynode", 1.0),
        "lambda":           raw_hparams.get("lambda_xgb", 1.0),
        "alpha":            raw_hparams.get("alpha_xgb", 0.0),
        "tree_method":      raw_hparams.get("tree_method", "auto"),
        "scale_pos_weight": raw_hparams.get("scale_pos_weight", 1.0),
        "objective":        raw_hparams.get("objective", "reg:squarederror"),
    }

    if "sketch_eps" in raw_hparams:
        params["sketch_eps"] = raw_hparams["sketch_eps"]
    if "base_score" in raw_hparams:
        params["base_score"] = raw_hparams["base_score"]
    if "eval_metric" in raw_hparams:
        params["eval_metric"] = raw_hparams["eval_metric"]
    if "seed" in raw_hparams:
        params["seed"] = raw_hparams["seed"]
    if raw_hparams.get("is_binary", True) is False:
        params["num_class"] = raw_hparams.get("num_classes", 2)

    # Data preparation
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df[label_col])
    dval   = xgb.DMatrix(val_df[feature_cols],   label=val_df[label_col])

    num_round = raw_hparams.get("num_round", 100)
    early_stop = raw_hparams.get("early_stopping_rounds", None)

    logger.info("Starting XGBoost training for %d rounds; early_stopping=%s", num_round, early_stop)
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=early_stop,
        verbose_eval=True
    )
    logger.info("XGBoost training complete. Best iteration: %d", bst.best_iteration)
    return bst


def save_artifacts(
    model: xgb.Booster,
    risk_map: Dict[str, Dict[Any, float]],
    impute_dict: Dict[str, float],
    output_dir: str = model_path
) -> None:
    """
    Save the trained XGBoost model, the risk mapping, and the imputation dictionary
    under /opt/ml/model so that SageMaker will package them into model.tar.gz.
    """
    os.makedirs(output_dir, exist_ok=True)

    # (1) Save XGBoost model
    model_file = os.path.join(output_dir, "xgboost-model.bst")
    model.save_model(model_file)
    logger.info("Saved XGBoost model to %s", model_file)

    # (2) Save risk_map as pickle
    risk_file = os.path.join(output_dir, "risk_mapping.pkl")
    with open(risk_file, "wb") as f:
        pkl.dump(risk_map, f)
    logger.info("Saved risk mapping to %s", risk_file)

    # (3) Save impute_dict as pickle
    impute_file = os.path.join(output_dir, "impute_dict.pkl")
    with open(impute_file, "wb") as f:
        pkl.dump(impute_dict, f)
    logger.info("Saved imputation dictionary to %s", impute_file)


def main():
    """
    Main entrypoint for training. Expects:
      - Hyperparameters JSON at /opt/ml/input/config/hyperparameters.json
      - Preprocessed CSVs under /opt/ml/input/data/train/, /val/, /test/
    """
    try:
        # 1) Load hyperparameters as plain dict
        hparams = load_hyperparameters(hparam_path)

        # 2) Load data
        train_csv = "train_processed_data.csv"
        val_csv   = "val_processed_data.csv"
        test_csv  = "test_processed_data.csv"

        logger.info("Loading training data from %s/%s", train_path, train_csv)
        train_df = load_csv_file(train_path, train_csv)

        logger.info("Loading validation data from %s/%s", val_path, val_csv)
        val_df = load_csv_file(val_path, val_csv)

        logger.info("Loading test data from %s/%s", test_path, test_csv)
        test_df = load_csv_file(test_path, test_csv)

        # 3) Extract label and feature columns
        label_col   = hparams.get("label_name", "label")
        cat_fields  = hparams.get("cat_field_list", [])
        num_fields  = hparams.get("tab_field_list", [])
        feature_cols = num_fields + cat_fields

        # 4) Build and apply risk mapping
        logger.info("Building risk mapping for categorical fields: %s", cat_fields)
        risk_map = build_risk_mapping(train_df, cat_fields, label_col)

        logger.info("Applying risk mapping to train/val/test")
        train_df = apply_risk_mapping(train_df, cat_fields, risk_map)
        val_df   = apply_risk_mapping(val_df,   cat_fields, risk_map)
        test_df  = apply_risk_mapping(test_df,  cat_fields, risk_map)

        # 5) Compute and apply numeric imputation
        logger.info("Computing numeric imputation dictionary for fields: %s", num_fields)
        impute_dict = compute_imputation_dict(train_df, num_fields)

        logger.info("Applying numeric imputation to train/val/test")
        train_df = apply_imputation(train_df, num_fields, impute_dict)
        val_df   = apply_imputation(val_df,   num_fields, impute_dict)
        test_df  = apply_imputation(test_df,  num_fields, impute_dict)

        # 6) Train XGBoost model
        bst = train_xgboost(
            raw_hparams=hparams,
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            label_col=label_col
        )

        # 7) Save model + artifacts
        save_artifacts(
            model=bst,
            risk_map=risk_map,
            impute_dict=impute_dict,
            output_dir=model_path
        )

        logger.info("All artifacts saved. Exiting successfully.")
        sys.exit(0)

    except Exception as e:
        logger.error("Exception during XGBoost training: %s", str(e))
        traceback.print_exc()
        # Write a failure file to signal SageMaker failure
        failure_path = os.path.join(model_path, "failure")
        try:
            with open(failure_path, "w") as f:
                f.write(f"Exception during training: {e}\n{traceback.format_exc()}")
            logger.info("Wrote failure log to %s", failure_path)
        except Exception:
            logger.error("Failed to write failure log.")
        sys.exit(255)


if __name__ == "__main__":
    main()
