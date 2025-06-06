#!/usr/bin/env python3
import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import pandas as pd
import numpy as np
import pickle as pkl
import xgboost as xgb

from pydantic import BaseModel, Field, model_validator

# -------------------------------------------------------------------------
# Import the RiskTableMappingProcessor from its new path
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
logger.addHandler(handler)

# -------------------------------------------------------------------------
# Pydantic V2 model: flatten everything (formerly in ModelHyperparameters)
# plus XGBoost-specific fields and risk-smoothing parameters.
# -------------------------------------------------------------------------
class XGBoostConfig(BaseModel):
    """
    Pydantic V2 model for all hyperparameters expected from hyperparameters.json,
    including:
      - full_field_list, cat_field_list, tab_field_list, etc.
      - XGBoost-specific parameters
      - Risk‐table smoothing parameters
    """

    # ----- From ModelHyperparameters (flattened) -----
    full_field_list: List[str] = Field(
        ..., description="All field names (unused except for completeness)."
    )
    cat_field_list: List[str] = Field(
        ..., description="List of categorical column names."
    )
    tab_field_list: List[str] = Field(
        ..., description="List of numeric column names."
    )
    categorical_features_to_encode: List[str] = Field(
        default_factory=list,
        description="List of categorical fields that require explicit encoding.",
    )
    id_name: str = Field(..., description="Name of the ID column.")
    label_name: str = Field(..., description="Name of the label column.")

    is_binary: bool = Field(default=True, description="Binary‐classification flag.")
    num_classes: int = Field(default=2, description="Number of classes.")
    multiclass_categories: List[Union[int, str]] = Field(
        default_factory=lambda: [0, 1],
        description="List of category labels (e.g. [0,1] for binary).",
    )
    class_weights: List[float] = Field(
        default_factory=lambda: [1.0, 1.0],
        description="List of class weights, length == num_classes.",
    )
    device: int = Field(default=-1, description="Device index (-1 for CPU).")

    model_class: str = Field(default="xgboost", description="Model identifier.")

    header: int = Field(default=0, description="Header row index for CSV files.")
    input_tab_dim: int = Field(default=0, description="Should equal len(tab_field_list).")

    lr: float = Field(default=3e-5, description="(Unused by XGB) Learning rate placeholder.")
    batch_size: int = Field(default=2, ge=1, le=256, description="(Unused by XGB) Batch size.")
    max_epochs: int = Field(default=3, ge=1, le=10, description="(Unused by XGB) Max epochs.")
    metric_choices: List[str] = Field(
        default_factory=lambda: ["f1_score", "auroc"], description="Metrics list."
    )
    optimizer: str = Field(default="SGD", description="(Unused by XGB) Optimizer name.")

    # ----- XGBoost-specific parameters -----
    booster: str = Field(default="gbtree", description="Which booster to use.")
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Step size shrinkage (learning_rate).")
    gamma: float = Field(default=0.0, ge=0.0, description="Min loss reduction to split leaf.")
    max_depth: int = Field(default=6, ge=0, description="Max tree depth (0 = unlimited).")
    min_child_weight: float = Field(default=1.0, ge=0.0, description="Min sum Hessian in a child.")
    max_delta_step: float = Field(default=0.0, description="Max delta step for weight estimation.")
    subsample: float = Field(default=1.0, gt=0.0, le=1.0, description="Subsample ratio of rows.")
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0, description="Subsample ratio of columns per tree.")
    colsample_bylevel: float = Field(default=1.0, gt=0.0, le=1.0, description="Cols per level.")
    colsample_bynode: float = Field(default=1.0, gt=0.0, le=1.0, description="Cols per split.")
    lambda_xgb: float = Field(default=1.0, ge=0.0, description="L2 regularization (reg_lambda).")
    alpha_xgb: float = Field(default=0.0, ge=0.0, description="L1 regularization (reg_alpha).")
    tree_method: str = Field(default="auto", description="Tree construction algorithm.")
    sketch_eps: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Sketch epsilon for 'approx'.")
    scale_pos_weight: float = Field(default=1.0, description="Balance positive/negative weights.")

    objective: str = Field(default="binary:logistic", description="Learning objective.")
    base_score: Optional[float] = Field(default=None, description="Initial prediction score.")
    eval_metric: Optional[Union[str, List[str]]] = Field(
        default=None, description="Evaluation metric(s)."
    )
    seed: Optional[int] = Field(default=None, description="Random seed.")

    # SageMaker XGBoost control
    num_round: int = Field(default=100, ge=1, description="Number of boosting rounds.")
    early_stopping_rounds: Optional[int] = Field(
        default=None, ge=1, description="Enable early stopping if provided."
    )

    # ----- Risk‐table smoothing parameters -----
    smooth_factor: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Smoothing factor for risk table."
    )
    count_threshold: int = Field(default=0, ge=0, description="Minimum count threshold.")

    # ---------------------------------------------------------------------
    # Pydantic V2 cross‐field checks (model_validator)
    # ---------------------------------------------------------------------
    @model_validator(mode="after")
    def _validate_all(cls, state: "XGBoostConfig") -> "XGBoostConfig":
        # 1) input_tab_dim must match len(tab_field_list)
        if state.input_tab_dim != len(state.tab_field_list):
            raise ValueError(
                f"input_tab_dim ({state.input_tab_dim}) != len(tab_field_list) ({len(state.tab_field_list)})"
            )

        # 2) For binary: num_classes == 2, multiclass_categories length == 2
        if state.is_binary:
            if state.num_classes != 2:
                raise ValueError("For binary classification, num_classes must be 2")
            if len(state.multiclass_categories) != 2:
                raise ValueError("For binary classification, multiclass_categories must have length 2")
        else:
            # multiclass: len(multiclass_categories) >= 2 and matches num_classes
            if len(state.multiclass_categories) < 2:
                raise ValueError("For multiclass, multiclass_categories must have at least 2 items")
            if state.num_classes != len(state.multiclass_categories):
                raise ValueError(
                    f"num_classes ({state.num_classes}) != len(multiclass_categories) ({len(state.multiclass_categories)})"
                )

        # 3) class_weights length must equal num_classes
        if len(state.class_weights) != state.num_classes:
            raise ValueError("Length of class_weights must equal num_classes")

        # 4) If early_stopping_rounds is set, eval_metric must be set
        if state.early_stopping_rounds is not None and not state.eval_metric:
            raise ValueError("'early_stopping_rounds' requires 'eval_metric' to be set")

        # 5) If objective starts with "multi:", num_classes >= 2
        if state.objective.startswith("multi:") and state.num_classes < 2:
            raise ValueError("For multiclass objective, num_classes must be >= 2")

        return state

    class Config:
        extra = "forbid"


# -------------------------------------------------------------------------
# Paths inside the SageMaker training container
# -------------------------------------------------------------------------
prefix          = "/opt/ml"
input_path      = os.path.join(prefix, "input", "data")
model_path      = os.path.join(prefix, "model")
hparam_path     = os.path.join(prefix, "input", "config", "hyperparameters.json")
checkpoint_path = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")

train_channel   = "train"
val_channel     = "val"
test_channel    = "test"

train_data_dir  = os.path.join(input_path, train_channel)
val_data_dir    = os.path.join(input_path, val_channel)
test_data_dir   = os.path.join(input_path, test_channel)


# -------------------------------------------------------------------------
# Utility: find the first data file in a folder (CSV/Parquet/JSON)
# -------------------------------------------------------------------------
def find_first_data_file(data_dir: str, exts: List[str] = [".csv", ".parquet", ".json"]) -> str:
    for fname in sorted(os.listdir(data_dir)):
        low = fname.lower()
        if any(low.endswith(e) for e in exts):
            return fname
    raise FileNotFoundError(f"No supported data file in {data_dir}")


# -------------------------------------------------------------------------
# Utility: read any supported file into a DataFrame
# -------------------------------------------------------------------------
def _read_any_data(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()

    # CSV / TSV / GZ
    if suffix in [".csv", ".tsv", ".gz"]:
        if suffix == ".gz":
            inner = p.stem.lower().split(".")[-1]
            delim = "," if inner == "csv" else "\t"
            # Sniff delimiter from first two lines
            import gzip
            with gzip.open(str(p), "rt") as f_in:
                sample = f_in.readline() + f_in.readline()
            import csv as _csv
            try:
                sep = _csv.Sniffer().sniff(sample).delimiter
            except Exception:
                sep = delim
            return pd.read_csv(str(p), sep=sep, compression="infer", dtype=object)

        else:
            delim = "," if suffix == ".csv" else "\t"
            with p.open("rt") as f_in:
                sample = f_in.readline() + f_in.readline()
            import csv as _csv
            try:
                sep = _csv.Sniffer().sniff(sample).delimiter
            except Exception:
                sep = delim
            return pd.read_csv(str(p), sep=sep, dtype=object)

    # JSON / JSON.GZ
    if suffix in [".json", ".json.gz"]:
        if suffix == ".json.gz":
            import gzip
            with gzip.open(str(p), "rt") as f_in:
                first_line = f_in.readline()
                try:
                    _ = json.loads(first_line)
                    return pd.read_json(str(p), lines=True, compression="infer", dtype=object)
                except Exception:
                    f_in.seek(0)
                    payload = json.load(f_in)
        else:
            with p.open("rt") as f_in:
                payload = json.load(f_in)
        if isinstance(payload, dict):
            return pd.json_normalize([payload])
        return pd.json_normalize(payload)

    # Parquet / Snappy
    if suffix in [".parquet", ".snappy.parquet"]:
        return pd.read_parquet(str(p))

    raise ValueError(f"Unsupported file extension: {suffix}")


# -------------------------------------------------------------------------
# Main training routine
# -------------------------------------------------------------------------
def main():
    # 1) Load hyperparameters.json
    try:
        with open(hparam_path, "r") as f:
            raw_hparams = json.load(f)
        logger.info("Loaded hyperparameters.json:")
        for k, v in raw_hparams.items():
            logger.info(f"  {k}: {v}")
        config = XGBoostConfig(**raw_hparams)
    except (FileNotFoundError, ValidationError) as err:
        logger.error(f"Failed to load/validate hyperparameters: {err}")
        sys.exit(1)

    # 2) Read train/val/test DataFrames
    train_file = find_first_data_file(train_data_dir)
    val_file   = find_first_data_file(val_data_dir)
    test_file  = find_first_data_file(test_data_dir)

    train_df = _read_any_data(os.path.join(train_data_dir, train_file))
    val_df   = _read_any_data(os.path.join(val_data_dir, val_file))
    test_df  = _read_any_data(os.path.join(test_data_dir, test_file))

    logger.info(f"Shapes → train: {train_df.shape}, val: {val_df.shape}, test: {test_df.shape}")

    # 3) Coerce label to integer 0..k-1
    lbl = config.label_name
    for df in (train_df, val_df, test_df):
        if lbl not in df.columns:
            raise KeyError(f"Label '{lbl}' not found in columns")
        if (
            not pd.api.types.is_integer_dtype(df[lbl])
            and not pd.api.types.is_float_dtype(df[lbl])
        ):
            uniques = sorted(df[lbl].dropna().unique())
            mapping = {val: idx for idx, val in enumerate(uniques)}
            df[lbl] = df[lbl].map(lambda x: mapping.get(x, np.nan))
        df[lbl] = pd.to_numeric(df[lbl], errors="coerce").astype("Int64")
    logger.info("Label column cast to integer dtype.")

    # 4) Build RiskTableMappingProcessor for each categorical var
    risk_processors: Dict[str, RiskTableMappingProcessor] = {}
    for var in config.cat_field_list:
        logger.info(f"Fitting risk table for '{var}' …")
        proc = RiskTableMappingProcessor(
            column_name     = var,
            label_name      = config.label_name,
            smooth_factor   = config.smooth_factor,
            count_threshold = config.count_threshold,
        )
        tmp = train_df[[var, config.label_name]].dropna(subset=[var, config.label_name])
        proc.fit(tmp)
        risk_processors[var] = proc

        # Save each var's risk table as both pickle and JSON
        outdir = Path(model_path) / "risk_tables"
        outdir.mkdir(parents=True, exist_ok=True)

        pkl_path  = outdir / f"risk_table_{var}.pkl"
        with open(pkl_path, "wb") as pf:
            pkl.dump(proc.get_risk_tables(), pf)

        json_path = outdir / f"risk_table_{var}.json"
        with open(json_path, "w") as jf:
            json.dump(proc.get_risk_tables(), jf, indent=2)

        logger.info(f"Saved risk tables for '{var}' → {pkl_path}")

    # 5) Transform each split through the fitted processors
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"Mapping categorical features in {split_name} set …")
        for var, proc in risk_processors.items():
            df[var] = proc.transform(df[var])

    # 6) Compute median-based imputation on numeric fields (train), then apply to all splits
    imp_dict: Dict[str, float] = {}
    for var in config.tab_field_list:
        if var not in train_df.columns:
            raise KeyError(f"Numeric column '{var}' not present in training data")
        median_val = float(train_df[var].median(skipna=True))
        imp_dict[var] = median_val

    for df in (train_df, val_df, test_df):
        for var, mval in imp_dict.items():
            df[var] = df[var].fillna(mval)

    # Save the imputation dictionary
    impute_outdir = Path(model_path) / "imputation"
    impute_outdir.mkdir(parents=True, exist_ok=True)
    impute_pkl = impute_outdir / "impute_dict.pkl"
    with open(impute_pkl, "wb") as f_imp:
        pkl.dump(imp_dict, f_imp)
    logger.info(f"Saved imputation dictionary → {impute_pkl}")

    # 7) Build DMatrixes for XGBoost
    y_train = train_df[ config.label_name ].astype(int).values
    X_train = train_df[ config.tab_field_list + config.cat_field_list ].astype(float).values

    y_val   = val_df[ config.label_name ].astype(int).values
    X_val   = val_df[ config.tab_field_list + config.cat_field_list ].astype(float).values

    y_test  = test_df[ config.label_name ].astype(int).values
    X_test  = test_df[ config.tab_field_list + config.cat_field_list ].astype(float).values

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    # 8) Configure XGBoost parameters
    xgb_params: Dict[str, Any] = {
        "objective":          config.objective,
        "eta":                config.eta,
        "gamma":              config.gamma,
        "max_depth":          config.max_depth,
        "lambda":             config.lambda_xgb,
        "alpha":              config.alpha_xgb,
        "subsample":          config.subsample,
        "colsample_bytree":   config.colsample_bytree,
        "scale_pos_weight":   config.scale_pos_weight,
        "tree_method":        config.tree_method,
    }
    logger.info(f"Starting XGBoost training with params: {xgb_params}, num_round={config.num_round}")

    evals = [(dtrain, "train"), (dval, "validation")]
    bst = xgb.train(
        params                = xgb_params,
        dtrain                = dtrain,
        num_boost_round       = config.num_round,
        evals                 = evals,
        early_stopping_rounds = config.early_stopping_rounds,
    )

    # 9) Save the final XGBoost model
    model_file = Path(model_path) / "xgboost_model.bst"
    bst.save_model(str(model_file))
    logger.info(f"Saved XGBoost model → {model_file}")

    # 10) (Optional) Dump feature‐importance to JSON
    fmap_json = Path(model_path) / "feature_importance.json"
    with open(fmap_json, "w") as f_fmap:
        json.dump(bst.get_fscore(), f_fmap, indent=2)
    logger.info(f"Saved feature‐importance → {fmap_json}")

    logger.info("Training script finished successfully.")


# -------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.error(f"Exception during training:\n{traceback.format_exc()}")
        sys.exit(1)

