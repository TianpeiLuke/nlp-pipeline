import os
import json
import argparse
import pandas as pd
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, f1_score
import xgboost as xgb

from processing.risk_table_processor import RiskTableMappingProcessor
from processing.numerical_imputation_processor import NumericalVariableImputationProcessor

def load_model_artifacts(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost_model.bst"))
    with open(os.path.join(model_dir, "risk_table_map.pkl"), "rb") as f:
        risk_tables = pkl.load(f)
    with open(os.path.join(model_dir, "impute_dict.pkl"), "rb") as f:
        impute_dict = pkl.load(f)
    with open(os.path.join(model_dir, "feature_columns.txt"), "r") as f:
        feature_columns = [line.strip().split(",")[1] for line in f if not line.startswith("#")]
    with open(os.path.join(model_dir, "hyperparameters.json"), "r") as f:
        hyperparams = json.load(f)
    return model, risk_tables, impute_dict, feature_columns, hyperparams

def preprocess_eval_data(df, feature_columns, risk_tables, impute_dict):
    # Apply risk table mapping
    for feature, risk_table in risk_tables.items():
        if feature in df.columns:
            proc = RiskTableMappingProcessor(
                column_name=feature,
                label_name="label",
                risk_tables=risk_table
            )
            df[feature] = proc.transform(df[feature])
    # Apply numerical imputation
    imputer = NumericalVariableImputationProcessor(imputation_dict=impute_dict)
    df = imputer.transform(df)
    # Ensure all features are numeric
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return df

def compute_metrics_binary(y_true, y_prob):
    # y_prob: N x 2, use class-1 prob
    y_score = y_prob[:, 1]
    metrics = {
        "auc_roc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
        "f1_score": f1_score(y_true, y_score > 0.5)
    }
    return metrics

def compute_metrics_multiclass(y_true, y_prob, n_classes):
    # y_prob: N x C
    metrics = {}
    # One-vs-rest for each class
    for i in range(n_classes):
        y_true_bin = (y_true == i).astype(int)
        y_score = y_prob[:, i]
        metrics[f"auc_roc_class_{i}"] = roc_auc_score(y_true_bin, y_score)
        metrics[f"average_precision_class_{i}"] = average_precision_score(y_true_bin, y_score)
        metrics[f"f1_score_class_{i}"] = f1_score(y_true_bin, y_score > 0.5)
    # Micro/macro average
    metrics["auc_roc_micro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="micro")
    metrics["auc_roc_macro"] = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    metrics["average_precision_micro"] = average_precision_score(y_true, y_prob, average="micro")
    metrics["average_precision_macro"] = average_precision_score(y_true, y_prob, average="macro")
    y_pred = np.argmax(y_prob, axis=1)
    metrics["f1_score_micro"] = f1_score(y_true, y_pred, average="micro")
    metrics["f1_score_macro"] = f1_score(y_true, y_pred, average="macro")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_type", type=str, required=True)
    args = parser.parse_args()

    # Environment variables
    ID_FIELD = os.environ.get("ID_FIELD", "id")
    LABEL_FIELD = os.environ.get("LABEL_FIELD", "label")

    # Input/output paths
    model_dir = "/opt/ml/processing/input/model"
    eval_data_dir = "/opt/ml/processing/input/eval_data"
    output_eval_dir = "/opt/ml/processing/output/eval"
    output_metrics_dir = "/opt/ml/processing/output/metrics"
    os.makedirs(output_eval_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)

    # Load model artifacts
    model, risk_tables, impute_dict, feature_columns, hyperparams = load_model_artifacts(model_dir)

    # Find eval data file (first .csv or .parquet)
    eval_files = sorted([f for f in Path(eval_data_dir).glob("**/*") if f.suffix in [".csv", ".parquet"]])
    if not eval_files:
        raise RuntimeError("No eval data file found in eval_data input.")
    eval_file = eval_files[0]
    df = pd.read_parquet(eval_file) if eval_file.suffix == ".parquet" else pd.read_csv(eval_file)

    # Preprocess eval data
    df = preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)

    # Get id, label, features
    id_col = ID_FIELD if ID_FIELD in df.columns else df.columns[0]
    label_col = LABEL_FIELD if LABEL_FIELD in df.columns else df.columns[1]
    y_true = df[label_col].values
    ids = df[id_col].values
    X = df[feature_columns].values

    # Predict
    dmatrix = xgb.DMatrix(X)
    y_prob = model.predict(dmatrix)
    if len(y_prob.shape) == 1:
        # Binary: convert to N x 2
        y_prob = np.column_stack([1 - y_prob, y_prob])

    # Compute metrics
    if hyperparams.get("is_binary", True) or y_prob.shape[1] == 2:
        metrics = compute_metrics_binary(y_true, y_prob)
    else:
        n_classes = y_prob.shape[1]
        metrics = compute_metrics_multiclass(y_true, y_prob, n_classes)

    # Output predictions with id, label, probs
    prob_cols = [f"prob_class_{i}" for i in range(y_prob.shape[1])]
    out_df = pd.DataFrame({
        id_col: ids,
        label_col: y_true
    })
    for i, col in enumerate(prob_cols):
        out_df[col] = y_prob[:, i]
    out_df.to_csv(os.path.join(output_eval_dir, "eval_predictions.csv"), index=False)

    # Output metrics as JSON
    with open(os.path.join(output_metrics_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
