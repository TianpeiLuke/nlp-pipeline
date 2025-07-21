#!/usr/bin/env python
"""Model Calibration Script for SageMaker Processing.

This script calibrates model prediction scores to accurate probabilities,
which is essential for risk-based decision-making and threshold setting.
It supports multiple calibration methods including GAM, Isotonic Regression,
and Platt Scaling, with options for monotonicity constraints.
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

# Import pygam for GAM implementation if available
try:
    from pygam import LogisticGAM, s
    HAS_PYGAM = True
except ImportError:
    HAS_PYGAM = False

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths from contract
INPUT_DATA_PATH = "/opt/ml/processing/input/eval_data"
INPUT_MODEL_PATH = "/opt/ml/processing/input/model"
INPUT_CODE_PATH = "/opt/ml/processing/input/code"
OUTPUT_CALIBRATION_PATH = "/opt/ml/processing/output/calibration"
OUTPUT_METRICS_PATH = "/opt/ml/processing/output/metrics"
OUTPUT_CALIBRATED_DATA_PATH = "/opt/ml/processing/output/calibrated_data"

# Get environment variables
CALIBRATION_METHOD = os.environ.get("CALIBRATION_METHOD", "gam").lower()
LABEL_FIELD = os.environ.get("LABEL_FIELD", "label")
SCORE_FIELD = os.environ.get("SCORE_FIELD", "prob_class_1")
MONOTONIC_CONSTRAINT = os.environ.get("MONOTONIC_CONSTRAINT", "True").lower() == "true"
GAM_SPLINES = int(os.environ.get("GAM_SPLINES", "10"))
ERROR_THRESHOLD = float(os.environ.get("ERROR_THRESHOLD", "0.05"))

def create_directories():
    """Create output directories if they don't exist."""
    os.makedirs(OUTPUT_CALIBRATION_PATH, exist_ok=True)
    os.makedirs(OUTPUT_METRICS_PATH, exist_ok=True)
    os.makedirs(OUTPUT_CALIBRATED_DATA_PATH, exist_ok=True)

def find_first_data_file(data_dir: str) -> str:
    """Find the first supported data file in directory.
    
    Args:
        data_dir: Directory to search for data files
        
    Returns:
        str: Path to the first supported data file found
        
    Raises:
        FileNotFoundError: If no supported data file is found
    """
    if not os.path.isdir(data_dir):
        return None
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith((".csv", ".parquet", ".json")):
            return os.path.join(data_dir, fname)
    return None

def load_data():
    """Load evaluation data with predictions.
    
    Returns:
        pd.DataFrame: Loaded evaluation data
        
    Raises:
        FileNotFoundError: If no data file is found
        ValueError: If required columns are missing
    """
    data_file = find_first_data_file(INPUT_DATA_PATH)
    if not data_file:
        raise FileNotFoundError(f"No data file found in {INPUT_DATA_PATH}")
    
    logger.info(f"Loading data from {data_file}")
    if data_file.endswith('.parquet'):
        df = pd.read_parquet(data_file)
    elif data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    else:
        raise ValueError(f"Unsupported file format: {data_file}")
    
    # Validate required columns
    if LABEL_FIELD not in df.columns:
        raise ValueError(f"Label field '{LABEL_FIELD}' not found in data")
    if SCORE_FIELD not in df.columns:
        raise ValueError(f"Score field '{SCORE_FIELD}' not found in data")
    
    logger.info(f"Loaded data with shape {df.shape}")
    return df

def load_model_info():
    """Load model information if available.
    
    Returns:
        Dict: Model hyperparameters and metadata
    """
    hyperparams_path = os.path.join(INPUT_MODEL_PATH, "hyperparameters.json")
    model_info = {}
    if os.path.exists(hyperparams_path):
        with open(hyperparams_path, "r") as f:
            model_info = json.load(f)
            logger.info(f"Loaded model hyperparameters from {hyperparams_path}")
    return model_info

def train_gam_calibration(scores: np.ndarray, labels: np.ndarray):
    """Train a GAM calibration model with optional monotonicity constraints.
    
    Args:
        scores: Raw prediction scores to calibrate
        labels: Ground truth binary labels (0/1)
        
    Returns:
        LogisticGAM: Trained GAM calibration model
        
    Raises:
        ImportError: If pygam is not installed
    """
    if not HAS_PYGAM:
        raise ImportError("pygam package is required for GAM calibration but not installed")
    
    scores = scores.reshape(-1, 1)  # Reshape for GAM
    
    # Configure GAM with monotonic constraint if specified
    if MONOTONIC_CONSTRAINT:
        gam = LogisticGAM(s(0, n_splines=GAM_SPLINES, constraints='monotonic_inc'))
        logger.info(f"Training GAM with monotonic constraint, {GAM_SPLINES} splines")
    else:
        gam = LogisticGAM(s(0, n_splines=GAM_SPLINES))
        logger.info(f"Training GAM without monotonic constraint, {GAM_SPLINES} splines")
    
    gam.fit(scores, labels)
    logger.info(f"GAM training complete, deviance: {gam.statistics_['deviance']}")
    return gam

def train_isotonic_calibration(scores: np.ndarray, labels: np.ndarray):
    """Train an isotonic regression calibration model.
    
    Args:
        scores: Raw prediction scores to calibrate
        labels: Ground truth binary labels (0/1)
        
    Returns:
        IsotonicRegression: Trained isotonic regression model
    """
    logger.info("Training isotonic regression calibration model")
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(scores, labels)
    logger.info("Isotonic regression training complete")
    return ir

def train_platt_scaling(scores: np.ndarray, labels: np.ndarray):
    """Train a Platt scaling (logistic regression) calibration model.
    
    Args:
        scores: Raw prediction scores to calibrate
        labels: Ground truth binary labels (0/1)
        
    Returns:
        LogisticRegression: Trained logistic regression model
    """
    logger.info("Training Platt scaling (logistic regression) calibration model")
    scores = scores.reshape(-1, 1)  # Reshape for LogisticRegression
    lr = LogisticRegression(C=1e5)  # High C for minimal regularization
    lr.fit(scores, labels)
    logger.info("Platt scaling training complete")
    return lr

def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Compute comprehensive calibration metrics including ECE, MCE, and reliability diagram.
    
    This function calculates:
    - Expected Calibration Error (ECE): weighted average of absolute calibration errors
    - Maximum Calibration Error (MCE): maximum calibration error across all bins
    - Reliability diagram data: points for plotting calibration curve
    - Bin statistics: detailed information about each probability bin
    - Brier score: quadratic scoring rule for probabilistic predictions
    - Preservation of discrimination: comparison of AUC before/after calibration
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dict: Dictionary containing calibration metrics
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Get bin assignments and counts
    bin_indices = np.minimum(n_bins - 1, (y_prob * n_bins).astype(int))
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_counts = bin_counts.astype(np.float64)
    
    # Compute mean predicted probability in each bin
    bin_probs = np.bincount(bin_indices, weights=y_prob, minlength=n_bins) / np.maximum(bin_counts, 1)
    
    # Compute mean true label in each bin
    bin_true = np.bincount(bin_indices, weights=y_true, minlength=n_bins) / np.maximum(bin_counts, 1)
    
    # Compute calibration errors per bin
    abs_errors = np.abs(bin_probs - bin_true)
    
    # Expected Calibration Error (weighted average of absolute errors)
    ece = np.sum(bin_counts / len(y_true) * abs_errors)
    
    # Maximum Calibration Error
    mce = np.max(abs_errors)
    
    # Brier score - quadratic scoring rule for probabilistic predictions
    brier = brier_score_loss(y_true, y_prob)
    
    # Discrimination preservation (AUC)
    auc = roc_auc_score(y_true, y_prob)
    
    # Create detailed bin information
    bins = []
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bins.append({
                "bin_index": i,
                "bin_start": i/n_bins,
                "bin_end": (i+1)/n_bins,
                "sample_count": int(bin_counts[i]),
                "mean_predicted": float(bin_probs[i]),
                "mean_true": float(bin_true[i]),
                "calibration_error": float(abs_errors[i]),
            })
    
    # Compile all metrics
    metrics = {
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
        "brier_score": float(brier),
        "auc_roc": float(auc),
        "reliability_diagram": {
            "true_probs": prob_true.tolist(),
            "pred_probs": prob_pred.tolist()
        },
        "bin_statistics": {
            "bin_counts": bin_counts.tolist(),
            "bin_predicted_probs": bin_probs.tolist(),
            "bin_true_probs": bin_true.tolist(),
            "calibration_errors": abs_errors.tolist(),
            "detailed_bins": bins
        },
        "num_samples": len(y_true),
        "num_bins": n_bins
    }
    
    return metrics

def plot_reliability_diagram(y_true: np.ndarray, y_prob_uncalibrated: np.ndarray, 
                           y_prob_calibrated: np.ndarray, n_bins: int = 10) -> str:
    """Create reliability diagram comparing uncalibrated and calibrated probabilities.
    
    Args:
        y_true: Ground truth binary labels (0/1)
        y_prob_uncalibrated: Uncalibrated prediction probabilities
        y_prob_calibrated: Calibrated prediction probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        str: Path to the saved figure
    """
    fig = plt.figure(figsize=(10, 8))
    
    # Plot calibration curves
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    # Plot uncalibrated curve
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_true, y_prob_uncalibrated, n_bins=n_bins
    )
    ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", label="Uncalibrated")
    
    # Plot calibrated curve
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_true, y_prob_calibrated, n_bins=n_bins
    )
    ax1.plot(prob_pred_cal, prob_true_cal, "s-", label="Calibrated")
    
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_title("Calibration Curve (Reliability Diagram)")
    ax1.legend(loc="lower right")
    
    # Plot histogram of predictions
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    
    ax2.hist(y_prob_uncalibrated, range=(0, 1), bins=n_bins, 
             label="Uncalibrated", alpha=0.5, edgecolor="k")
    ax2.hist(y_prob_calibrated, range=(0, 1), bins=n_bins, 
             label="Calibrated", alpha=0.5, edgecolor="r")
    ax2.set_xlabel("Mean predicted probability")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center")
    
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(OUTPUT_METRICS_PATH, "reliability_diagram.png")
    plt.savefig(figure_path)
    plt.close(fig)
    
    return figure_path

def main():
    """Main entry point for the calibration script."""
    try:
        logger.info("Starting model calibration")
        
        # Create output directories
        create_directories()
        
        # Load data and model info
        df = load_data()
        model_info = load_model_info()
        
        # Extract features and target
        y_true = df[LABEL_FIELD].values
        y_prob_uncalibrated = df[SCORE_FIELD].values
        
        # Select and train calibration model
        if CALIBRATION_METHOD == "gam":
            if not HAS_PYGAM:
                logger.warning("pygam not installed, falling back to Platt scaling")
                calibrator = train_platt_scaling(y_prob_uncalibrated, y_true)
            else:
                calibrator = train_gam_calibration(y_prob_uncalibrated, y_true)
        elif CALIBRATION_METHOD == "isotonic":
            calibrator = train_isotonic_calibration(y_prob_uncalibrated, y_true)
        elif CALIBRATION_METHOD == "platt":
            calibrator = train_platt_scaling(y_prob_uncalibrated, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {CALIBRATION_METHOD}")
        
        # Apply calibration to get calibrated probabilities
        if isinstance(calibrator, IsotonicRegression):
            y_prob_calibrated = calibrator.transform(y_prob_uncalibrated)
        elif isinstance(calibrator, LogisticRegression):
            y_prob_calibrated = calibrator.predict_proba(y_prob_uncalibrated.reshape(-1, 1))[:, 1]
        else:  # GAM
            y_prob_calibrated = calibrator.predict_proba(y_prob_uncalibrated.reshape(-1, 1))
        
        # Compute calibration metrics for before and after
        uncalibrated_metrics = compute_calibration_metrics(y_true, y_prob_uncalibrated)
        calibrated_metrics = compute_calibration_metrics(y_true, y_prob_calibrated)
        
        # Create visualization
        plot_path = plot_reliability_diagram(y_true, y_prob_uncalibrated, y_prob_calibrated)
        
        # Create comprehensive metrics report
        metrics_report = {
            "calibration_method": CALIBRATION_METHOD,
            "uncalibrated": uncalibrated_metrics,
            "calibrated": calibrated_metrics,
            "improvement": {
                "ece_reduction": uncalibrated_metrics["expected_calibration_error"] - calibrated_metrics["expected_calibration_error"],
                "mce_reduction": uncalibrated_metrics["maximum_calibration_error"] - calibrated_metrics["maximum_calibration_error"],
                "brier_reduction": uncalibrated_metrics["brier_score"] - calibrated_metrics["brier_score"],
                "auc_change": calibrated_metrics["auc_roc"] - uncalibrated_metrics["auc_roc"],
            },
            "visualization_paths": {
                "reliability_diagram": plot_path
            },
            "config": {
                "label_field": LABEL_FIELD,
                "score_field": SCORE_FIELD,
                "monotonic_constraint": MONOTONIC_CONSTRAINT,
                "gam_splines": GAM_SPLINES,
                "error_threshold": ERROR_THRESHOLD
            }
        }
        
        # Save metrics report
        metrics_path = os.path.join(OUTPUT_METRICS_PATH, "calibration_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_report, f, indent=2)
        
        # Save calibrator model
        calibrator_path = os.path.join(OUTPUT_CALIBRATION_PATH, "calibration_model.joblib")
        joblib.dump(calibrator, calibrator_path)
        
        # Add calibrated scores to dataframe and save
        df["calibrated_" + SCORE_FIELD] = y_prob_calibrated
        output_path = os.path.join(OUTPUT_CALIBRATED_DATA_PATH, "calibrated_data.parquet")
        df.to_parquet(output_path, index=False)
        
        # Write summary
        summary = {
            "status": "success",
            "calibration_method": CALIBRATION_METHOD,
            "uncalibrated_ece": uncalibrated_metrics["expected_calibration_error"],
            "calibrated_ece": calibrated_metrics["expected_calibration_error"],
            "improvement_percentage": (1 - calibrated_metrics["expected_calibration_error"] / max(uncalibrated_metrics["expected_calibration_error"], 1e-10)) * 100,
            "output_files": {
                "metrics": metrics_path,
                "calibrator": calibrator_path,
                "calibrated_data": output_path
            }
        }
        
        summary_path = os.path.join(OUTPUT_CALIBRATION_PATH, "calibration_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Check if calibration improved by error threshold
        if summary["improvement_percentage"] < 0:
            logger.warning("Calibration did not improve expected calibration error!")
        elif summary["improvement_percentage"] < 5:
            logger.warning("Calibration only marginally improved expected calibration error")
            
        logger.info(f"Calibration complete. ECE reduced from {uncalibrated_metrics['expected_calibration_error']:.4f} to {calibrated_metrics['expected_calibration_error']:.4f}")
        logger.info(f"All outputs saved to: {OUTPUT_CALIBRATION_PATH}, {OUTPUT_METRICS_PATH}, and {OUTPUT_CALIBRATED_DATA_PATH}")
        
    except Exception as e:
        logger.error(f"Error in model calibration: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
