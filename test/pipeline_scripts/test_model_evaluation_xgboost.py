import unittest
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import pickle as pkl
import json
from pathlib import Path

# Import the main evaluation function and helpers
from src.pipeline_scripts import model_evaluation_xgboost

class TestModelEvaluationXGBoost(unittest.TestCase):
    """Unit tests for the model_evaluation_xgboost.py script."""

    def setUp(self):
        """Set up a temporary directory structure and dummy artifacts."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.temp_dir, "model")
        self.eval_data_dir = os.path.join(self.temp_dir, "eval_data")
        self.output_eval_dir = os.path.join(self.temp_dir, "output_eval")
        self.output_metrics_dir = os.path.join(self.temp_dir, "output_metrics")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.eval_data_dir, exist_ok=True)
        os.makedirs(self.output_eval_dir, exist_ok=True)
        os.makedirs(self.output_metrics_dir, exist_ok=True)

        # Create dummy model artifacts
        # Save a dummy xgboost model
        import xgboost as xgb
        X = np.random.rand(10, 2)
        y = np.random.randint(0, 2, 10)
        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train({'objective': 'binary:logistic'}, dtrain, num_boost_round=2)
        model.save_model(os.path.join(self.model_dir, "xgboost_model.bst"))

        # Save dummy risk table and impute dict
        with open(os.path.join(self.model_dir, "risk_table_map.pkl"), "wb") as f:
            pkl.dump({}, f)
        with open(os.path.join(self.model_dir, "impute_dict.pkl"), "wb") as f:
            pkl.dump({}, f)
        # Save feature columns
        with open(os.path.join(self.model_dir, "feature_columns.txt"), "w") as f:
            f.write("# Feature columns in exact order required for XGBoost model inference\n")
            f.write("0,feature1\n1,feature2\n")
        # Save hyperparameters
        with open(os.path.join(self.model_dir, "hyperparameters.json"), "w") as f:
            json.dump({"is_binary": True}, f)

        # Create dummy eval data
        df = pd.DataFrame({
            "feature1": np.random.rand(10),
            "feature2": np.random.rand(10),
            "id": range(10),
            "label": np.random.randint(0, 2, 10)
        })
        eval_file = os.path.join(self.eval_data_dir, "eval.csv")
        df.to_csv(eval_file, index=False)

    def tearDown(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_load_model_artifacts(self):
        """Test loading model artifacts."""
        model, risk_tables, impute_dict, feature_columns, hyperparams = model_evaluation_xgboost.load_model_artifacts(self.model_dir)
        self.assertTrue(hasattr(model, "predict"))
        self.assertEqual(feature_columns, ["feature1", "feature2"])
        self.assertIn("is_binary", hyperparams)

    def test_preprocess_eval_data(self):
        """Test preprocessing eval data (no-op for empty risk/impute dicts)."""
        df = pd.DataFrame({"feature1": [1.0], "feature2": [2.0]})
        feature_columns = ["feature1", "feature2"]
        risk_tables = {}
        impute_dict = {}
        processed = model_evaluation_xgboost.preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)
        self.assertEqual(list(processed.columns), feature_columns)

    def test_evaluate_model_and_outputs(self):
        """Test the full evaluation and output generation."""
        # Load model artifacts
        model, risk_tables, impute_dict, feature_columns, hyperparams = model_evaluation_xgboost.load_model_artifacts(self.model_dir)
        # Load eval data
        df = pd.read_csv(os.path.join(self.eval_data_dir, "eval.csv"))
        # Preprocess
        df_proc = model_evaluation_xgboost.preprocess_eval_data(df, feature_columns, risk_tables, impute_dict)
        id_col, label_col = model_evaluation_xgboost.get_id_label_columns(df_proc, "id", "label")
        # Evaluate
        model_evaluation_xgboost.evaluate_model(
            model, df_proc, feature_columns, id_col, label_col, hyperparams,
            self.output_eval_dir, self.output_metrics_dir
        )
        # Check outputs
        pred_file = os.path.join(self.output_eval_dir, "eval_predictions.csv")
        metrics_file = os.path.join(self.output_metrics_dir, "metrics.json")
        self.assertTrue(os.path.exists(pred_file))
        self.assertTrue(os.path.exists(metrics_file))
        preds = pd.read_csv(pred_file)
        self.assertIn("id", preds.columns)
        self.assertIn("label", preds.columns)
        with open(metrics_file) as f:
            metrics = json.load(f)
        self.assertIn("auc_roc", metrics)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
