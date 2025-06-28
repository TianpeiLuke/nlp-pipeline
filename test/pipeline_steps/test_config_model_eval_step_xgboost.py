import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the config class to be tested
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters


class TestXGBoostModelEvalConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, valid configuration for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'model_evaluation_xgboost.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy model evaluation script for testing\n')
            f.write('print("This is a dummy script")\n')
            
        # Create a proper XGBoostModelHyperparameters instance
        self.hyperparameters = XGBoostModelHyperparameters(
            id_name="id",
            label_name="label",
            # Other required fields with defaults will be used
        )
        
        # Create a minimal valid config
        self.config = XGBoostModelEvalConfig(
            processing_entry_point='model_evaluation_xgboost.py',
            processing_source_dir=self.temp_dir,
            processing_instance_count=1,
            processing_volume_size=30,
            pipeline_name='test-pipeline',
            job_type='validation',
            hyperparameters=self.hyperparameters,
            # Other fields will use defaults
        )

    def test_default_values(self):
        """Test that default values are set correctly."""
        # Check default values for fields with defaults
        self.assertEqual(self.config.processing_entry_point, 'model_evaluation_xgboost.py')
        self.assertEqual(self.config.xgboost_framework_version, '1.5-1')
        self.assertEqual(self.config.job_type, 'validation')
        
        # Check default input_names
        self.assertIn("model_input", self.config.input_names)
        self.assertIn("eval_data_input", self.config.input_names)
        
        # Check default output_names
        self.assertIn("eval_output", self.config.output_names)
        self.assertIn("metrics_output", self.config.output_names)
        
        # Check default eval_metric_choices
        self.assertIn("auc", self.config.eval_metric_choices)
        self.assertIn("average_precision", self.config.eval_metric_choices)
        self.assertIn("f1_score", self.config.eval_metric_choices)

    def test_validation_job_type(self):
        """Test that job_type validation works correctly."""
        # Valid job types
        valid_job_types = ["training", "calibration", "validation", "test"]
        
        for job_type in valid_job_types:
            config = XGBoostModelEvalConfig(
                processing_entry_point='model_evaluation_xgboost.py',
                processing_source_dir=self.temp_dir,
                processing_instance_count=1,
                processing_volume_size=30,
                pipeline_name='test-pipeline',
                job_type=job_type,
                hyperparameters=self.hyperparameters,
            )
            self.assertEqual(config.job_type, job_type)
        
        # Invalid job type
        with self.assertRaises(ValueError):
            XGBoostModelEvalConfig(
                processing_entry_point='model_evaluation_xgboost.py',
                processing_source_dir=self.temp_dir,
                processing_instance_count=1,
                processing_volume_size=30,
                pipeline_name='test-pipeline',
                job_type='invalid_job_type',
                hyperparameters=self.hyperparameters,
            )

    def test_validation_hyperparameters(self):
        """Test that hyperparameters validation works correctly."""
        # Invalid hyperparameters (not an XGBoostModelHyperparameters instance)
        with self.assertRaises(ValueError):
            XGBoostModelEvalConfig(
                processing_entry_point='model_evaluation_xgboost.py',
                processing_source_dir=self.temp_dir,
                processing_instance_count=1,
                processing_volume_size=30,
                pipeline_name='test-pipeline',
                job_type='validation',
                hyperparameters="not_a_hyperparameters_instance",
            )

    def test_validation_processing_entry_point(self):
        """Test that processing_entry_point validation works correctly."""
        # Missing processing_entry_point
        with self.assertRaises(ValueError):
            XGBoostModelEvalConfig(
                processing_entry_point=None,
                processing_source_dir=self.temp_dir,
                processing_instance_count=1,
                processing_volume_size=30,
                pipeline_name='test-pipeline',
                job_type='validation',
                hyperparameters=self.hyperparameters,
            )

    def test_set_default_names(self):
        """Test that default input and output names are set if not provided."""
        # Create config with None for input_names and output_names
        config = XGBoostModelEvalConfig(
            processing_entry_point='model_evaluation_xgboost.py',
            processing_source_dir=self.temp_dir,
            processing_instance_count=1,
            processing_volume_size=30,
            pipeline_name='test-pipeline',
            job_type='validation',
            hyperparameters=self.hyperparameters,
            input_names=None,
            output_names=None,
        )
        
        # Check that default input_names were set
        self.assertIsNotNone(config.input_names)
        self.assertIn("model_input", config.input_names)
        self.assertIn("eval_data_input", config.input_names)
        
        # Check that default output_names were set
        self.assertIsNotNone(config.output_names)
        self.assertIn("eval_output", config.output_names)
        self.assertIn("metrics_output", config.output_names)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_get_script_path(self):
        """Test that get_script_path returns the correct path."""
        # When processing_entry_point is set
        script_path = self.config.get_script_path()
        self.assertTrue(script_path.endswith('model_evaluation_xgboost.py'))
        
        # Create a different script file in the temporary directory
        different_script = 'different_script.py'
        different_script_path = os.path.join(self.temp_dir, different_script)
        with open(different_script_path, 'w') as f:
            f.write('# Another dummy script for testing\n')
            f.write('print("This is another dummy script")\n')
            
        # Test with a different processing_entry_point value
        original_value = self.config.processing_entry_point
        self.config.processing_entry_point = different_script
        self.assertTrue(self.config.get_script_path().endswith(different_script))
        
        # Restore original value
        self.config.processing_entry_point = original_value

    def test_custom_eval_metric_choices(self):
        """Test that custom eval_metric_choices can be set."""
        custom_metrics = ["accuracy", "precision", "recall"]
        
        config = XGBoostModelEvalConfig(
            processing_entry_point='model_evaluation_xgboost.py',
            processing_source_dir=self.temp_dir,
            processing_instance_count=1,
            processing_volume_size=30,
            pipeline_name='test-pipeline',
            job_type='validation',
            hyperparameters=self.hyperparameters,
            eval_metric_choices=custom_metrics,
        )
        
        self.assertEqual(config.eval_metric_choices, custom_metrics)
        self.assertIn("accuracy", config.eval_metric_choices)
        self.assertIn("precision", config.eval_metric_choices)
        self.assertIn("recall", config.eval_metric_choices)

    def test_custom_input_output_names(self):
        """Test that custom input and output names can be set."""
        custom_input_names = {
            "model_input": "Custom model input",
            "eval_data_input": "Custom eval data input",
            "extra_input": "Extra input channel"
        }
        
        custom_output_names = {
            "eval_output": "Custom eval output",
            "metrics_output": "Custom metrics output",
            "extra_output": "Extra output channel"
        }
        
        config = XGBoostModelEvalConfig(
            processing_entry_point='model_evaluation_xgboost.py',
            processing_source_dir=self.temp_dir,
            processing_instance_count=1,
            processing_volume_size=30,
            pipeline_name='test-pipeline',
            job_type='validation',
            hyperparameters=self.hyperparameters,
            input_names=custom_input_names,
            output_names=custom_output_names,
        )
        
        # Check input_names
        self.assertEqual(config.input_names, custom_input_names)
        self.assertEqual(config.input_names["model_input"], "Custom model input")
        self.assertEqual(config.input_names["eval_data_input"], "Custom eval data input")
        self.assertEqual(config.input_names["extra_input"], "Extra input channel")
        
        # Check output_names
        self.assertEqual(config.output_names, custom_output_names)
        self.assertEqual(config.output_names["eval_output"], "Custom eval output")
        self.assertEqual(config.output_names["metrics_output"], "Custom metrics output")
        self.assertEqual(config.output_names["extra_output"], "Extra output channel")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
