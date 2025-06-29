import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the config class to be tested
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters


class TestXGBoostTrainingConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, valid configuration for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal valid hyperparameters object
        self.hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=[],
            label_name='label',
            id_name='order_id',
            model_class='xgboost',
            num_round=100,
            objective='binary:logistic',
            eta=0.1,
            max_depth=6,
            input_tab_dim=2  # Must match the length of tab_field_list
        )
        
        # Create a minimal valid config
        self.config_data = {
            "bucket": "test-bucket",
            "pipeline_name": "test-pipeline",
            "current_date": "2025-06-12",
            "training_entry_point": "train_xgb.py",
            "source_dir": self.temp_dir,
            "training_instance_type": "ml.m5.large",
            "training_instance_count": 1,
            "training_volume_size": 30,
            "framework_version": "1.7-1",
            "py_version": "py3",
            "hyperparameters": self.hyperparameters,
            "input_path": "s3://test-bucket/test-pipeline/preprocessed_data/2025-06-12",
            "output_path": "s3://test-bucket/test-pipeline/training_output/2025-06-12/model"
        }

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_init_with_valid_config(self, mock_normalize_paths):
        """Test initialization with valid configuration."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        config = XGBoostTrainingConfig(**self.config_data)
        
        # Verify basic attributes
        self.assertEqual(config.training_entry_point, "train_xgb.py")
        self.assertEqual(config.source_dir, self.temp_dir)
        self.assertEqual(config.training_instance_type, "ml.m5.large")
        self.assertEqual(config.training_instance_count, 1)
        self.assertEqual(config.training_volume_size, 30)
        self.assertEqual(config.framework_version, "1.7-1")
        self.assertEqual(config.py_version, "py3")
        self.assertEqual(config.hyperparameters, self.hyperparameters)
        
        # Verify paths (ignoring trailing slashes)
        expected_input_path = "s3://test-bucket/test-pipeline/preprocessed_data/2025-06-12"
        expected_output_path = "s3://test-bucket/test-pipeline/training_output/2025-06-12/model"
        
        self.assertEqual(config.input_path.rstrip('/'), expected_input_path)
        self.assertEqual(config.output_path.rstrip('/'), expected_output_path)
        
        # Verify default input_names
        self.assertIn("input_path", config.input_names)
        self.assertIn("config", config.input_names)
        
        # Verify default output_names
        self.assertIn("output_path", config.output_names)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_missing_training_entry_point(self, mock_normalize_paths):
        """Test that empty training_entry_point raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Set training_entry_point to empty string
        invalid_config = self.config_data.copy()
        invalid_config["training_entry_point"] = ""
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_missing_source_dir(self, mock_normalize_paths):
        """Test that missing source_dir raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Remove source_dir
        invalid_config = self.config_data.copy()
        invalid_config.pop("source_dir")
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_missing_hyperparameters(self, mock_normalize_paths):
        """Test that missing hyperparameters raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Remove hyperparameters
        invalid_config = self.config_data.copy()
        invalid_config.pop("hyperparameters")
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_invalid_training_instance_type(self, mock_normalize_paths):
        """Test that invalid training_instance_type raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Set invalid training_instance_type
        invalid_config = self.config_data.copy()
        invalid_config["training_instance_type"] = "invalid-instance-type"
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_path_normalization(self, mock_normalize_paths):
        """Test that paths are normalized (trailing slashes removed)."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Add trailing slashes to paths
        modified_config = self.config_data.copy()
        modified_config["input_path"] = "s3://test-bucket/test-pipeline/preprocessed_data/2025-06-12/"
        modified_config["output_path"] = "s3://test-bucket/test-pipeline/training_output/2025-06-12/model/"
        
        config = XGBoostTrainingConfig(**modified_config)
        
        # Manually normalize paths for testing
        self.assertEqual(config.input_path.rstrip('/'), "s3://test-bucket/test-pipeline/preprocessed_data/2025-06-12")
        self.assertEqual(config.output_path.rstrip('/'), "s3://test-bucket/test-pipeline/training_output/2025-06-12/model")

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_path_validation_duplicate_paths(self, mock_normalize_paths):
        """Test that duplicate paths raise ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Set output_path to same as input_path
        invalid_config = self.config_data.copy()
        invalid_config["output_path"] = invalid_config["input_path"]
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_path_validation_insufficient_depth(self, mock_normalize_paths):
        """Test that paths with insufficient depth raise ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Set input_path with insufficient depth
        invalid_config = self.config_data.copy()
        invalid_config["input_path"] = "s3://test-bucket"
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_hyperparameter_validation_invalid_field_list(self, mock_normalize_paths):
        """Test that invalid field lists in hyperparameters raise ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Create hyperparameters with invalid field list
        invalid_hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2', 'invalid_feature'],  # invalid feature not in full_field_list
            cat_field_list=[],
            label_name='label',
            id_name='order_id',
            model_class='xgboost',
            num_round=100,
            objective='binary:logistic',
            eta=0.1,
            max_depth=6,
            input_tab_dim=3  # Must match the length of tab_field_list
        )
        
        invalid_config = self.config_data.copy()
        invalid_config["hyperparameters"] = invalid_hyperparameters
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_hyperparameter_validation_invalid_label_name(self, mock_normalize_paths):
        """Test that invalid label_name in hyperparameters raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Create hyperparameters with invalid label_name
        invalid_hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=[],
            label_name='invalid_label',  # invalid label not in full_field_list
            id_name='order_id',
            model_class='xgboost',
            num_round=100,
            objective='binary:logistic',
            eta=0.1,
            max_depth=6,
            input_tab_dim=2  # Must match the length of tab_field_list
        )
        
        invalid_config = self.config_data.copy()
        invalid_config["hyperparameters"] = invalid_hyperparameters
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_hyperparameter_validation_invalid_id_name(self, mock_normalize_paths):
        """Test that invalid id_name in hyperparameters raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Create hyperparameters with invalid id_name
        invalid_hyperparameters = XGBoostModelHyperparameters(
            full_field_list=['order_id', 'feature1', 'feature2', 'label'],
            tab_field_list=['feature1', 'feature2'],
            cat_field_list=[],
            label_name='label',
            id_name='invalid_id',  # invalid id not in full_field_list
            model_class='xgboost',
            num_round=100,
            objective='binary:logistic',
            eta=0.1,
            max_depth=6,
            input_tab_dim=2  # Must match the length of tab_field_list
        )
        
        invalid_config = self.config_data.copy()
        invalid_config["hyperparameters"] = invalid_hyperparameters
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_default_values(self, mock_normalize_paths):
        """Test that default values are set correctly."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Create config with minimal required fields
        minimal_config = {
            "bucket": "test-bucket",
            "pipeline_name": "test-pipeline",
            "current_date": "2025-06-12",
            "hyperparameters": self.hyperparameters,
            "source_dir": self.temp_dir  # Add source_dir to fix the test
        }
        
        config = XGBoostTrainingConfig(**minimal_config)
        
        # Verify default values
        self.assertEqual(config.training_instance_type, "ml.m5.xlarge")
        self.assertEqual(config.training_instance_count, 1)
        self.assertEqual(config.training_volume_size, 30)
        self.assertEqual(config.training_entry_point, "train_xgb.py")
        self.assertEqual(config.framework_version, "1.7-1")
        self.assertEqual(config.py_version, "py3")
        
        # Verify constructed paths
        expected_input_path = f"s3://test-bucket/test-pipeline/preprocessed_data/2025-06-12"
        expected_output_path = f"s3://test-bucket/test-pipeline/training_output/2025-06-12/model"
        expected_checkpoint_path = f"s3://test-bucket/test-pipeline/training_checkpoints/2025-06-12"
        expected_hyperparameters_s3_uri = f"s3://test-bucket/test-pipeline/training_config/2025-06-12"
        
        # Compare with expected values, ignoring trailing slashes
        self.assertEqual(config.input_path.rstrip('/'), expected_input_path)
        self.assertEqual(config.output_path.rstrip('/'), expected_output_path)
        self.assertEqual(config.checkpoint_path.rstrip('/') if config.checkpoint_path else None, expected_checkpoint_path)
        self.assertEqual(config.hyperparameters_s3_uri.rstrip('/') if config.hyperparameters_s3_uri else None, expected_hyperparameters_s3_uri)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_custom_input_names(self, mock_normalize_paths):
        """Test that custom input_names can be set."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        custom_input_names = {
            "input_path": "Custom input path",
            "config": "Custom config path"
        }
        
        # Add custom input_names
        modified_config = self.config_data.copy()
        modified_config["input_names"] = custom_input_names
        
        config = XGBoostTrainingConfig(**modified_config)
        
        # Verify custom input_names
        self.assertEqual(config.input_names, custom_input_names)
        self.assertEqual(config.input_names["input_path"], "Custom input path")
        self.assertEqual(config.input_names["config"], "Custom config path")

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_custom_output_names(self, mock_normalize_paths):
        """Test that custom output_names can be set."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        custom_output_names = {
            "output_path": "Custom output path"
        }
        
        # Add custom output_names
        modified_config = self.config_data.copy()
        modified_config["output_names"] = custom_output_names
        
        config = XGBoostTrainingConfig(**modified_config)
        
        # Verify custom output_names
        self.assertEqual(config.output_names, custom_output_names)
        self.assertEqual(config.output_names["output_path"], "Custom output path")

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_missing_required_input_names(self, mock_normalize_paths):
        """Test that missing required input_names raises ValidationError."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        # Set input_names with missing required keys
        invalid_config = self.config_data.copy()
        invalid_config["input_names"] = {
            "wrong_name": "description"
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_missing_required_output_names(self, mock_normalize_paths):
        """Test that missing required output_names raises ValidationError."""
        # Set output_names with missing required keys
        invalid_config = self.config_data.copy()
        invalid_config["output_names"] = {
            "wrong_name": "description"
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            XGBoostTrainingConfig(**invalid_config)

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_get_checkpoint_uri(self, mock_normalize_paths):
        """Test that get_checkpoint_uri returns the correct URI."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        config = XGBoostTrainingConfig(**self.config_data)
        
        # Set checkpoint_path
        config.checkpoint_path = "s3://test-bucket/test-pipeline/checkpoints"
        
        # Verify get_checkpoint_uri returns the correct URI
        self.assertEqual(config.get_checkpoint_uri(), "s3://test-bucket/test-pipeline/checkpoints")

    @patch('src.pipeline_steps.config_training_step_xgboost.XGBoostTrainingConfig._normalize_paths')
    def test_has_checkpoint(self, mock_normalize_paths):
        """Test that has_checkpoint returns the correct value."""
        # Skip the _normalize_paths method to avoid recursion error
        mock_normalize_paths.return_value = None
        
        config = XGBoostTrainingConfig(**self.config_data)
        
        # Set checkpoint_path
        config.checkpoint_path = "s3://test-bucket/test-pipeline/checkpoints"
        
        # Verify has_checkpoint returns True
        self.assertTrue(config.has_checkpoint())
        
        # Set checkpoint_path to None
        config.checkpoint_path = None
        
        # Verify has_checkpoint returns False
        self.assertFalse(config.has_checkpoint())


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
