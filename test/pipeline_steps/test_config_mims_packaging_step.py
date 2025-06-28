import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the config class to be tested
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig


class TestPackageStepConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, valid configuration for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'mims_package.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy MIMS packaging script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a minimal valid config
        self.config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "processing_entry_point": "mims_package.py",
            "processing_source_dir": self.temp_dir,
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "processing_instance_type_large": "ml.m5.4xlarge",
            "processing_instance_type_small": "ml.m5.large",
            "use_large_processing_instance": False,
            "processing_framework_version": "0.23-1"
        }

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = PackageStepConfig(**self.config_data)
        
        # Verify basic attributes
        self.assertEqual(config.processing_entry_point, "mims_package.py")
        self.assertEqual(config.processing_source_dir, self.temp_dir)
        self.assertEqual(config.processing_instance_count, 1)
        self.assertEqual(config.processing_volume_size, 30)
        self.assertEqual(config.processing_instance_type_large, "ml.m5.4xlarge")
        self.assertEqual(config.processing_instance_type_small, "ml.m5.large")
        self.assertEqual(config.use_large_processing_instance, False)
        self.assertEqual(config.processing_framework_version, "0.23-1")
        
        # Verify default input_names
        self.assertIn("model_input", config.input_names)
        self.assertIn("inference_scripts_input", config.input_names)
        
        # Verify default output_names
        self.assertIn("packaged_model_output", config.output_names)

    def test_missing_processing_entry_point(self):
        """Test that empty processing_entry_point raises ValidationError."""
        # Set processing_entry_point to empty string
        invalid_config = self.config_data.copy()
        invalid_config["processing_entry_point"] = ""
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            PackageStepConfig(**invalid_config)

    def test_missing_processing_source_dir(self):
        """Test that missing processing_source_dir raises ValidationError when no source_dir is provided."""
        # Remove processing_source_dir
        invalid_config = self.config_data.copy()
        invalid_config.pop("processing_source_dir")
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            PackageStepConfig(**invalid_config)

    def test_source_dir_fallback(self):
        """Test that source_dir is used as fallback when processing_source_dir is not provided."""
        # Remove processing_source_dir but add source_dir
        modified_config = self.config_data.copy()
        modified_config.pop("processing_source_dir")
        modified_config["source_dir"] = self.temp_dir
        
        # Should not raise ValidationError
        config = PackageStepConfig(**modified_config)
        
        # Verify source_dir is used
        self.assertEqual(config.get_effective_source_dir(), self.temp_dir)

    def test_custom_input_names(self):
        """Test that custom input_names can be set."""
        custom_input_names = {
            "model_input": "Custom model input",
            "inference_scripts_input": "Custom inference scripts input"
        }
        
        # Add custom input_names
        modified_config = self.config_data.copy()
        modified_config["input_names"] = custom_input_names
        
        config = PackageStepConfig(**modified_config)
        
        # Verify custom input_names
        self.assertEqual(config.input_names, custom_input_names)
        self.assertEqual(config.input_names["model_input"], "Custom model input")
        self.assertEqual(config.input_names["inference_scripts_input"], "Custom inference scripts input")

    def test_custom_output_names(self):
        """Test that custom output_names can be set."""
        custom_output_names = {
            "packaged_model_output": "Custom packaged model output"
        }
        
        # Add custom output_names
        modified_config = self.config_data.copy()
        modified_config["output_names"] = custom_output_names
        
        config = PackageStepConfig(**modified_config)
        
        # Verify custom output_names
        self.assertEqual(config.output_names, custom_output_names)
        self.assertEqual(config.output_names["packaged_model_output"], "Custom packaged model output")

    def test_missing_required_input_names(self):
        """Test that missing required input_names raises ValidationError."""
        # Set input_names with missing required keys
        invalid_config = self.config_data.copy()
        invalid_config["input_names"] = {
            "wrong_name": "description"
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            PackageStepConfig(**invalid_config)

    def test_missing_required_output_names(self):
        """Test that missing required output_names raises ValidationError."""
        # Set output_names with missing required keys
        invalid_config = self.config_data.copy()
        invalid_config["output_names"] = {
            "wrong_name": "description"
        }
        
        # Should raise ValidationError
        with self.assertRaises(ValueError):
            PackageStepConfig(**invalid_config)

    def test_get_script_path(self):
        """Test that get_script_path returns the correct path."""
        config = PackageStepConfig(**self.config_data)
        
        # Verify script path
        script_path = config.get_script_path()
        self.assertTrue(script_path.endswith('mims_package.py'))
        
        # Create a different script file in the temporary directory
        different_script = 'different_script.py'
        different_script_path = os.path.join(self.temp_dir, different_script)
        with open(different_script_path, 'w') as f:
            f.write('# Another dummy script for testing\n')
            f.write('print("This is another dummy script")\n')
            
        # Test with a different processing_entry_point value
        config.processing_entry_point = different_script
        self.assertTrue(config.get_script_path().endswith(different_script))

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    def test_validate_entry_point_paths(self, mock_is_file, mock_exists):
        """Test that validate_entry_point_paths validates the entry point exists."""
        # Mock Path.exists and Path.is_file to return True
        mock_exists.return_value = True
        mock_is_file.return_value = True
        
        # Should not raise any exceptions
        config = PackageStepConfig(**self.config_data)
        
        # Mock Path.is_file to return False
        mock_is_file.return_value = False
        
        # Should raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            PackageStepConfig(**self.config_data)

    def test_get_instance_type(self):
        """Test that get_instance_type returns the correct instance type."""
        config = PackageStepConfig(**self.config_data)
        
        # Test with use_large_processing_instance = False
        self.assertEqual(config.get_instance_type(), "ml.m5.large")
        
        # Test with use_large_processing_instance = True
        config.use_large_processing_instance = True
        self.assertEqual(config.get_instance_type(), "ml.m5.4xlarge")
        
        # Test with explicit size parameter
        self.assertEqual(config.get_instance_type("small"), "ml.m5.large")
        self.assertEqual(config.get_instance_type("large"), "ml.m5.4xlarge")
        
        # Test with invalid size parameter
        with self.assertRaises(ValueError):
            config.get_instance_type("invalid_size")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
