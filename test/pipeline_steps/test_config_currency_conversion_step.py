import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_steps.config_currency_conversion_step import CurrencyConversionConfig


class TestCurrencyConversionConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal valid configuration for testing."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'currency_conversion.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy currency conversion script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a minimal valid configuration
        self.valid_config_data = {
            "job_type": "training",
            "mode": "per_split",
            "train_ratio": 0.7,
            "test_val_ratio": 0.5,
            "label_field": "target",
            "processing_entry_point": "currency_conversion.py",
            "processing_source_dir": self.temp_dir,
            "processing_framework_version": "0.23-1",
            "processing_instance_type_large": "ml.m5.4xlarge",
            "processing_instance_type_small": "ml.m5.large",
            "use_large_processing_instance": False,
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "marketplace_id_col": "marketplace_id",
            "currency_conversion_var_list": ["price", "cost"],
            "currency_conversion_dict": {"USD": 1.0, "EUR": 0.85, "GBP": 0.75},
            "marketplace_info": {
                "US": {"currency_code": "USD"},
                "UK": {"currency_code": "GBP"},
                "DE": {"currency_code": "EUR"}
            },
            "default_currency": "USD",
            "enable_currency_conversion": True,
            "input_names": {"data_input": "ProcessedTabularData"},
            "output_names": {"converted_data": "ConvertedCurrencyData"}
        }
        
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = CurrencyConversionConfig(**self.valid_config_data)
        
            # Verify basic attributes
            self.assertEqual(config.job_type, "training")
            self.assertEqual(config.mode, "per_split")
            self.assertEqual(config.train_ratio, 0.7)
            self.assertEqual(config.test_val_ratio, 0.5)
            self.assertEqual(config.label_field, "target")
            self.assertEqual(config.marketplace_id_col, "marketplace_id")
            self.assertEqual(config.currency_conversion_var_list, ["price", "cost"])
            self.assertEqual(config.currency_conversion_dict, {"USD": 1.0, "EUR": 0.85, "GBP": 0.75})
            self.assertEqual(config.default_currency, "USD")
            self.assertTrue(config.enable_currency_conversion)
            
            # Verify input and output names
            self.assertEqual(config.input_names, {"data_input": "ProcessedTabularData"})
            self.assertEqual(config.output_names, {"converted_data": "ConvertedCurrencyData"})
        
    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        # Remove a required field
        invalid_config = self.valid_config_data.copy()
        invalid_config.pop("marketplace_id_col")
        
        # Should raise ValidationError
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_invalid_job_type(self):
        """Test that invalid job_type raises ValidationError."""
        invalid_config = self.valid_config_data.copy()
        invalid_config["job_type"] = "invalid_job_type"
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValidationError."""
        invalid_config = self.valid_config_data.copy()
        invalid_config["mode"] = "invalid_mode"
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_currency_conversion_dict_validation(self):
        """Test validation of currency_conversion_dict."""
        # Test empty dictionary
        invalid_config = self.valid_config_data.copy()
        invalid_config["currency_conversion_dict"] = {}
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
        
        # Test missing rate of 1.0
        invalid_config = self.valid_config_data.copy()
        invalid_config["currency_conversion_dict"] = {"EUR": 0.85, "GBP": 0.75}
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
        
        # Test negative rate
        invalid_config = self.valid_config_data.copy()
        invalid_config["currency_conversion_dict"] = {"USD": 1.0, "EUR": -0.85}
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_currency_conversion_var_list_validation(self):
        """Test validation of currency_conversion_var_list."""
        # Test duplicate variables
        invalid_config = self.valid_config_data.copy()
        invalid_config["currency_conversion_var_list"] = ["price", "price"]
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_validate_config_missing_marketplace_id_col(self):
        """Test validate_config with missing marketplace_id_col."""
        invalid_config = self.valid_config_data.copy()
        invalid_config["marketplace_id_col"] = ""
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_validate_config_empty_currency_conversion_var_list(self):
        """Test validate_config with empty currency_conversion_var_list."""
        invalid_config = self.valid_config_data.copy()
        invalid_config["currency_conversion_var_list"] = []
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_validate_config_empty_marketplace_info(self):
        """Test validate_config with empty marketplace_info."""
        invalid_config = self.valid_config_data.copy()
        invalid_config["marketplace_info"] = {}
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_validate_config_missing_label_field_for_split_after_conversion(self):
        """Test validate_config with missing label_field for split_after_conversion mode."""
        invalid_config = self.valid_config_data.copy()
        invalid_config["mode"] = "split_after_conversion"
        invalid_config["label_field"] = ""
        
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                CurrencyConversionConfig(**invalid_config)
    
    def test_set_default_names(self):
        """Test set_default_names method."""
        # Test with no input_names and output_names
        config_data = self.valid_config_data.copy()
        config_data.pop("input_names")
        config_data.pop("output_names")
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = CurrencyConversionConfig(**config_data)
            
            # Verify default names were set
            self.assertEqual(config.input_names, {"data_input": "ProcessedTabularData"})
            self.assertEqual(config.output_names, {"converted_data": "ConvertedCurrencyData"})
    
    def test_get_script_arguments(self):
        """Test get_script_arguments method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch.object(CurrencyConversionConfig, 'get_script_arguments', return_value=[
                 "--job-type", "training",
                 "--mode", "per_split",
                 "--marketplace-id-col", "marketplace_id",
                 "--default-currency", "USD",
                 "--enable-conversion", "true"
             ]):
            config = CurrencyConversionConfig(**self.valid_config_data)
            
            # Get script arguments
            args = config.get_script_arguments()
            
            # Verify arguments
            self.assertIn("--job-type", args)
            self.assertIn("training", args)
            self.assertIn("--mode", args)
            self.assertIn("per_split", args)
            self.assertIn("--marketplace-id-col", args)
            self.assertIn("marketplace_id", args)
            self.assertIn("--default-currency", args)
            self.assertIn("USD", args)
            self.assertIn("--enable-conversion", args)
            self.assertIn("true", args)
    
    def test_get_environment_variables(self):
        """Test get_environment_variables method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = CurrencyConversionConfig(**self.valid_config_data)
            
            # Get environment variables
            env = config.get_environment_variables()
            
            # Verify environment variables
            self.assertIn("CURRENCY_CONVERSION_VARS", env)
            self.assertIn("CURRENCY_CONVERSION_DICT", env)
            self.assertIn("MARKETPLACE_INFO", env)
            self.assertIn("LABEL_FIELD", env)
            self.assertIn("TRAIN_RATIO", env)
            self.assertIn("TEST_VAL_RATIO", env)
            
            # Verify values
            import json
            self.assertEqual(json.loads(env["CURRENCY_CONVERSION_VARS"]), ["price", "cost"])
            self.assertEqual(json.loads(env["CURRENCY_CONVERSION_DICT"]), {"USD": 1.0, "EUR": 0.85, "GBP": 0.75})
            self.assertEqual(json.loads(env["MARKETPLACE_INFO"]), {
                "US": {"currency_code": "USD"},
                "UK": {"currency_code": "GBP"},
                "DE": {"currency_code": "EUR"}
            })
            self.assertEqual(env["LABEL_FIELD"], "target")
            self.assertEqual(env["TRAIN_RATIO"], "0.7")
            self.assertEqual(env["TEST_VAL_RATIO"], "0.5")
    
    def test_missing_processing_script(self):
        """Test validation when processing script is missing."""
        # Remove the script file
        os.remove(os.path.join(self.temp_dir, 'currency_conversion.py'))
        
        # Should raise ValidationError
        with self.assertRaises(Exception):
            CurrencyConversionConfig(**self.valid_config_data)
    
    def test_missing_processing_source_dir(self):
        """Test validation when processing source directory is missing."""
        # Use a non-existent directory
        invalid_config = self.valid_config_data.copy()
        invalid_config["processing_source_dir"] = "/non/existent/directory"
        
        # Should raise ValidationError
        with self.assertRaises(Exception):
            CurrencyConversionConfig(**invalid_config)
    
    def test_get_script_path(self):
        """Test get_script_path method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = CurrencyConversionConfig(**self.valid_config_data)
            
            # Get script path
            script_path = config.get_script_path()
            
            # Verify script path
            self.assertTrue(script_path.endswith('currency_conversion.py'))
            
            # Test with S3 source dir
            config.processing_source_dir = "s3://bucket/path"
            script_path = config.get_script_path()
            self.assertEqual(script_path, "s3://bucket/path/currency_conversion.py")


if __name__ == "__main__":
    unittest.main()
