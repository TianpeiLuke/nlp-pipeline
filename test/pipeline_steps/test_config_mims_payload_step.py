import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pipeline_steps.config_mims_payload_step import PayloadConfig
from src.pipeline_steps.config_mims_registration_step import VariableType


class TestPayloadConfig(unittest.TestCase):
    def setUp(self):
        """Set up a minimal valid configuration for testing."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'inference.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy inference script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a minimal valid configuration
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "model_owner": "test-team",
            "model_registration_domain": "BuyerSellerMessaging",
            "model_registration_objective": "TestObjective",
            "source_model_inference_content_types": ["text/csv"],
            "source_model_inference_response_types": ["application/json"],
            "source_model_inference_output_variable_list": {"score": VariableType.NUMERIC},
            "source_model_inference_input_variable_list": {
                "feature1": VariableType.NUMERIC, 
                "feature2": VariableType.TEXT
            },
            "source_dir": self.temp_dir,
            "inference_entry_point": "inference.py"
        }
        
    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Verify basic attributes
            self.assertEqual(config.bucket, "test-bucket")
            self.assertEqual(config.author, "test-author")
            self.assertEqual(config.pipeline_name, "test-pipeline")
            self.assertEqual(config.model_owner, "test-team")
            self.assertEqual(config.model_registration_domain, "BuyerSellerMessaging")
            self.assertEqual(config.model_registration_objective, "TestObjective")
            
            # Verify default values
            self.assertEqual(config.expected_tps, 2)
            self.assertEqual(config.max_latency_in_millisecond, 800)
            self.assertEqual(config.max_acceptable_error_rate, 0.2)
            self.assertEqual(config.default_numeric_value, 0.0)
            self.assertEqual(config.default_text_value, "DEFAULT_TEXT")
            self.assertIsNone(config.special_field_values)
        
    def test_missing_required_field(self):
        """Test that missing required fields raise ValidationError."""
        # Remove a required field
        invalid_config = self.valid_config_data.copy()
        invalid_config.pop("model_registration_objective")
        
        # Should raise ValidationError
        with self.assertRaises(Exception):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                PayloadConfig(**invalid_config)
            
    def test_construct_payload_path(self):
        """Test that sample_payload_s3_key is constructed if not provided during initialization."""
        # Create a config with sample_payload_s3_key explicitly set to None
        config_data = self.valid_config_data.copy()
        config_data["sample_payload_s3_key"] = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Verify that sample_payload_s3_key was constructed by the model_validator
            expected_key = f"mods/payload/payload_{config.pipeline_name}_{config.pipeline_version}_{config.model_registration_objective}.tar.gz"
            self.assertEqual(config.sample_payload_s3_key, expected_key)
        
    def test_ensure_payload_path(self):
        """Test the ensure_payload_path method."""
        # Create a config with sample_payload_s3_key explicitly set to None
        config_data = self.valid_config_data.copy()
        config_data["sample_payload_s3_key"] = None
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Reset sample_payload_s3_key to None (since the model_validator would have set it)
            config.sample_payload_s3_key = None
            
            # Call ensure_payload_path
            config.ensure_payload_path()
            
            # Verify that sample_payload_s3_key was set correctly
            expected_key = f"mods/payload/payload_{config.pipeline_name}_{config.pipeline_version}_{config.model_registration_objective}.tar.gz"
            self.assertEqual(config.sample_payload_s3_key, expected_key)
        
    def test_get_full_payload_path(self):
        """Test get_full_payload_path method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Verify full payload path
            expected_path = f"s3://{config.bucket}/{config.sample_payload_s3_key}"
            self.assertEqual(config.get_full_payload_path(), expected_path)
        
    def test_get_field_default_value_numeric(self):
        """Test get_field_default_value method for numeric fields."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Verify default value for numeric field
            value = config.get_field_default_value("feature1", VariableType.NUMERIC)
            self.assertEqual(value, str(config.default_numeric_value))
        
    def test_get_field_default_value_text(self):
        """Test get_field_default_value method for text fields."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Verify default value for text field
            value = config.get_field_default_value("feature2", VariableType.TEXT)
            self.assertEqual(value, config.default_text_value)
        
    def test_get_field_default_value_special_field(self):
        """Test get_field_default_value method for special text fields."""
        # Add special field values
        config_data = self.valid_config_data.copy()
        config_data["special_field_values"] = {
            "feature2": "Special value with timestamp: {timestamp}"
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Verify special field value includes timestamp
            value = config.get_field_default_value("feature2", VariableType.TEXT)
            self.assertIn("Special value with timestamp:", value)
            self.assertNotIn("{timestamp}", value)  # Placeholder should be replaced
        
    def test_get_field_default_value_invalid_type(self):
        """Test get_field_default_value method with invalid variable type."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Should raise ValueError for invalid type
            with self.assertRaises(ValueError):
                config.get_field_default_value("feature1", "INVALID_TYPE")
            
    def test_generate_csv_payload_dict_format(self):
        """Test generate_csv_payload method with dictionary format input variables."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Generate CSV payload
            csv_payload = config.generate_csv_payload()
            
            # Verify CSV format
            values = csv_payload.split(",")
            self.assertEqual(len(values), 2)  # Two input variables
            self.assertEqual(values[0], str(config.default_numeric_value))  # feature1 (NUMERIC)
            self.assertEqual(values[1], config.default_text_value)  # feature2 (TEXT)
        
    def test_generate_csv_payload_list_format(self):
        """Test generate_csv_payload method with list format input variables."""
        # Use list format for input variables
        config_data = self.valid_config_data.copy()
        config_data["source_model_inference_input_variable_list"] = [
            ["feature1", VariableType.NUMERIC],
            ["feature2", VariableType.TEXT]
        ]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Generate CSV payload
            csv_payload = config.generate_csv_payload()
            
            # Verify CSV format
            values = csv_payload.split(",")
            self.assertEqual(len(values), 2)  # Two input variables
            self.assertEqual(values[0], str(config.default_numeric_value))  # feature1 (NUMERIC)
            self.assertEqual(values[1], config.default_text_value)  # feature2 (TEXT)
        
    def test_generate_json_payload_dict_format(self):
        """Test generate_json_payload method with dictionary format input variables."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Generate JSON payload
            json_payload = config.generate_json_payload()
            
            # Parse JSON
            payload_dict = json.loads(json_payload)
            
            # Verify JSON format
            self.assertEqual(len(payload_dict), 2)  # Two input variables
            self.assertEqual(payload_dict["feature1"], str(config.default_numeric_value))  # feature1 (NUMERIC)
            self.assertEqual(payload_dict["feature2"], config.default_text_value)  # feature2 (TEXT)
        
    def test_generate_json_payload_list_format(self):
        """Test generate_json_payload method with list format input variables."""
        # Use list format for input variables
        config_data = self.valid_config_data.copy()
        config_data["source_model_inference_input_variable_list"] = [
            ["feature1", VariableType.NUMERIC],
            ["feature2", VariableType.TEXT]
        ]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Generate JSON payload
            json_payload = config.generate_json_payload()
            
            # Parse JSON
            payload_dict = json.loads(json_payload)
            
            # Verify JSON format
            self.assertEqual(len(payload_dict), 2)  # Two input variables
            self.assertEqual(payload_dict["feature1"], str(config.default_numeric_value))  # feature1 (NUMERIC)
            self.assertEqual(payload_dict["feature2"], config.default_text_value)  # feature2 (TEXT)
        
    def test_generate_sample_payloads_csv(self):
        """Test generate_sample_payloads method with CSV content type."""
        # Use CSV content type
        config_data = self.valid_config_data.copy()
        config_data["source_model_inference_content_types"] = ["text/csv"]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Generate sample payloads
            payloads = config.generate_sample_payloads()
            
            # Verify payloads
            self.assertEqual(len(payloads), 1)  # One content type
            
            # Verify CSV payload
            csv_payload = payloads[0]
            self.assertEqual(csv_payload["content_type"], "text/csv")
            self.assertIsNotNone(csv_payload["payload"])
            self.assertIn(",", csv_payload["payload"])
        
    def test_generate_sample_payloads_json(self):
        """Test generate_sample_payloads method with JSON content type."""
        # Use JSON content type
        config_data = self.valid_config_data.copy()
        config_data["source_model_inference_content_types"] = ["application/json"]
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
            # Generate sample payloads
            payloads = config.generate_sample_payloads()
            
            # Verify payloads
            self.assertEqual(len(payloads), 1)  # One content type
            
            # Verify JSON payload
            json_payload = payloads[0]
            self.assertEqual(json_payload["content_type"], "application/json")
            self.assertIsNotNone(json_payload["payload"])
            self.assertIn("{", json_payload["payload"])
            self.assertIn("}", json_payload["payload"])
        
    def test_generate_sample_payloads_unsupported_content_type(self):
        """Test generate_sample_payloads method with unsupported content type."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Set unsupported content type
            config.source_model_inference_content_types = ["application/unsupported"]
            
            # Should raise ValueError for unsupported content type
            with self.assertRaises(ValueError):
                config.generate_sample_payloads()
            
    def test_save_payloads(self):
        """Test save_payloads method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Create temporary directory for test
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save payloads
                file_paths = config.save_payloads(temp_dir)
                
                # Verify file paths
                self.assertEqual(len(file_paths), 1)  # One content type
                self.assertTrue(file_paths[0].exists())
                self.assertTrue(str(file_paths[0]).endswith(".csv"))  # CSV file
            
    @patch("boto3.client")
    def test_upload_payloads_to_s3(self, mock_boto3_client):
        """Test upload_payloads_to_s3 method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Mock S3 client
            mock_s3_client = MagicMock()
            mock_boto3_client.return_value = mock_s3_client
            
            # Create temporary directory and file for test
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / "test_payload.csv"
                with open(temp_file, "w") as f:
                    f.write("test,payload")
                
                # Upload payload
                s3_uri = config.upload_payloads_to_s3([temp_file])
                
                # Verify S3 URI
                expected_uri = f"s3://{config.bucket}/{config.sample_payload_s3_key}"
                self.assertEqual(s3_uri, expected_uri)
                
                # Verify S3 client was called
                mock_s3_client.upload_file.assert_called_once()
            
    def test_upload_payloads_to_s3_no_files(self):
        """Test upload_payloads_to_s3 method with no files."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Should raise ValueError for no files
            with self.assertRaises(ValueError):
                config.upload_payloads_to_s3([])
            
    @patch("tempfile.TemporaryDirectory")
    @patch.object(PayloadConfig, "save_payloads")
    @patch.object(PayloadConfig, "upload_payloads_to_s3")
    def test_generate_and_upload_payloads(self, mock_upload, mock_save, mock_temp_dir):
        """Test generate_and_upload_payloads method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Mock temporary directory
            mock_temp_dir_instance = MagicMock()
            mock_temp_dir.return_value.__enter__.return_value = "/tmp/test"
            
            # Mock save_payloads
            mock_file_paths = [Path("/tmp/test/payload.csv")]
            mock_save.return_value = mock_file_paths
            
            # Mock upload_payloads_to_s3
            expected_s3_uri = f"s3://{config.bucket}/{config.sample_payload_s3_key}"
            mock_upload.return_value = expected_s3_uri
            
            # Generate and upload payloads
            s3_uri = config.generate_and_upload_payloads()
            
            # Verify S3 URI
            self.assertEqual(s3_uri, expected_s3_uri)
            
            # Verify methods were called
            mock_save.assert_called_once()
            mock_upload.assert_called_once_with(mock_file_paths)
        
    def test_validate_special_fields_success(self):
        """Test validate_special_fields with valid special fields."""
        # Add special field values for existing TEXT field
        config_data = self.valid_config_data.copy()
        config_data["special_field_values"] = {
            "feature2": "Special value with timestamp: {timestamp}"
        }
        
        # Should not raise any exceptions
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**config_data)
        
    def test_validate_special_fields_non_existent_field(self):
        """Test validate_special_fields with non-existent field."""
        # Add special field values for non-existent field
        config_data = self.valid_config_data.copy()
        config_data["special_field_values"] = {
            "non_existent_field": "Special value"
        }
        
        # Should raise ValueError for non-existent field
        with self.assertRaises(ValueError):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                PayloadConfig(**config_data)
            
    def test_validate_special_fields_non_text_field(self):
        """Test validate_special_fields with non-TEXT field."""
        # Add special field values for NUMERIC field
        config_data = self.valid_config_data.copy()
        config_data["special_field_values"] = {
            "feature1": "Special value"  # feature1 is NUMERIC
        }
        
        # Should raise ValueError for non-TEXT field
        with self.assertRaises(ValueError):
            with patch('pathlib.Path.exists', return_value=True), \
                 patch('pathlib.Path.is_file', return_value=True):
                PayloadConfig(**config_data)
            
    def test_model_dump(self):
        """Test model_dump method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
        
            # Dump model to dict
            config_dict = config.model_dump()
            
            # Verify dict contains all expected keys
            for key in self.valid_config_data:
                self.assertIn(key, config_dict)
                
            # Verify Path is converted to string
            if isinstance(config.payload_script_path, Path):
                self.assertIsInstance(config_dict["payload_script_path"], str)
                
    def test_get_effective_source_dir(self):
        """Test get_effective_source_dir method."""
        # Test with no payload_source_dir (should fall back to source_dir)
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
            
            # Verify effective source dir is source_dir
            self.assertEqual(config.get_effective_source_dir(), self.temp_dir)
            
            # Test with payload_source_dir set
            payload_source_dir = "/path/to/payload/source"
            config.payload_source_dir = payload_source_dir
            
            # Verify effective source dir is payload_source_dir
            self.assertEqual(config.get_effective_source_dir(), payload_source_dir)
            
    def test_get_script_path(self):
        """Test get_script_path method."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True):
            config = PayloadConfig(**self.valid_config_data)
            
            # Verify script path
            script_path = config.get_script_path()
            self.assertTrue(script_path.endswith('inference.py'))
            
            # Test with S3 source dir
            config.source_dir = "s3://bucket/path"
            
            # Verify script path with S3 source dir
            script_path = config.get_script_path()
            self.assertEqual(script_path, "s3://bucket/path/inference.py")
            
            # Test with no inference_entry_point
            config.inference_entry_point = None
            
            # Verify script path is None
            self.assertIsNone(config.get_script_path())


if __name__ == "__main__":
    unittest.main()
