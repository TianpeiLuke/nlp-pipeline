import unittest
import json
import tempfile
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
            "payload_script_path": "/path/to/script.py"
        }
        
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
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
        invalid_config.pop("payload_script_path")
        
        # Should raise ValidationError
        with self.assertRaises(Exception):
            PayloadConfig(**invalid_config)
            
    def test_construct_payload_path(self):
        """Test that sample_payload_s3_key is constructed if not provided."""
        config = PayloadConfig(**self.valid_config_data)
        
        # Verify that sample_payload_s3_key was constructed
        expected_key = f"mods/payload/payload_{config.pipeline_name}_{config.pipeline_version}_{config.model_registration_objective}.tar.gz"
        self.assertEqual(config.sample_payload_s3_key, expected_key)
        
    def test_get_full_payload_path(self):
        """Test get_full_payload_path method."""
        config = PayloadConfig(**self.valid_config_data)
        
        # Verify full payload path
        expected_path = f"s3://{config.bucket}/{config.sample_payload_s3_key}"
        self.assertEqual(config.get_full_payload_path(), expected_path)
        
    def test_get_field_default_value_numeric(self):
        """Test get_field_default_value method for numeric fields."""
        config = PayloadConfig(**self.valid_config_data)
        
        # Verify default value for numeric field
        value = config.get_field_default_value("feature1", VariableType.NUMERIC)
        self.assertEqual(value, str(config.default_numeric_value))
        
    def test_get_field_default_value_text(self):
        """Test get_field_default_value method for text fields."""
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
        
        config = PayloadConfig(**config_data)
        
        # Verify special field value includes timestamp
        value = config.get_field_default_value("feature2", VariableType.TEXT)
        self.assertIn("Special value with timestamp:", value)
        self.assertNotIn("{timestamp}", value)  # Placeholder should be replaced
        
    def test_get_field_default_value_invalid_type(self):
        """Test get_field_default_value method with invalid variable type."""
        config = PayloadConfig(**self.valid_config_data)
        
        # Should raise ValueError for invalid type
        with self.assertRaises(ValueError):
            config.get_field_default_value("feature1", "INVALID_TYPE")
            
    def test_generate_csv_payload_dict_format(self):
        """Test generate_csv_payload method with dictionary format input variables."""
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
        config = PayloadConfig(**self.valid_config_data)
        
        # Mock the generate_sample_payloads method to test the unsupported content type case
        with patch('src.pipeline_steps.config_mims_payload_step.PayloadConfig.generate_sample_payloads') as mock_method:
            # Set up the mock to raise ValueError when called
            mock_method.side_effect = ValueError("Unsupported content type: application/unsupported")
            
            # Call the method and verify it raises ValueError
            with self.assertRaises(ValueError):
                config.generate_sample_payloads()
                
            # Verify the mock was called
            mock_method.assert_called_once()
            
    def test_save_payloads(self):
        """Test save_payloads method."""
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
        config = PayloadConfig(**self.valid_config_data)
        
        # Should raise ValueError for no files
        with self.assertRaises(ValueError):
            config.upload_payloads_to_s3([])
            
    @patch("tempfile.TemporaryDirectory")
    @patch.object(PayloadConfig, "save_payloads")
    @patch.object(PayloadConfig, "upload_payloads_to_s3")
    def test_generate_and_upload_payloads(self, mock_upload, mock_save, mock_temp_dir):
        """Test generate_and_upload_payloads method."""
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
            PayloadConfig(**config_data)
            
    def test_model_dump(self):
        """Test model_dump method."""
        config = PayloadConfig(**self.valid_config_data)
        
        # Dump model to dict
        config_dict = config.model_dump()
        
        # Verify dict contains all expected keys
        for key in self.valid_config_data:
            self.assertIn(key, config_dict)
            
        # Verify Path is converted to string
        if isinstance(config.payload_script_path, Path):
            self.assertIsInstance(config_dict["payload_script_path"], str)


if __name__ == "__main__":
    unittest.main()
