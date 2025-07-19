#!/usr/bin/env python3
"""
Test script to verify that the PayloadConfig class works correctly without 
accessing any private fields directly.
"""

import sys
import os
import logging
import unittest
from unittest import mock
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()

# Import the necessary modules
from src.pipeline_steps.config_mims_payload_step import PayloadConfig
from src.config_field_manager.type_aware_config_serializer import serialize_config


class TestPayloadConfig(unittest.TestCase):
    """Tests for the PayloadConfig class without direct access to private fields."""
    
    def setUp(self):
        """Set up test environment with valid local paths."""
        # Get the project root path
        self.project_root = Path(__file__).resolve().parent.parent.parent
        
        # Source directory to use for testing
        self.source_dir = str(self.project_root / "src" / "pipeline_scripts")
        
        # Create a standard config instance for tests
        self.config = PayloadConfig(
            bucket="test-bucket",
            author="test-author",
            region="NA",
            pipeline_name="test-pipeline",
            pipeline_description="Test pipeline description",
            pipeline_version="0.1.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline",
            source_dir=self.source_dir,
            processing_source_dir=self.source_dir,
            processing_entry_point="mims_payload.py",
            model_registration_objective="test-objective"
        )
    
    def test_initial_path_is_none(self):
        """Test that the payload path is initially None."""
        # The path should be None before ensure_payload_path is called
        self.assertIsNone(self.config.sample_payload_s3_key)
    
    def test_ensure_payload_path_creates_path(self):
        """Test that ensure_payload_path creates a path."""
        # Initially the path should be None
        self.assertIsNone(self.config.sample_payload_s3_key)
        
        # Call ensure_payload_path
        logger.info("Calling ensure_payload_path...")
        self.config.ensure_payload_path()
        
        # Now the path should be set
        self.assertIsNotNone(self.config.sample_payload_s3_key)
        logger.info(f"Generated path: {self.config.sample_payload_s3_key}")
    
    def test_path_format(self):
        """Test that the generated path follows the expected format."""
        # Call ensure_payload_path to generate the path
        self.config.ensure_payload_path()
        
        # Get the generated path
        path = self.config.sample_payload_s3_key
        self.assertIsNotNone(path)
        
        # Verify format
        expected_prefix = f"mods/payload/payload_{self.config.pipeline_name}_{self.config.pipeline_version}_{self.config.model_registration_objective}"
        self.assertTrue(
            path.startswith(expected_prefix),
            f"Expected S3 key to start with '{expected_prefix}', got '{path}'"
        )
        self.assertTrue(path.endswith(".tar.gz"))
    
    def test_path_stability(self):
        """Test that calling ensure_payload_path multiple times doesn't change the path."""
        # Generate path
        self.config.ensure_payload_path()
        original_path = self.config.sample_payload_s3_key
        self.assertIsNotNone(original_path)
        
        # Call multiple times
        logger.info("Calling ensure_payload_path multiple times...")
        for i in range(3):
            self.config.ensure_payload_path()
            current_path = self.config.sample_payload_s3_key
            logger.info(f"Call {i+1} result: {current_path}")
            
            # Path should remain stable
            self.assertEqual(current_path, original_path)
    
    def test_full_path_construction(self):
        """Test that get_full_payload_path combines bucket and path correctly."""
        # Ensure path is generated
        self.config.ensure_payload_path()
        path = self.config.sample_payload_s3_key
        self.assertIsNotNone(path)
        
        # Get full path
        full_path = self.config.get_full_payload_path()
        logger.info(f"Full path: {full_path}")
        
        # Verify format
        self.assertEqual(full_path, f"s3://{self.config.bucket}/{path}")
        
    def test_path_generation_on_get_full_path(self):
        """Test that get_full_payload_path generates a path if needed."""
        # Initially path should be None
        self.assertIsNone(self.config.sample_payload_s3_key)
        
        # Call get_full_payload_path
        full_path = self.config.get_full_payload_path()
        
        # Now path should be set
        self.assertIsNotNone(self.config.sample_payload_s3_key)
        self.assertEqual(full_path, f"s3://{self.config.bucket}/{self.config.sample_payload_s3_key}")
        
    def test_path_generation_on_upload(self):
        """Test that upload_payloads_to_s3 generates a path if needed."""
        # Create a mock file
        with mock.patch('boto3.client'):
            with mock.patch('tarfile.open'):
                # Mock a list of payload files
                mock_files = [Path("mock/path1.txt"), Path("mock/path2.txt")]
                
                # Initially path should be None
                self.assertIsNone(self.config.sample_payload_s3_key)
                
                # This will raise an exception due to mocking, but we're just checking path generation
                try:
                    self.config.upload_payloads_to_s3(mock_files)
                except Exception:
                    pass
                
                # Path should now be set
                self.assertIsNotNone(self.config.sample_payload_s3_key)
        
    def test_serialization_excludes_private_fields(self):
        """Test that serialization excludes private fields."""
        # Ensure path is generated
        self.config.ensure_payload_path()
        self.assertIsNotNone(self.config.sample_payload_s3_key)
        
        # Serialize the config
        serialized = serialize_config(self.config)
        
        # Private field should not be in serialized data
        self.assertNotIn("_sample_payload_s3_key", serialized)
        
        # But public fields should be included
        self.assertIn("bucket", serialized)
        self.assertIn("pipeline_name", serialized)
    
    def test_payload_generation_methods(self):
        """Test that payload generation methods work correctly."""
        # Add test input variables
        self.config.source_model_inference_input_variable_list = {
            "field1": "NUMERIC",
            "field2": "TEXT"
        }
        
        # Test CSV payload
        csv = self.config.generate_csv_payload()
        self.assertIsInstance(csv, str)
        values = csv.split(",")
        self.assertEqual(len(values), 2)
        
        # Test JSON payload
        json_str = self.config.generate_json_payload()
        self.assertIsInstance(json_str, str)
        payload = json.loads(json_str)
        self.assertIn("field1", payload)
        self.assertIn("field2", payload)


if __name__ == "__main__":
    unittest.main()
