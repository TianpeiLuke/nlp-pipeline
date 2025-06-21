"""
Unit tests for the upload_s3 module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os
from pathlib import Path
import tempfile

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bedrock.upload_s3 import (
    get_s3_client,
    upload_to_s3,
    upload_directory_to_s3
)


class TestUploadS3(unittest.TestCase):
    """Test cases for the upload_s3 module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create some test files
        self.test_file = self.test_dir / "test_file.txt"
        self.test_file.write_text("Test content")
        
        # Create a subdirectory with a file
        self.test_subdir = self.test_dir / "subdir"
        self.test_subdir.mkdir()
        self.test_subfile = self.test_subdir / "test_subfile.txt"
        self.test_subfile.write_text("Test content in subdirectory")

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    @patch('boto3.client')
    def test_get_s3_client(self, mock_client):
        """Test getting an S3 client."""
        mock_client.return_value = "mock_s3_client"
        result = get_s3_client()
        self.assertEqual(result, "mock_s3_client")
        mock_client.assert_called_once_with('s3')

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_to_s3_success(self, mock_get_client):
        """Test successful upload to S3."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function
        result = upload_to_s3(
            file_path=self.test_file,
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertTrue(result)
        mock_client.upload_file.assert_called_once()
        
        # Check the arguments
        args, kwargs = mock_client.upload_file.call_args
        self.assertEqual(args[0], str(self.test_file))
        self.assertEqual(args[1], "test-bucket")
        self.assertEqual(args[2], "test-prefix/test_file.txt")
        self.assertEqual(kwargs['ExtraArgs'], {'ContentType': 'application/octet-stream'})

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_to_s3_with_content_type(self, mock_get_client):
        """Test upload to S3 with custom content type."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function
        result = upload_to_s3(
            file_path=self.test_file,
            bucket="test-bucket",
            s3_prefix="test-prefix",
            content_type="text/plain"
        )
        
        # Check the result
        self.assertTrue(result)
        
        # Check the content type
        args, kwargs = mock_client.upload_file.call_args
        self.assertEqual(kwargs['ExtraArgs'], {'ContentType': 'text/plain'})

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_to_s3_with_extra_args(self, mock_get_client):
        """Test upload to S3 with extra arguments."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function
        result = upload_to_s3(
            file_path=self.test_file,
            bucket="test-bucket",
            s3_prefix="test-prefix",
            extra_args={'ACL': 'public-read'}
        )
        
        # Check the result
        self.assertTrue(result)
        
        # Check the extra args
        args, kwargs = mock_client.upload_file.call_args
        self.assertEqual(
            kwargs['ExtraArgs'], 
            {'ContentType': 'application/octet-stream', 'ACL': 'public-read'}
        )

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_to_s3_file_not_found(self, mock_get_client):
        """Test upload to S3 with non-existent file."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function with a non-existent file
        result = upload_to_s3(
            file_path=self.test_dir / "non_existent_file.txt",
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertFalse(result)
        mock_client.upload_file.assert_not_called()

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_to_s3_not_a_file(self, mock_get_client):
        """Test upload to S3 with a directory instead of a file."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function with a directory
        result = upload_to_s3(
            file_path=self.test_dir,
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertFalse(result)
        mock_client.upload_file.assert_not_called()

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_to_s3_client_error(self, mock_get_client):
        """Test upload to S3 with client error."""
        # Create a mock S3 client that raises an error
        mock_client = MagicMock()
        mock_client.upload_file.side_effect = Exception("Test error")
        mock_get_client.return_value = mock_client
        
        # Call the function
        result = upload_to_s3(
            file_path=self.test_file,
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertFalse(result)
        mock_client.upload_file.assert_called_once()

    @patch('src.bedrock.upload_s3.upload_to_s3')
    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_directory_to_s3_success(self, mock_get_client, mock_upload):
        """Test successful directory upload to S3."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the upload_to_s3 function to return True
        mock_upload.return_value = True
        
        # Call the function
        result = upload_directory_to_s3(
            local_dir=self.test_dir,
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertEqual(len(result), 2)  # Two files should be uploaded
        self.assertEqual(mock_upload.call_count, 2)

    @patch('src.bedrock.upload_s3.upload_to_s3')
    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_directory_to_s3_with_content_type_mapping(self, mock_get_client, mock_upload):
        """Test directory upload with content type mapping."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the upload_to_s3 function to return True
        mock_upload.return_value = True
        
        # Create a content type mapping
        content_type_mapping = {
            '.txt': 'text/plain',
            '.json': 'application/json'
        }
        
        # Call the function
        result = upload_directory_to_s3(
            local_dir=self.test_dir,
            bucket="test-bucket",
            s3_prefix="test-prefix",
            content_type_mapping=content_type_mapping
        )
        
        # Check the result
        self.assertEqual(len(result), 2)  # Two files should be uploaded
        self.assertEqual(mock_upload.call_count, 2)
        
        # Check that the content type was passed correctly
        for call in mock_upload.call_args_list:
            kwargs = call[1]  # Get the keyword arguments
            file_path = kwargs.get('file_path', '')
            if '.txt' in str(file_path):
                self.assertEqual(kwargs.get('content_type'), 'text/plain')

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_directory_to_s3_directory_not_found(self, mock_get_client):
        """Test directory upload with non-existent directory."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function with a non-existent directory
        result = upload_directory_to_s3(
            local_dir=self.test_dir / "non_existent_dir",
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertEqual(result, [])

    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_directory_to_s3_not_a_directory(self, mock_get_client):
        """Test directory upload with a file instead of a directory."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Call the function with a file
        result = upload_directory_to_s3(
            local_dir=self.test_file,
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertEqual(result, [])

    @patch('src.bedrock.upload_s3.upload_to_s3')
    @patch('src.bedrock.upload_s3.get_s3_client')
    def test_upload_directory_to_s3_partial_success(self, mock_get_client, mock_upload):
        """Test directory upload with some files failing."""
        # Create a mock S3 client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the upload_to_s3 function to return True for the first file and False for the second
        mock_upload.side_effect = [True, False]
        
        # Call the function
        result = upload_directory_to_s3(
            local_dir=self.test_dir,
            bucket="test-bucket",
            s3_prefix="test-prefix"
        )
        
        # Check the result
        self.assertEqual(len(result), 1)  # Only one file should be successfully uploaded
        self.assertEqual(mock_upload.call_count, 2)


if __name__ == '__main__':
    unittest.main()
