"""
Unit tests for the main module.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import pandas as pd
import tempfile
from pathlib import Path
import argparse

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bedrock.main import (
    setup_logging,
    load_dataframe,
    save_summary,
    process_with_checkpoint,
    parse_args,
    main
)


class TestMain(unittest.TestCase):
    """Test cases for the main module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame
        self.sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'category': ['TrueDNR', 'FalseDNR', 'TrueDNR'],
            'confidence_score': [0.95, 0.85, 0.92],
            'message_evidence': [['Evidence 1', 'Evidence 2'], ['Evidence 3'], ['Evidence 4']],
            'shipping_evidence': [['Shipping 1'], ['Shipping 2'], ['Shipping 3']]
        })
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test CSV file
        self.test_csv = self.test_dir / "test_data.csv"
        self.sample_df.to_csv(self.test_csv, index=False)
        
        # Create a test Parquet file
        self.test_parquet = self.test_dir / "test_data.parquet"
        self.sample_df.to_parquet(self.test_parquet, index=False)

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    @patch('logging.basicConfig')
    def test_setup_logging(self, mock_basicConfig):
        """Test setting up logging."""
        # Test with default log level
        setup_logging()
        mock_basicConfig.assert_called_once()
        
        # Reset mock
        mock_basicConfig.reset_mock()
        
        # Test with custom log level
        setup_logging("DEBUG")
        mock_basicConfig.assert_called_once()

    def test_load_dataframe_csv(self):
        """Test loading a DataFrame from a CSV file."""
        # Load the test CSV file
        df = load_dataframe(str(self.test_csv))
        
        # Check that the DataFrame was loaded correctly
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['id', 'category', 'confidence_score', 'message_evidence', 'shipping_evidence'])

    def test_load_dataframe_parquet(self):
        """Test loading a DataFrame from a Parquet file."""
        # Load the test Parquet file
        df = load_dataframe(str(self.test_parquet))
        
        # Check that the DataFrame was loaded correctly
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['id', 'category', 'confidence_score', 'message_evidence', 'shipping_evidence'])

    def test_load_dataframe_nonexistent(self):
        """Test loading a DataFrame from a nonexistent file."""
        # Try to load a nonexistent file
        with self.assertRaises(FileNotFoundError):
            load_dataframe(str(self.test_dir / "nonexistent.csv"))

    def test_load_dataframe_unsupported(self):
        """Test loading a DataFrame from an unsupported file format."""
        # Create a test file with an unsupported extension
        test_txt = self.test_dir / "test_data.txt"
        with open(test_txt, 'w') as f:
            f.write("This is not a supported format")
        
        # Try to load the unsupported file
        with self.assertRaises(ValueError):
            load_dataframe(str(test_txt))

    @patch('src.bedrock.main.upload_to_s3')
    def test_save_summary(self, mock_upload):
        """Test saving a summary of the processing results."""
        # Mock upload_to_s3 to return True
        mock_upload.return_value = True
        
        # Save summary without S3 upload
        summary_path = save_summary(
            df=self.sample_df,
            output_dir=str(self.test_dir),
            s3_bucket=None
        )
        
        # Check that the summary file was created
        self.assertTrue(os.path.exists(summary_path))
        
        # Check that upload_to_s3 was not called
        mock_upload.assert_not_called()
        
        # Save summary with S3 upload
        summary_path = save_summary(
            df=self.sample_df,
            output_dir=str(self.test_dir),
            s3_bucket="test-bucket"
        )
        
        # Check that the summary file was created
        self.assertTrue(os.path.exists(summary_path))
        
        # Check that upload_to_s3 was called
        mock_upload.assert_called_once()

    @patch('src.bedrock.main.batch_process_dataframe')
    def test_process_with_checkpoint(self, mock_batch_process):
        """Test processing data with checkpoint support."""
        # Mock batch_process_dataframe to return our sample DataFrame
        mock_batch_process.return_value = self.sample_df
        
        # Process with checkpoint
        result_df = process_with_checkpoint(
            df=self.sample_df,
            output_dir=str(self.test_dir),
            batch_size=10,
            max_workers=5,
            checkpoint_file=str(self.test_dir / "checkpoint.json")
        )
        
        # Check that batch_process_dataframe was called
        mock_batch_process.assert_called_once()
        
        # Check that the result is our sample DataFrame
        self.assertIs(result_df, self.sample_df)
        
        # Check that the results file was created
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "results.csv")))

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_args(self, mock_parse_args):
        """Test parsing command-line arguments."""
        # Mock parse_args to return a namespace with our test values
        mock_args = argparse.Namespace(
            input_file=str(self.test_csv),
            output_dir=str(self.test_dir),
            batch_size=10,
            max_workers=5,
            s3_bucket="test-bucket",
            checkpoint_file="",
            log_level="INFO",
            partition_cols=""
        )
        mock_parse_args.return_value = mock_args
        
        # Parse arguments
        args = parse_args()
        
        # Check that the arguments were parsed correctly
        self.assertEqual(args.input_file, str(self.test_csv))
        self.assertEqual(args.output_dir, str(self.test_dir))
        self.assertEqual(args.batch_size, 10)
        self.assertEqual(args.max_workers, 5)
        self.assertEqual(args.s3_bucket, "test-bucket")
        self.assertEqual(args.checkpoint_file, "")
        self.assertEqual(args.log_level, "INFO")
        self.assertEqual(args.partition_cols, "")

    @patch('src.bedrock.main.parse_args')
    @patch('src.bedrock.main.setup_logging')
    @patch('src.bedrock.main.load_dataframe')
    @patch('src.bedrock.main.process_with_checkpoint')
    @patch('src.bedrock.main.process_and_merge_results')
    @patch('src.bedrock.main.save_summary')
    def test_main_with_checkpoint(self, mock_save_summary, mock_process_merge, 
                                 mock_process_checkpoint, mock_load_df, 
                                 mock_setup_logging, mock_parse_args):
        """Test the main function with checkpoint."""
        # Mock parse_args to return a namespace with our test values
        mock_args = argparse.Namespace(
            input_file=str(self.test_csv),
            output_dir=str(self.test_dir),
            batch_size=10,
            max_workers=5,
            s3_bucket="test-bucket",
            checkpoint_file=str(self.test_dir / "checkpoint.json"),
            log_level="INFO",
            partition_cols=""
        )
        mock_parse_args.return_value = mock_args
        
        # Mock load_dataframe to return our sample DataFrame
        mock_load_df.return_value = self.sample_df
        
        # Mock process_with_checkpoint to return our sample DataFrame
        mock_process_checkpoint.return_value = self.sample_df
        
        # Call main
        result = main()
        
        # Check that the functions were called
        mock_parse_args.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_load_df.assert_called_once()
        mock_process_checkpoint.assert_called_once()
        mock_process_merge.assert_not_called()
        mock_save_summary.assert_called_once()
        
        # Check that main returned 0 (success)
        self.assertEqual(result, 0)

    @patch('src.bedrock.main.parse_args')
    @patch('src.bedrock.main.setup_logging')
    @patch('src.bedrock.main.load_dataframe')
    @patch('src.bedrock.main.process_with_checkpoint')
    @patch('src.bedrock.main.process_and_merge_results')
    @patch('src.bedrock.main.save_summary')
    def test_main_without_checkpoint(self, mock_save_summary, mock_process_merge, 
                                    mock_process_checkpoint, mock_load_df, 
                                    mock_setup_logging, mock_parse_args):
        """Test the main function without checkpoint."""
        # Mock parse_args to return a namespace with our test values
        mock_args = argparse.Namespace(
            input_file=str(self.test_csv),
            output_dir=str(self.test_dir),
            batch_size=10,
            max_workers=5,
            s3_bucket="test-bucket",
            checkpoint_file="",
            log_level="INFO",
            partition_cols=""
        )
        mock_parse_args.return_value = mock_args
        
        # Mock load_dataframe to return our sample DataFrame
        mock_load_df.return_value = self.sample_df
        
        # Mock process_and_merge_results to return our sample DataFrame
        mock_process_merge.return_value = self.sample_df
        
        # Call main
        result = main()
        
        # Check that the functions were called
        mock_parse_args.assert_called_once()
        mock_setup_logging.assert_called_once()
        mock_load_df.assert_called_once()
        mock_process_checkpoint.assert_not_called()
        mock_process_merge.assert_called_once()
        mock_save_summary.assert_called_once()
        
        # Check that main returned 0 (success)
        self.assertEqual(result, 0)

    @patch('src.bedrock.main.parse_args')
    @patch('src.bedrock.main.load_dataframe')
    def test_main_error(self, mock_load_df, mock_parse_args):
        """Test the main function with an error."""
        # Mock parse_args to return a namespace with our test values
        mock_args = argparse.Namespace(
            input_file=str(self.test_csv),
            output_dir=str(self.test_dir),
            batch_size=10,
            max_workers=5,
            s3_bucket="test-bucket",
            checkpoint_file="",
            log_level="INFO",
            partition_cols=""
        )
        mock_parse_args.return_value = mock_args
        
        # Mock load_dataframe to raise an exception
        mock_load_df.side_effect = ValueError("Test error")
        
        # Call main
        result = main()
        
        # Check that the functions were called
        mock_parse_args.assert_called_once()
        mock_load_df.assert_called_once()
        
        # Check that main returned 1 (error)
        self.assertEqual(result, 1)


if __name__ == '__main__':
    unittest.main()
