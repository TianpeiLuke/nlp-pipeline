"""
Unit tests for the bedrock_batch_process_merge module.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import pandas as pd
import pyarrow as pa
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bedrock.bedrock_batch_process_merge import (
    save_batch_results,
    merge_batch_results,
    process_and_merge_results
)


class TestBedrockBatchProcessMerge(unittest.TestCase):
    """Test cases for the bedrock_batch_process_merge module."""

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
        
        # Create subdirectories
        self.batch_dir = self.test_dir / "batches"
        self.merged_dir = self.test_dir / "merged"
        self.stats_dir = self.test_dir / "stats"
        
        self.batch_dir.mkdir(exist_ok=True)
        self.merged_dir.mkdir(exist_ok=True)
        self.stats_dir.mkdir(exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()

    @patch('pyarrow.parquet.write_to_dataset')
    @patch('pandas.DataFrame.to_parquet')
    def test_save_batch_results_without_partitions(self, mock_to_parquet, mock_write_dataset):
        """Test saving batch results without partitioning."""
        # Call the function
        result = save_batch_results(
            df=self.sample_df,
            base_dir=str(self.batch_dir),
            batch_id="test_batch",
            partition_cols=None
        )
        
        # Check that to_parquet was called
        mock_to_parquet.assert_called_once()
        
        # Check that write_to_dataset was not called
        mock_write_dataset.assert_not_called()
        
        # Check that the result is a string
        self.assertIsInstance(result, str)
        
        # Check that the result contains the batch_id
        self.assertIn("test_batch", result)

    @patch('pyarrow.parquet.write_to_dataset')
    @patch('pandas.DataFrame.to_parquet')
    def test_save_batch_results_with_partitions(self, mock_to_parquet, mock_write_dataset):
        """Test saving batch results with partitioning."""
        # Call the function
        result = save_batch_results(
            df=self.sample_df,
            base_dir=str(self.batch_dir),
            batch_id="test_batch",
            partition_cols=["category"]
        )
        
        # Check that to_parquet was not called
        mock_to_parquet.assert_not_called()
        
        # Check that write_to_dataset was called
        mock_write_dataset.assert_called_once()
        
        # Check that the result is a string
        self.assertIsInstance(result, str)
        
        # Check that the result contains the batch_id
        self.assertIn("test_batch", result)

    @patch('pandas.read_parquet')
    @patch('pandas.concat')
    @patch('pyarrow.parquet.write_to_dataset')
    @patch('pandas.DataFrame.to_parquet')
    def test_merge_batch_results_without_partitions(self, mock_to_parquet, mock_write_dataset, 
                                                   mock_concat, mock_read_parquet):
        """Test merging batch results without partitioning."""
        # Create a test batch file
        test_batch_file = self.batch_dir / "batch_test_20250101_120000.parquet"
        with open(test_batch_file, 'w') as f:
            f.write("dummy content")
        
        # Mock read_parquet to return our sample DataFrame
        mock_read_parquet.return_value = self.sample_df
        
        # Mock concat to return our sample DataFrame
        mock_concat.return_value = self.sample_df
        
        # Call the function
        result = merge_batch_results(
            base_dir=str(self.batch_dir),
            output_dir=str(self.merged_dir),
            pattern="batch_*.parquet",
            partition_cols=None
        )
        
        # Check that read_parquet was called
        mock_read_parquet.assert_called()
        
        # Check that concat was called
        mock_concat.assert_called_once()
        
        # Check that to_parquet was called
        mock_to_parquet.assert_called_once()
        
        # Check that write_to_dataset was not called
        mock_write_dataset.assert_not_called()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

    @patch('pandas.read_parquet')
    @patch('pandas.concat')
    @patch('pyarrow.parquet.write_to_dataset')
    @patch('pandas.DataFrame.to_parquet')
    def test_merge_batch_results_with_partitions(self, mock_to_parquet, mock_write_dataset, 
                                                mock_concat, mock_read_parquet):
        """Test merging batch results with partitioning."""
        # Create a test batch file
        test_batch_file = self.batch_dir / "batch_test_20250101_120000.parquet"
        with open(test_batch_file, 'w') as f:
            f.write("dummy content")
        
        # Mock read_parquet to return our sample DataFrame
        mock_read_parquet.return_value = self.sample_df
        
        # Mock concat to return our sample DataFrame
        mock_concat.return_value = self.sample_df
        
        # Call the function
        result = merge_batch_results(
            base_dir=str(self.batch_dir),
            output_dir=str(self.merged_dir),
            pattern="batch_*.parquet",
            partition_cols=["category"]
        )
        
        # Check that read_parquet was called
        mock_read_parquet.assert_called()
        
        # Check that concat was called
        mock_concat.assert_called_once()
        
        # Check that to_parquet was not called
        mock_to_parquet.assert_not_called()
        
        # Check that write_to_dataset was called
        mock_write_dataset.assert_called_once()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

    @patch('src.bedrock.bedrock_batch_process_merge.batch_process_dataframe')
    @patch('src.bedrock.bedrock_batch_process_merge.save_batch_results')
    @patch('src.bedrock.bedrock_batch_process_merge.merge_batch_results')
    @patch('src.bedrock.bedrock_batch_process_merge.upload_to_s3')
    def test_process_and_merge_results(self, mock_upload, mock_merge, mock_save, mock_batch_process):
        """Test the complete workflow."""
        # Mock batch_process_dataframe to return our sample DataFrame
        mock_batch_process.return_value = self.sample_df
        
        # Mock save_batch_results to return a file path
        mock_save.return_value = str(self.batch_dir / "batch_test_20250101_120000.parquet")
        
        # Mock merge_batch_results to return our sample DataFrame
        mock_merge.return_value = self.sample_df
        
        # Mock upload_to_s3 to return True
        mock_upload.return_value = True
        
        # Call the function
        result = process_and_merge_results(
            df=self.sample_df,
            base_dir=str(self.test_dir),
            s3_bucket="test-bucket",
            batch_size=10
        )
        
        # Check that batch_process_dataframe was called
        mock_batch_process.assert_called_once()
        
        # Check that save_batch_results was called
        mock_save.assert_called_once()
        
        # Check that merge_batch_results was called
        mock_merge.assert_called_once()
        
        # Check that upload_to_s3 was called at least once
        mock_upload.assert_called()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)

    @patch('src.bedrock.bedrock_batch_process_merge.batch_process_dataframe')
    @patch('src.bedrock.bedrock_batch_process_merge.save_batch_results')
    @patch('src.bedrock.bedrock_batch_process_merge.merge_batch_results')
    def test_process_and_merge_results_without_s3(self, mock_merge, mock_save, mock_batch_process):
        """Test the workflow without S3 upload."""
        # Mock batch_process_dataframe to return our sample DataFrame
        mock_batch_process.return_value = self.sample_df
        
        # Mock save_batch_results to return a file path
        mock_save.return_value = str(self.batch_dir / "batch_test_20250101_120000.parquet")
        
        # Mock merge_batch_results to return our sample DataFrame
        mock_merge.return_value = self.sample_df
        
        # Call the function without an S3 bucket
        result = process_and_merge_results(
            df=self.sample_df,
            base_dir=str(self.test_dir),
            s3_bucket="",  # Empty string means no S3 upload
            batch_size=10
        )
        
        # Check that batch_process_dataframe was called
        mock_batch_process.assert_called_once()
        
        # Check that save_batch_results was called
        mock_save.assert_called_once()
        
        # Check that merge_batch_results was called
        mock_merge.assert_called_once()
        
        # Check that the result is a DataFrame
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
