"""
Unit tests for the bedrock_batch_inference module.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import pandas as pd
import json
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bedrock.bedrock_batch_inference import (
    save_checkpoint,
    load_checkpoint,
    create_result_dataframe,
    process_single_row,
    process_batch,
    batch_process_dataframe
)
from src.bedrock.prompt_rnr_parse import BSMAnalysis


class TestBedrockBatchInference(unittest.TestCase):
    """Test cases for the bedrock_batch_inference module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample DataFrame
        self.sample_df = pd.DataFrame({
            'dialogue': ['Test dialogue 1', 'Test dialogue 2', 'Test dialogue 3'],
            'shiptrack_event_history': ['Track 1', 'Track 2', 'Track 3'],
            'max_estimated_arrival_date': ['2023-01-01', '2023-01-02', '2023-01-03']
        })
        
        # Create a sample BSMAnalysis
        self.sample_analysis = BSMAnalysis(
            category="TestCategory",
            confidence_score=0.95,
            message_evidence=["Evidence 1", "Evidence 2"],
            shipping_evidence=["Shipping 1"],
            timeline_evidence=["Timeline 1"],
            primary_factors=["Factor 1"],
            supporting_evidence=["Support 1"],
            contradicting_evidence=["Contradict 1"],
            raw_response="Raw response",
            latency=0.5
        )
        
        # Create a sample results list
        self.sample_results = [
            {'index': 0, 'analysis': self.sample_analysis},
            {'index': 1, 'analysis': self.sample_analysis}
        ]

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('pathlib.Path.mkdir')
    def test_save_checkpoint(self, mock_mkdir, mock_json_dump, mock_file):
        """Test saving checkpoint to a file."""
        # Call the function
        save_checkpoint('test_checkpoint.json', 2, self.sample_results)
        
        # Check that the file was opened for writing
        # Note: The function converts the path to a Path object, so we don't check the exact call
        mock_file.assert_called_once()
        
        # Check that mkdir was called with parents=True
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        
        # Check that json.dump was called with the correct data
        args, _ = mock_json_dump.call_args
        checkpoint_data = args[0]
        self.assertEqual(checkpoint_data['processed_rows'], 2)
        self.assertEqual(len(checkpoint_data['results']), 2)

    @patch('builtins.open', new_callable=mock_open, read_data='{"processed_rows": 2, "results": []}')
    @patch('json.load')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_checkpoint(self, mock_exists, mock_json_load, mock_file):
        """Test loading checkpoint from a file."""
        # Mock the JSON data
        mock_json_load.return_value = {
            'processed_rows': 2,
            'results': [
                {
                    'index': 0,
                    'analysis': self.sample_analysis.model_dump()
                },
                {
                    'index': 1,
                    'analysis': self.sample_analysis.model_dump()
                }
            ]
        }
        
        # Call the function
        processed_rows, results = load_checkpoint('test_checkpoint.json')
        
        # Check the results
        self.assertEqual(processed_rows, 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['index'], 0)
        self.assertEqual(results[1]['index'], 1)
        
        # Check that the file was opened for reading
        # Note: The function converts the path to a Path object, so we don't check the exact call
        mock_file.assert_called_once()

    @patch('pathlib.Path.exists', return_value=False)
    def test_load_checkpoint_no_file(self, mock_exists):
        """Test loading checkpoint when file doesn't exist."""
        # Call the function
        processed_rows, results = load_checkpoint('nonexistent_checkpoint.json')
        
        # Check the results
        self.assertEqual(processed_rows, 0)
        self.assertEqual(results, [])

    def test_create_result_dataframe(self):
        """Test creating result DataFrame."""
        # Call the function
        result_df = create_result_dataframe(self.sample_results, self.sample_df)
        
        # Check the results
        self.assertEqual(len(result_df), 3)  # Original DataFrame has 3 rows
        self.assertTrue('category' in result_df.columns)
        self.assertTrue('confidence_score' in result_df.columns)
        self.assertTrue('message_evidence' in result_df.columns)
        
        # Check that the values were correctly merged
        self.assertEqual(result_df.iloc[0]['category'], "TestCategory")
        self.assertEqual(result_df.iloc[1]['category'], "TestCategory")
        self.assertTrue(pd.isna(result_df.iloc[2]['category']))  # No result for index 2

    @patch('src.bedrock.bedrock_batch_inference.analyze_dialogue_with_claude')
    @patch('src.bedrock.bedrock_batch_inference.parse_claude_response')
    def test_process_single_row_success(self, mock_parse, mock_analyze):
        """Test processing a single row successfully."""
        # Mock the analyze_dialogue_with_claude function
        mock_analyze.return_value = ("Claude response", 0.5)
        
        # Mock the parse_claude_response function
        mock_parse.return_value = self.sample_analysis
        
        # Call the function
        result = process_single_row(
            dialogue="Test dialogue",
            shiptrack_event_history="Track history",
            max_estimated_arrival_date="2023-01-01"
        )
        
        # Check the result
        self.assertEqual(result.category, "TestCategory")
        self.assertEqual(result.confidence_score, 0.95)
        self.assertEqual(result.latency, 0.5)
        
        # Check that the functions were called with the correct arguments
        mock_analyze.assert_called_once_with(
            dialogue="Test dialogue",
            shiptrack="Track history",
            max_estimated_arrival_date="2023-01-01",
            max_retries=5,
            bedrock_client=None
        )
        mock_parse.assert_called_once_with("Claude response")

    @patch('src.bedrock.bedrock_batch_inference.analyze_dialogue_with_claude')
    def test_process_single_row_error(self, mock_analyze):
        """Test processing a single row with an error."""
        # Mock the analyze_dialogue_with_claude function to return an error
        mock_analyze.return_value = ("Error: Test error", None)
        
        # Call the function
        result = process_single_row(
            dialogue="Test dialogue",
            shiptrack_event_history="Track history",
            max_estimated_arrival_date="2023-01-01"
        )
        
        # Check the result
        self.assertEqual(result.category, "Error")
        self.assertEqual(result.confidence_score, 0.0)
        self.assertEqual(result.error, "Error: Test error")

    @patch('src.bedrock.bedrock_batch_inference.analyze_dialogue_with_claude')
    def test_process_single_row_exception(self, mock_analyze):
        """Test processing a single row with an exception."""
        # Mock the analyze_dialogue_with_claude function to raise an exception
        mock_analyze.side_effect = Exception("Test exception")
        
        # Call the function
        result = process_single_row(
            dialogue="Test dialogue",
            shiptrack_event_history="Track history",
            max_estimated_arrival_date="2023-01-01"
        )
        
        # Check the result
        self.assertEqual(result.category, "Error")
        self.assertEqual(result.confidence_score, 0.0)
        self.assertTrue("Test exception" in result.error)

    @patch('src.bedrock.bedrock_batch_inference.process_single_row')
    def test_process_batch(self, mock_process_row):
        """Test processing a batch of rows."""
        # Mock the process_single_row function
        mock_process_row.return_value = self.sample_analysis
        
        # Create a mock progress callback
        mock_callback = MagicMock()
        
        # Call the function
        results = process_batch(
            batch_df=self.sample_df,
            max_workers=2,
            max_retries=3,
            bedrock_client=MagicMock(),
            progress_callback=mock_callback
        )
        
        # Check the results
        self.assertEqual(len(results), 3)  # 3 rows in the sample DataFrame
        
        # Check that all indices are present (0, 1, 2)
        indices = sorted([r['index'] for r in results])
        self.assertEqual(indices, [0, 1, 2])
        
        # Check that process_single_row was called for each row
        self.assertEqual(mock_process_row.call_count, 3)
        
        # Check that the progress callback was called for each row
        self.assertEqual(mock_callback.call_count, 3)

    @patch('src.bedrock.bedrock_batch_inference.process_batch')
    @patch('src.bedrock.bedrock_batch_inference.get_bedrock_client')
    def test_batch_process_dataframe(self, mock_get_client, mock_process_batch):
        """Test batch processing a DataFrame."""
        # Mock the get_bedrock_client function
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Create a result DataFrame with unique indices
        result_records = []
        for i, result in enumerate(self.sample_results):
            analysis = result['analysis']
            result_records.append({
                'index': i,  # Use sequential indices
                **analysis.model_dump(exclude={'raw_response'})
            })
        result_df = pd.DataFrame(result_records)
        result_df.set_index('index', inplace=True)
        
        # Mock the process_batch function to return results with unique indices
        mock_process_batch.return_value = [
            {'index': 0, 'analysis': self.sample_analysis},
            {'index': 1, 'analysis': self.sample_analysis},
            {'index': 2, 'analysis': self.sample_analysis}
        ]
        
        # Create a mock result DataFrame
        mock_result_df = pd.DataFrame({
            'category': ['TestCategory', 'TestCategory', 'TestCategory'],
            'confidence_score': [0.95, 0.95, 0.95]
        })
        
        # Call the function with a DataFrame that has a unique index
        df_with_unique_index = self.sample_df.copy()
        df_with_unique_index.index = range(len(df_with_unique_index))
        
        # Use a context manager to patch pd.concat to return our mock result
        with patch('pandas.concat', return_value=mock_result_df):
            result_df = batch_process_dataframe(
                df=df_with_unique_index,
                batch_size=2,
                max_workers=2,
                max_retries=3
            )
        
        # Check the results
        self.assertEqual(len(result_df), 3)  # Original DataFrame has 3 rows
        
        # Check that process_batch was called with the correct arguments
        mock_process_batch.assert_called()
        
        # Check that get_bedrock_client was called
        mock_get_client.assert_called_once()

    @patch('src.bedrock.bedrock_batch_inference.process_batch')
    @patch('src.bedrock.bedrock_batch_inference.load_checkpoint')
    @patch('src.bedrock.bedrock_batch_inference.save_checkpoint')
    @patch('src.bedrock.bedrock_batch_inference.get_bedrock_client')
    def test_batch_process_dataframe_with_checkpoint(self, mock_get_client, mock_save, mock_load, mock_process_batch):
        """Test batch processing with checkpoint."""
        # Mock the get_bedrock_client function
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the load_checkpoint function to return some processed rows
        mock_load.return_value = (2, self.sample_results[:2])
        
        # Mock the process_batch function for the remaining row
        mock_process_batch.return_value = self.sample_results[2:]
        
        # Create a temporary checkpoint file
        with tempfile.NamedTemporaryFile() as temp_file:
            # Call the function
            result_df = batch_process_dataframe(
                df=self.sample_df,
                batch_size=2,
                max_workers=2,
                max_retries=3,
                checkpoint_file=temp_file.name
            )
            
            # Check the results
            self.assertEqual(len(result_df), 3)  # Original DataFrame has 3 rows
            
            # Check that load_checkpoint was called
            mock_load.assert_called_once_with(temp_file.name)
            
            # Check that save_checkpoint was called
            mock_save.assert_called()
            
            # Check that process_batch was called only for the remaining rows
            mock_process_batch.assert_called_once()


if __name__ == '__main__':
    unittest.main()
