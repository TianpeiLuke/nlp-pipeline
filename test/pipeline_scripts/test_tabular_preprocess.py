# test/test_tabular_preprocess.py
import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import gzip
import json
from pathlib import Path

# Import the functions to be tested
from src.pipeline_scripts.tabular_preprocess import (
    combine_shards,
    impute_single_variable,
    parallel_imputation,
    _read_file_to_df,
    peek_json_format,
    main as preprocess_main  # Import the main logic function
)

class TestTabularPreprocessHelpers(unittest.TestCase):
    """Unit tests for the helper functions in the preprocessing script."""
    def setUp(self):
        """Set up a temporary directory to act as a mock filesystem."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def tearDown(self):
        """Remove the temporary directory after tests are complete."""
        shutil.rmtree(self.temp_dir)

    # --- Helper methods to create test files ---
    def _create_csv_shard(self, filename, data, gzipped=False, delimiter=','):
        """Helper to create a CSV/TSV shard file."""
        path = self.temp_path / filename
        df = pd.DataFrame(data)
        if gzipped:
            with gzip.open(path, 'wt', newline='') as f:
                df.to_csv(f, index=False, sep=delimiter)
        else:
            df.to_csv(path, index=False, sep=delimiter)
        return path

    def _create_json_shard(self, filename, data, lines=True, gzipped=False):
        """Helper to create a JSON shard file."""
        path = self.temp_path / filename
        open_func = gzip.open if gzipped else open
        mode = 'wt'
        with open_func(path, mode) as f:
            if lines:
                if data: # Avoid writing if data is empty
                    for record in data:
                        f.write(json.dumps(record) + '\n')
            else:
                json.dump(data, f)
        return path

    def _create_parquet_shard(self, filename, data):
        """Helper to create a Parquet shard file."""
        path = self.temp_path / filename
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        return path

    # --- Tests for file processing and utility functions ---
    def test_combine_shards_concat_error(self):
        """Test that combine_shards raises an error if concatenation fails."""
        self._create_csv_shard("part-00000.csv", [{"a": 1}])
        self._create_csv_shard("part-00001.csv", [{"b": 2}]) # Different columns will cause concat issues
        
        # Mocking concat to raise a generic exception to test the handler
        with patch('pandas.concat', side_effect=ValueError("Concat failed")):
            with self.assertRaisesRegex(RuntimeError, "Failed to concatenate shards"):
                combine_shards(self.temp_dir)

    def test_read_json_single_object(self):
        """Test reading a JSON file with a single top-level object."""
        json_path = self._create_json_shard("single.json", {"a": 1, "b": "c"}, lines=False)
        df = _read_file_to_df(json_path)
        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df.iloc[0]['b'], 'c')

    def test_peek_json_format_empty_file(self):
        """Test peek_json_format with an empty file raises an error."""
        empty_path = self.temp_path / "empty.json"
        empty_path.touch()
        with self.assertRaisesRegex(RuntimeError, "Empty file"):
            peek_json_format(empty_path)

    @patch('src.pipeline_scripts.tabular_preprocess.csv.Sniffer.sniff', side_effect=Exception("Sniffing failed"))
    def test_detect_separator_fallback(self, mock_sniffer):
        """Test that the separator detection falls back to comma on error."""
        from src.pipeline_scripts.tabular_preprocess import _detect_separator_from_sample
        self.assertEqual(_detect_separator_from_sample("a\tb\n1\t2"), ",")

    def test_read_gzipped_parquet(self):
        """Test reading a gzipped Parquet file, which requires temporary decompression."""
        parquet_path = self.temp_path / "test.parquet"
        gzipped_parquet_path = self.temp_path / "test.parquet.gz"
        
        df_orig = pd.DataFrame([{"a": 10, "b": "world"}])
        df_orig.to_parquet(parquet_path)

        with open(parquet_path, 'rb') as f_in, gzip.open(gzipped_parquet_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        
        df_read = _read_file_to_df(gzipped_parquet_path)
        pd.testing.assert_frame_equal(df_orig, df_read)

    def test_read_unsupported_gzipped_file(self):
        """Test reading a gzipped file with an unsupported inner extension."""
        unsupported_gz_path = self.temp_path / "test.txt.gz"
        with gzip.open(unsupported_gz_path, 'wt') as f:
            f.write("some data")
        with self.assertRaises(ValueError):
            _read_file_to_df(unsupported_gz_path)

class TestMainFunction(unittest.TestCase):
    """Tests for the main preprocessing logic."""
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create dummy data file
        self.df = pd.DataFrame({
            "feature1": np.random.rand(100),
            "feature2__DOT__val": np.random.rand(100), # Test column renaming
            "label": np.random.choice(['A', 'B'], 100)
        })
        self.df.to_csv(os.path.join(self.input_dir, 'part-00000.csv'), index=False)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_main_training_split(self):
        """Test the main logic for a 'training' data_type, verifying the three-way split."""
        preprocess_main(
            data_type='training',
            label_field='label',
            train_ratio=0.6,
            test_val_ratio=0.5,
            input_dir=self.input_dir,
            output_dir=self.output_dir
        )
        # Verify outputs
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'train')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'test')))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'val')))
        
        train_df = pd.read_csv(os.path.join(self.output_dir, 'train', 'train_processed_data.csv'))
        test_df = pd.read_csv(os.path.join(self.output_dir, 'test', 'test_processed_data.csv'))
        val_df = pd.read_csv(os.path.join(self.output_dir, 'val', 'val_processed_data.csv'))

        # Check split sizes (60 train, 20 test, 20 val)
        self.assertEqual(len(train_df), 60)
        self.assertEqual(len(test_df), 20)
        self.assertEqual(len(val_df), 20)
        
        # Check column renaming
        self.assertIn("feature2.val", train_df.columns)
        # Check label encoding
        self.assertTrue(pd.api.types.is_integer_dtype(train_df['label']))

    def test_main_validation_mode(self):
        """Test main logic for a non-training data_type, ensuring no split occurs."""
        preprocess_main(
            data_type='validation',
            label_field='label',
            train_ratio=0.8,
            test_val_ratio=0.5,
            input_dir=self.input_dir,
            output_dir=self.output_dir
        )
        # Verify single output folder is created with the name of the data_type
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, 'validation')))
        self.assertFalse(os.path.exists(os.path.join(self.output_dir, 'train')))

        val_df = pd.read_csv(os.path.join(self.output_dir, 'validation', 'validation_processed_data.csv'))
        # Should contain all original data
        self.assertEqual(len(val_df), 100)
    
    def test_main_label_not_found_error(self):
        """Test that main raises a RuntimeError if the label field is not found."""
        with self.assertRaisesRegex(RuntimeError, "Label field 'wrong_label' not found"):
            preprocess_main(
                data_type='training',
                label_field='wrong_label',
                train_ratio=0.8,
                test_val_ratio=0.5,
                input_dir=self.input_dir,
                output_dir=self.output_dir
            )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

