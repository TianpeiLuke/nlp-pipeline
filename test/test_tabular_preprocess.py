# test/test_tabular_preprocess.py
import unittest
from unittest.mock import patch
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
import gzip
import json

# Import the functions to be tested
from src.pipeline_scripts.tabular_preprocess import (
    combine_shards,
    impute_single_variable,
    parallel_imputation,
    _read_file_to_df
)

class TestTabularPreprocessHelpers(unittest.TestCase):
    def setUp(self):
        """Set up a temporary directory to act as a mock filesystem."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory after tests are complete."""
        shutil.rmtree(self.temp_dir)

    # --- Helper methods to create test files ---
    def _create_csv_shard(self, filename, data, gzipped=False):
        """Helper to create a CSV shard file."""
        path = os.path.join(self.temp_dir, filename)
        df = pd.DataFrame(data)
        if gzipped:
            with gzip.open(path, 'wt', newline='') as f:
                df.to_csv(f, index=False)
        else:
            df.to_csv(path, index=False)
        return path

    def _create_json_shard(self, filename, data, lines=True, gzipped=False):
        """Helper to create a JSON shard file."""
        path = os.path.join(self.temp_dir, filename)
        open_func = gzip.open if gzipped else open
        mode = 'wt'
        with open_func(path, mode) as f:
            if lines:
                for record in data:
                    f.write(json.dumps(record) + '\n')
            else:
                json.dump(data, f)
        return path

    def _create_parquet_shard(self, filename, data):
        """Helper to create a Parquet shard file."""
        path = os.path.join(self.temp_dir, filename)
        df = pd.DataFrame(data)
        df.to_parquet(path, index=False)
        return path

    # --- Tests for combine_shards ---
    def test_combine_shards_multiple_formats(self):
        """
        Test if combine_shards correctly reads and merges various file formats.
        """
        self._create_csv_shard("part-00000.csv", [{"a": 1, "b": "x"}])
        self._create_json_shard("part-00001.json.gz", [{"a": 2, "b": "y"}], gzipped=True)
        self._create_parquet_shard("part-00002.parquet", [{"a": 3, "b": "z"}])
        
        combined_df = combine_shards(self.temp_dir)
        
        self.assertEqual(len(combined_df), 3)
        self.assertEqual(list(combined_df.columns), ["a", "b"])
        self.assertTrue(pd.api.types.is_integer_dtype(combined_df['a']))
        combined_df = combined_df.sort_values(by='a').reset_index(drop=True)
        self.assertEqual(combined_df.iloc[0]['b'], 'x')
        self.assertEqual(combined_df.iloc[1]['b'], 'y')
        self.assertEqual(combined_df.iloc[2]['b'], 'z')

    def test_combine_shards_no_files_found(self):
        """Test that an error is raised if no valid shards are found."""
        with self.assertRaises(RuntimeError) as cm:
            combine_shards(self.temp_dir)
        self.assertIn("No CSV/JSON/Parquet shards found", str(cm.exception))
        
    # --- Tests for imputation functions ---
    def test_impute_single_variable(self):
        """Test the helper function for single variable imputation."""
        data = {'col1': [1, np.nan, 3, np.nan]}
        df_chunk = pd.DataFrame(data)
        impute_dict = {'col1': -1.0}
        
        args = (df_chunk, 'col1', impute_dict)
        result_series = impute_single_variable(args)
        
        expected = pd.Series([1.0, -1.0, 3.0, -1.0], name='col1')
        pd.testing.assert_series_equal(result_series, expected)

    @patch('src.pipeline_scripts.tabular_preprocess.cpu_count', return_value=2)
    def test_parallel_imputation(self, mock_cpu_count):
        """Test the parallel imputation wrapper function."""
        data = {
            'num1': [1.0, 2.0, np.nan, 4.0],
            'num2': [np.nan, 20.0, 30.0, 40.0],
            'cat1': ['A', 'B', 'A', 'C']
        }
        df = pd.DataFrame(data)
        num_vars = ['num1', 'num2']
        impute_dict = {'num1': 0.0, 'num2': -99.0}
        
        imputed_df = parallel_imputation(df.copy(), num_vars, impute_dict, n_workers=2)
        
        self.assertFalse(imputed_df['num1'].hasnans)
        self.assertFalse(imputed_df['num2'].hasnans)
        self.assertEqual(imputed_df.loc[2, 'num1'], 0.0)
        self.assertEqual(imputed_df.loc[0, 'num2'], -99.0)
        # Ensure categorical column is unchanged
        pd.testing.assert_series_equal(df['cat1'], imputed_df['cat1'])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

