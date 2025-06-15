import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import logging
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processing.numerical_binning_processor import NumericalBinningProcessor


class TestNumericalBinningProcessor(unittest.TestCase):

    PROCESSOR_MODULE_LOGGER_NAME = "src.processing.numerical_binning_processor"

    def setUp(self):
        self.column_to_bin = 'value'
        self.data = pd.DataFrame({
            self.column_to_bin: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10.0, np.nan, 100.0],
            'feature2': list(range(13))
        })
        self.data_single_value = pd.DataFrame({
            self.column_to_bin: [5.0] * 10 
        })
        self.data_few_unique = pd.DataFrame({
            self.column_to_bin: [1.0, 1.0, 1.0, 2.0, 2.0, 10.0, 10.0] 
        })
        self.data_all_nan_for_fit = pd.DataFrame({
            self.column_to_bin: [np.nan, np.nan, np.nan]
        })
        self.data_qcut_fail = pd.DataFrame({ 
            self.column_to_bin: [1.0] * 10
        })


    def test_initialization_defaults(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin)
        self.assertEqual(processor.column_name, self.column_to_bin)
        self.assertEqual(processor.n_bins_requested, 5)
        self.assertEqual(processor.strategy, 'quantile')
        self.assertIsNone(processor.bin_labels_config)
        self.assertEqual(processor.output_column_name, f"{self.column_to_bin}_binned")
        self.assertEqual(processor.handle_missing_value, "as_is")
        self.assertEqual(processor.handle_out_of_range, "boundary_bins")
        self.assertFalse(processor.is_fitted)

    def test_fit_raises_errors_on_invalid_input(self):
        """Test that fit raises errors for invalid input types or missing columns."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin)
        with self.assertRaisesRegex(TypeError, "requires a pandas DataFrame"):
            processor.fit([1, 2, 3]) # Should be a DataFrame
        with self.assertRaisesRegex(ValueError, "not found in input data"):
            processor.fit(pd.DataFrame({'wrong_column': [1, 2]}))

    @patch('src.processing.numerical_binning_processor.logger')
    def test_fit_with_all_nan_data(self, mock_logger):
        """Test the fit method when the column contains only NaN values."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=5)
        processor.fit(self.data_all_nan_for_fit)
        
        # Assert that the warning method was called and contained the correct message
        mock_logger.warning.assert_called_once()
        self.assertIn("has no valid data", mock_logger.warning.call_args[0][0])
        
        self.assertTrue(processor.is_fitted)
        self.assertEqual(processor.n_bins_actual_, 1)
        np.testing.assert_array_equal(processor.bin_edges_, np.array([-np.inf, np.inf]))

    def test_transform_with_interval_labels(self):
        """Test the transform method when bin_labels is False."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=2, strategy='equal-width', bin_labels=False)
        processor.fit(self.data)
        
        series_to_transform = pd.Series([1.0, 50.0, 100.0])
        transformed = processor.transform(series_to_transform)
        
        self.assertTrue(isinstance(transformed.dtype, pd.CategoricalDtype))
        expected_codes = pd.Series([0, 0, 1], dtype='int8')
        pd.testing.assert_series_equal(transformed.cat.codes, expected_codes)

    def test_transform_raises_error_on_invalid_input_type(self):
        """Test that transform raises a TypeError for unsupported input types."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin)
        processor.fit(self.data)
        with self.assertRaisesRegex(TypeError, "must be a pandas DataFrame or Series"):
            processor.transform([1, 2, 3])

    def test_load_params_with_missing_keys(self):
        """Test that load_params raises a ValueError if required keys are missing."""
        invalid_params = {"column_name": "value", "n_bins_requested": 5}
        with self.assertRaisesRegex(ValueError, "are missing required keys"):
            NumericalBinningProcessor.load_params(invalid_params)

    def test_process_with_nan_as_is(self):
        """Test the process method with handle_missing_value set to 'as_is'."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, handle_missing_value="as_is")
        processor.fit(self.data)
        self.assertIsNone(processor.process(np.nan))

    def test_fit_with_single_unique_value(self):
        """Test fit method with data containing a single unique value."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        processor.fit(self.data_single_value)
        self.assertTrue(processor.is_fitted)
        self.assertEqual(processor.n_bins_actual_, 1)
        np.testing.assert_array_equal(processor.bin_edges_, np.array([4.999995, 5.000005]))  # Updated expected bin edges

    def test_fit_with_few_unique_values(self):
        """Test fit method with data containing few unique values."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=5, strategy='quantile')
        processor.fit(self.data_few_unique)
        self.assertTrue(processor.is_fitted)
        self.assertLessEqual(processor.n_bins_actual_, processor.n_bins_requested)

    def test_process_out_of_range_value(self):
        """Test process method with values outside the fitted range."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width', handle_out_of_range="OutOfRange")
        processor.fit(self.data)
        self.assertEqual(processor.process(-100), "OutOfRange")
        self.assertEqual(processor.process(1000), "OutOfRange")

    def test_process_boundary_bins(self):
        """Test process method with handle_out_of_range set to 'boundary_bins'."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width', handle_out_of_range="boundary_bins")
        processor.fit(self.data)
        self.assertEqual(processor.process(-100), "Bin_0")
        self.assertEqual(processor.process(1000), "Bin_2")

    def test_transform_with_dataframe(self):
        """Test transform method with a DataFrame input."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        processor.fit(self.data)
        transformed = processor.transform(self.data)
        self.assertIn(f"{self.column_to_bin}_binned", transformed.columns)
        self.assertTrue(isinstance(transformed[f"{self.column_to_bin}_binned"].dtype, pd.CategoricalDtype))  # Ensure categorical dtype
        self.assertEqual(transformed[f"{self.column_to_bin}_binned"].cat.categories.tolist(), ["Bin_0", "Bin_1", "Bin_2"])  # Verify categories

    def test_transform_with_series(self):
        """Test transform method with a Series input."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        processor.fit(self.data)
        series_to_transform = pd.Series([1.0, 50.0, 100.0])
        transformed = processor.transform(series_to_transform)
        self.assertTrue(isinstance(transformed.dtype, pd.CategoricalDtype))  # Ensure categorical dtype
        self.assertEqual(transformed.cat.categories.tolist(), ["Bin_0", "Bin_1", "Bin_2"])  # Verify categories

    def test_transform_with_missing_column(self):
        """Test transform method raises error when column is missing in DataFrame."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        processor.fit(self.data)
        with self.assertRaisesRegex(ValueError, "not found in input DataFrame"):
            processor.transform(pd.DataFrame({'wrong_column': [1, 2, 3]}))

    def test_save_and_load_params(self):
        """Test saving and loading processor parameters."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        processor.fit(self.data)
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.save_params(temp_dir)
            loaded_processor = NumericalBinningProcessor.load_params(Path(temp_dir) / f"numerical_binning_processor_{self.column_to_bin}_params.json")
            self.assertTrue(loaded_processor.is_fitted)
            self.assertEqual(loaded_processor.column_name, processor.column_name)
            np.testing.assert_array_equal(loaded_processor.bin_edges_, processor.bin_edges_)

    def test_get_params(self):
        """Test get_params method returns correct parameters."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        processor.fit(self.data)
        params = processor.get_params()
        self.assertEqual(params["column_name"], self.column_to_bin)
        self.assertEqual(params["n_bins_requested"], 3)
        self.assertEqual(params["strategy"], 'equal-width')
        self.assertIsNotNone(params["bin_edges"])

    def test_initialization_custom(self):
        labels = ['Low', 'Med', 'High']
        processor = NumericalBinningProcessor(
            column_name="age", n_bins=3, strategy='equal-width', bin_labels=labels,
            output_column_name="age_group", handle_missing_value="MissingAge",
            handle_out_of_range="AgeOutOfRange"
        )
        self.assertEqual(processor.column_name, "age")
        self.assertEqual(processor.n_bins_requested, 3)
        self.assertEqual(processor.strategy, 'equal-width')
        self.assertEqual(processor.bin_labels_config, labels)
        self.assertEqual(processor.output_column_name, "age_group")
        self.assertEqual(processor.handle_missing_value, "MissingAge")
        self.assertEqual(processor.handle_out_of_range, "AgeOutOfRange")

    def test_fit_equal_width(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width', bin_labels=True)
        processor.fit(self.data)
        self.assertTrue(processor.is_fitted)
        self.assertIsNotNone(processor.bin_edges_)
        self.assertEqual(len(processor.bin_edges_), processor.n_bins_actual_ + 1)
        self.assertEqual(processor.n_bins_actual_, 3)

    def test_process_value(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width', bin_labels=True)
        processor.fit(self.data) 
        self.assertEqual(processor(50.0), 'Bin_1') 

    @patch('src.processing.numerical_binning_processor.logger')
    def test_fit_quantile_fallback(self, mock_logger):
        """Test quantile binning falls back to equal-width on failure."""
        # This data will cause a ValueError in qcut, triggering the fallback
        data_for_fallback = pd.DataFrame({self.column_to_bin: [1, 2, 3, 4, 100]})
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=4, strategy='quantile')
        
        # Mock qcut to simulate failure
        with patch('pandas.qcut', side_effect=ValueError("Bin edges must be unique")):
            processor.fit(data_for_fallback)
        
        # Check that the fallback warning was logged
        self.assertTrue(any("Falling back to equal-width" in call.args[0] for call in mock_logger.warning.call_args_list))
        # The public strategy remains, but the internal behavior changes
        self.assertEqual(processor.strategy, 'quantile') 
        self.assertTrue(processor.is_fitted)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
