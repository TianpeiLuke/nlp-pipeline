# test/test_numerical_binning_processor.py
import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import logging
import sys
import os

# Configure logging for tests to capture warnings
def setUpModule():
    logging.basicConfig(level=logging.WARNING, force=True)


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

    def test_fit_with_all_nan_data(self):
        """Test the fit method when the column contains only NaN values."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=5)
        with self.assertLogs(self.PROCESSOR_MODULE_LOGGER_NAME, level='WARNING') as log:
            processor.fit(self.data_all_nan_for_fit)
        
        self.assertIn("has no valid data", log.output[0])
        self.assertTrue(processor.is_fitted)
        self.assertEqual(processor.n_bins_actual_, 1)
        np.testing.assert_array_equal(processor.bin_edges_, np.array([-np.inf, np.inf]))

    def test_transform_with_interval_labels(self):
        """Test the transform method when bin_labels is False."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=2, strategy='equal-width', bin_labels=False)
        processor.fit(self.data)
        
        series_to_transform = pd.Series([1.0, 50.0, 100.0])
        transformed = processor.transform(series_to_transform)
        
        # Check that the output is a categorical series
        self.assertTrue(isinstance(transformed.dtype, pd.CategoricalDtype))
        
        # When bin_labels=False, the output should be integer codes for the bins.
        # Data range is 1-100. With 2 bins, split is at 50.5. [1, 50] -> bin 0, [100] -> bin 1.
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
        invalid_params = {"column_name": "value", "n_bins_requested": 5} # Missing strategy, bin_edges, etc.
        with self.assertRaisesRegex(ValueError, "are missing required keys"):
            NumericalBinningProcessor.load_params(invalid_params)

    def test_process_with_nan_as_is(self):
        """Test the process method with handle_missing_value set to 'as_is'."""
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, handle_missing_value="as_is")
        processor.fit(self.data)
        self.assertIsNone(processor.process(np.nan))

    # --- Other existing tests remain unchanged ---
    
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

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
