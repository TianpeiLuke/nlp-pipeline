import unittest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import logging

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
            self.column_to_bin: [1.0] * 10 # This data will NOT cause qcut to raise ValueError with duplicates='drop'
                                          # It will trigger "fewer unique values" and "bin adjustment" warnings.
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

    def test_initialization_invalid_params(self):
        with self.assertRaisesRegex(ValueError, "column_name must be a non-empty string"):
            NumericalBinningProcessor(column_name="")
        with self.assertRaisesRegex(ValueError, "n_bins must be a positive integer"):
            NumericalBinningProcessor(column_name="test", n_bins=0)
        with self.assertRaisesRegex(ValueError, "strategy must be either 'quantile' or 'equal-width'"):
            NumericalBinningProcessor(column_name="test", strategy="invalid_strategy")
        with self.assertRaisesRegex(ValueError, "bin_labels must be a list of strings, boolean, or None"):
            NumericalBinningProcessor(column_name="test", bin_labels="not_a_list_or_bool")
        try:
            NumericalBinningProcessor(column_name="test", n_bins=3, bin_labels=['a','b'])
        except ValueError: 
            self.fail("ValueError for mismatched bin_labels length should be in fit, not init.")


    def test_fit_equal_width(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width', bin_labels=True)
        processor.fit(self.data)
        self.assertTrue(processor.is_fitted)
        self.assertIsNotNone(processor.bin_edges_)
        self.assertEqual(len(processor.bin_edges_), processor.n_bins_actual_ + 1)
        self.assertEqual(processor.n_bins_actual_, 3)
        self.assertEqual(processor.actual_labels_, ['Bin_0', 'Bin_1', 'Bin_2'])
        self.assertAlmostEqual(processor.min_fitted_value_, 1.0)
        self.assertAlmostEqual(processor.max_fitted_value_, 100.0)

    def test_fit_quantile(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=4, strategy='quantile', bin_labels=False)
        processor.fit(self.data) 
        self.assertTrue(processor.is_fitted)
        self.assertIsNotNone(processor.bin_edges_)
        self.assertTrue(1 <= processor.n_bins_actual_ <= 4)
        self.assertFalse(processor.actual_labels_) 

    def test_fit_quantile_behavior_with_few_uniques(self): # Renamed from test_fit_quantile_fallback
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=4, strategy='quantile')
        # self.data_qcut_fail has only 1 unique value.
        # Expect "fewer unique values", "Could not create valid bins", and "Number of bins ... adjusted" warnings.
        with self.assertLogs(level='WARNING') as log_capture: # Capture from root
            processor.fit(self.data_qcut_fail)
        
        # Filter for messages from the specific logger
        module_logs = [record.getMessage() for record in log_capture.records if record.name == self.PROCESSOR_MODULE_LOGGER_NAME]

        self.assertTrue(any("fewer unique values" in message for message in module_logs), "Missing 'fewer unique values' warning.")
        self.assertTrue(any("Could not create valid bins" in message for message in module_logs), "Missing 'Could not create valid bins' warning.")
        self.assertTrue(any("Number of bins for column 'value' was adjusted" in message for message in module_logs), "Missing 'Number of bins adjusted' warning.")
        
        # The "Quantile binning failed... Falling back" message should NOT be present for this data
        self.assertFalse(any("Quantile binning failed" in message and "Falling back" in message for message in module_logs),
                         "'Quantile binning failed... Falling back' log should not occur for this data.")
        
        self.assertEqual(processor.strategy, 'quantile') # Strategy itself doesn't change unless qcut fails with ValueError
        self.assertTrue(processor.is_fitted)
        self.assertEqual(processor.n_bins_actual_, 1) # Should default to 1 bin

    def test_fit_single_unique_value(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width')
        with self.assertLogs(level='WARNING') as log_capture: # Capture from root
            processor.fit(self.data_single_value)
        
        module_logs = [record.getMessage() for record in log_capture.records if record.name == self.PROCESSOR_MODULE_LOGGER_NAME]
        self.assertTrue(any("single unique value" in message and "Creating one bin" in message for message in module_logs))
        self.assertEqual(processor.n_bins_actual_, 1)
        self.assertEqual(len(processor.bin_edges_), 2) 
        self.assertEqual(processor.actual_labels_, ['Bin_0'])

    def test_fit_custom_labels(self):
        labels = ['GroupA', 'GroupB']
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=2, strategy='equal-width', bin_labels=labels)
        processor.fit(self.data) 
        self.assertEqual(processor.actual_labels_, labels)
        self.assertEqual(processor.n_bins_actual_, 2)


    def test_fit_custom_labels_mismatch_warning_and_behavior(self):
        processor = NumericalBinningProcessor(
            column_name=self.column_to_bin, 
            n_bins=5, 
            strategy='quantile', 
            bin_labels=['Label1', 'Label2'] 
        )
        with self.assertLogs(level='WARNING') as log_capture: # Capture from root
            processor.fit(self.data_few_unique) 
        
        module_logs = [record.getMessage() for record in log_capture.records if record.name == self.PROCESSOR_MODULE_LOGGER_NAME]
        
        label_mismatch_warning_present = any("Provided bin_labels length" in message for message in module_logs)
        bin_adjustment_warning_present = any("Number of bins for column" in message for message in module_logs)
        
        self.assertTrue(label_mismatch_warning_present or bin_adjustment_warning_present, 
                        "Expected a warning for either bin adjustment or label mismatch.")

        # Logic for checking actual_labels_ after potential warnings
        if processor.n_bins_actual_ == 2 and processor.actual_labels_ == ['Label1', 'Label2']:
            pass # Labels matched the adjusted number of bins
        else:
            # If labels didn't match n_bins_actual_, default labels should have been used
            self.assertEqual(processor.actual_labels_, [f"Bin_{i}" for i in range(processor.n_bins_actual_)])


    def test_process_value(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin, n_bins=3, strategy='equal-width', bin_labels=True)
        processor.fit(self.data) 
        
        self.assertEqual(processor.process(1.0), 'Bin_0')
        self.assertEqual(processor.process(34.0), 'Bin_0') 
        self.assertEqual(processor.process(34.00001), 'Bin_1')
        self.assertEqual(processor.process(67.0), 'Bin_1')
        self.assertEqual(processor.process(67.00001), 'Bin_2')
        self.assertEqual(processor.process(100.0), 'Bin_2')
        self.assertEqual(processor(50.0), 'Bin_1') 

    def test_process_nan_and_out_of_range(self):
        processor = NumericalBinningProcessor(
            column_name=self.column_to_bin, n_bins=2, strategy='equal-width',
            handle_missing_value="MISSING_VAL", handle_out_of_range="OOR_VAL"
        )
        # Fit on data [1..10]
        fit_data = self.data[self.data[self.column_to_bin].between(1, 10, inclusive='both')].copy()
        processor.fit(fit_data)
        # Expected: min_fitted=1, max_fitted=10. Edges for 2 bins: [1, 5.5, 10]

        self.assertEqual(processor.process(np.nan), "MISSING_VAL")
        self.assertEqual(processor.process(0.0), "OOR_VAL")   # Below min_fitted_value_
        self.assertEqual(processor.process(101.0), "OOR_VAL") # Above max_fitted_value_
        self.assertEqual(processor.process(3.0), "Bin_0") # Within range

        processor_boundary = NumericalBinningProcessor(
            column_name=self.column_to_bin, n_bins=2, strategy='equal-width', 
            handle_out_of_range="boundary_bins", handle_missing_value="as_is"
        )
        processor_boundary.fit(fit_data)
        self.assertEqual(processor_boundary.process(0.0), "Bin_0") # Should go to the first bin
        self.assertEqual(processor_boundary.process(101.0), "Bin_1")# Should go to the last bin
        self.assertIsNone(processor_boundary.process(np.nan)) # "as_is" for missing

    def test_transform_series(self):
        processor = NumericalBinningProcessor(
            column_name=self.column_to_bin, n_bins=2, strategy='equal-width', 
            bin_labels=['Low', 'High'], handle_missing_value="SpecialMissing",
            handle_out_of_range="BoundaryValueAssigned" 
        )
        processor.fit(self.data[self.data[self.column_to_bin].between(1, 10, inclusive='both')])
        
        series_to_transform = pd.Series([1.0, 3.0, 5.5, 6.0, 10.0, np.nan, 0.0, 100.0])
        transformed = processor.transform(series_to_transform)
        
        expected_values = ['Low', 'Low', 'Low', 'High', 'High', 'SpecialMissing', 'BoundaryValueAssigned', 'BoundaryValueAssigned']
        expected_pd_series = pd.Series(expected_values, dtype=str)
        
        pd.testing.assert_series_equal(transformed, expected_pd_series, check_dtype=False)

    def test_transform_dataframe(self):
        output_col = "value_group"
        processor = NumericalBinningProcessor(
            column_name=self.column_to_bin, n_bins=2, strategy='equal-width', 
            bin_labels=True, output_column_name=output_col,
            handle_missing_value="DF_MISSING", handle_out_of_range="DF_OOR"
        )
        processor.fit(self.data[self.data[self.column_to_bin].between(1, 10, inclusive='both')])
        
        df_to_transform = pd.DataFrame({
            self.column_to_bin: [1.0, 6.0, 10.0, np.nan, 0.0, 100.0],
            'other': ['a', 'b', 'c', 'd', 'e', 'f']
        })
        transformed_df = processor.transform(df_to_transform)

        self.assertIn(output_col, transformed_df.columns)
        self.assertIn(self.column_to_bin, transformed_df.columns)
        
        expected_bins = ['Bin_0', 'Bin_1', 'Bin_1', 'DF_MISSING', 'DF_OOR', 'DF_OOR']
        pd.testing.assert_series_equal(
            transformed_df[output_col], 
            pd.Series(expected_bins, name=output_col, dtype=str),
            check_dtype=False
        )

    def test_transform_dataframe_replace_original(self):
        processor = NumericalBinningProcessor(
            column_name=self.column_to_bin, n_bins=2, strategy='equal-width', 
            bin_labels=True, output_column_name=self.column_to_bin 
        )
        processor.fit(self.data[self.data[self.column_to_bin].between(1, 10, inclusive='both')])
        df_to_transform = pd.DataFrame({self.column_to_bin: [1.0, 6.0]})
        transformed_df = processor.transform(df_to_transform)
        expected_bins = ['Bin_0', 'Bin_1']
        pd.testing.assert_series_equal(
            transformed_df[self.column_to_bin],
            pd.Series(expected_bins, name=self.column_to_bin, dtype=str),
            check_dtype=False
        )

    def test_runtime_error_if_not_fitted(self):
        processor = NumericalBinningProcessor(column_name=self.column_to_bin)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            processor.process(5.0)
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            processor.transform(pd.Series([1.0,2.0,3.0]))
        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            processor.save_params(Path("."))

    def test_save_load_params_file_and_dict(self):
        processor = NumericalBinningProcessor(
            column_name=self.column_to_bin, n_bins=3, strategy='equal-width', 
            bin_labels=['X','Y','Z'], handle_missing_value="SAVED_MISSING",
            handle_out_of_range="SAVED_OOR"
        )
        processor.fit(self.data)
        original_params = processor.get_params()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            processor.save_params(tmpdir_path)
            
            expected_filename = tmpdir_path / f"{processor.processor_name}_{processor.column_name}_params.json"
            self.assertTrue(expected_filename.exists())

            loaded_processor_file = NumericalBinningProcessor.load_params(expected_filename)
            self.assertTrue(loaded_processor_file.is_fitted)
            self.assertEqual(original_params["column_name"], loaded_processor_file.column_name)
            self.assertEqual(original_params["n_bins_actual"], loaded_processor_file.n_bins_actual_)
            self.assertEqual(original_params["strategy"], loaded_processor_file.strategy)
            np.testing.assert_array_almost_equal(original_params["bin_edges"], loaded_processor_file.bin_edges_)
            self.assertEqual(original_params["actual_labels"], loaded_processor_file.actual_labels_)
            self.assertEqual(original_params["handle_missing_value"], loaded_processor_file.handle_missing_value)
            self.assertEqual(original_params["handle_out_of_range"], loaded_processor_file.handle_out_of_range)

        loaded_processor_dict = NumericalBinningProcessor.load_params(original_params)
        self.assertTrue(loaded_processor_dict.is_fitted)
        self.assertEqual(original_params["column_name"], loaded_processor_dict.column_name)
        np.testing.assert_array_almost_equal(original_params["bin_edges"], loaded_processor_dict.bin_edges_)
        self.assertEqual(original_params["actual_labels"], loaded_processor_dict.actual_labels_)

    def test_get_name_from_base(self):
        processor = NumericalBinningProcessor(column_name="test")
        self.assertEqual(processor.get_name(), "numerical_binning_processor")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
