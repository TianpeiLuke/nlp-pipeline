import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json


from src.processing.binning_processor import BinningProcessor


class TestBinningProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test data and common objects."""
        self.column_to_bin = 'category'
        self.label_column = 'label'

        self.test_data = pd.DataFrame({
            self.column_to_bin: ['A', 'A', 'A', 'B', 'B', 'C', 'D', 'D', 'E', 'F', np.nan],
            self.label_column:    [1,   1,   0,   1,   0,   1,   0,   0,   0,  -1,  1],
            'other_col': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
        })
        
        # Pre-computed risk tables for testing (based on filtered data for 'category')
        # A (2/3=0.666...), B (1/2=0.5), C (1/1=1.0), D (0/2=0.0), E (0/1=0.0)
        # Overall mean of (1,1,0,1,0,1,0,0,0) = 4/9
        self.risk_tables_for_category_col = {
            "bins": {
                "A": 2/3,
                "B": 0.5,
                "C": 1.0,
                "D": 0.0,
                "E": 0.0 
            },
            "default_bin": 4/9 
        }

    def test_initialization(self):
        """Test processor initialization."""
        # Test basic initialization
        processor = BinningProcessor(column_name=self.column_to_bin, label_name=self.label_column)
        self.assertEqual(processor.column_name, self.column_to_bin)
        self.assertEqual(processor.label_name, self.label_column)
        self.assertFalse(processor.is_fitted)
        self.assertEqual(processor.get_name(), 'binning_processor')
        
        # Test initialization with risk tables
        processor_with_tables = BinningProcessor(
            column_name=self.column_to_bin,
            label_name=self.label_column,
            risk_tables=self.risk_tables_for_category_col
        )
        self.assertTrue(processor_with_tables.is_fitted)
        self.assertEqual(processor_with_tables.risk_tables, self.risk_tables_for_category_col)

        # Test initialization error with empty column_name
        with self.assertRaisesRegex(ValueError, "column_name must be a non-empty string"):
            BinningProcessor(column_name="", label_name=self.label_column)

    def test_validation_risk_tables_structure(self):
        """Test input validation for the risk_tables structure."""
        with self.assertRaisesRegex(ValueError, "Risk tables must be a dictionary"):
            BinningProcessor(column_name="cat", label_name="lbl", risk_tables="not_a_dict")
        with self.assertRaisesRegex(ValueError, "Risk tables must contain 'bins' and 'default_bin' keys"):
            BinningProcessor(column_name="cat", label_name="lbl", risk_tables={"invalid_key": "format"})
        with self.assertRaisesRegex(ValueError, "Risk tables 'bins' must be a dictionary"):
            BinningProcessor(column_name="cat", label_name="lbl", risk_tables={"bins": "not_a_dict", "default_bin": 0.5})
        with self.assertRaisesRegex(ValueError, "Risk tables 'default_bin' must be a number"):
            BinningProcessor(column_name="cat", label_name="lbl", risk_tables={"bins": {}, "default_bin": "not_a_number"})

    def test_fit_and_risk_calculation(self):
        """Test fitting the processor and the calculated risk values."""
        processor = BinningProcessor(
            column_name=self.column_to_bin,
            label_name=self.label_column,
            smooth_factor=0.0, # No smoothing for direct comparison
            count_threshold=0  # No threshold for direct comparison
        )
        processor.fit(self.test_data)
        
        self.assertTrue(processor.is_fitted)
        self.assertIn("bins", processor.risk_tables)
        self.assertIn("default_bin", processor.risk_tables)
        
        self.assertAlmostEqual(processor.risk_tables["default_bin"], self.risk_tables_for_category_col["default_bin"], places=5)

        expected_bins = self.risk_tables_for_category_col["bins"]
        self.assertEqual(len(processor.risk_tables["bins"]), len(expected_bins)) # Ensure same number of keys
        for category_key, expected_risk in expected_bins.items():
            self.assertIn(category_key, processor.risk_tables["bins"])
            self.assertAlmostEqual(processor.risk_tables["bins"][category_key], expected_risk, places=5)
        
        self.assertNotIn("F", processor.risk_tables["bins"]) 
        self.assertNotIn("nan", processor.risk_tables["bins"])

    def test_fit_with_smoothing_and_threshold(self):
        """Test fitting with smoothing and count threshold."""
        processor = BinningProcessor(
            column_name=self.column_to_bin,
            label_name=self.label_column,
            smooth_factor=0.1,
            count_threshold=2 # Category E has count 1, D has count 2
        )
        processor.fit(self.test_data)
        self.assertTrue(processor.is_fitted)
        
        default_risk_val = processor.risk_tables["default_bin"]
        self.assertAlmostEqual(processor.risk_tables["bins"]["E"], default_risk_val, places=5,
                               msg="Category E (count 1) should get default risk due to threshold.")
        self.assertNotAlmostEqual(processor.risk_tables["bins"]["D"], default_risk_val, places=5,
                                  msg="Category D (count 2) should have its risk smoothed, not default.")
        self.assertNotAlmostEqual(processor.risk_tables["bins"]["A"], default_risk_val, places=5,
                                  msg="Category A (count 3) should have its risk smoothed, not default.")

    def test_fit_errors(self):
        """Test error conditions during fit."""
        processor_bad_label = BinningProcessor(column_name=self.column_to_bin, label_name="invalid_label_name")
        with self.assertRaisesRegex(ValueError, "Label variable 'invalid_label_name' not found"):
            processor_bad_label.fit(self.test_data)

        processor_bad_col = BinningProcessor(column_name="non_existent_column", label_name=self.label_column)
        with self.assertRaisesRegex(ValueError, "Column to bin 'non_existent_column' not found"):
            processor_bad_col.fit(self.test_data)

        empty_df_for_fit = pd.DataFrame({self.column_to_bin: ['X'], self.label_column: [-1]})
        processor_empty_data = BinningProcessor(column_name=self.column_to_bin, label_name=self.label_column)
        processor_empty_data.fit(empty_df_for_fit) # Should now handle empty filtered data
        self.assertTrue(processor_empty_data.is_fitted)
        self.assertEqual(processor_empty_data.risk_tables["bins"], {})
        self.assertEqual(processor_empty_data.risk_tables["default_bin"], 0.5) # Default for no valid labels


    def test_process_single_value(self):
        """Test processing single values (via __call__ or direct process)."""
        processor = BinningProcessor(
            column_name=self.column_to_bin,
            label_name=self.label_column,
            risk_tables=self.risk_tables_for_category_col
        )
        self.assertAlmostEqual(processor.process("A"), self.risk_tables_for_category_col["bins"]["A"], places=5)
        self.assertAlmostEqual(processor("B"), self.risk_tables_for_category_col["bins"]["B"], places=5) # Test __call__
        self.assertAlmostEqual(processor.process("UNKNOWN_CATEGORY"), self.risk_tables_for_category_col["default_bin"], places=5)
        self.assertAlmostEqual(processor.process(123), self.risk_tables_for_category_col["default_bin"], places=5)
        self.assertAlmostEqual(processor.process(np.nan), self.risk_tables_for_category_col["default_bin"], places=5)

    def test_transform_series_and_dataframe(self):
        """Test transforming pandas Series and DataFrame."""
        processor = BinningProcessor(
            column_name=self.column_to_bin,
            label_name=self.label_column,
            risk_tables=self.risk_tables_for_category_col
        )
        
        # Test Series
        test_series = pd.Series(['A', 'C', 'UNKNOWN', np.nan], name=self.column_to_bin)
        result_series = processor.transform(test_series)
        expected_s_values = [
            self.risk_tables_for_category_col["bins"]["A"],
            self.risk_tables_for_category_col["bins"]["C"],
            self.risk_tables_for_category_col["default_bin"],
            self.risk_tables_for_category_col["default_bin"]
        ]
        expected_pd_series = pd.Series(expected_s_values, name=self.column_to_bin)
        pd.testing.assert_series_equal(result_series, expected_pd_series, atol=1e-5)
        
        # Test DataFrame
        test_df_input = pd.DataFrame({
            self.column_to_bin: ['A', 'B', 'UNKNOWN', np.nan],
            'other_col': [10, 20, 30, 40]
        })
        result_df = processor.transform(test_df_input)
        
        expected_df_output = test_df_input.copy()
        expected_df_col_values = [
            self.risk_tables_for_category_col["bins"]["A"],
            self.risk_tables_for_category_col["bins"]["B"],
            self.risk_tables_for_category_col["default_bin"],
            self.risk_tables_for_category_col["default_bin"]
        ]
        expected_df_output[self.column_to_bin] = expected_df_col_values
        pd.testing.assert_frame_equal(result_df, expected_df_output, atol=1e-5)

        # Test transform with a column not present in DataFrame
        df_missing_column = pd.DataFrame({'another_col': ["val1", "val2"]})
        with self.assertRaisesRegex(ValueError, f"Column '{self.column_to_bin}' not found"):
            processor.transform(df_missing_column)


    def test_save_load_risk_tables(self):
        """Test saving and loading risk tables."""
        processor = BinningProcessor(
            column_name=self.column_to_bin,
            label_name=self.label_column,
            risk_tables=self.risk_tables_for_category_col
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            processor.save_risk_tables(tmpdir_path)
            
            # Updated filenames based on processor_name and column_name
            expected_pkl_filename = f"{processor.get_name()}_{processor.column_name}_risk_tables.pkl"
            expected_json_filename = f"{processor.get_name()}_{processor.column_name}_risk_tables.json"
            
            pkl_file_path = tmpdir_path / expected_pkl_filename
            json_file_path = tmpdir_path / expected_json_filename

            self.assertTrue(pkl_file_path.exists())
            self.assertTrue(json_file_path.exists())
            
            new_processor = BinningProcessor(column_name=self.column_to_bin, label_name=self.label_column)
            new_processor.load_risk_tables(pkl_file_path) # Load from the correctly named file
            
            self.assertTrue(new_processor.is_fitted)
            self.assertEqual(processor.risk_tables["default_bin"], new_processor.risk_tables["default_bin"])
            # Compare bins dictionaries
            self.assertDictEqual(processor.risk_tables["bins"], new_processor.risk_tables["bins"])

            # Verify JSON content
            with open(json_file_path, 'r') as f:
                loaded_json_data = json.load(f)
            self.assertAlmostEqual(loaded_json_data["default_bin"], self.risk_tables_for_category_col["default_bin"], places=5)
            self.assertEqual(len(loaded_json_data["bins"]), len(self.risk_tables_for_category_col["bins"]))


    def test_runtime_errors_if_not_fitted(self):
        """Test error handling for operations before fitting."""
        processor = BinningProcessor(column_name=self.column_to_bin, label_name=self.label_column)
        
        # For process()
        # Actual message: "BinningProcessor must be fitted or initialized with risk tables before processing."
        process_error_regex = r"BinningProcessor must be fitted or initialized with risk tables before processing\."
        with self.assertRaisesRegex(RuntimeError, process_error_regex):
            processor.process("A")
        
        # For transform()
        # Actual message: "BinningProcessor must be fitted or initialized with risk tables before transforming."
        transform_error_regex = r"BinningProcessor must be fitted or initialized with risk tables before transforming\."
        with self.assertRaisesRegex(RuntimeError, transform_error_regex):
            processor.transform(pd.Series(['A']))
        
        # For get_risk_tables()
        # Actual message: "BinningProcessor has not been fitted or initialized with risk tables."
        get_tables_error_regex = r"BinningProcessor has not been fitted or initialized with risk tables\."
        with self.assertRaisesRegex(RuntimeError, get_tables_error_regex):
            processor.get_risk_tables()
        
        # For save_risk_tables()
        # Actual message: "Cannot save risk tables before fitting or initialization with risk tables."
        save_tables_error_regex = r"Cannot save risk tables before fitting or initialization with risk tables\."
        with self.assertRaisesRegex(RuntimeError, save_tables_error_regex):
            processor.save_risk_tables(Path("dummy_path_for_test"))


    def test_smoothing_effect_logic(self):
        """Test the effect of smoothing factor more precisely."""
        data_for_smooth_test = pd.DataFrame({
            'my_feature': ['X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y'], 
            'my_target':  [1,   1,   0,   1,   1,   1,   0,   0]
        })
        # Overall mean for this data: (2 positive for X + 3 positive for Y) / (3 X's + 5 Y's) = 5/8 = 0.625

        proc_no_smooth = BinningProcessor(column_name='my_feature', label_name='my_target', smooth_factor=0.0, count_threshold=0)
        proc_no_smooth.fit(data_for_smooth_test)
        risk_X_no_smooth = proc_no_smooth.risk_tables["bins"]['X'] # Expected: 2/3
        risk_Y_no_smooth = proc_no_smooth.risk_tables["bins"]['Y'] # Expected: 3/5
        self.assertAlmostEqual(risk_X_no_smooth, 2/3)
        self.assertAlmostEqual(risk_Y_no_smooth, 3/5)

        # With smoothing (smooth_factor=1.0 means smooth_samples = N = 8)
        # Default_risk = 0.625
        # For X: count=3, risk_raw=2/3. smooth_risk_X = (3 * 2/3 + 8 * 0.625) / (3 + 8) = (2 + 5) / 11 = 7/11
        proc_full_smooth = BinningProcessor(column_name='my_feature', label_name='my_target', smooth_factor=1.0, count_threshold=0)
        proc_full_smooth.fit(data_for_smooth_test)
        risk_X_full_smooth = proc_full_smooth.risk_tables["bins"]['X']
        self.assertAlmostEqual(risk_X_full_smooth, 7/11, places=5)

        # For Y: count=5, risk_raw=3/5. smooth_risk_Y = (5 * 3/5 + 8 * 0.625) / (5 + 8) = (3 + 5) / 13 = 8/13
        risk_Y_full_smooth = proc_full_smooth.risk_tables["bins"]['Y']
        self.assertAlmostEqual(risk_Y_full_smooth, 8/13, places=5)

        default_overall_risk = proc_full_smooth.risk_tables["default_bin"] # Should be 0.625
        # Check that smoothed risk is between raw risk and default_risk (or equal if raw == default)
        self.assertTrue(
            min(risk_X_no_smooth, default_overall_risk) - 1e-9 <= risk_X_full_smooth <= max(risk_X_no_smooth, default_overall_risk) + 1e-9
        )
        self.assertTrue(
            min(risk_Y_no_smooth, default_overall_risk) - 1e-9 <= risk_Y_full_smooth <= max(risk_Y_no_smooth, default_overall_risk) + 1e-9
        )


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Keep exit=False for interactive environments
