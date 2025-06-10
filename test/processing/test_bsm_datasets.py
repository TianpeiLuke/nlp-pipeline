# test/test_bsm_datasets.py
import unittest
import pandas as pd
import numpy as np
import torch
import tempfile
import os
import shutil
from src.processing.bsm_datasets import BSMDataset
from src.processing.processors import Processor

# Define dummy pipeline processors for testing in this file.
class DummyProcessor(Processor):
    """A dummy processor that appends '_dummy' to the input."""
    def __init__(self):
        super().__init__()
        self.processor_name = "dummy_processor"
    def process(self, input_text: str):
        return input_text + "_dummy"

class DummyTokenizationProcessor(Processor):
    """A dummy tokenization processor that simulates tokenization by splitting text."""
    def __init__(self):
        super().__init__()
        self.processor_name = "dummy_tokenization_processor"
    def process(self, input_text: str):
        tokens = input_text.split()
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

# --- Unit Tests for BSMDataset ---
class TestBSMDataset(unittest.TestCase):
    def setUp(self):
        """Set up a dummy dataframe and temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a dummy dataframe 
        self.data = {
            "dialogue": [
                "[bom] Hello world! [eom] [bom] How are you? [eom]",
                "[bom] This is a test. [eom]",
                None # To test missing value handling
            ],
            "label": ["cat", "dog", "cat"],
            "cat_var": ["red", "blue", ""],
            "ttm_conc_count": ["0.0", "2.0", "NaN"],
            "net_conc_amt": ["129.0674", "793.7500", "-1.0"]
        }
        self.df = pd.DataFrame(self.data)
        
        # Base configuration
        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "full_field_list": list(self.df.columns),
            "cat_field_list": ["label", "cat_var", "dialogue"],
            "tab_field_list": ["ttm_conc_count", "net_conc_amt"],
            "need_language_detect": False
        }
        
        # Default dataset instance
        self.dataset = BSMDataset(config=self.config, dataframe=self.df)
        dummy_pipeline = DummyProcessor() >> DummyTokenizationProcessor()
        self.dataset.add_pipeline("dialogue", dummy_pipeline)

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)

    def test_load_dataframe(self):
        """Tests initialization from a pandas DataFrame."""
        self.assertIsInstance(self.dataset.DataReader, pd.DataFrame)
        self.assertEqual(len(self.dataset), 3)
        self.assertEqual(set(self.dataset.full_field_list), set(self.df.columns))
    
    def test_load_from_csv_file(self):
        """Tests initialization from a CSV file."""
        csv_path = os.path.join(self.temp_dir, "test_data.csv")
        self.df.to_csv(csv_path, index=False)

        file_dataset = BSMDataset(config=self.config, file_dir=self.temp_dir, filename="test_data.csv")
        self.assertEqual(len(file_dataset), 3)
        self.assertEqual(file_dataset.DataReader.shape, self.df.shape)
        # Check if a column is correctly read
        self.assertEqual(file_dataset.DataReader.iloc[1]["label"], "dog")

    def test_initialization_error(self):
        """Tests that a TypeError is raised if no data source is provided."""
        with self.assertRaises(TypeError):
            BSMDataset(config=self.config)

    def test_fill_missing_value(self):
        """Tests the fill_missing_value method."""
        # Create a df with NaNs to test imputation
        data_with_nan = {
            "dialogue": ["text1", None],
            "label": [1, None],
            "cat_var": ["A", "B"],
            "ttm_conc_count": [10.5, np.nan]
        }
        df_nan = pd.DataFrame(data_with_nan)
        dataset_nan = BSMDataset(config=self.config, dataframe=df_nan)
        
        # Run imputation
        dataset_nan.fill_missing_value()
        
        # Check results
        imputed_df = dataset_nan.DataReader
        self.assertEqual(imputed_df.loc[1, 'label'], 0) # Fills with 0 for label
        self.assertEqual(imputed_df.loc[1, 'ttm_conc_count'], -1.0) # Fills with -1.0 for numeric
        # Corrected assertion: .astype(str) converts a None object to the string 'None'.
        self.assertEqual(imputed_df.loc[1, 'dialogue'], 'None') 

    def test_dynamic_setters(self):
        """Tests the dynamic setter methods for config attributes."""
        self.dataset.set_label_field_name("new_label")
        self.assertEqual(self.dataset.label_name, "new_label")

        self.dataset.set_text_field_name("new_text")
        self.assertEqual(self.dataset.text_name, "new_text")
        
        self.dataset.set_cat_field_list(["new_cat"])
        self.assertEqual(self.dataset.cat_field_list, ["new_cat"])

        with self.assertRaises(TypeError):
            self.dataset.set_full_field_list("not a list")

    def test_add_invalid_pipeline(self):
        """Tests that adding a non-processor object raises a TypeError."""
        with self.assertRaises(TypeError):
            self.dataset.add_pipeline("dialogue", "not a processor")

    def test_getitem_with_tensor_index(self):
        """Tests fetching an item using a torch.Tensor as the index."""
        item = self.dataset[torch.tensor(1)]
        self.assertIn("dialogue_processed", item)
        self.assertEqual(item['label'], 'dog')

    def test_numerical_fields_conversion(self):
        """Check that numerical fields are correctly converted to numbers."""
        row0 = self.dataset.DataReader.iloc[0]
        self.assertTrue(np.issubdtype(type(row0["ttm_conc_count"]), np.floating))
        self.assertTrue(np.issubdtype(type(row0["net_conc_amt"]), np.floating))
        self.assertEqual(row0["ttm_conc_count"], 0.0)
        self.assertEqual(row0["net_conc_amt"], 129.0674)

    def test_add_pipeline(self):
        """Tests the add_pipeline method and processor composition."""
        self.assertIn("dialogue", self.dataset.processor_pipelines)
        raw_text = self.df.iloc[0]["dialogue"]
        processed = self.dataset.processor_pipelines["dialogue"](raw_text)
        expected_raw = raw_text + "_dummy"
        expected = {"input_ids": expected_raw.split(),
                    "attention_mask": [1] * len(expected_raw.split())}
        self.assertEqual(processed, expected)
    
    def test___getitem__(self):
        """Tests fetching and processing a single item."""
        item = self.dataset[0]
        self.assertIn("dialogue_processed", item)
        expected_str = self.df.iloc[0]["dialogue"] + "_dummy"
        expected_tokens = expected_str.split()
        expected = {"input_ids": expected_tokens, "attention_mask": [1] * len(expected_tokens)}
        self.assertEqual(item["dialogue_processed"], expected)
        # Ensure the original text field is deleted
        self.assertNotIn("dialogue", item)

    def test_field_without_pipeline(self):
        """Check that fields with no pipeline remain unchanged."""
        item = self.dataset[1]
        self.assertIn("label", item)
        self.assertNotIn("label_processed", item)
        self.assertEqual(item["label"], self.df.iloc[1]["label"])


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
