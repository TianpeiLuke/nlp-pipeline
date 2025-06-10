# test/test_bsm_datasets.py
import unittest
import pandas as pd
import numpy as np
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
        # Create a dummy dataframe with 'dialogue', 'label', and 'cat_var' fields.
        data = {
            "dialogue": [
                "[bom] Hello world! [eom] [bom] How are you? [eom]",
                "[bom] This is a test. [eom]"
            ],
            "label": ["cat", "dog"],
            "cat_var": ["red", "blue"],
            # Numerical fields as strings in the CSV
            "ttm_conc_count": ["0.0", "2.0"],
            "net_conc_amt": ["129.0674", "793.7500"]
        }
        self.df = pd.DataFrame(data)
        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "full_field_list": list(self.df.columns),
            "cat_field_list": list(set(self.df.columns) - {"ttm_conc_count", "net_conc_amt"}),
            "tab_field_list": ["ttm_conc_count", "net_conc_amt"],
            "need_language_detect": False
        }
        self.dataset = BSMDataset(config=self.config, dataframe=self.df)
        # Compose a dummy pipeline for dialogue: DummyProcessor then DummyTokenizationProcessor.
        dummy_pipeline = DummyProcessor() >> DummyTokenizationProcessor()
        self.dataset.add_pipeline("dialogue", dummy_pipeline)

    def test_load_dataframe(self):
        self.assertIsInstance(self.dataset.DataReader, pd.DataFrame)
        self.assertEqual(len(self.dataset), 2)
        self.assertEqual(set(self.dataset.full_field_list), set(self.df.columns))
    
    def test_numerical_fields_conversion(self):
        # Check that numerical fields in tab_field_list are converted to numbers.
        # Access the processed data through the instance, not by re-calling the method
        row0 = self.dataset.DataReader.iloc[0]
        # Instead of checking with isinstance(..., float), use np.issubdtype.
        self.assertTrue(np.issubdtype(type(row0["ttm_conc_count"]), np.floating))
        self.assertTrue(np.issubdtype(type(row0["net_conc_amt"]), np.floating))
        self.assertEqual(row0["ttm_conc_count"], 0.0)
        self.assertEqual(row0["net_conc_amt"], 129.0674)

    def test_add_pipeline(self):
        self.assertIn("dialogue", self.dataset.processor_pipelines)
        raw_text = self.df.iloc[0]["dialogue"]
        processed = self.dataset.processor_pipelines["dialogue"](raw_text)
        expected_raw = raw_text + "_dummy"
        expected = {"input_ids": expected_raw.split(),
                    "attention_mask": [1] * len(expected_raw.split())}
        self.assertEqual(processed, expected)
    
    def test___getitem__(self):
        item = self.dataset[0]
        self.assertIn("dialogue_processed", item)
        expected_str = self.df.iloc[0]["dialogue"] + "_dummy"
        expected_tokens = expected_str.split()
        expected = {"input_ids": expected_tokens, "attention_mask": [1] * len(expected_tokens)}
        self.assertEqual(item["dialogue_processed"], expected)
        # Ensure the original text field is deleted
        self.assertNotIn("dialogue", item)

    def test_field_without_pipeline(self):
        # Check that fields with no pipeline remain unchanged.
        item = self.dataset[1]
        self.assertIn("label", item)
        self.assertNotIn("label_processed", item)
        self.assertEqual(item["label"], self.df.iloc[1]["label"])


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
