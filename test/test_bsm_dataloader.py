# test/test_bsm_dataloader.py
import unittest
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from src.processing.bsm_dataloader import build_collate_batch
from src.processing.bsm_datasets import BSMDataset
from src.processing.processors import (
    TextNormalizationProcessor, HTMLNormalizerProcessor, EmojiRemoverProcessor,
    DialogueSplitterProcessor, DialogueChunkerProcessor, DummyTokenizer,
    Processor
)
from src.processing.categorical_label_processor import CategoricalLabelProcessor


# Define dummy processors for testing non-categorical pipelines.
class DummyProcessor(Processor):
    """A dummy processor that appends '_dummy' to the input."""
    def __init__(self):
        super().__init__()
        self.processor_name = "dummy_processor"
    def process(self, input_text: str):
        return input_text + "_dummy"

class DummyTokenizationProcessor(Processor):
    """A dummy tokenization processor that simulates tokenization by splitting text into chunks."""
    def __init__(self):
        super().__init__()
        self.processor_name = "dummy_tokenization_processor"

    def process(self, input_text):
        def tokenize(chunk):
            tokens = chunk.split()
            input_ids = [ord(t[0]) if t else 0 for t in tokens]  # Convert tokens to dummy integer IDs
            return {"input_ids": input_ids, "attention_mask": [1] * len(tokens)}

        if isinstance(input_text, list):
            return [tokenize(chunk) for chunk in input_text]
        else:
            return [tokenize(input_text)]


class TestBSMDataset(unittest.TestCase):
    def setUp(self):
        data = {
            "dialogue": [
                "[bom] Hello world! [eom] [bom] How are you? [eom]",
                "[bom] This is a test. [eom]"
            ],
            "label": ["cat", "dog"],
            "cat_var": ["red", "blue"]
        }
        self.df = pd.DataFrame(data)
        self.config = {
            "text_name": "dialogue",
            "label_name": "label",
            "full_field_list": ["dialogue", "label", "cat_var"]
        }
        self.dataset = BSMDataset(config=self.config, dataframe=self.df)
        dummy_pipeline = DummyProcessor() >> DummyTokenizationProcessor()
        self.dataset.add_pipeline("dialogue", dummy_pipeline)

    def test_load_dataframe(self):
        self.assertIsInstance(self.dataset.DataReader, pd.DataFrame)
        self.assertEqual(len(self.dataset), 2)
        self.assertEqual(self.dataset.full_field_list, ["dialogue", "label", "cat_var"])

    def test_add_pipeline(self):
        self.assertIn("dialogue", self.dataset.processor_pipelines)
        raw_text = self.df.iloc[0]["dialogue"]
        processed = self.dataset.processor_pipelines["dialogue"](raw_text)
        expected_tokens = raw_text.split()
        expected_ids = [ord(t[0]) if t else 0 for t in expected_tokens]
        expected = [{"input_ids": expected_ids, "attention_mask": [1] * len(expected_tokens)}]
        self.assertEqual(processed, expected)

    def test___getitem__(self):
        item = self.dataset[0]
        self.assertIn("dialogue_processed", item)
        expected_tokens = self.df.iloc[0]["dialogue"].split()
        expected_ids = [ord(t[0]) if t else 0 for t in expected_tokens]
        expected = [{"input_ids": expected_ids, "attention_mask": [1] * len(expected_tokens)}]
        self.assertEqual(item["dialogue_processed"], expected)

    def test_field_without_pipeline(self):
        item = self.dataset[1]
        self.assertIn("label", item)
        self.assertNotIn("label_processed", item)
        self.assertEqual(item["label"], self.df.iloc[1]["label"])

    def test_long_dialogue_chunking_and_tokenization(self):
        dummy_tokenizer = DummyTokenizer()
        pipeline = DialogueSplitterProcessor() >> DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=5) >> DummyTokenizationProcessor()
        self.dataset.add_pipeline("dialogue", pipeline)

        dialogue = (
            "[bom] a b c d [eom] "
            "[bom] e f [eom] "
            "[bom] g h i j [eom]"
        )
        df_long = pd.DataFrame({"dialogue": [dialogue], "label": ["cat"], "cat_var": ["red"]})
        dataset_long = BSMDataset(config=self.config, dataframe=df_long, processor_pipelines={"dialogue": pipeline})

        item = dataset_long[0]
        self.assertEqual(len(item["dialogue_processed"]), 3)
        self.assertEqual(item["dialogue_processed"][0]["attention_mask"], [1, 1, 1, 1])

    def test_mixed_chunk_counts_in_batch(self):
        dummy_pipeline = DummyProcessor() >> DummyTokenizationProcessor()
        self.dataset.add_pipeline("dialogue", dummy_pipeline)

        data = {
            "dialogue": [
                "[bom] Hello world! [eom] [bom] How are you? [eom]",  # 2 chunks
                "[bom] One chunk only [eom]"  # 1 chunk
            ],
            "label": ["cat", "dog"],
            "cat_var": ["red", "blue"]
        }
        df = pd.DataFrame(data)
        dataset = BSMDataset(config=self.config, dataframe=df, processor_pipelines={"dialogue": dummy_pipeline})
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=build_collate_batch())
        batch = next(iter(dataloader))
        self.assertIn("dialogue_processed_input_ids", batch)
        self.assertIn("dialogue_processed_attention_mask", batch)
        self.assertEqual(batch["dialogue_processed_input_ids"].shape[0], 2)
        self.assertEqual(batch["dialogue_processed_input_ids"].dim(), 3)  # batch, chunks, tokens

    def test_categorical_label_processor(self):
        processor = CategoricalLabelProcessor(initial_categories=["cat", "dog"], update_on_new=True)
        self.assertEqual(processor("cat"), 0)
        self.assertEqual(processor("dog"), 1)
        self.assertEqual(processor("bird"), 2)
        self.assertEqual(processor("bird"), 2)

        processor_no_update = CategoricalLabelProcessor(initial_categories=["cat", "dog"], update_on_new=False, unknown_label=-1)
        self.assertEqual(processor_no_update("bird"), -1)

    def test_composed_processor_names(self):
        pipeline = TextNormalizationProcessor() >> HTMLNormalizerProcessor() >> EmojiRemoverProcessor()
        expected_names = [
            "text_normalization_processor",
            "html_normalizer_processor",
            "emoji_remover_processor"
        ]
        self.assertEqual(pipeline.function_name_list, expected_names)

    def test_dataloader_integration(self):
        collate_fn = build_collate_batch()
        data = {
            "dialogue": [
                "[bom] Hello world! [eom]",
                "[bom] How are you? [eom]"
            ],
            "label": [1, 0],
            "cat_var": ["red", "blue"]
        }
        df = pd.DataFrame(data)
        config = {
            "text_name": "dialogue",
            "label_name": "label",
            "full_field_list": ["dialogue", "label", "cat_var"]
        }
        dummy_pipeline = DummyProcessor() >> DummyTokenizationProcessor()
        dataset = BSMDataset(config=config, dataframe=df, processor_pipelines={"dialogue": dummy_pipeline})
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(dataloader))

        self.assertIsInstance(batch, dict)
        self.assertIn("dialogue_processed_input_ids", batch)
        self.assertIn("dialogue_processed_attention_mask", batch)
        self.assertEqual(batch["dialogue_processed_input_ids"].shape[0], 2)
        self.assertEqual(batch["dialogue_processed_attention_mask"].shape[0], 2)

            
            
if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
