import unittest
from transformers import AutoTokenizer
from src.processing.bert_tokenize_processor import TokenizationProcessor


class TestTokenizationProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.processor = TokenizationProcessor(
            tokenizer=self.tokenizer,
            add_special_tokens=True,
            max_length=8,
            truncation=True,
            padding="max_length",
            input_ids_key="input_ids",
            attention_mask_key="attention_mask",
        )

    def test_normal_input(self):
        input_chunks = ["Hello world", "How are you?"]
        output = self.processor.process(input_chunks)
        self.assertEqual(len(output), 2)
        self.assertTrue(all("input_ids" in x and "attention_mask" in x for x in output))
        self.assertTrue(all(len(x["input_ids"]) == 8 for x in output))

    def test_empty_input_chunk(self):
        input_chunks = [""]
        output = self.processor.process(input_chunks)
        self.assertEqual(len(output), 0)  # Now should be skipped

    def test_whitespace_input_chunk(self):
        input_chunks = ["   "]
        output = self.processor.process(input_chunks)
        self.assertEqual(len(output), 0)  # Now should be skipped

    def test_mixed_valid_and_empty(self):
        input_chunks = ["Test input", "", "Another one"]
        output = self.processor.process(input_chunks)
        self.assertEqual(len(output), 2)
        self.assertTrue(all(len(x["input_ids"]) == 8 for x in output))

    def test_chunk_list_with_empty_strings_only(self):
        input_chunks = ["", "   ", "\n"]
        output = self.processor.process(input_chunks)
        self.assertEqual(len(output), 0)

    def test_fully_empty_chunk_list(self):
        input_chunks = []
        output = self.processor.process(input_chunks)
        self.assertEqual(len(output), 0)


if __name__ == '__main__':
    unittest.main()
