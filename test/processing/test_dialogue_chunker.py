import unittest
from typing import List
from src.processing.bsm_processor import (
    DialogueChunkerProcessor
)
from transformers import AutoTokenizer


class TestDialogueChunkerProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.chunker = DialogueChunkerProcessor(tokenizer=self.tokenizer, max_tokens=10, truncate=True, max_total_chunks=3)

    def test_basic_chunking(self):
        messages = ["Hello", "How are you?", "I am fine", "Thanks"]
        chunks = self.chunker.process(messages)
        self.assertIsInstance(chunks, list)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))
        self.assertGreater(len(chunks), 0)

    def test_empty_messages(self):
        messages = []
        chunks = self.chunker.process(messages)
        self.assertEqual(chunks, ["."])

    def test_all_empty_messages(self):
        messages = ["   ", ""]
        chunks = self.chunker.process(messages)
        self.assertEqual(chunks, ["."])

    def test_truncation_limit(self):
        long_messages = ["This is a long message." for _ in range(20)]
        chunks = self.chunker.process(long_messages)
        self.assertLessEqual(len(chunks), self.chunker.max_total_chunks)

    def test_chunk_content(self):
        messages = ["One", "Two", "Three"]
        chunks = self.chunker.process(messages)
        self.assertTrue(all(isinstance(chunk, str) and chunk.strip() for chunk in chunks))

if __name__ == "__main__":
    unittest.main()
