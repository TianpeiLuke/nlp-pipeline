import unittest
from typing import List
from src.processing.processors import (
    DialogueSplitterProcessor
)


class TestDialogueSplitterProcessor(unittest.TestCase):
    def setUp(self):
        self.default_splitter = DialogueSplitterProcessor()
        self.min_length_splitter = DialogueSplitterProcessor(min_length=5)

    def test_basic_split(self):
        input_text = "[bom]Hello[eom][bom]World[eom]"
        output = self.default_splitter.process(input_text)
        self.assertEqual(output, ["Hello", "World"])

    def test_extra_whitespace(self):
        input_text = "[bom]   Hello   [eom][bom]   [eom][bom]World[eom]"
        output = self.default_splitter.process(input_text)
        self.assertEqual(output, ["Hello", "World"])

    def test_empty_messages_filtered(self):
        input_text = "[bom]   [eom][bom]     [eom][bom]  valid  [eom]"
        output = self.default_splitter.process(input_text)
        self.assertEqual(output, ["valid"])

    def test_min_length_filtering(self):
        input_text = "[bom]hi[eom][bom]hello[eom][bom]greetings[eom]"
        output = self.min_length_splitter.process(input_text)
        self.assertEqual(output, ["hello", "greetings"])  # "hi" is too short

    def test_no_matches(self):
        input_text = "This text has no bom/eom structure."
        output = self.default_splitter.process(input_text)
        self.assertEqual(output, [])

    def test_mixed_valid_and_invalid(self):
        input_text = "[bom]short[eom][bom]  [eom][bom]   long enough   [eom]"
        splitter = DialogueSplitterProcessor(min_length=6)
        output = splitter.process(input_text)
        self.assertEqual(output, ["long enough"])

if __name__ == "__main__":
    unittest.main()
