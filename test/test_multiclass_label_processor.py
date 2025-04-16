import unittest
import torch
from typing import List
from transformers import AutoTokenizer  # Assuming you have this
from ..src.processing.processors import Processor  # Import your Processor base class
from ..src.processing.multiclass_label_processor import MultiClassLabelProcessor  # Import the class to be tested



class TestMultiClassLabelProcessor(unittest.TestCase):

    def test_single_string_label(self):
        processor = MultiClassLabelProcessor()
        label = "cat"
        output = processor.process(label)
        self.assertTrue(torch.equal(output, torch.tensor([0])))

    def test_single_int_label(self):
        processor = MultiClassLabelProcessor()
        label = 1
        output = processor.process(label)
        self.assertTrue(torch.equal(output, torch.tensor([0])))

    def test_single_float_label(self):
        processor = MultiClassLabelProcessor()
        label = 2.5
        output = processor.process(label)
        self.assertTrue(torch.equal(output, torch.tensor([0])))

    def test_label_list(self):
        processor = MultiClassLabelProcessor()
        labels = ["dog", "cat", "bird"]
        output = processor.process(labels)
        expected = torch.tensor([0, 1, 2])
        self.assertTrue(torch.equal(output, expected))

    def test_dynamic_label_mapping(self):
        processor = MultiClassLabelProcessor()
        processor.process("A")
        processor.process("B")
        processor.process("C")
        self.assertEqual(processor.label_to_id, {"A": 0, "B": 1, "C": 2})
        self.assertEqual(processor.id_to_label, ["A", "B", "C"])

    def test_one_hot_encoding(self):
        processor = MultiClassLabelProcessor(one_hot=True)
        processor.process(["apple", "banana"])
        output = processor.process("banana")
        expected = torch.tensor([[0.0, 1.0]])
        self.assertTrue(torch.allclose(output, expected))

    def test_label_list_with_numbers(self):
        processor = MultiClassLabelProcessor()
        labels = [0, 1, 2]
        output = processor.process(labels)
        expected = torch.tensor([0, 1, 2])
        self.assertTrue(torch.equal(output, expected))

    def test_consistent_mapping(self):
        processor = MultiClassLabelProcessor()
        processor.process(["A", "B"])
        out1 = processor.process("A")
        out2 = processor.process("B")
        self.assertEqual(out1.item(), 0)
        self.assertEqual(out2.item(), 1)

    def test_reuse_with_defined_label_list(self):
        processor = MultiClassLabelProcessor(label_list=["red", "green", "blue"])
        output = processor.process("green")
        self.assertTrue(torch.equal(output, torch.tensor([1])))

    def test_unseen_label_extends_mapping(self):
        processor = MultiClassLabelProcessor()
        _ = processor.process("x")
        _ = processor.process("y")
        _ = processor.process("z")
        output = processor.process("new")
        self.assertEqual(output.item(), 3)
        self.assertEqual(processor.label_to_id["new"], 3)


if __name__ == "__main__":
    unittest.main()
