# test/test_gensim_tokenize_processor.py
import unittest
import numpy as np
from gensim.models import KeyedVectors

# Assuming the processor is in a 'src' directory structure
from src.processing.gensim_tokenize_processor import FastTextEmbeddingProcessor

class TestFastTextEmbeddingProcessor(unittest.TestCase):
    def setUp(self):
        """Set up a mock KeyedVectors object and the processor."""
        # Create a mock KeyedVectors model for testing
        # This avoids needing a real FastText model file
        self.dim = 5  # Use a small dimension for testing
        self.kv = KeyedVectors(vector_size=self.dim)
        
        # Add some dummy words and their vectors
        self.vocab = {
            "hello": np.array([1.0] * self.dim),
            "world": np.array([2.0] * self.dim),
            "test": np.array([3.0] * self.dim),
        }
        self.kv.add_vectors(list(self.vocab.keys()), list(self.vocab.values()))

        # Instantiate the processor to be tested
        self.processor = FastTextEmbeddingProcessor(
            keyed_vectors=self.kv,
            max_length=10,
            pad_to_max_length=True
        )

    def test_basic_embedding_and_padding(self):
        """
        Tests if words are correctly embedded and the output is padded.
        """
        input_chunks = ["hello world"]
        processed_output = self.processor.process(input_chunks)

        # There should be one dictionary in the output list
        self.assertEqual(len(processed_output), 1)
        
        result = processed_output[0]
        
        # Check that the output has the correct keys
        self.assertIn("embeddings", result)
        self.assertIn("attention_mask", result)

        embeddings = result["embeddings"]
        mask = result["attention_mask"]

        # Check dimensions - should be (max_length, dim)
        self.assertEqual(len(embeddings), self.processor.max_length)
        self.assertEqual(len(embeddings[0]), self.dim)
        
        # Check embedding values for known words
        self.assertEqual(embeddings[0], self.vocab["hello"].tolist())
        self.assertEqual(embeddings[1], self.vocab["world"].tolist())
        
        # Check padding values
        self.assertEqual(embeddings[2], [0.0] * self.dim) # Padded embedding
        
        # Check attention mask
        expected_mask = [1, 1] + [0] * (self.processor.max_length - 2)
        self.assertEqual(mask, expected_mask)

    def test_unknown_words(self):
        """
        Tests if unknown words are mapped to zero vectors and mask is 0.
        """
        input_chunks = ["hello unknown word"]
        processed_output = self.processor.process(input_chunks)
        result = processed_output[0]

        embeddings = result["embeddings"]
        mask = result["attention_mask"]
        
        # 'hello' should be known
        self.assertEqual(embeddings[0], self.vocab["hello"].tolist())
        self.assertEqual(mask[0], 1)

        # 'unknown' and 'word' should be zero vectors
        self.assertEqual(embeddings[1], [0.0] * self.dim)
        self.assertEqual(mask[1], 0)
        self.assertEqual(embeddings[2], [0.0] * self.dim)
        self.assertEqual(mask[2], 0)
        
    def test_truncation(self):
        """
        Tests if input longer than max_length is correctly truncated.
        """
        # Create an input with 12 words
        long_sentence = " ".join(["hello"] * 12)
        input_chunks = [long_sentence]
        
        processed_output = self.processor.process(input_chunks)
        result = processed_output[0]
        
        embeddings = result["embeddings"]
        mask = result["attention_mask"]

        # The length of embeddings and mask should be exactly max_length
        self.assertEqual(len(embeddings), self.processor.max_length)
        self.assertEqual(len(mask), self.processor.max_length)
        
        # All tokens in the mask should be 1, as it was all real words before truncation
        self.assertEqual(sum(mask), self.processor.max_length)

    def test_multiple_chunks(self):
        """
        Tests if the processor handles a list with multiple text chunks.
        """
        input_chunks = ["hello world", "test"]
        processed_output = self.processor.process(input_chunks)

        # The output list should contain two dictionaries
        self.assertEqual(len(processed_output), 2)

        # Check the first chunk's processing
        result1 = processed_output[0]
        self.assertEqual(result1["embeddings"][0], self.vocab["hello"].tolist())
        self.assertEqual(result1["attention_mask"], [1, 1] + [0] * 8)

        # Check the second chunk's processing
        result2 = processed_output[1]
        self.assertEqual(result2["embeddings"][0], self.vocab["test"].tolist())
        self.assertEqual(result2["attention_mask"], [1] + [0] * 9)

    def test_no_padding(self):
        """
        Tests functionality when padding is disabled.
        """
        processor_no_pad = FastTextEmbeddingProcessor(
            keyed_vectors=self.kv,
            max_length=10,
            pad_to_max_length=False # Disable padding
        )
        
        input_chunks = ["hello world test"]
        processed_output = processor_no_pad.process(input_chunks)
        result = processed_output[0]
        
        embeddings = result["embeddings"]
        mask = result["attention_mask"]

        # Length should match the number of words, not max_length
        self.assertEqual(len(embeddings), 3)
        self.assertEqual(len(mask), 3)
        
        # The mask should be all 1s as there's no padding
        self.assertEqual(mask, [1, 1, 1])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)