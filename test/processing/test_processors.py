import unittest
# Import the actual processors to be tested
from src.processing.processors import (
    TextNormalizationProcessor, HTMLNormalizerProcessor, EmojiRemoverProcessor,
    DialogueSplitterProcessor, DialogueChunkerProcessor,
    Processor
)
from src.processing.bert_tokenize_processor import TokenizationProcessor
from src.processing.categorical_label_processor import CategoricalLabelProcessor


# --- Dummy classes for testing ---

class DummyTokenizer:
    """
    A dummy tokenizer that mimics Hugging Face tokenizers for testing.
    It splits text by space and is callable.
    """
    def encode(self, text, add_special_tokens=False):
        """Simulates encoding text into tokens."""
        if not text:
            return []
        return text.split()

    def __call__(self, text, **kwargs):
        """
        Makes the tokenizer callable, returning a dictionary of tokens.
        Accepts and ignores extra keyword arguments like max_length, padding, etc.
        """
        tokens = self.encode(text)
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}

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


# --- Unit Tests for Processors ---
class TestProcessors(unittest.TestCase):
    def test_text_normalization_processor(self):
        processor = TextNormalizationProcessor()
        input_text = "   HeLLo    WOrld!   "
        expected = "hello world!"
        self.assertEqual(processor(input_text), expected)

    def test_html_normalizer_processor(self):
        processor = HTMLNormalizerProcessor()
        input_html = "<html><body><p>Hello <b>World!</b></p></body></html>"
        expected = "Hello World!"
        self.assertEqual(processor(input_html), expected)
    
    def test_emoji_remover_processor(self):
        processor = EmojiRemoverProcessor()
        input_text = "Hello 😊! How are you? 🚀"
        expected = "Hello ! How are you? "
        self.assertEqual(processor(input_text).strip(), expected.strip())

    def test_dialogue_splitter_processor(self):
        processor = DialogueSplitterProcessor()
        dialogue = (
            "[bom] Message one content. [eom] Some extra text "
            "[bom]  Message two content here.  [eom]"
        )
        expected = ["Message one content.", "Message two content here."]
        self.assertEqual(processor(dialogue), expected)

    def test_dialogue_chunker_processor_with_dummy_tokenizer(self):
        dummy_tokenizer = DummyTokenizer()
        chunker = DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=5)
        messages = ["a b", "c d e", "f", "g h i j", "k"]
        expected = ["a b c d e", "f g h i j", "k"]
        result = chunker(messages)
        self.assertEqual(result, expected)
        for chunk in result:
            tokens = dummy_tokenizer.encode(chunk, add_special_tokens=False)
            self.assertLessEqual(len(tokens), 5)
            
    def test_dialogue_chunker_handles_empty_inputs(self):
        """
        Tests that the chunker returns ['.'] for empty/whitespace-only inputs,
        which reflects the current implementation.
        """
        dummy_tokenizer = DummyTokenizer()
        chunker = DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=5)
        
        # Test with a completely empty list of messages
        result_empty = chunker([])
        self.assertEqual(result_empty, ['.'])
        
        # Test with a list containing only empty or whitespace strings
        result_whitespace = chunker(["", "   ", "\t"])
        self.assertEqual(result_whitespace, ['.'])

    def test_tokenization_processor_with_dummy_tokenizer(self):
        dummy_tokenizer = DummyTokenizer()
        processor = TokenizationProcessor(tokenizer=dummy_tokenizer, add_special_tokens=False)
        chunks = ["hello world", "this is a test"]
        expected = [
            {"input_ids": ["hello", "world"], "attention_mask": [1, 1]},
            {"input_ids": ["this", "is", "a", "test"], "attention_mask": [1, 1, 1, 1]}
        ]
        result = processor(chunks)
        self.assertEqual(result, expected)

    def test_full_pipeline(self):
        dummy_tokenizer = DummyTokenizer()
        full_processor = (
            TextNormalizationProcessor() 
            >> DialogueSplitterProcessor() 
            >> DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=10)
        )
        dialogue = (
            "[bom]  Hello THERE, How are you?  [eom] "
            "[bom] I'M FINE, THANKS! [eom]"
        )
        expected = ["hello there, how are you? i'm fine, thanks!"]
        self.assertEqual(full_processor(dialogue), expected)
 
    def test_chunking_long_dialogue(self):
        dummy_tokenizer = DummyTokenizer()
        # Compose pipeline with max_tokens=5 to force chunking.
        full_processor = (
            TextNormalizationProcessor() 
            >> DialogueSplitterProcessor() 
            >> DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=5)
            >> TokenizationProcessor(dummy_tokenizer, add_special_tokens=False)
        )
        dialogue = (
            "[bom] a b c d [eom] "  # 4 tokens
            "[bom] e f [eom] "       # 2 tokens
            "[bom] g h i j [eom]"     # 4 tokens
        )
        # Expected behavior: chunker combines messages until max_tokens is reached.
        expected = [
            {"input_ids": ["a", "b", "c", "d"], "attention_mask": [1,1,1,1]},
            {"input_ids": ["e", "f"], "attention_mask": [1,1]},
            {"input_ids": ["g", "h", "i", "j"], "attention_mask": [1,1,1,1]}
        ]
        self.assertEqual(full_processor(dialogue), expected)
    
    def test_full_pipeline_with_html_and_emoji_removal(self):
        dummy_tokenizer = DummyTokenizer()
        full_processor = (
            HTMLNormalizerProcessor() 
            >> EmojiRemoverProcessor() 
            >> TextNormalizationProcessor() 
            >> DialogueSplitterProcessor() 
            >> DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=20)
            >> TokenizationProcessor(dummy_tokenizer, add_special_tokens=False)
        )
        dialogue = (
            "[bom] <p>Hi 😊 there!</p> [eom] "
            "[bom] <div>How are you? 🚀</div> [eom]"
        )
        expected = [{"input_ids": ["hi", "there!", "how", "are", "you?"], "attention_mask": [1, 1, 1, 1, 1]}]
        self.assertEqual(full_processor(dialogue), expected)

    def test_empty_dialogue(self):
        dummy_tokenizer = DummyTokenizer()
        full_processor = (
            HTMLNormalizerProcessor() 
            >> EmojiRemoverProcessor() 
            >> TextNormalizationProcessor() 
            >> DialogueSplitterProcessor() 
            >> DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=20)
            >> TokenizationProcessor(dummy_tokenizer, add_special_tokens=False)
        )
        dialogue = ""
        # The chunker returns ['.'], which the tokenizer processes.
        expected = [{"input_ids": ["."], "attention_mask": [1]}]
        self.assertEqual(full_processor(dialogue), expected)

    def test_single_long_message_chunk_boundary(self):
        dummy_tokenizer = DummyTokenizer()
        message = "one two three four five"
        full_processor = (
            TextNormalizationProcessor() 
            >> DialogueSplitterProcessor() 
            >> DialogueChunkerProcessor(tokenizer=dummy_tokenizer, max_tokens=5)
        )
        dialogue = f"[bom] {message} [eom]"
        expected = [message]
        self.assertEqual(full_processor(dialogue), expected)

    def test_composed_processor_names(self):
        # Create a composed pipeline using multiple processors.
        pipeline = TextNormalizationProcessor() >> HTMLNormalizerProcessor() >> EmojiRemoverProcessor()
        expected_names = [
            "text_normalization_processor",
            "html_normalizer_processor",
            "emoji_remover_processor"
        ]
        # Check that the composed pipeline's function_name_list matches the expected list.
        self.assertEqual(pipeline.function_name_list, expected_names)

        

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)