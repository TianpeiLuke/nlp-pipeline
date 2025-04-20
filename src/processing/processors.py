import re

from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod
from bs4 import BeautifulSoup
from .constants import *



import warnings
warnings.filterwarnings("ignore") 

# --- Base Processor Classes ---
class Processor(ABC):
    processor_name: str
    function_name_list: List[str]
    
    def __init__(self):
        self.processor_name = 'processor'
        self.function_name_list = []
    
    def get_name(self) -> str:
        return self.processor_name
    
    def __call__(self, input_text):
        return self.process(input_text)
    
    @abstractmethod
    def process(self, input_text):
        pass
    
    # Use the >> operator to compose processors.
    def __rshift__(self, other):
        # If self is already a ComposedProcessor, we merge its processors with 'other'
        if isinstance(self, ComposedProcessor):
            return ComposedProcessor(self.processors + [other])
        return ComposedProcessor([self, other])


class ComposedProcessor(Processor):
    def __init__(self, processors: List[Processor]):
        super().__init__()
        self.processors = processors
        # Set function_name_list to a list of the names of each processor.
        self.function_name_list = [p.get_name() for p in processors]
    
    def process(self, input_text):
        for processor in self.processors:
            input_text = processor(input_text)
        return input_text
    
#=====================================================================================
class IdentityProcessor(Processor):
    '''
    An identity processor return a copy of input message itself
    '''
    def __init__(self):
        self.processor_name = 'identity'
    
    
    def process(self, x):
        return x    
    

#=====================================================================================
# Processor 1: Text Normalization
class TextNormalizationProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "text_normalization_processor"

    def process(self, input_text: str):
        # Basic normalization: trim and lowercase.
        normalized = input_text.strip().lower()
        # Collapse multiple spaces into one.
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

    
# Processor 1: Text Normalization
class TextUpperProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "text_upper_processor"

    def process(self, input_text: str):
        # Basic normalization: trim and lowercase.
        normalized = input_text.strip().upper()
        # Collapse multiple spaces into one.
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized

#=====================================================================================
# Processor 2: Dialogue Splitting
class DialogueSplitterProcessor(Processor):
    def __init__(self, min_length: int = 1):
        """
        Args:
            min_length: Minimum number of non-whitespace characters required to keep a message.
        """
        super().__init__()
        self.processor_name = "dialogue_splitter_processor"
        self.min_length = min_length

    def process(self, input_text: str):
        """
        Splits the dialogue into individual messages based on [bom] and [eom] delimiters.
        Returns:
            List of message strings.
        """
        pattern = r'\[bom\](.*?)\[eom\]'
        raw_messages = re.findall(pattern, input_text, flags=re.DOTALL)

        # Strip whitespace and filter out short/empty messages
        messages = [
            msg.strip()
            for msg in raw_messages
            if msg.strip() and len(msg.strip()) >= self.min_length
        ]

        return messages

    
#=====================================================================================
# Processor 3: Dialogue Chunker
class DialogueChunkerProcessor(Processor):
    def __init__(self, 
                 tokenizer, 
                 max_tokens=512, 
                 truncate: bool = False,  # Added truncate parameter
                 max_total_chunks: Optional[int] = 5
                ):
        """
        Args:
            tokenizer: A Hugging Face AutoTokenizer instance.
            max_tokens: Maximum token count per chunk.
        """
        super().__init__()
        self.processor_name = "dialogue_chunker_processor"
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.max_total_chunks = max_total_chunks
        self.truncate = truncate

    def process(self, messages: List[str]):
        """
        Chunks a list of messages into groups such that each chunk's token count (without special tokens)
        does not exceed the max_tokens limit.
        
        Returns:
            List of dialogue chunks (each chunk is a concatenated string of messages).
        """
        chunks = []
        current_chunk = []
        current_tokens = 0
        num_chunks = 0  # Track the number of chunks created

        for msg in messages:
            # Count tokens using the HF AutoTokenizer; avoid adding special tokens here
            token_count = len(self.tokenizer.encode(msg, add_special_tokens=False))
            # If adding this message would exceed limit, save current chunk and start a new one.
            if current_tokens + token_count > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk).strip())
                    num_chunks += 1 # Increment chunk count
                    if self.max_total_chunks is not None and self.truncate and num_chunks >= self.max_total_chunks:
                        break  # Stop if max_total_chunks is reached
                current_chunk = [msg]
                current_tokens = token_count
            else:
                current_chunk.append(msg)
                current_tokens += token_count

            if self.max_total_chunks is not None and self.truncate and num_chunks >= self.max_total_chunks:
                break # Stop if max_total_chunks is reached

        if current_chunk and (not self.truncate or num_chunks < self.max_total_chunks):
            chunks.append(" ".join(current_chunk).strip())
            num_chunks += 1

        # Ensure at least one non-empty chunk exists
        if not chunks:
            chunks = ["."]
        elif all(not chunk.strip() for chunk in chunks):
            chunks = ["."]

        return chunks
    
    
#====================================================================================
# --- Processor 4: Emoji Remover ---
class EmojiRemoverProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = 'emoji_remover_processor'
        # Define a regex pattern that captures common emoji ranges.
        self.emoji_pattern = re.compile(
            "["                                     
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"  # dingbats
            u"\U000024C2-\U0001F251" 
            "]+", 
            flags=re.UNICODE
        )

    def process(self, input_text: str) -> str:
        """
        Removes emojis from the input text.
        """
        return self.emoji_pattern.sub('', input_text)

    
#=====================================================================================   
# --- Processor 2: HTML Normalization ---
class HTMLNormalizerProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.processor_name = "html_normalizer_processor"
    
    def process(self, input_text: str) -> str:
        """
        Converts HTML content to raw text using BeautifulSoup.
        """
        # Parse the HTML content
        soup = BeautifulSoup(input_text, "html.parser")
        # Get text content with whitespace normalized.
        raw_text = soup.get_text(separator=" ", strip=True)
        return raw_text               