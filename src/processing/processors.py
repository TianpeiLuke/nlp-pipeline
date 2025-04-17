import re
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import sklearn
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from typing import List,  Union, Dict, Optional
from abc import ABC, abstractmethod
import re
from bs4 import BeautifulSoup


from .constants import *
from urllib.request import urlretrieve



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
    def __init__(self):
        super().__init__()
        self.processor_name = "dialogue_splitter_processor"

    def process(self, input_text: str):
        """
        Splits the dialogue into individual messages based on [bom] and [eom] delimiters.
        Returns:
            List of message strings.
        """
        pattern = r'\[bom\](.*?)\[eom\]'
        raw_messages = re.findall(pattern, input_text, flags=re.DOTALL)
        # Strip extra whitespace from each message
        messages = [msg.strip() for msg in raw_messages if msg.strip()]
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

        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())
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
  
    
#====================================================================================================    
class LabelProcessor(Processor):
    '''
    '''
    label_encoder: sklearn.preprocessing._label.LabelEncoder
    label_encoder_dict: dict
    
    def __init__(self, label_encoder: Optional[sklearn.preprocessing._label.LabelEncoder] = None,
                 label_encoder_dict: Dict[Union[str, int], int] = None
                ):
        self.processor_name = 'label_encoding_process'
        if not label_encoder or not isinstance(label_encoder, sklearn.preprocessing._label.LabelEncoder):
            self.label_encoder = LabelEncoder()
        else:
            self.label_encoder = label_encoder
            
        try:
            self.label_encoder_dict = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_).tolist()))
        except AttributeError:
            if label_encoder_dict:
                self.label_encoder_dict = label_encoder_dict
            else:
                self.label_encoder_dict = dict()
        
        self.function_handler_list = [
                                      self.label_transform, 
                                     ]
        self.function_name_list = [
                                   'label_transform', 
                                  ]
        
    def label_encoder_fit(self, label_df: Union[list, pd.core.series.Series]) -> None:
        if not isinstance(label_df, list) and \
            not isinstance(label_df, pd.core.series.Series) and \
            not isinstance(label_df, pd.core.frame.DataFrame):
            raise TypeError('Input must be list or pandas.core.series.Series or pandas.core.frame.DataFrame')
        self.label_encoder.fit(label_df)
        self.label_encoder_dict = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_).tolist()))

        
    def label_transform(self, label: Union[str, int, list, pd.core.series.Series]) -> Union[int, List[int]]:
        if not isinstance(label, list) and \
            not isinstance(label, pd.core.series.Series) and \
            not isinstance(label, pd.core.frame.DataFrame):
                
            if isinstance(label, str) or isinstance(label, int):
                label = [label]
                try:
                    result = self.label_encoder.transform(label)
                except ValueError:
                    result = [-1]
                return result[0]
            else:
                raise TypeError('Input must be list or pandas.core.series.Series')
        else:
            try:
                transformed_label = self.label_encoder.transform(label)
            except ValueError:
                transformed_label = -1*np.ones(label.shape)
            return transformed_label
    
    def process(self, label: Union[str, int,  list, pd.core.series.Series], **kwargs) -> Union[int, List[int]]:
        if not self.label_encoder_dict or len(self.label_encoder_dict) == 0:
            return label
        else:
            return self.label_transform(label)
    

#====================================================================================================    
class OrdinalProcessor(Processor):
    '''
    Remove quotations of previous emails from Gmail, Hotmail etc.
    '''    
    def __init__(self, encoder: Optional[sklearn.preprocessing._encoders.OrdinalEncoder] = None,
                 encoder_dict: Dict[Union[str, int], int] = None
                ):
        self.processor_name = 'categorical_encoder'
        if not encoder or not isinstance(encoder, sklearn.preprocessing._encoders.OrdinalEncoder):
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        else:
            self.encoder = encoder
            
        try:
            self.encoder_dict = dict(zip(list(self.encoder.categories_[0]), 
                                              self.encoder.transform(self.encoder.categories_[0].reshape(-1, 1)).reshape(1, -1).tolist()[0]
                                    ))
        except AttributeError:
            if encoder_dict:
                self.encoder_dict = encoder_dict
            else:
                self.encoder_dict = dict()
        
        self.function_handler_list = [
                                      self.feature_transform, 
                                     ]
        self.function_name_list = [
                                   'feature_transform', 
                                  ]
        
    def encoder_fit(self, df: Union[list, pd.core.series.Series]) -> None:
        df_array = None
        if isinstance(df, list):
            df_array = np.array(df).reshape(-1, 1)
        elif isinstance(df, pd.core.series.Series):
            df_array = df.to_numpy().reshape(-1, 1)
        elif isinstance(df, pd.core.frame.DataFrame):
            df_array = df.to_numpy()
        else:
            raise TypeError('Input must be list or pandas.core.series.Series or pandas.core.frame.DataFrame')
            
        self.encoder.fit(df_array)
        self.encoder_dict = dict(zip(list(self.encoder.categories_[0]), 
                                          self.encoder.transform(self.encoder.categories_[0].reshape(-1, 1)).reshape(1, -1).tolist()[0]
                                ))

        
    def feature_transform(self, 
                          categorical: Union[str, int, list, pd.core.series.Series, pd.core.frame.DataFrame]) -> Union[int, List[int]]:
        if isinstance(categorical, str) or isinstance(categorical, int):
            categorical_array = [[categorical]]
        elif isinstance(categorical, list):
            categorical_array = np.array(categorical).reshape(-1, 1)
        elif isinstance(categorical, pd.core.series.Series):
            categorical_array = categorical.to_numpy().reshape(-1, 1)
        elif isinstance(categorical, pd.core.frame.DataFrame):
            categorical_array = categorical.to_numpy()
        else:
            raise TypeError('Input must be int, str, list, pandas.core.series.Series or pandas.core.frame.DataFrame')
            
        transformed_label = self.encoder.transform(categorical_array)
        if isinstance(categorical, str) or isinstance(categorical, int):
            transformed_label = transformed_label[0][0]
        elif isinstance(categorical, list) or isinstance(categorical, pd.core.series.Series):
            transformed_label = transformed_label.reshape(1, -1)[0]
            
        return transformed_label
    
    def process(self, 
                categorical: Union[str, int,  list, pd.core.series.Series, pd.core.frame.DataFrame], 
                **kwargs) -> Union[int, List[int]]:
        if not self.encoder_dict or len(self.encoder_dict) == 0:
            return categorical
        else:
            return self.feature_transform(categorical)
        
    

#=======================

# --- Dummy Tokenizer for Unit Testing (can also be used by others) ---
class DummyTokenizer:
    """A dummy tokenizer that splits text by whitespace."""
    def encode(self, text, add_special_tokens=False):
        return text.split()
    
    def __call__(self, text, add_special_tokens=True, max_length=None, truncation=True, padding="longest", return_attention_mask=True):
        tokens = text.split()
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}