"""
Name generator utilities for pipeline naming.

This module provides utilities for generating pipeline names with consistent formats.
"""

import random
import string
import logging

logger = logging.getLogger(__name__)

def generate_random_word(length: int = 4) -> str:
    """
    Generate a random word of specified length.
    
    Args:
        length: Length of the random word
        
    Returns:
        Random string of specified length
    """
    # Using uppercase letters for better readability in names
    return ''.join(random.choices(string.ascii_uppercase, k=length))

def generate_pipeline_name(base_name: str, version: str = "1.0") -> str:
    """
    Generate a pipeline name with the format:
    {base_name}-{random_word}-{version}-pipeline
    
    Args:
        base_name: Base name for the pipeline
        version: Version string to include in the name
        
    Returns:
        A string with the generated pipeline name
    """
    # Generate random 4-letter word
    random_word = generate_random_word(4)
    
    # Combine all parts
    return f"{base_name}-{random_word}-{version}-pipeline"
