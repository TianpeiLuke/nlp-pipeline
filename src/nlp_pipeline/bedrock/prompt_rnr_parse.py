
"""
Claude Response Parser Module

This module provides functionality to parse structured responses from Claude
into a validated Pydantic model, extracting key information from the LLM output.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ValidationError


# Set up logging
logger = logging.getLogger(__name__)


class BSMAnalysis(BaseModel):
    """
    Structured container for analysis results with validation.
    
    This model represents the parsed output from Claude's analysis of buyer-seller
    messaging data, with fields for category classification, confidence scores,
    and various types of evidence.
    """
    category: str = Field(default="Unknown", description="Classification category")
    confidence_score: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0 and 1"
    )
    message_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from message content"
    )
    shipping_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from shipping information"
    )
    timeline_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence from timeline analysis"
    )
    primary_factors: List[str] = Field(
        default_factory=list,
        description="Primary factors for the classification"
    )
    supporting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence supporting the classification"
    )
    contradicting_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence contradicting the classification"
    )
    raw_response: str = Field(
        default="",
        description="Original raw response from Claude"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if parsing failed"
    )
    latency: Optional[float] = Field(
        default=None,
        description="API call latency in seconds"
    )

    class Config:
        validate_assignment = True
        

def extract_evidence_section(response: str, section_name: str) -> List[str]:
    """
    Extract evidence items from a specific section of the response.
    
    Args:
        response (str): The full response text from Claude
        section_name (str): The name of the section to extract (e.g., "Message Evidence")
        
    Returns:
        List[str]: List of evidence items extracted from the section
    """
    # Look for section starting with * and ending before the next section
    pattern = fr"\* {section_name}:.*?(?=(?:\n\s*\* |\n\d\.|\Z))"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return []
    
    # Split by [sep] token and clean
    evidence_text = match.group(0)
    # Use positive lookbehind to ensure we get content after [sep]
    items = re.findall(r'(?<=\[sep\])(.*?)(?=(?:\[sep\]|\n\s*\*|\Z))', evidence_text, re.DOTALL)
    return [item.strip() for item in items if item.strip()]


def extract_category(response: str) -> str:
    """
    Extract the category from the response.
    
    Args:
        response (str): The full response text from Claude
        
    Returns:
        str: The extracted category or "Unknown" if not found
    """
    category_match = re.search(r'Category:\s*([A-Za-z_]+(?:\s*\([^)]*\))?)', response)
    return category_match.group(1) if category_match else "Unknown"


def extract_confidence_score(response: str) -> float:
    """
    Extract the confidence score from the response.
    
    Args:
        response (str): The full response text from Claude
        
    Returns:
        float: The extracted confidence score or 0.0 if not found
    """
    confidence_match = re.search(r'Confidence Score:\s*(0?\.\d+|1\.00?)', response)
    return float(confidence_match.group(1)) if confidence_match else 0.0


def create_error_analysis(response: str, error: Exception) -> BSMAnalysis:
    """
    Create a BSMAnalysis object with error information when full parsing fails.
    
    Args:
        response (str): The full response text from Claude
        error (Exception): The exception that occurred during parsing
        
    Returns:
        BSMAnalysis: A BSMAnalysis object with basic information and error details
    """
    try:
        # Try to extract basic information even if full parsing fails
        category = extract_category(response)
        confidence_score = extract_confidence_score(response)
        
        # For test compatibility, keep the category as "Unknown" if that's what was extracted
        return BSMAnalysis(
            category=category,
            confidence_score=confidence_score,
            raw_response=response,
            error=f"Partial parse only: {str(error)}"
        )
    except Exception as e:
        logger.error(f"Failed to create error analysis: {str(e)}")
        return BSMAnalysis(
            category="Unknown",  # Changed from "Error" to "Unknown" for test compatibility
            confidence_score=0.0,
            raw_response=response,
            error=f"Failed to parse response: {str(error)}, secondary error: {str(e)}"
        )


def parse_claude_response(response: str) -> BSMAnalysis:
    """
    Parse Claude's response with [sep] tokens using Pydantic validation.
    
    Args:
        response (str): The full response text from Claude
        
    Returns:
        BSMAnalysis: A validated BSMAnalysis object containing the parsed information
    """
    if not response or not isinstance(response, str):
        logger.error(f"Invalid response: {type(response)}")
        return BSMAnalysis(
            category="Error",
            confidence_score=0.0,
            raw_response=str(response),
            error="Invalid or empty response"
        )
    
    try:
        # Extract category and confidence score
        category = extract_category(response)
        confidence_score = extract_confidence_score(response)

        # Extract evidence for each section
        message_evidence = extract_evidence_section(response, "Message Evidence")
        shipping_evidence = extract_evidence_section(response, "Shipping Evidence")
        timeline_evidence = extract_evidence_section(response, "Timeline Evidence")
        primary_factors = extract_evidence_section(response, "Primary Factors")
        supporting_evidence = extract_evidence_section(response, "Supporting Evidence")
        contradicting_evidence = extract_evidence_section(response, "Contradicting Evidence")

        # Create BSMAnalysis instance
        analysis = BSMAnalysis(
            category=category,
            confidence_score=confidence_score,
            message_evidence=message_evidence,
            shipping_evidence=shipping_evidence,
            timeline_evidence=timeline_evidence,
            primary_factors=primary_factors,
            supporting_evidence=supporting_evidence,
            contradicting_evidence=contradicting_evidence,
            raw_response=response
        )

        # Validate the results
        if not analysis.category or analysis.category == "Unknown":
            logger.warning("Missing category in response")
            if confidence_score == 0.0:
                raise ValueError("Missing required fields: category and confidence score")
            else:
                # Set error for missing category but with confidence score
                analysis.error = "Missing category in response"
        
        if analysis.confidence_score == 0.0:
            logger.warning("Missing confidence score in response")

        return analysis

    except Exception as e:
        logger.error(f"Parsing error: {str(e)}")
        logger.debug(f"Raw response: {response[:200]}...")  # Log first 200 chars of response
        
        return create_error_analysis(response, e)


if __name__ == '__main__':
    """Test the parser with sample output"""
    # Configure logging for the test
    logging.basicConfig(level=logging.INFO)
    
    test_response = """
1. Category: TrueDNR
2. Confidence Score: 0.95
3. Key Evidence:
   * Message Evidence:
     [sep] Customer message: "Package never arrived"
     [sep] Seller response: "Tracking shows delivered"
   * Shipping Evidence:
     [sep] EVENT_301: Delivered on 2024-03-15
     [sep] No subsequent delivery scans
   * Timeline Evidence:
     [sep] Delivery scan on March 15
     [sep] Customer complaint on March 16
4. Reasoning:
   * Primary Factors:
     [sep] Tracking shows delivery but customer claims non-receipt
     [sep] Timeline supports DNR claim
   * Supporting Evidence:
     [sep] Customer checked with neighbors
     [sep] No alternative delivery location found
   * Contradicting Evidence:
     [sep] None
    """
    
    result = parse_claude_response(test_response)
    print("Parsed Result:")
    print(f"Category: {result.category}")
    print(f"Confidence: {result.confidence_score}")
    print("Message Evidence:", result.message_evidence)
    print("Shipping Evidence:", result.shipping_evidence)
    print("Timeline Evidence:", result.timeline_evidence)
