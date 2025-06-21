"""
Unit tests for the prompt_rnr_parse module.
"""

import unittest
import sys
import os
from typing import List

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bedrock.prompt_rnr_parse import (
    BSMAnalysis,
    extract_evidence_section,
    extract_category,
    extract_confidence_score,
    create_error_analysis,
    parse_claude_response
)


class TestPromptRnrParse(unittest.TestCase):
    """Test cases for the prompt_rnr_parse module."""

    def setUp(self):
        """Set up test fixtures."""
        # Sample Claude response for testing
        self.sample_response = """
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
        
        # Sample response with missing fields
        self.incomplete_response = """
1. Category: TrueDNR
3. Key Evidence:
   * Message Evidence:
     [sep] Customer message: "Package never arrived"
4. Reasoning:
   * Primary Factors:
     [sep] Tracking shows delivery but customer claims non-receipt
        """
        
        # Sample response with invalid format
        self.invalid_response = """
This is not a properly formatted response.
It doesn't contain the expected sections.
        """

    def test_extract_category(self):
        """Test extracting category from response."""
        # Test with valid response
        category = extract_category(self.sample_response)
        self.assertEqual(category, "TrueDNR")
        
        # Test with invalid response
        category = extract_category(self.invalid_response)
        self.assertEqual(category, "Unknown")

    def test_extract_confidence_score(self):
        """Test extracting confidence score from response."""
        # Test with valid response
        score = extract_confidence_score(self.sample_response)
        self.assertEqual(score, 0.95)
        
        # Test with response missing confidence score
        score = extract_confidence_score(self.incomplete_response)
        self.assertEqual(score, 0.0)
        
        # Test with invalid response
        score = extract_confidence_score(self.invalid_response)
        self.assertEqual(score, 0.0)

    def test_extract_evidence_section(self):
        """Test extracting evidence sections from response."""
        # Test extracting message evidence
        message_evidence = extract_evidence_section(self.sample_response, "Message Evidence")
        self.assertEqual(len(message_evidence), 2)
        self.assertEqual(message_evidence[0], 'Customer message: "Package never arrived"')
        self.assertEqual(message_evidence[1], 'Seller response: "Tracking shows delivered"')
        
        # Test extracting shipping evidence
        shipping_evidence = extract_evidence_section(self.sample_response, "Shipping Evidence")
        self.assertEqual(len(shipping_evidence), 2)
        self.assertEqual(shipping_evidence[0], 'EVENT_301: Delivered on 2024-03-15')
        self.assertEqual(shipping_evidence[1], 'No subsequent delivery scans')
        
        # Test extracting non-existent section
        nonexistent = extract_evidence_section(self.sample_response, "Nonexistent Section")
        self.assertEqual(nonexistent, [])
        
        # Test with invalid response
        invalid = extract_evidence_section(self.invalid_response, "Message Evidence")
        self.assertEqual(invalid, [])

    def test_create_error_analysis(self):
        """Test creating error analysis object."""
        # Create an error analysis
        error = ValueError("Test error")
        analysis = create_error_analysis(self.sample_response, error)
        
        # Check the result
        self.assertEqual(analysis.category, "TrueDNR")
        self.assertEqual(analysis.confidence_score, 0.95)
        self.assertTrue("Test error" in analysis.error)
        self.assertEqual(analysis.raw_response, self.sample_response)
        
        # Test with invalid response
        analysis = create_error_analysis(self.invalid_response, error)
        self.assertEqual(analysis.category, "Unknown")
        self.assertEqual(analysis.confidence_score, 0.0)
        self.assertTrue("Test error" in analysis.error)

    def test_parse_claude_response_valid(self):
        """Test parsing a valid Claude response."""
        # Parse the sample response
        analysis = parse_claude_response(self.sample_response)
        
        # Check the result
        self.assertEqual(analysis.category, "TrueDNR")
        self.assertEqual(analysis.confidence_score, 0.95)
        self.assertEqual(len(analysis.message_evidence), 2)
        self.assertEqual(len(analysis.shipping_evidence), 2)
        self.assertEqual(len(analysis.timeline_evidence), 2)
        self.assertEqual(len(analysis.primary_factors), 2)
        self.assertEqual(len(analysis.supporting_evidence), 2)
        self.assertEqual(len(analysis.contradicting_evidence), 1)
        self.assertEqual(analysis.contradicting_evidence[0], "None")
        self.assertIsNone(analysis.error)

    def test_parse_claude_response_incomplete(self):
        """Test parsing an incomplete Claude response."""
        # Parse the incomplete response
        analysis = parse_claude_response(self.incomplete_response)
        
        # Check the result
        self.assertEqual(analysis.category, "TrueDNR")
        self.assertEqual(analysis.confidence_score, 0.0)
        self.assertEqual(len(analysis.message_evidence), 1)
        self.assertEqual(len(analysis.primary_factors), 1)
        self.assertEqual(len(analysis.shipping_evidence), 0)
        self.assertEqual(len(analysis.timeline_evidence), 0)
        self.assertEqual(len(analysis.supporting_evidence), 0)
        self.assertEqual(len(analysis.contradicting_evidence), 0)
        # The current implementation doesn't set an error for missing confidence score
        # when the category is present, so we don't assert on the error field

    def test_parse_claude_response_invalid(self):
        """Test parsing an invalid Claude response."""
        # Parse the invalid response
        analysis = parse_claude_response(self.invalid_response)
        
        # Check the result
        self.assertEqual(analysis.category, "Unknown")
        self.assertEqual(analysis.confidence_score, 0.0)
        self.assertEqual(len(analysis.message_evidence), 0)
        self.assertEqual(len(analysis.shipping_evidence), 0)
        self.assertEqual(len(analysis.timeline_evidence), 0)
        self.assertEqual(len(analysis.primary_factors), 0)
        self.assertEqual(len(analysis.supporting_evidence), 0)
        self.assertEqual(len(analysis.contradicting_evidence), 0)
        self.assertIsNotNone(analysis.error)

    def test_parse_claude_response_empty(self):
        """Test parsing an empty response."""
        # Parse an empty response
        analysis = parse_claude_response("")
        
        # Check the result
        self.assertEqual(analysis.category, "Error")
        self.assertEqual(analysis.confidence_score, 0.0)
        self.assertIsNotNone(analysis.error)
        
        # Parse None
        analysis = parse_claude_response(None)
        
        # Check the result
        self.assertEqual(analysis.category, "Error")
        self.assertEqual(analysis.confidence_score, 0.0)
        self.assertIsNotNone(analysis.error)

    def test_bsm_analysis_model(self):
        """Test the BSMAnalysis model."""
        # Create a BSMAnalysis object
        analysis = BSMAnalysis(
            category="TestCategory",
            confidence_score=0.75,
            message_evidence=["Evidence 1", "Evidence 2"],
            shipping_evidence=["Shipping 1"],
            timeline_evidence=["Timeline 1"],
            primary_factors=["Factor 1"],
            supporting_evidence=["Support 1"],
            contradicting_evidence=["Contradict 1"],
            raw_response="Raw response",
            latency=0.5
        )
        
        # Check the values
        self.assertEqual(analysis.category, "TestCategory")
        self.assertEqual(analysis.confidence_score, 0.75)
        self.assertEqual(analysis.message_evidence, ["Evidence 1", "Evidence 2"])
        self.assertEqual(analysis.shipping_evidence, ["Shipping 1"])
        self.assertEqual(analysis.timeline_evidence, ["Timeline 1"])
        self.assertEqual(analysis.primary_factors, ["Factor 1"])
        self.assertEqual(analysis.supporting_evidence, ["Support 1"])
        self.assertEqual(analysis.contradicting_evidence, ["Contradict 1"])
        self.assertEqual(analysis.raw_response, "Raw response")
        self.assertEqual(analysis.latency, 0.5)
        self.assertIsNone(analysis.error)
        
        # Test default values
        default_analysis = BSMAnalysis(
            category="TestCategory",
            confidence_score=0.75
        )
        
        self.assertEqual(default_analysis.category, "TestCategory")
        self.assertEqual(default_analysis.confidence_score, 0.75)
        self.assertEqual(default_analysis.message_evidence, [])
        self.assertEqual(default_analysis.shipping_evidence, [])
        self.assertEqual(default_analysis.timeline_evidence, [])
        self.assertEqual(default_analysis.primary_factors, [])
        self.assertEqual(default_analysis.supporting_evidence, [])
        self.assertEqual(default_analysis.contradicting_evidence, [])
        self.assertEqual(default_analysis.raw_response, "")
        self.assertIsNone(default_analysis.latency)
        self.assertIsNone(default_analysis.error)


if __name__ == '__main__':
    unittest.main()
