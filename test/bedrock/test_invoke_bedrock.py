"""
Unit tests for the invoke_bedrock module.
"""

import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# Add the src directory to the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bedrock.invoke_bedrock import (
    load_prompt_template,
    exponential_backoff,
    get_bedrock_client,
    analyze_dialogue_with_claude
)


class TestInvokeBedrock(unittest.TestCase):
    """Test cases for the invoke_bedrock module."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_dialogue = "Customer: My package never arrived.\nSeller: The tracking shows it was delivered."
        self.sample_shiptrack = "EVENT_301: Delivered on 2023-01-01"
        self.sample_max_date = "2023-01-05"
        
        # Sample Claude response
        self.sample_claude_response = {
            "content": [
                {
                    "text": """
1. Category: TrueDNR
2. Confidence Score: 0.95
3. Key Evidence:
   * Message Evidence:
     [sep] Customer claims package never arrived
     [sep] Seller points to tracking showing delivered
   * Shipping Evidence:
     [sep] EVENT_301: Delivered on 2023-01-01
   * Timeline Evidence:
     [sep] Delivery date: 2023-01-01
     [sep] Max estimated arrival: 2023-01-05
4. Reasoning:
   * Primary Factors:
     [sep] Tracking shows delivered but customer claims non-receipt
   * Supporting Evidence:
     [sep] Timeline is consistent with a DNR claim
   * Contradicting Evidence:
     [sep] None
                    """
                }
            ]
        }

    @patch('builtins.open', new_callable=mock_open, read_data="Test prompt template {dialogue}")
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_prompt_template(self, mock_exists, mock_file):
        """Test loading a prompt template."""
        # Call the function
        template = load_prompt_template("test_template.txt")
        
        # Check the result
        self.assertEqual(template, "Test prompt template {dialogue}")
        
        # Check that the file was opened
        mock_file.assert_called_once()

    @patch('pathlib.Path.exists', return_value=False)
    def test_load_prompt_template_not_found(self, mock_exists):
        """Test loading a non-existent prompt template."""
        # Call the function and check that it raises FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            load_prompt_template("nonexistent_template.txt")

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        # Test with different attempt numbers
        backoff_1 = exponential_backoff(1)
        backoff_2 = exponential_backoff(2)
        backoff_3 = exponential_backoff(3)
        
        # Check that backoff increases with attempt number
        self.assertLess(backoff_1, backoff_2)
        self.assertLess(backoff_2, backoff_3)
        
        # Test with max_delay
        backoff_max = exponential_backoff(10, max_delay=5)
        self.assertLessEqual(backoff_max, 6)  # 5 + max jitter of 1

    @patch('boto3.client')
    def test_get_bedrock_client(self, mock_client):
        """Test getting a Bedrock client."""
        # Mock the boto3.client function
        mock_client.return_value = "mock_bedrock_client"
        
        # Call the function
        client = get_bedrock_client()
        
        # Check the result
        self.assertEqual(client, "mock_bedrock_client")
        
        # Check that boto3.client was called with the correct arguments
        mock_client.assert_called_once_with(service_name='bedrock-runtime')

    @patch('src.bedrock.invoke_bedrock.load_prompt_template')
    @patch('src.bedrock.invoke_bedrock.get_bedrock_client')
    def test_analyze_dialogue_with_claude_success(self, mock_get_client, mock_load_template):
        """Test successful analysis with Claude."""
        # Mock the load_prompt_template function
        mock_load_template.return_value = "Test prompt with {dialogue}"
        
        # Mock the Bedrock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock the response from invoke_model
        mock_response = MagicMock()
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps(self.sample_claude_response)
        mock_response.__getitem__.return_value = mock_response_body
        mock_response.get.return_value = mock_response_body
        mock_client.invoke_model.return_value = mock_response
        
        # Call the function
        result, latency = analyze_dialogue_with_claude(
            dialogue=self.sample_dialogue,
            shiptrack=self.sample_shiptrack,
            max_estimated_arrival_date=self.sample_max_date
        )
        
        # Check the result
        self.assertIn("Category: TrueDNR", result)
        self.assertIsNotNone(latency)
        
        # Check that the client was called with the correct arguments
        mock_client.invoke_model.assert_called_once()
        args, kwargs = mock_client.invoke_model.call_args
        self.assertEqual(kwargs['modelId'], 'anthropic.claude-3-sonnet-20240229-v1:0')
        
        # Check that the body contains the dialogue
        body = json.loads(kwargs['body'])
        self.assertIn("messages", body)
        self.assertEqual(len(body["messages"]), 1)
        self.assertEqual(body["messages"][0]["role"], "user")

    @patch('src.bedrock.invoke_bedrock.load_prompt_template')
    def test_analyze_dialogue_template_not_found(self, mock_load_template):
        """Test analysis when template is not found."""
        # Mock the load_prompt_template function to raise FileNotFoundError
        mock_load_template.side_effect = FileNotFoundError("Template not found")
        
        # Call the function
        result, latency = analyze_dialogue_with_claude(
            dialogue=self.sample_dialogue
        )
        
        # Check the result
        self.assertTrue(result.startswith("Error:"))
        self.assertIsNone(latency)

    @patch('src.bedrock.invoke_bedrock.load_prompt_template')
    @patch('src.bedrock.invoke_bedrock.get_bedrock_client')
    def test_analyze_dialogue_with_claude_client_error(self, mock_get_client, mock_load_template):
        """Test analysis with client error."""
        # Mock the load_prompt_template function
        mock_load_template.return_value = "Test prompt with {dialogue}"
        
        # Mock the Bedrock client to raise ClientError
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Create a ClientError
        error_response = {
            'Error': {
                'Code': 'InternalServerError',
                'Message': 'Test error'
            }
        }
        mock_client.invoke_model.side_effect = ClientError(error_response, 'InvokeModel')
        
        # Call the function
        result, latency = analyze_dialogue_with_claude(
            dialogue=self.sample_dialogue,
            max_retries=1
        )
        
        # Check the result
        self.assertTrue(result.startswith("Error:"))
        self.assertIsNone(latency)

    @patch('src.bedrock.invoke_bedrock.load_prompt_template')
    @patch('src.bedrock.invoke_bedrock.get_bedrock_client')
    @patch('time.sleep')  # Mock sleep to avoid waiting in tests
    def test_analyze_dialogue_with_claude_throttling(self, mock_sleep, mock_get_client, mock_load_template):
        """Test analysis with throttling exception and retry."""
        # Mock the load_prompt_template function
        mock_load_template.return_value = "Test prompt with {dialogue}"
        
        # Mock the Bedrock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Create a ThrottlingException for the first call
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        throttling_error = ClientError(error_response, 'InvokeModel')
        
        # Mock the response for the second call
        mock_response = MagicMock()
        mock_response_body = MagicMock()
        mock_response_body.read.return_value = json.dumps(self.sample_claude_response)
        mock_response.__getitem__.return_value = mock_response_body
        mock_response.get.return_value = mock_response_body
        
        # Set up the side effects: first call raises error, second call succeeds
        mock_client.invoke_model.side_effect = [throttling_error, mock_response]
        
        # Call the function
        result, latency = analyze_dialogue_with_claude(
            dialogue=self.sample_dialogue,
            max_retries=2
        )
        
        # Check the result
        self.assertIn("Category: TrueDNR", result)
        self.assertIsNotNone(latency)
        
        # Check that invoke_model was called twice
        self.assertEqual(mock_client.invoke_model.call_count, 2)
        
        # Check that sleep was called once for the retry
        mock_sleep.assert_called_once()

    @patch('src.bedrock.invoke_bedrock.load_prompt_template')
    @patch('src.bedrock.invoke_bedrock.get_bedrock_client')
    @patch('time.sleep')  # Mock sleep to avoid waiting in tests
    def test_analyze_dialogue_max_retries_exceeded(self, mock_sleep, mock_get_client, mock_load_template):
        """Test analysis with max retries exceeded."""
        # Mock the load_prompt_template function
        mock_load_template.return_value = "Test prompt with {dialogue}"
        
        # Mock the Bedrock client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Create a ThrottlingException
        error_response = {
            'Error': {
                'Code': 'ThrottlingException',
                'Message': 'Rate exceeded'
            }
        }
        throttling_error = ClientError(error_response, 'InvokeModel')
        
        # Make all calls raise ThrottlingException
        mock_client.invoke_model.side_effect = throttling_error
        
        # Call the function with max_retries=2
        result, latency = analyze_dialogue_with_claude(
            dialogue=self.sample_dialogue,
            max_retries=2
        )
        
        # Check the result
        self.assertTrue(result.startswith("Error: Max retries reached"))
        self.assertIsNone(latency)
        
        # Check that invoke_model was called twice
        self.assertEqual(mock_client.invoke_model.call_count, 2)
        
        # Check that sleep was called once for the retry
        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == '__main__':
    unittest.main()
