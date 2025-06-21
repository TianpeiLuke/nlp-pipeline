"""
AWS Bedrock Claude Invocation Module

This module provides functions to interact with AWS Bedrock Claude models,
handling prompt template loading, API calls with retry logic, and response processing.
"""

import json
import boto3
import time
import logging
from botocore.exceptions import ClientError
import random
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union


# Set up logging
logger = logging.getLogger(__name__)


def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the prompt_repo folder.

    Args:
        template_name (str): Name of the template file (e.g., "claude_prompt.txt").

    Returns:
        str: The content of the prompt template.
    
    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    prompt_repo_path = Path(__file__).parent / "prompt_repo"
    template_path = prompt_repo_path / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template '{template_name}' not found in {prompt_repo_path}")

    with open(template_path, "r") as file:
        return file.read()


def exponential_backoff(attempt: int, max_delay: int = 32) -> float:
    """
    Calculate exponential backoff time with jitter.

    Args:
        attempt (int): The current retry attempt number.
        max_delay (int): Maximum delay in seconds.

    Returns:
        float: The calculated backoff time in seconds.
    """
    delay = min(max_delay, (2 ** attempt) + random.uniform(0, 1))
    return delay


def get_bedrock_client() -> boto3.client:
    """
    Create and return a boto3 client for AWS Bedrock.
    
    Returns:
        boto3.client: Configured Bedrock client.
    """
    return boto3.client(service_name='bedrock-runtime')


def analyze_dialogue_with_claude(
    dialogue: str,
    shiptrack: Optional[str] = None,
    max_estimated_arrival_date: Optional[str] = None,
    max_retries: int = 5,
    template_name: str = "prompt_rnr_2025_03.txt",
    model_id: str = 'anthropic.claude-3-sonnet-20240229-v1:0',
    bedrock_client: Optional[boto3.client] = None
) -> Tuple[str, Optional[float]]:
    """
    Analyze dialogue using Claude via AWS Bedrock with retry logic.
    
    Args:
        dialogue (str): The dialogue text to analyze.
        shiptrack (Optional[str]): Shipping tracking information (if available).
        max_estimated_arrival_date (Optional[str]): Maximum estimated arrival date (if available).
        max_retries (int): Maximum number of retry attempts.
        template_name (str): Name of the prompt template file to use.
        model_id (str): AWS Bedrock model ID to use.
        bedrock_client (Optional[boto3.client]): Pre-configured Bedrock client (for testing).
        
    Returns:
        Tuple[str, Optional[float]]: Tuple containing (result, latency).
            If an error occurs, result will be an error message and latency will be None.
    """
    # Load the prompt template
    try:
        prompt_template = load_prompt_template(template_name)
    except FileNotFoundError as e:
        logger.error(f"Failed to load prompt template: {str(e)}")
        return f"Error: {str(e)}", None
    
    # Get or create Bedrock client
    if bedrock_client is None:
        bedrock_client = get_bedrock_client()
    
    # Format the prompt with available information
    prompt_kwargs = {"dialogue": dialogue}
    if shiptrack:
        prompt_kwargs["shiptrack"] = shiptrack
    if max_estimated_arrival_date:
        prompt_kwargs["max_estimated_arrival_date"] = max_estimated_arrival_date
    
    try:
        formatted_prompt = prompt_template.format(**prompt_kwargs)
    except KeyError as e:
        error_msg = f"Error formatting prompt template: {str(e)}"
        logger.error(error_msg)
        return f"Error: {error_msg}", None
    
    # Prepare request body
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": formatted_prompt
            }
        ]
    }

    # Retry logic
    attempt = 0
    while attempt < max_retries:
        try:
            start_time = time.time()

            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response.get('body').read())
            result = response_body.get('content', [{}])[0].get('text', '')

            end_time = time.time()
            latency = end_time - start_time
            
            logger.info(f"Successfully invoked Bedrock model. Latency: {latency:.2f}s")
            return result, latency

        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                attempt += 1
                if attempt == max_retries:
                    error_msg = f"Max retries ({max_retries}) reached. Failing."
                    logger.error(error_msg)
                    return f"Error: Max retries reached - {str(e)}", None

                delay = exponential_backoff(attempt)
                logger.warning(f"Rate limit reached. Waiting {delay:.2f} seconds before retry {attempt}/{max_retries}")
                time.sleep(delay)
            else:
                logger.error(f"AWS client error: {str(e)}")
                return f"Error: {str(e)}", None

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return f"Error: {str(e)}", None
