import json
import boto3
import time
from botocore.exceptions import ClientError
import random
from pathlib import Path


# Function to load prompt template from the prompt_repo folder
def load_prompt_template(template_name: str) -> str:
    """
    Load a prompt template from the prompt_repo folder.

    Args:
        template_name (str): Name of the template file (e.g., "claude_prompt.txt").

    Returns:
        str: The content of the prompt template.
    """
    prompt_repo_path = Path(__file__).parent / "prompt_repo"
    template_path = prompt_repo_path / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template '{template_name}' not found in {prompt_repo_path}")

    with open(template_path, "r") as file:
        return file.read()


# Load the prompt template dynamically
prompt_template = load_prompt_template("prompt_rnr_2025_03.txt")


def exponential_backoff(attempt, max_delay=32):
    """
    Calculate exponential backoff time with jitter
    """
    delay = min(max_delay, (2 ** attempt) + random.uniform(0, 1))
    return delay


def analyze_dialogue_with_claude(dialogue, max_retries=5):
    """
    Analyze dialogue using Claude via AWS Bedrock with retry logic
    """
    bedrock = boto3.client(service_name='bedrock-runtime')

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": prompt_template.format(dialogue=dialogue)
            }
        ]
    }

    attempt = 0
    while attempt < max_retries:
        try:
            start_time = time.time()

            response = bedrock.invoke_model(
                modelId='anthropic.claude-3-sonnet-20240229-v1:0',
                body=json.dumps(body)
            )

            response_body = json.loads(response.get('body').read())
            result = response_body.get('content', [{}])[0].get('text', '')

            end_time = time.time()
            latency = end_time - start_time

            return result, latency

        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                attempt += 1
                if attempt == max_retries:
                    print(f"Max retries ({max_retries}) reached. Failing.")
                    return f"Error: Max retries reached - {str(e)}", None

                delay = exponential_backoff(attempt)
                print(f"Rate limit reached. Waiting {delay:.2f} seconds before retry {attempt}/{max_retries}")
                time.sleep(delay)
            else:
                return f"Error: {str(e)}", None

        except Exception as e:
            return f"Error: {str(e)}", None