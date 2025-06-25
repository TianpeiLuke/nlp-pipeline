#!/usr/bin/env python3
"""
Script to generate MIMS payload samples.
This script is designed to run in a ProcessingStep, taking configuration parameters
from command-line arguments or environment variables.

When running in a ProcessingStep:
- The ProcessingStep will automatically handle the transfer of data between steps
- The script should write outputs to the directory specified by --output-dir
- The ProcessingStep will automatically upload the contents of this directory to S3
"""

import json
import logging
import os
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Union, Optional
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define VariableType enum
class VariableType(str, Enum):
    """Enum for variable types."""
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"


def get_field_default_value(field_name: str, var_type: str, 
                           default_numeric_value: float, 
                           default_text_value: str,
                           special_field_values: Optional[Dict[str, str]] = None) -> str:
    """
    Get default value for a field.
    
    Args:
        field_name: Name of the field
        var_type: Type of the field (NUMERIC or TEXT)
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Optional dictionary of special TEXT fields and their template values
        
    Returns:
        String representation of the field value
    """
    if var_type == "TEXT":
        if special_field_values and field_name in special_field_values:
            template = special_field_values[field_name]
            try:
                return template.format(
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            except KeyError as e:
                raise ValueError(f"Invalid placeholder in template for field '{field_name}': {str(e)}")
        return default_text_value
    elif var_type == "NUMERIC":
        return str(default_numeric_value)
    else:
        raise ValueError(f"Unknown variable type: {var_type}")


def generate_csv_payload(input_variables: Dict[str, str],
                        default_numeric_value: float,
                        default_text_value: str,
                        special_field_values: Optional[Dict[str, str]] = None) -> str:
    """
    Generate CSV format payload.
    
    Args:
        input_variables: Dictionary mapping input variable names to their types
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Optional dictionary of special TEXT fields and their template values
        
    Returns:
        Comma-separated string of values
    """
    values = []
    
    for field_name, var_type in input_variables.items():
        values.append(get_field_default_value(
            field_name, 
            var_type, 
            default_numeric_value, 
            default_text_value, 
            special_field_values
        ))
            
    return ",".join(values)


def generate_json_payload(input_variables: Dict[str, str],
                         default_numeric_value: float,
                         default_text_value: str,
                         special_field_values: Optional[Dict[str, str]] = None) -> str:
    """
    Generate JSON format payload.
    
    Args:
        input_variables: Dictionary mapping input variable names to their types
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Optional dictionary of special TEXT fields and their template values
        
    Returns:
        JSON string with field names and values
    """
    payload = {}
    
    for field_name, var_type in input_variables.items():
        payload[field_name] = get_field_default_value(
            field_name, 
            var_type, 
            default_numeric_value, 
            default_text_value, 
            special_field_values
        )
            
    return json.dumps(payload)


def generate_sample_payloads(content_types: List[str],
                            input_variables: Dict[str, str],
                            default_numeric_value: float,
                            default_text_value: str,
                            special_field_values: Optional[Dict[str, str]] = None) -> List[Dict[str, Union[str, dict]]]:
    """
    Generate sample payloads for each content type.
    
    Args:
        content_types: List of content types supported by the model
        input_variables: Dictionary mapping input variable names to their types
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Optional dictionary of special TEXT fields and their template values
        
    Returns:
        List of dictionaries containing content type and payload
    """
    payloads = []

    for content_type in content_types:
        payload_info = {
            "content_type": content_type,
            "payload": None
        }
    
        if content_type == "text/csv":
            payload_info["payload"] = generate_csv_payload(
                input_variables, 
                default_numeric_value, 
                default_text_value, 
                special_field_values
            )
        elif content_type == "application/json":
            payload_info["payload"] = generate_json_payload(
                input_variables, 
                default_numeric_value, 
                default_text_value, 
                special_field_values
            )
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        payloads.append(payload_info)
    
    return payloads


def save_payloads(output_dir: Path,
                 content_types: List[str],
                 input_variables: Dict[str, str],
                 default_numeric_value: float,
                 default_text_value: str,
                 special_field_values: Optional[Dict[str, str]] = None) -> List[Path]:
    """
    Generate and save payloads to files.
    
    Args:
        output_dir: Directory to save payload files
        content_types: List of content types supported by the model
        input_variables: Dictionary mapping input variable names to their types
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Optional dictionary of special TEXT fields and their template values
        
    Returns:
        List of paths to created payload files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    payloads = generate_sample_payloads(
        content_types, 
        input_variables, 
        default_numeric_value, 
        default_text_value, 
        special_field_values
    )
    
    for i, payload_info in enumerate(payloads):
        content_type = payload_info["content_type"]
        payload = payload_info["payload"]
        
        # Determine file extension and name
        ext = ".csv" if content_type == "text/csv" else ".json"
        file_name = f"payload_{content_type.replace('/', '_')}_{i}{ext}"
        file_path = output_dir / file_name
        
        # Save payload
        with open(file_path, "w") as f:
            f.write(payload)
            
        file_paths.append(file_path)
        logger.info(f"Created payload file: {file_path}")
        
        # Display file content for verification
        logger.info(f"Content of {file_path.name}:\n{payload}")
        
    return file_paths




def parse_input_variables(input_vars_json: str) -> Dict[str, str]:
    """
    Parse input variables from JSON string.
    
    Args:
        input_vars_json: JSON string representing input variables
        
    Returns:
        Dictionary mapping input variable names to their types
    """
    try:
        return json.loads(input_vars_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse input variables JSON: {str(e)}")
        raise ValueError(f"Invalid JSON for input variables: {str(e)}")


def parse_special_field_values(special_fields_json: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Parse special field values from JSON string.
    
    Args:
        special_fields_json: JSON string representing special field values
        
    Returns:
        Dictionary mapping field names to their template values
    """
    if not special_fields_json:
        return None
        
    try:
        return json.loads(special_fields_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse special field values JSON: {str(e)}")
        raise ValueError(f"Invalid JSON for special field values: {str(e)}")


def parse_content_types(content_types_str: str) -> List[str]:
    """
    Parse content types from comma-separated string.
    
    Args:
        content_types_str: Comma-separated string of content types
        
    Returns:
        List of content types
    """
    return [ct.strip() for ct in content_types_str.split(',')]


def main():
    """Main function to generate payload samples."""
    parser = argparse.ArgumentParser(description='Generate MIMS payload samples')
    
    # Required arguments
    parser.add_argument('--output-dir', type=str, default='./payload_samples',
                        help='Directory to save payload files')
    parser.add_argument('--content-types', type=str, required=True,
                        help='Comma-separated list of content types (e.g., "text/csv,application/json")')
    parser.add_argument('--input-variables', type=str, required=True,
                        help='JSON string mapping input variable names to their types')
    
    # Optional arguments
    parser.add_argument('--default-numeric-value', type=float, default=0.0,
                        help='Default value for numeric fields')
    parser.add_argument('--default-text-value', type=str, default='DEFAULT_TEXT',
                        help='Default value for text fields')
    parser.add_argument('--special-field-values', type=str,
                        help='JSON string mapping special TEXT fields to their template values')
    
    # Parse arguments or use environment variables
    args = parser.parse_args()
    
    # Use environment variables as fallback for required arguments
    if not args.content_types and 'CONTENT_TYPES' in os.environ:
        args.content_types = os.environ['CONTENT_TYPES']
    if not args.input_variables and 'INPUT_VARIABLES' in os.environ:
        args.input_variables = os.environ['INPUT_VARIABLES']
    
    # Use environment variables as fallback for optional arguments
    if not args.output_dir and 'OUTPUT_DIR' in os.environ:
        args.output_dir = os.environ['OUTPUT_DIR']
    if not args.default_numeric_value and 'DEFAULT_NUMERIC_VALUE' in os.environ:
        args.default_numeric_value = float(os.environ['DEFAULT_NUMERIC_VALUE'])
    if not args.default_text_value and 'DEFAULT_TEXT_VALUE' in os.environ:
        args.default_text_value = os.environ['DEFAULT_TEXT_VALUE']
    if not args.special_field_values and 'SPECIAL_FIELD_VALUES' in os.environ:
        args.special_field_values = os.environ['SPECIAL_FIELD_VALUES']
    
    # Validate required arguments
    missing_args = []
    if not args.content_types:
        missing_args.append('content-types')
    if not args.input_variables:
        missing_args.append('input-variables')
    
    if missing_args:
        logger.error(f"Missing required arguments: {', '.join(missing_args)}")
        parser.print_help()
        sys.exit(1)
    
    # Parse complex arguments
    content_types = parse_content_types(args.content_types)
    input_variables = parse_input_variables(args.input_variables)
    special_field_values = parse_special_field_values(args.special_field_values)
    
    
    # Generate and save payloads
    logger.info(f"Generating payload samples in {args.output_dir}")
    payload_files = save_payloads(
        Path(args.output_dir),
        content_types,
        input_variables,
        args.default_numeric_value,
        args.default_text_value,
        special_field_values
    )
    
    logger.info(f"Successfully generated {len(payload_files)} payload samples in {args.output_dir}")
    logger.info("These files will be automatically uploaded to S3 by the ProcessingStep")


if __name__ == "__main__":
    main()
