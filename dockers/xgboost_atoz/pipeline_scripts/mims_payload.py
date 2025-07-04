#!/usr/bin/env python
"""
MIMS Payload Generation Processing Script

This script reads field information from hyperparameters extracted from model.tar.gz,
extracts configuration from environment variables,
and creates payload files for model inference.
"""
import json
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from enum import Enum
from typing import List, Dict, Any, Union
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for environment variable names
ENV_CONTENT_TYPES = "CONTENT_TYPES"
ENV_DEFAULT_NUMERIC_VALUE = "DEFAULT_NUMERIC_VALUE"
ENV_DEFAULT_TEXT_VALUE = "DEFAULT_TEXT_VALUE"
ENV_SPECIAL_FIELD_PREFIX = "SPECIAL_FIELD_"

# Fixed input/output directories
INPUT_MODEL_DIR = "/opt/ml/processing/input/model"
OUTPUT_DIR = "/opt/ml/processing/output"
PAYLOAD_SAMPLE_DIR = os.path.join(OUTPUT_DIR, "payload_sample")
PAYLOAD_METADATA_DIR = os.path.join(OUTPUT_DIR, "payload_metadata")
WORKING_DIRECTORY = "/tmp/mims_payload_work"

class VariableType(str, Enum):
    """Type of variable in model input/output"""
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"

def create_model_variable_list(
    full_field_list: List[str],
    tab_field_list: List[str],
    cat_field_list: List[str],
    label_name: str = "label",
    id_name: str = "id"
) -> List[List[str]]:
    """
    Creates a list of [variable_name, variable_type] pairs.
    
    Args:
        full_field_list: List of all field names
        tab_field_list: List of numeric/tabular field names
        cat_field_list: List of categorical field names
        label_name: Name of the label column (default: "label")
        id_name: Name of the ID column (default: "id")
        
    Returns:
        List[List[str]]: List of [variable_name, type] pairs where type is 'NUMERIC' or 'TEXT'
    """
    model_var_list = []

    for field in full_field_list:
        # Skip label and id fields
        if field in [label_name, id_name]:
            continue

        # Determine field type
        if field in tab_field_list:
            field_type = 'NUMERIC'
        elif field in cat_field_list:
            field_type = 'TEXT'
        else:
            # For any fields not explicitly categorized, default to TEXT
            field_type = 'TEXT'
        
        # Add [field_name, field_type] pair
        model_var_list.append([field, field_type])

    return model_var_list

def extract_hyperparameters_from_tarball() -> Dict:
    """Extract and load hyperparameters from model artifacts"""
    # The builder step has been updated to use the directory as destination, not model.tar.gz
    # But we'll keep the name for backward compatibility and handle both cases
    input_model_path = os.path.join(INPUT_MODEL_DIR, "model.tar.gz")
    input_model_dir = INPUT_MODEL_DIR
    logger.info(f"Looking for hyperparameters in model artifacts")
    
    # Create temporary directory for extraction
    os.makedirs(WORKING_DIRECTORY, exist_ok=True)
    
    hyperparams_path = None
    
    # First check if model.tar.gz exists and is a file (original case)
    if os.path.exists(input_model_path) and os.path.isfile(input_model_path):
        logger.info(f"Found model.tar.gz file at {input_model_path}")
        try:
            # Extract just the hyperparameters.json file from tarball
            with tarfile.open(input_model_path, "r:gz") as tar:
                # Check if hyperparameters.json exists in the tarball
                hyperparams_info = None
                for member in tar.getmembers():
                    if member.name == 'hyperparameters.json':
                        hyperparams_info = member
                        break
                
                if not hyperparams_info:
                    # List contents for debugging
                    contents = [m.name for m in tar.getmembers()]
                    logger.error(f"hyperparameters.json not found in tarball. Contents: {contents}")
                    # Don't raise error here, continue checking other locations
                else:
                    # Extract only the hyperparameters file
                    tar.extract(hyperparams_info, WORKING_DIRECTORY)
                    hyperparams_path = os.path.join(WORKING_DIRECTORY, "hyperparameters.json")
        except Exception as e:
            logger.warning(f"Error processing model.tar.gz as tarfile: {e}")
            # Continue to other methods
    
    # Next check if model.tar.gz exists but is a directory (the error case we're fixing)
    if hyperparams_path is None and os.path.exists(input_model_path) and os.path.isdir(input_model_path):
        logger.info(f"{input_model_path} is a directory, looking for hyperparameters.json inside")
        direct_hyperparams_path = os.path.join(input_model_path, "hyperparameters.json")
        if os.path.exists(direct_hyperparams_path):
            logger.info(f"Found hyperparameters.json directly in the model.tar.gz directory")
            hyperparams_path = direct_hyperparams_path
    
    # Finally check if hyperparameters.json exists directly in the input model directory
    if hyperparams_path is None:
        logger.info(f"Looking for hyperparameters.json directly in {input_model_dir}")
        direct_hyperparams_path = os.path.join(input_model_dir, "hyperparameters.json")
        if os.path.exists(direct_hyperparams_path):
            logger.info(f"Found hyperparameters.json directly in the input model directory")
            hyperparams_path = direct_hyperparams_path
    
    # If we still haven't found it, search recursively
    if hyperparams_path is None:
        logger.info(f"Searching recursively for hyperparameters.json in {input_model_dir}")
        for root, dirs, files in os.walk(input_model_dir):
            if "hyperparameters.json" in files:
                hyperparams_path = os.path.join(root, "hyperparameters.json")
                logger.info(f"Found hyperparameters.json at {hyperparams_path}")
                break
    
    # If still not found, raise error
    if hyperparams_path is None:
        logger.error(f"hyperparameters.json not found in any location")
        # List directory contents for debugging
        contents = []
        for root, dirs, files in os.walk(input_model_dir):
            for file in files:
                contents.append(os.path.join(root, file))
        logger.error(f"Directory contents: {contents}")
        raise FileNotFoundError("hyperparameters.json not found in model artifacts")
    
    # Load the hyperparameters
    with open(hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
    
    # Copy to working directory if not already there
    if not hyperparams_path.startswith(WORKING_DIRECTORY):
        import shutil
        dest_path = os.path.join(WORKING_DIRECTORY, "hyperparameters.json")
        shutil.copy2(hyperparams_path, dest_path)
        
    logger.info(f"Successfully loaded hyperparameters: {list(hyperparams.keys())}")
    return hyperparams

def get_environment_content_types() -> List[str]:
    """Get content types from environment variables."""
    content_types_str = os.environ.get(ENV_CONTENT_TYPES, "application/json")
    return [ct.strip() for ct in content_types_str.split(',')]

def get_environment_default_numeric_value() -> float:
    """Get default numeric value from environment variables."""
    try:
        return float(os.environ.get(ENV_DEFAULT_NUMERIC_VALUE, "0.0"))
    except ValueError:
        logger.warning(f"Invalid {ENV_DEFAULT_NUMERIC_VALUE}, using default 0.0")
        return 0.0

def get_environment_default_text_value() -> str:
    """Get default text value from environment variables."""
    return os.environ.get(ENV_DEFAULT_TEXT_VALUE, "DEFAULT_TEXT")

def get_environment_special_fields() -> Dict[str, str]:
    """Get special field values from environment variables."""
    special_fields = {}
    for env_var, env_value in os.environ.items():
        if env_var.startswith(ENV_SPECIAL_FIELD_PREFIX):
            field_name = env_var[len(ENV_SPECIAL_FIELD_PREFIX):].lower()
            special_fields[field_name] = env_value
    return special_fields

def get_field_default_value(
    field_name: str, 
    var_type: str, 
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str]
) -> str:
    """Get default value for a field"""
    if var_type == "TEXT" or var_type == VariableType.TEXT:
        if field_name in special_field_values:
            template = special_field_values[field_name]
            try:
                return template.format(
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
            except KeyError as e:
                raise ValueError(f"Invalid placeholder in template for field '{field_name}': {str(e)}")
        return default_text_value
    elif var_type == "NUMERIC" or var_type == VariableType.NUMERIC:
        return str(default_numeric_value)
    else:
        raise ValueError(f"Unknown variable type: {var_type}")

def generate_csv_payload(
    input_vars,
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str]
) -> str:
    """
    Generate CSV format payload following the order in input_vars.
    
    Returns:
        Comma-separated string of values
    """
    values = []
    
    if isinstance(input_vars, dict):
        # Dictionary format
        for field_name, var_type in input_vars.items():
            values.append(get_field_default_value(
                field_name, var_type, default_numeric_value, default_text_value, special_field_values
            ))
    else:
        # List format
        for field_name, var_type in input_vars:
            values.append(get_field_default_value(
                field_name, var_type, default_numeric_value, default_text_value, special_field_values
            ))
            
    return ",".join(values)

def generate_json_payload(
    input_vars,
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str]
) -> str:
    """
    Generate JSON format payload using input_vars.
    
    Returns:
        JSON string with field names and values
    """
    payload = {}
    
    if isinstance(input_vars, dict):
        # Dictionary format
        for field_name, var_type in input_vars.items():
            payload[field_name] = get_field_default_value(
                field_name, var_type, default_numeric_value, default_text_value, special_field_values
            )
    else:
        # List format
        for field_name, var_type in input_vars:
            payload[field_name] = get_field_default_value(
                field_name, var_type, default_numeric_value, default_text_value, special_field_values
            )
            
    return json.dumps(payload)

def generate_sample_payloads(
    input_vars,
    content_types: List[str],
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str]
) -> List[Dict[str, Union[str, dict]]]:
    """
    Generate sample payloads for each content type.
    
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
                input_vars, default_numeric_value, default_text_value, special_field_values
            )
        elif content_type == "application/json":
            payload_info["payload"] = generate_json_payload(
                input_vars, default_numeric_value, default_text_value, special_field_values
            )
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        payloads.append(payload_info)
    
    return payloads

def save_payloads(
    output_dir: str,
    input_vars,
    content_types: List[str],
    default_numeric_value: float,
    default_text_value: str,
    special_field_values: Dict[str, str]
) -> List[str]:
    """
    Save payloads to files.
    
    Args:
        output_dir: Directory to save payload files
        input_vars: Source model inference input variable list
        content_types: List of content types to generate payloads for
        default_numeric_value: Default value for numeric fields
        default_text_value: Default value for text fields
        special_field_values: Dictionary of special field values
            
    Returns:
        List of paths to created payload files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_paths = []
    payloads = generate_sample_payloads(
        input_vars, content_types, default_numeric_value, default_text_value, special_field_values
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
            
        file_paths.append(str(file_path))
        logger.info(f"Created payload file: {file_path}")
        
    return file_paths

def create_payload_archive(payload_files: List[str]) -> str:
    """
    Create a tar.gz archive containing only payload files (not metadata).
    
    Args:
        payload_files: List of paths to payload files
        
    Returns:
        Path to the created archive
    """
    # Create archive in the output directory
    archive_path = os.path.join(OUTPUT_DIR, "payload.tar.gz")
    
    with tarfile.open(archive_path, "w:gz") as tar:
        # Add payload files
        for file_path in payload_files:
            # Add file to archive with basename as name
            tar.add(file_path, arcname=os.path.basename(file_path))
    
    logger.info(f"Created payload archive: {archive_path}")
    return archive_path

def main():
    """Main entry point for the script."""
    # Extract hyperparameters from model tarball
    hyperparams = extract_hyperparameters_from_tarball()
    
    # Extract field information from hyperparameters
    full_field_list = hyperparams.get('full_field_list', [])
    tab_field_list = hyperparams.get('tab_field_list', [])
    cat_field_list = hyperparams.get('cat_field_list', [])
    label_name = hyperparams.get('label_name', 'label')
    id_name = hyperparams.get('id_name', 'id')
    
    # Create variable list
    adjusted_full_field_list = tab_field_list + cat_field_list
    var_type_list = create_model_variable_list(
        adjusted_full_field_list, tab_field_list, cat_field_list, 
        label_name, id_name
    )
    
    # Get parameters from environment variables
    content_types = get_environment_content_types()
    default_numeric_value = get_environment_default_numeric_value()
    default_text_value = get_environment_default_text_value()
    special_field_values = get_environment_special_fields()
    
    # Extract pipeline name and version from hyperparams
    pipeline_name = hyperparams.get('pipeline_name', 'default_pipeline')
    pipeline_version = hyperparams.get('pipeline_version', '1.0.0')
    model_objective = hyperparams.get('model_registration_objective', None)
    
    # Create output directories
    os.makedirs(PAYLOAD_SAMPLE_DIR, exist_ok=True)
    os.makedirs(PAYLOAD_METADATA_DIR, exist_ok=True)
    
    # Generate and save payloads to the sample directory
    payload_file_paths = save_payloads(
        PAYLOAD_SAMPLE_DIR,
        var_type_list,
        content_types,
        default_numeric_value,
        default_text_value,
        special_field_values
    )
    
    # Save metadata about the payloads to metadata directory
    metadata = {
        'input_var_list': var_type_list,
        'content_types': content_types,
        'pipeline_name': pipeline_name,
        'pipeline_version': pipeline_version,
        'model_objective': model_objective
    }
    
    metadata_file_path = os.path.join(PAYLOAD_METADATA_DIR, 'payload_metadata.json')
    with open(metadata_file_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create tar.gz archive of only payload files (not metadata)
    archive_path = create_payload_archive(payload_file_paths)
    
    logger.info(f"MIMS payload generation complete.")
    logger.info(f"Payload files saved to: {PAYLOAD_SAMPLE_DIR}")
    logger.info(f"Metadata files saved to: {PAYLOAD_METADATA_DIR}")
    logger.info(f"Payload archive saved to: {archive_path}")

if __name__ == '__main__':
    main()
