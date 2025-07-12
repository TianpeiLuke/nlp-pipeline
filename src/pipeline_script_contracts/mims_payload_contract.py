"""
MIMS Payload Script Contract

Defines the contract for the MIMS payload generation script that creates
sample payloads and metadata for model inference testing.
"""

from .base_script_contract import ScriptContract

MIMS_PAYLOAD_CONTRACT = ScriptContract(
    entry_point="mims_payload.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model"
    },
    expected_output_paths={
        "payload_sample": "/opt/ml/processing/output/payload.tar.gz",
        "payload_metadata": "/opt/ml/processing/output/payload_metadata"
    },
    required_env_vars=[
        # No strictly required environment variables - script has defaults
    ],
    optional_env_vars={
        "CONTENT_TYPES": "application/json",
        "DEFAULT_NUMERIC_VALUE": "0.0",
        "DEFAULT_TEXT_VALUE": "DEFAULT_TEXT",
        # Special field environment variables follow pattern SPECIAL_FIELD_<fieldname>
    },
    framework_requirements={
        "python": ">=3.7"
        # Uses only standard library modules: json, logging, os, tarfile, tempfile, pathlib, enum, typing, datetime
    },
    description="""
    MIMS payload generation script that:
    1. Extracts hyperparameters from model artifacts (model.tar.gz or directory)
    2. Creates model variable list from field information
    3. Generates sample payloads in multiple formats (JSON, CSV)
    4. Creates payload metadata for inference testing
    5. Archives payload files for deployment
    
    Input Structure:
    - /opt/ml/processing/input/model: Model artifacts containing hyperparameters.json
    
    Output Structure:
    - /opt/ml/processing/output/payload_sample/: Sample payload files
    - /opt/ml/processing/output/payload_metadata/: Payload metadata files
    - /opt/ml/processing/output/payload.tar.gz: Archived payload files
    
    Environment Variables:
    - CONTENT_TYPES: Comma-separated list of content types (default: "application/json")
    - DEFAULT_NUMERIC_VALUE: Default value for numeric fields (default: "0.0")
    - DEFAULT_TEXT_VALUE: Default value for text fields (default: "DEFAULT_TEXT")
    - SPECIAL_FIELD_<fieldname>: Custom values for specific fields
    """
)
