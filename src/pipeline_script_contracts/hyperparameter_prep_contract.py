"""
Hyperparameter Preparation Script Contract

Defines the contract for the hyperparameter preparation script that serializes
hyperparameters to JSON and uploads them to S3.
"""

from .base_script_contract import ScriptContract

HYPERPARAMETER_PREP_CONTRACT = ScriptContract(
    entry_point="hyperparameter_prep.py",  # Virtual entry point for the Lambda function
    expected_input_paths={},  # No input paths since it uses the hyperparameters directly from config
    expected_output_paths={
        "hyperparameters_s3_uri": "/opt/ml/processing/output/hyperparameters"  # Virtual output path
    },
    required_env_vars=[],  # No required environment variables
    optional_env_vars={},
    framework_requirements={},  # No framework requirements for Lambda
    description="""
    Hyperparameter preparation that:
    1. Serializes hyperparameters from config to JSON format
    2. Uploads the JSON to an S3 location
    3. Returns the S3 URI of the uploaded hyperparameters
    
    This contract is unique because it doesn't represent a processing script
    but rather a Lambda function that operates on the hyperparameters directly
    from the configuration.
    
    Output:
    - S3 URI pointing to the uploaded hyperparameters.json file
    """
)
