"""
Contract for dummy training step that copies a pretrained model.tar.gz to output location.

This script contract defines the expected input and output paths, environment variables,
and framework requirements for the DummyTraining step, which copies a pretrained model
to make it available for downstream packaging and registration steps.
"""

from .base_script_contract import ScriptContract

DUMMY_TRAINING_CONTRACT = ScriptContract(
    entry_point="dummy_training.py",
    expected_input_paths={
        "pretrained_model_path": "/opt/ml/processing/input/model/model.tar.gz"
    },
    expected_output_paths={
        "model_input": "/opt/ml/processing/output/model"  # Matches specification logical name
    },
    required_env_vars=[],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "pathlib": ">=1.0.0"
    },
    description="Contract for dummy training step that copies a pretrained model.tar.gz to output location"
)
