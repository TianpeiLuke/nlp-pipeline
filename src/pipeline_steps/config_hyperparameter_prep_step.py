from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Dict, Optional, Any
from pathlib import Path
import logging

from .config_base import BasePipelineConfig
from .hyperparameters_xgboost import XGBoostModelHyperparameters

logger = logging.getLogger(__name__)


class HyperparameterPrepConfig(BasePipelineConfig):
    """
    Configuration for the Hyperparameter Preparation step.
    
    This step is responsible for serializing hyperparameters to JSON and uploading
    them to S3, making them available for the training step. It uses a Lambda function
    to perform this task, which is more lightweight than a ProcessingStep.
    """

    # XGBoost hyperparameters
    hyperparameters: XGBoostModelHyperparameters = Field(
        description="XGBoost model hyperparameters to be prepared and uploaded to S3."
    )

    # S3 path for the hyperparameter output
    hyperparameters_s3_uri: str = Field(
        description="S3 URI prefix under which hyperparameters.json will be uploaded.",
        pattern=r'^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._-]+)*$'
    )

    # Input/Output names
    input_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {},
        description="Mapping of input channel names to their descriptions."
    )
    
    output_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "hyperparameters_output": "HyperparametersOutput"
        },
        description="Mapping of output channel names to their descriptions."
    )

    # Lambda function timeout in seconds
    lambda_timeout: int = Field(
        default=60,
        description="Timeout for the Lambda function in seconds."
    )

    # Lambda function memory size in MB
    lambda_memory_size: int = Field(
        default=128,
        description="Memory size for the Lambda function in MB."
    )

    class Config(BasePipelineConfig.Config):
        arbitrary_types_allowed = True
        validate_assignment = True

    @model_validator(mode="after")
    def validate_hyperparameters_and_channels(self) -> "HyperparameterPrepConfig":
        """Validate hyperparameters and channel configurations."""

        # Set default output names if None or empty
        if not self.output_names:
            self.output_names = {
                "hyperparameters_output": "HyperparametersOutput"
            }

        # Validate required output channel
        if "hyperparameters_output" not in self.output_names:
            raise ValueError("output_names must contain key 'hyperparameters_output'")

        return self

    @model_validator(mode='before')
    @classmethod
    def _construct_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs S3 paths if they are not explicitly provided."""
        values = super()._construct_base_attributes(values)

        bucket = values.get('bucket')
        current_date = values.get('current_date')
        pipeline_name = values.get('pipeline_name', 'DefaultPipeline')

        # Default S3 prefix under which the builder will write `hyperparameters.json`
        if not values.get('hyperparameters_s3_uri'):
            values['hyperparameters_s3_uri'] = (
                f"s3://{bucket}/{pipeline_name}/training_config/"
                f"{current_date}"
            )

        return values
