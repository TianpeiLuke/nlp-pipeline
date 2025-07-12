from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_base import BasePipelineConfig
from .hyperparameters_xgboost import XGBoostModelHyperparameters

# Import the script contract
from ..pipeline_script_contracts.hyperparameter_prep_contract import HYPERPARAMETER_PREP_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract

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
    def validate_hyperparameters(self) -> "HyperparameterPrepConfig":
        """Validate hyperparameters."""
        if not self.hyperparameters:
            raise ValueError("hyperparameters must be provided and non-empty")
            
        if not self.hyperparameters_s3_uri:
            raise ValueError("hyperparameters_s3_uri must be provided and non-empty")
            
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
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The hyperparameter preparation script contract
        """
        return HYPERPARAMETER_PREP_CONTRACT
