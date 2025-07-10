from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .hyperparameters_xgboost import XGBoostModelHyperparameters
from .config_base import BasePipelineConfig


class XGBoostTrainingConfig(BasePipelineConfig):
    """
    Configuration specific to the SageMaker XGBoost Training Step.
    This version is streamlined to work with specification-driven architecture.
    Input/output paths are now provided via step specifications and dependencies.
    """
    # Instance configuration
    training_instance_type: str = Field(default='ml.m5.xlarge', description="Instance type for XGBoost training job.")
    training_instance_count: int = Field(default=1, ge=1, description="Number of instances for XGBoost training job.")
    training_volume_size: int = Field(default=30, ge=1, description="Volume size (GB) for training instances.")

    # Script mode configuration
    training_entry_point: str = Field(default='train_xgb.py', description="Entry point script for XGBoost training.")

    # Framework versions for SageMaker XGBoost container
    framework_version: str = Field(default="1.7-1", description="SageMaker XGBoost framework version.")
    py_version: str = Field(default="py3", description="Python version for the SageMaker XGBoost container.")

    # XGBoost specific hyperparameters object
    hyperparameters: XGBoostModelHyperparameters

    class Config(BasePipelineConfig.Config):
        pass


    @model_validator(mode='after')
    def validate_hyperparameter_fields(self) -> 'XGBoostTrainingConfig':
        """
        Validate field lists from hyperparameters.
        """
        # Validate hyperparameters presence
        if not self.hyperparameters:
            raise ValueError("XGBoost hyperparameters must be provided.")

        # Validate field lists
        all_fields = set(self.hyperparameters.full_field_list)
        
        # Check tab_field_list
        if not set(self.hyperparameters.tab_field_list).issubset(all_fields):
            raise ValueError("All fields in tab_field_list must be in full_field_list (from hyperparameters).")
        
        # Check cat_field_list
        if not set(self.hyperparameters.cat_field_list).issubset(all_fields):
            raise ValueError("All fields in cat_field_list must be in full_field_list (from hyperparameters).")
        
        # Check label_name
        if self.hyperparameters.label_name not in all_fields:
            raise ValueError(
                f"label_name '{self.hyperparameters.label_name}' must be in full_field_list (from hyperparameters)."
            )
        
        # Check id_name
        if self.hyperparameters.id_name not in all_fields:
            raise ValueError(
                f"id_name '{self.hyperparameters.id_name}' must be in full_field_list (from hyperparameters)."
            )

        return self

    
    @field_validator('training_instance_type')
    @classmethod
    def _validate_sagemaker_xgboost_instance_type(cls, v: str) -> str:
        # Common CPU instances for XGBoost. XGBoost can also use GPU instances (e.g., ml.g4dn, ml.g5)
        # if tree_method='gpu_hist' is used and framework supports it.
        valid_cpu_instances = [
            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
            "ml.m5.12xlarge", "ml.m5.24xlarge",
            "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge",
            "ml.c5.9xlarge", "ml.c5.18xlarge",
        ]
        valid_gpu_instances = [ # For GPU accelerated XGBoost
            "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g4dn.4xlarge", 
            "ml.g4dn.8xlarge", "ml.g4dn.12xlarge", "ml.g4dn.16xlarge",
            "ml.g5.xlarge", "ml.g5.2xlarge", "ml.g5.4xlarge",
            "ml.g5.8xlarge", "ml.g5.12xlarge", "ml.g5.16xlarge",
             "ml.p3.2xlarge" # Older but sometimes used
        ]
        valid_instances = valid_cpu_instances + valid_gpu_instances
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type for XGBoost: {v}. "
                f"Must be one of: {', '.join(valid_instances)}"
            )
        return v
