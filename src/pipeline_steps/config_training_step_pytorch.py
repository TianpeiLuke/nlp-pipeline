from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .hyperparameters_base import ModelHyperparameters
from .config_base import BasePipelineConfig


class PyTorchTrainingConfig(BasePipelineConfig):
    """Configuration specific to the SageMaker Training Step."""
    # Input/output names for training with default values
    input_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "input_path": "data"  # KEY: logical name, VALUE: script input name
        },
        description="Mapping of logical input channel names to their script input names."
    )
    
    output_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "model_output": "ModelArtifacts",
            "metrics_output": "TrainingMetrics",
            "training_job_name": "TrainingJobName"
        },
        description="Mapping of output channel names to their descriptions."
    )
    
    # S3 paths with updated pattern
    input_path: str = Field(
        description="S3 path for input data",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )
    output_path: str = Field(
        description="S3 path for output data",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional S3 path for model checkpoints",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )

    training_instance_type: str = Field(default='ml.g5.12xlarge', description="Instance type for training job.")
    training_instance_count: int = Field(default=1, ge=1, description="Number of instances for training job.")
    training_volume_size: int = Field(default=30, ge=1, description="Volume size (GB) for training instances.")
    training_entry_point: str = Field(default='train.py', description="Entry point script for training.")
    # source_dir is inherited from BasePipelineConfig, assumed to contain training_entry_point

    # Framework versions for SageMaker PyTorch container
    framework_version: str = Field(default="1.12.0", description="SageMaker PyTorch framework version.")
    py_version: str = Field(default="py38", description="Python version for the SageMaker PyTorch container.")

    # Hyperparameters are now a separate object, linked at the MasterWorkflowConfig level
    # Add reference to hyperparameters
    hyperparameters: Optional[ModelHyperparameters] = Field(None, description="Model hyperparameters")

    class Config(BasePipelineConfig.Config): # Inherit base config settings
        pass

    @model_validator(mode='before')
    @classmethod
    def _construct_training_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs S3 paths specific to training if not provided."""
        # Ensure base attributes like bucket and current_date are present from BasePipelineConfig's validator
        # (which should have run if this ModelConfig is part of a hierarchy correctly parsed)
        # For standalone instantiation, ensure they are passed or defaulted.
        bucket = values.get('bucket')
        current_date = values.get('current_date')
        pipeline_name = values.get('pipeline_name', 'DefaultPipeline')

        if not bucket or not current_date: # Should be set by BasePipelineConfig._construct_base_attributes
            # This might indicate an issue in how Pydantic calls validators in hierarchy
            # or how the master config passes/constructs these.
            # For safety, one might re-fetch/default them if BasePipelineConfig's validator hasn't run yet for these values.
            pass # Assuming they are already populated by BasePipelineConfig validator

        if 'input_path' not in values or values['input_path'] is None:
            values['input_path'] = f"s3://{bucket}/{pipeline_name}/training_input/{current_date}"
        if 'output_path' not in values or values['output_path'] is None:
            values['output_path'] = f"s3://{bucket}/{pipeline_name}/training_output/{current_date}/model"
        if 'checkpoint_path' not in values: # Allow explicit None
             values['checkpoint_path'] = f"s3://{bucket}/{pipeline_name}/training_checkpoints/{current_date}"
             
        # Normalize all paths to ensure no trailing slashes
        for path_key in ['input_path', 'output_path', 'checkpoint_path']:
            if path_key in values and values.get(path_key):
                values[path_key] = values[path_key].rstrip('/')
                
        return values

    @model_validator(mode='after')
    def _validate_training_paths_logic(self) -> 'PyTorchTrainingConfig':
        """Validates S3 path requirements for training."""
        # Example: Ensure input_path and output_path are different
        paths = {
            'input_path': self.input_path,
            'output_path': self.output_path
        }
        if self.checkpoint_path:
            paths['checkpoint_path'] = self.checkpoint_path

        # Check for uniqueness
        if len(set(paths.values())) != len(paths):
            raise ValueError("All paths (input, output, checkpoint) must be different")
            
        # Validate minimum path depths
        min_depth = 2
        for path_name, path in paths.items():
            depth = len(path.split('/')[3:])
            if depth < min_depth:
                raise ValueError(f"{path_name} must have at least {min_depth} levels of hierarchy")

        return self


    @model_validator(mode='after')
    def validate_field_lists(self) -> 'PyTorchTrainingConfig':
        """Validate field lists from hyperparameters"""
        if not self.hyperparameters:
            raise ValueError("hyperparameters must be provided")

        # Check if all fields in tab_field_list and cat_field_list are in full_field_list
        all_fields = set(self.hyperparameters.full_field_list)
        
        if not set(self.hyperparameters.tab_field_list).issubset(all_fields):
            raise ValueError("All fields in tab_field_list must be in full_field_list")
            
        if not set(self.hyperparameters.cat_field_list).issubset(all_fields):
            raise ValueError("All fields in cat_field_list must be in full_field_list")
        
        # Check if label_name and id_name are in full_field_list
        if self.hyperparameters.label_name not in all_fields:
            raise ValueError(
                f"label_name '{self.hyperparameters.label_name}' must be in full_field_list"
            )
            
        if self.hyperparameters.id_name not in all_fields:
            raise ValueError(
                f"id_name '{self.hyperparameters.id_name}' must be in full_field_list"
            )

        return self


    @model_validator(mode='after')
    def set_default_names(self) -> 'PyTorchTrainingConfig':
        """Ensure default input and output names are set if not provided."""
        if not self.input_names:
            self.input_names = {
                "input_path": "data"
            }
        
        if not self.output_names:
            self.output_names = {
                "model_output": "ModelArtifacts",
                "metrics_output": "TrainingMetrics",
                "training_job_name": "TrainingJobName"
            }
        
        return self
    
    @field_validator('training_instance_type')
    @classmethod
    def _validate_sagemaker_training_instance_type(cls, v: str) -> str:
        valid_instances = [
            "ml.g4dn.16xlarge", 
            "ml.g5.12xlarge", 
            "ml.g5.16xlarge",
            "ml.p3.8xlarge", 
            "ml.m5.12xlarge",
            "ml.p3.16xlarge"
        ]
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type: {v}. "
                f"Must be one of: {', '.join(valid_instances)}"
            )
        return v

    
    def get_checkpoint_uri(self) -> Optional[str]:
        return self.checkpoint_path

    def has_checkpoint(self) -> bool:
        return self.checkpoint_path is not None
