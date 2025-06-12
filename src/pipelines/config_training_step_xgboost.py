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
    This assumes script mode for XGBoost to align with the builder's design.
    If using built-in XGBoost without a script, entry_point/source_dir would be optional.
    """
    # ----------------------------------------------------
    # 1) S3‐related paths (input/output/checkpoint)
    # ----------------------------------------------------
    input_path: str = Field(
        default=None,
        description="S3 prefix where training data (train/val) is located.",
        pattern=r"^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._-]+)*$",
    )
    output_path: str = Field(
        default=None,
        description="S3 prefix where model artifacts (model.tar.gz) will be saved.",
        pattern=r"^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._-]+)*$",
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional S3 prefix for saving intermediate checkpoints.",
        pattern=r"^s3://[a-zA-Z0-9.-]+(?:/[a-zA-Z0-9._-]+)*$",
    )

    # ----------------------------------------------------
    # 2) Instance configuration for the XGBoost job
    # ----------------------------------------------------
    training_instance_type: str = Field(
        default="ml.m5.xlarge",
        description="Instance type for the XGBoost training job.",
    )
    training_instance_count: int = Field(
        default=1, ge=1, description="Number of instances for the XGBoost training job."
    )
    training_volume_size: int = Field(
        default=30, ge=1, description="EBS volume size (GB) for training instances."
    )

    # ----------------------------------------------------
    # 3) Script mode settings
    # ----------------------------------------------------
    training_entry_point: str = Field(
        default="train_xgb.py",
        description="Relative path (within source_dir) to the training script.",
    )
    # source_dir is inherited from BasePipelineConfig, should contain training_entry_point

    # ----------------------------------------------------
    # 4) XGBoost‐container versions
    # ----------------------------------------------------
    framework_version: str = Field(
        default="1.7-1", description="SageMaker XGBoost container version (e.g. '1.7-1')."
    )
    py_version: str = Field(
        default="py3", description="Python version for the XGBoost container."
    )


    # XGBoost specific hyperparameters
    hyperparameters: XGBoostModelHyperparameters = Field(
        ...,
        description="All XGBoost‐specific and data‐schema hyperparameters.",
    )

    class Config(BasePipelineConfig.Config): # Inherit base config settings
        extra = "forbid"

    # ----------------------------------------------------
    # 6) Before‐validation hook: ensure BasePipelineConfig defaults exist,
    #    then construct sensible default S3 prefixes if user omitted them.
    # ----------------------------------------------------
    @model_validator(mode="before")
    @classmethod
    def _construct_training_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # 6.1) First apply BasePipelineConfig’s before‐hook to fill bucket/author/etc.
        values = super()._construct_base_attributes(values)

        bucket        = values.get("bucket")
        current_date  = values.get("current_date")
        pipeline_name = values.get("pipeline_name", "DefaultPipeline")

        # 6.2) If the user did not explicitly set `input_path`, build a default:
        if not values.get("input_path"):
            values["input_path"] = (
                f"s3://{bucket}/{pipeline_name}/training_input/{current_date}"
            )

        # 6.3) Similarly for `output_path`:
        if not values.get("output_path"):
            values["output_path"] = (
                f"s3://{bucket}/{pipeline_name}/training_output/{current_date}/model"
            )

        # 6.4) If checkpoint_path is entirely missing, give it a default.
        #      (If the user explicitly set checkpoint_path=None, keep it None.)
        if "checkpoint_path" not in values or values.get("checkpoint_path") is None:
            values["checkpoint_path"] = (
                f"s3://{bucket}/{pipeline_name}/training_checkpoints/{current_date}"
            )

        return values

    @model_validator(mode='after')
    def _validate_training_paths_logic(self) -> 'XGBoostTrainingConfig':
        """Validates S3 path requirements for training."""
        paths_to_check: Dict[str, Optional[str]] = {
            'input_path': self.input_path,
            'output_path': self.output_path
        }
        if self.checkpoint_path: # Only add if it's not None
            paths_to_check['checkpoint_path'] = self.checkpoint_path

        # Filter out None paths before checking for uniqueness
        defined_paths = {k: v for k, v in paths_to_check.items() if v is not None}

        if len(set(defined_paths.values())) != len(defined_paths):
            # Identify which paths are duplicated for a clearer error message
            from collections import Counter
            path_counts = Counter(defined_paths.values())
            duplicates = {path: count for path, count in path_counts.items() if count > 1}
            raise ValueError(f"All defined paths (input, output, checkpoint) must be unique. Duplicates found: {duplicates}")
            
        min_depth = 2 # S3 bucket + at least one prefix level
        for path_name, path_value in defined_paths.items():
            if path_value: # Should always be true due to filter, but defensive
                # s3://bucket/prefix1/prefix2 -> ['bucket', 'prefix1', 'prefix2'] -> length 3
                depth = len(path_value.replace("s3://", "").split('/'))
                if depth < min_depth:
                    raise ValueError(
                        f"{path_name} ('{path_value}') must have at least {min_depth} levels of hierarchy (bucket + prefix)."
                    )
        return self
    
    # ----------------------------------------------------
    # 7) After‐validation: ensure none of the S3 paths collide
    # ----------------------------------------------------
    @model_validator(mode="after")
    def _validate_s3_paths_unique(self) -> "XGBoostTrainingConfig":
        defined = {
            "input_path": self.input_path,
            "output_path": self.output_path,
        }
        if self.checkpoint_path:
            defined["checkpoint_path"] = self.checkpoint_path

        # If any two of these prefixes are identical, that would be bad.
        vals = list(defined.values())
        if len(vals) != len(set(vals)):
            # Find duplicates for a clearer error
            from collections import Counter

            cnts = Counter(vals)
            duplicates = {k: v for k, v in cnts.items() if v > 1}
            raise ValueError(
                f"input_path, output_path, and checkpoint_path must all be distinct. "
                f"Duplicates found: {duplicates}"
            )

        # Also check each has at least “bucket + one prefix” (depth ≥ 2)
        for name, uri in defined.items():
            # e.g. "s3://mybucket/my/prefix" → ["mybucket","my","prefix"] length=3
            parts = uri.replace("s3://", "").split("/")
            if len(parts) < 2:
                raise ValueError(
                    f"'{name}' (‘{uri}’) must have at least bucket + one prefix level."
                )

        return self
    
    # ----------------------------------------------------
    # 8) Validate hyperparameter‐related consistency
    # ----------------------------------------------------
    @model_validator(mode="after")
    def _validate_hyperpartition_consistency(self) -> "XGBoostTrainingConfig":
        # Ensure the hyperparameters block was provided
        if not self.hyperparameters:
            raise ValueError("You must supply a non‐empty `hyperparameters` (XGBoostConfig).")

        hp = self.hyperparameters

        # 8.1) Every field in tab_field_list and cat_field_list must appear in full_field_list
        full_set = set(hp.full_field_list)
        if not set(hp.tab_field_list).issubset(full_set):
            raise ValueError(
                "Every tabular field in `hp.tab_field_list` must exist in `hp.full_field_list`."
            )
        if not set(hp.cat_field_list).issubset(full_set):
            raise ValueError(
                "Every categorical field in `hp.cat_field_list` must exist in `hp.full_field_list`."
            )

        # 8.2) label_name and id_name must also be in full_field_list
        if hp.label_name not in full_set:
            raise ValueError(
                f"`hp.label_name` (‘{hp.label_name}’) must be one of `hp.full_field_list`."
            )
        if hp.id_name not in full_set:
            raise ValueError(
                f"`hp.id_name` (‘{hp.id_name}’) must be one of `hp.full_field_list`."
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

    # ----------------------------------------------------
    # 10) Convenience methods
    # ----------------------------------------------------
    def get_checkpoint_uri(self) -> Optional[str]:
        """Return the checkpoint S3 URI (or None if disabled)."""
        return self.checkpoint_path

    def has_checkpoint(self) -> bool:
        """Return True if a checkpoint path is configured, False otherwise."""
        return self.checkpoint_path is not None