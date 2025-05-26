from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__) # Ensure logger is defined at module level


from .config_processing_step_base import ProcessingStepConfigBase


class PackageStepConfig(ProcessingStepConfigBase):
    """Configuration for a model packaging step."""
    packaging_entry_point: str = Field(
        default="mims_package.py", # Default name of the script
        description="Entry point script for packaging, located within the 'source_dir' from BasePipelineConfig."
    )

    # Preference for processing instance size for this specific packaging step
    use_large_processing_instance: bool = Field(
        default=False,
        description="Set to True to use 'processing_instance_type_large', otherwise 'processing_instance_type_small' will be used."
    )
    
    packaging_script_arguments: Optional[List[str]] = Field(
        default=None,
        description="Optional arguments for the packaging script."
    )

    # Input names for clarity in the ProcessingStep definition
    model_input_name_in_job: str = Field(default="model_input", description="Input name for model artifacts in the processing job.")
    scripts_input_name_in_job: str = Field(default="inference_scripts_input", description="Input name for inference scripts in the processing job.")
    
    # Output name for clarity
    packaged_model_output_name_from_job: str = Field(default="packaged_model_output", description="Output name for the packaged model from the processing job.")


    class Config(ProcessingStepConfigBase.Config):
        pass

    @field_validator('packaging_entry_point')
    @classmethod
    def _validate_packaging_entry_point_is_relative(cls, v: str) -> str:
        if not v:
            raise ValueError("packaging_entry_point cannot be empty.")
        if Path(v).is_absolute() or v.startswith('/') or v.startswith('s3://'):
            raise ValueError(f"packaging_entry_point ('{v}') must be a relative path within source_dir, not absolute or S3 URI.")
        return v
    
    @model_validator(mode='after')
    def _validate_entry_point_paths(self, info: ValidationInfo) -> 'PackageStepConfig':
        # 1. Validate packaging_entry_point against the (already resolved) self.source_dir
        if not self.source_dir:
            # This should have been caught by BasePipelineConfig if source_dir was mandatory there
            # or if a default wasn't set. If source_dir is optional in Base and None here,
            # then packaging_entry_point cannot be validated against it locally.
            if self.packaging_entry_point and not self.packaging_entry_point.startswith('s3://'):
                 raise ValueError("source_dir must be defined to locate local packaging_entry_point.")
        elif self.source_dir.startswith('s3://'):
            logger.info(
                f"PackageStepConfig: source_dir ('{self.source_dir}') is S3. "
                f"Assuming packaging_entry_point '{self.packaging_entry_point}' exists within it."
            )
            # The effective code path for SKLearnProcessor will be:
            # s3://bucket/path/to/source_dir/packaging_entry_point.py
            # This assumes SKLearnProcessor's 'code' argument can handle a single S3 script.
            # More typically, for S3, 'code' points to a tar.gz.
            # If source_dir is S3 and contains an uncompressed script, this needs careful handling
            # by the builder when setting the 'code' argument for SKLearnProcessor.
            # For now, this validator assumes the builder will handle it.
        else: # Local source_dir (is absolute and validated from BasePipelineConfig)
            script_full_path = Path(self.source_dir) / self.packaging_entry_point
            if not script_full_path.is_file():
                raise FileNotFoundError(
                    f"Packaging entry point script '{self.packaging_entry_point}' "
                    f"not found within resolved source_dir '{self.source_dir}'. "
                    f"Looked at: '{script_full_path}'"
                )
            logger.info(f"PackageStepConfig: Validated packaging_entry_point '{script_full_path}' exists.")
            # Unlike source_dir, we don't necessarily mutate packaging_entry_point itself,
            # as it's meant to be relative to source_dir. The builder will combine them.
            
        return self
