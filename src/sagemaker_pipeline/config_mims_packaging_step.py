from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .config_processing_step_base import ProcessingStepConfigBase


class PackageStepConfig(ProcessingStepConfigBase):
    """Configuration for a model packaging step."""
    packaging_entry_point: str = Field(
        default="mims_package.py", # Default name of the script
        description="Entry point script for packaging, located within the 'source_dir' from BasePipelineConfig."
    )

    # Path to an additional directory of inference code needed as input by the packaging script
    # (e.g., 'online_inference_src' relative to notebook_root, or an S3 URI).
    # If None, this input might be omitted by the builder.
    inference_code_input_path: Optional[str] = Field(
        None,
        description="Path to additional inference code directory to be mounted as an input to the packaging job."
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
    inference_scripts_input_name_in_job: str = Field(default="inference_scripts_input", description="Input name for inference scripts in the processing job.")
    
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

    @field_validator('inference_code_input_path')
    @classmethod
    def _validate_inference_code_input_path_format(cls, v: Optional[str]) -> Optional[str]:
        if v and not (v.startswith('s3://') or isinstance(v, str)): # Basic check
            # Further validation if needed, but existence is checked by builder
            pass
        return v