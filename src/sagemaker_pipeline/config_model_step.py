from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .config_base import BasePipelineConfig


class ModelCreationConfig(BasePipelineConfig): # Renamed from ModelStepConfig for clarity
    """Configuration specific to the SageMaker Model creation (for inference)."""
    inference_instance_type: str = Field(default='ml.m5.large', description="Instance type for inference endpoint/transform job.")
    inference_entry_point: str = Field(default='inference.py', description="Entry point script for inference.")
    # source_dir is inherited from BasePipelineConfig, assumed to contain inference_entry_point
    # framework_version, py_version are inherited from BasePipelineConfig

    # Endpoint / Container specific settings
    initial_instance_count: int = Field(default=1, ge=1, description="Initial instance count for endpoint (used by EndpointConfig).")
    container_startup_health_check_timeout: int = Field(default=300, ge=60, description="Container startup health check timeout (seconds).")
    container_memory_limit: int = Field(default=6144, ge=1024, description="Container memory limit (MB).")
    data_download_timeout: int = Field(default=900, ge=60, description="Model data download timeout (seconds).")
    inference_memory_limit: int = Field(default=6144, ge=1024, description="Inference memory limit (MB).")
    max_concurrent_invocations: int = Field(default=10, ge=1, description="Max concurrent invocations per instance.") # Increased default
    max_payload_size: int = Field(default=6, ge=1, le=100, description="Max payload size (MB) for inference.") # Increased range

    class Config(BasePipelineConfig.Config):
        pass

    @field_validator('inference_instance_type')
    @classmethod
    def _validate_sagemaker_inference_instance_type(cls, v: str) -> str:
        if not v.startswith('ml.'):
            raise ValueError(f"Invalid inference instance type: {v}. Must start with 'ml.'")
        return v
        
    @model_validator(mode='after')
    def _validate_memory_constraints(self) -> 'ModelCreationConfig':
        if self.inference_memory_limit > self.container_memory_limit:
            raise ValueError("Inference memory limit cannot exceed container memory limit.")
        return self