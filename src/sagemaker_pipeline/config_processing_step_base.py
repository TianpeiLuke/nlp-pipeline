from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .config_base import BasePipelineConfig


class ProcessingStepConfigBase(BasePipelineConfig):
    """Base configuration for SageMaker Processing Steps."""
    processing_instance_count: int = Field(default=1, ge=1)
    processing_volume_size: int = Field(default=30, ge=10) # GB
    # Which instance type to use can be decided by subclass or further fields
    # processing_instance_type_small from original ModelConfig -> maps to a specific instance type
    # processing_instance_type_large from original ModelConfig -> maps to a specific instance type
    default_processing_instance_type: str = Field(default='ml.m5.large', description="Default instance type for this processing step.")

    class Config(BasePipelineConfig.Config):
        pass