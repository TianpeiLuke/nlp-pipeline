# src/pipeline_steps/config_batch_transform_step.py

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, Dict, Any
from .config_base import BasePipelineConfig


class BatchTransformStepConfig(BasePipelineConfig):
    """
    Configuration for a generic SageMaker BatchTransform step.
    Inherits all the BasePipelineConfig attributes (bucket, region, etc.)
    and adds just what's needed to drive a TransformStep.
    """
    

    # 1) Which slice are we scoring?
    job_type: str = Field(
        ...,
        description="One of 'training','testing','validation','calibration' to indicate which slice to transform"
    )

    # 2) Input / output S3 URIs
    batch_input_location: str = Field(
        ...,
        description="S3 URI prefix for the input data (e.g. 's3://my-bucket/test/')"
    )
    batch_output_location: str = Field(
        ...,
        description="S3 URI prefix for the transform outputs (e.g. 's3://my-bucket/test-scores/')"
    )

    # 3) Compute sizing
    transform_instance_type: str = Field(
        default="ml.m5.large",
        description="Instance type for the BatchTransform job"
    )
    transform_instance_count: int = Field(
        default=1, ge=1,
        description="Number of instances for the BatchTransform job"
    )

    # 4) Content negotiation & splitting
    content_type: str = Field(
        default="text/csv",
        description="MIME type of the input data"
    )
    accept: str = Field(
        default="text/csv",
        description="Response MIME type so output_fn knows how to format"
    )
    split_type: str = Field(
        default="Line",
        description="How to split the input file (must match your container’s input_fn)"
    )
    assemble_with: Optional[str] = Field(
        default="Line",
        description="How to re‐assemble input+output when join_source='Input'"
    )

    # 5) Optional JMESPath filters
    input_filter: Optional[str] = Field(
        default="$[1:]",
        description="JMESPath filter on each input record (e.g. '$[1:]')"
    )
    output_filter: Optional[str] = Field(
        default="$[-1]",
        description="JMESPath filter on each joined record (e.g. '$[-1]')"
    )

    # 6) Join strategy
    join_source: str = Field(
        default="Input",
        description="Whether to join on the 'Input' or 'Output' stream"
    )

    # 7) Input/output names for batch transform
    input_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "model_name": "The name of the SageMaker model (string or Properties)"
        },
        description="Mapping of input channel names to their descriptions."
    )
    
    output_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "transform_output": "S3 location of the batch transform output",
            "batch_transform_job_name": "Name of the batch transform job"
        },
        description="Mapping of output channel names to their descriptions."
    )

    class Config(BasePipelineConfig.Config):
        """inherit all your BasePipelineConfig config settings"""
        pass

    @field_validator("job_type")
    def _validate_job_type(cls, v: str) -> str:
        allowed = {"training", "testing", "validation", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("batch_input_location", "batch_output_location")
    def _validate_s3_uri(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError(f"must be a valid S3 URI, got '{v}'")
        return v

    @field_validator("transform_instance_type")
    def _validate_instance_type(cls, v: str) -> str:
        if not v.startswith("ml."):
            raise ValueError(f"invalid instance type '{v}', must start with 'ml.'")
        return v

    @model_validator(mode='after')
    def validate_config(self) -> 'BatchTransformStepConfig':
        """Validate join and assemble configurations."""
        split = self.split_type
        assemble = self.assemble_with
        join = self.join_source
        if join == "Input" and assemble and assemble != split:
            raise ValueError("when join_source='Input', assemble_with must equal split_type")
        return self

    @model_validator(mode='after')
    def set_default_names(self) -> 'BatchTransformStepConfig':
        """Ensure default input and output names are set if not provided."""
        if not self.input_names:
            self.input_names = {
                "model_name": "The name of the SageMaker model (string or Properties)"
            }
        
        if not self.output_names:
            self.output_names = {
                "transform_output": "S3 location of the batch transform output",
                "batch_transform_job_name": "Name of the batch transform job"
            }
        
        return self
