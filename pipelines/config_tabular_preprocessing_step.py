from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional, Any
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from .hyperparameters_base import ModelHyperparameters

logger = logging.getLogger(__name__)


class TabularPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Tabular Preprocessing step.
    Inherits from ProcessingStepConfigBase.

    In addition to the usual fields, it now defines:
      - train_ratio    : float in (0,1) fraction for train vs (test+val)
      - test_val_ratio : float in (0,1) fraction for test vs val within the holdout
    """

    # 1) Entry point for the preprocessing script (relative to processing_source_dir)
    processing_entry_point: str = Field(
        default="tabular_preprocess.py",
        description="Relative path (within processing_source_dir) to the tabular preprocessing script."
    )

    # 2) Full set of model hyperparameters, of which we only use label_name here
    hyperparameters: ModelHyperparameters = Field(
        default_factory=ModelHyperparameters,
        description="Model hyperparameters (only label_name is used by the preprocessing step)."
    )

    # 3) Which data_type are we processing?
    job_type: str = Field(
        default='training',
        description="One of ['training','validation','testing','calibration']"
    )

    # 4) Train/Test+Val split ratios (floats in (0,1))
    train_ratio: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of data to allocate to the training set (only used if data_type=='training')."
    )
    test_val_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of the holdout to allocate to the test set vs. validation (only if data_type=='training')."
    )

    # 5) Exactly one input channel, two output channels
    input_names: Dict[str, str] = Field(
        default_factory=lambda: {"data_input": "RawData"},
        description="Mapping of input channel name to its description. (Must contain 'data_input'.)"
    )
    output_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "processed_data": "ProcessedTabularData",
            "full_data":      "FullTabularData"
        },
        description="Mapping of output channel names to their descriptions. (Must contain 'processed_data' & 'full_data'.)"
    )

    class Config(ProcessingStepConfigBase.Config):
        arbitrary_types_allowed = True
        validate_assignment = True

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        """
        Ensure processing_entry_point is a non‐empty relative path.
        """
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non‐empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError("processing_entry_point must be a relative path within source directory")
        return v

    @field_validator("job_type")
    @classmethod
    def validate_data_type(cls, v: str) -> str:
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("train_ratio", "test_val_ratio")
    @classmethod
    def validate_ratios(cls, v: float) -> float:
        """
        Ensure the ratio is strictly between 0 and 1 (not including 0 or 1).
        """
        if not (0.0 < v < 1.0):
            raise ValueError(f"Split ratio must be strictly between 0 and 1, got {v}")
        return v

    @model_validator(mode="after")
    def validate_label_and_channels(self) -> "TabularPreprocessingConfig":
        """
        Cross‐field checks:
         - hyperparameters.label_name must be nonempty and not whitespace.
         - input_names must contain 'data_input'
         - output_names must contain 'processed_data' and 'full_data'
        """
        hp = self.hyperparameters

        if not hp.label_name or not hp.label_name.strip():
            raise ValueError("hyperparameters.label_name must be provided and non‐empty")

        if "data_input" not in self.input_names:
            raise ValueError("input_names must contain key 'data_input'")

        if "processed_data" not in self.output_names or "full_data" not in self.output_names:
            raise ValueError("output_names must contain keys 'processed_data' and 'full_data'")

        return self