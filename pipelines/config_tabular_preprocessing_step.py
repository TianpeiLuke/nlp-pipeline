from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from .hyperparameters_base import ModelHyperparameters

logger = logging.getLogger(__name__)


class TabularPreprocessingConfig(ProcessingStepConfigBase):
    """
    Configuration for the Tabular Preprocessing processing step.
    Inherits from ProcessingStepConfigBase.

    Replaces separate numeric_fields / categorical_fields with a single
    ModelHyperparameters object that contains tab_field_list and cat_field_list.
    """

    # Override entry point to our preprocess script
    processing_entry_point: str = Field(
        default="preprocess.py",
        description="Name of the preprocessing script (relative to processing_source_dir)."
    )

    # Include full set of model hyperparameters
    hyperparameters: ModelHyperparameters = Field(
        default_factory=ModelHyperparameters,
        description="Model hyperparameters (includes tab_field_list, cat_field_list, label_name, etc.)"
    )

    # Number of parallel workers
    n_workers: int = Field(
        default=50,
        ge=1,
        description="Number of parallel worker processes for imputation and mapping."
    )

    # Input/Output channel names
    input_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "data_input": "RawData",
            "config_input": "PreprocessingConfig"
        },
        description="Mapping of input channel names to their descriptions."
    )
    output_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "processed_data": "ProcessedTabularData",
            "full_data": "FullTabularData"
        },
        description="Mapping of output channel names to their descriptions."
    )

    class Config(ProcessingStepConfigBase.Config):
        arbitrary_types_allowed = True
        validate_assignment = True

    @field_validator("n_workers")
    @classmethod
    def validate_n_workers_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_workers must be ≥ 1")
        return v

    @field_validator("processing_entry_point")
    @classmethod
    def validate_entry_point_relative(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v.strip():
            raise ValueError("processing_entry_point must be a non‐empty relative path")
        if Path(v).is_absolute() or v.startswith("/") or v.startswith("s3://"):
            raise ValueError("processing_entry_point must be a relative path within source directory")
        return v

    @model_validator(mode="after")
    def validate_hyperparameters_and_fields(self) -> "TabularPreprocessingConfig":
        """
        Cross‐field checks:
         - Ensure that tab_field_list and cat_field_list come from hyperparameters
         - Ensure label_name is not included in tab_field_list or cat_field_list
         - Ensure input_names contains 'data_input' and 'config_input'
         - Ensure output_names contains 'processed_data' and 'full_data'
        """
        hp = self.hyperparameters

        # Ensure label_name is not in tab or cat lists
        if hp.label_name in hp.tab_field_list or hp.label_name in hp.cat_field_list:
            raise ValueError("hyperparameters.label_name must not appear in tab_field_list or cat_field_list")

        # Check required I/O channel keys
        if "data_input" not in self.input_names or "config_input" not in self.input_names:
            raise ValueError("input_names must contain keys 'data_input' and 'config_input'")
        if "processed_data" not in self.output_names or "full_data" not in self.output_names:
            raise ValueError("output_names must contain keys 'processed_data' and 'full_data'")

        return self
