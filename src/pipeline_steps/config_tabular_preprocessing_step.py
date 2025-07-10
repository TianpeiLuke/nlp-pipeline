from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Optional, Any, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from .hyperparameters_base import ModelHyperparameters

# Import contract
from ..pipeline_script_contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract

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

    # Note: input_names and output_names have been removed and replaced with script contract

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
    def validate_label(self) -> "TabularPreprocessingConfig":
        """Validate label name."""
        # Validate label name
        if not self.hyperparameters.label_name or not self.hyperparameters.label_name.strip():
            raise ValueError("hyperparameters.label_name must be provided and non‐empty")
            
        return self
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The tabular preprocessing script contract
        """
        return TABULAR_PREPROCESS_CONTRACT
        
    def get_script_path(self) -> str:
        """
        Get script path with priority order:
        1. Use processing_entry_point if provided
        2. Fall back to script_contract.entry_point if available
        
        Always combines with effective source directory.
        
        Returns:
            Script path or None if no entry point can be determined
        """
        # Determine which entry point to use
        entry_point = None
        
        # First priority: Use processing_entry_point if provided
        if self.processing_entry_point:
            entry_point = self.processing_entry_point
        # Second priority: Use contract entry point
        elif hasattr(self, 'script_contract') and self.script_contract and hasattr(self.script_contract, 'entry_point'):
            entry_point = self.script_contract.entry_point
        else:
            return None
        
        # Get the effective source directory
        effective_source_dir = self.get_effective_source_dir()
        if not effective_source_dir:
            return entry_point  # No source dir, just return entry point
        
        # Combine source dir with entry point
        if effective_source_dir.startswith('s3://'):
            full_path = f"{effective_source_dir.rstrip('/')}/{entry_point}"
        else:
            full_path = str(Path(effective_source_dir) / entry_point)
        
        return full_path
