from pydantic import Field, field_validator, model_validator
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

from .config_processing_step_base import ProcessingStepConfigBase


class CurrencyConversionConfig(ProcessingStepConfigBase):
    """
    Configuration for currency conversion processing step,
    chained immediately after TabularPreprocessingStep.
    """
    # --- Splitting / chaining fields ---
    job_type: str = Field(
        default="validation",
        description="One of ['training','validation','testing','calibration']"
    )
    mode: str = Field(
        default="per_split",
        description="One of ['per_split','split_after_conversion']"
    )
    train_ratio: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Train fraction when split_after_conversion"
    )
    test_val_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Test vs val split within holdout"
    )
    label_field: str = Field(
        ..., description="Label column name for stratified splitting"
    )

    # --- Processing entry & sizing (inherits other fields) ---
    processing_entry_point: str = Field(
        default="currency_conversion.py",
        description="Entry point script for currency conversion."
    )
    use_large_processing_instance: bool = Field(
        default=False,
        description="Whether to use large instance type."
    )

    # --- Currency conversion parameters ---
    marketplace_id_col: str = Field(..., description="Column with marketplace IDs")
    currency_col: Optional[str] = Field(
        default=None,
        description="Optional column with currency codes; else infer from marketplace_info"
    )
    currency_conversion_var_list: List[str] = Field(
        default_factory=list,
        description="Which numeric columns to convert"
    )
    currency_conversion_dict: Dict[str, float] = Field(
        ..., description="Map currency code → conversion rate"
    )
    marketplace_info: Dict[str, Dict[str, str]] = Field(
        ..., description="Map marketplace ID → {'currency_code':...}"
    )
    enable_currency_conversion: bool = Field(
        default=True,
        description="Turn off conversion if False"
    )
    default_currency: str = Field(
        default="USD",
        description="Fallback currency code"
    )
    skip_invalid_currencies: bool = Field(
        default=False,
        description="If True, fill invalid codes with default_currency"
    )

    # --- IO channel names (must match TabularPreprocessingStep outputs) ---
    input_names: Dict[str, str] = Field(
        default_factory=lambda: {"data_input": "ProcessedTabularData"},
        description="Should match TabularPreprocessingConfig.output_names['processed_data']"
    )
    output_names: Dict[str, str] = Field(
        default_factory=lambda: {"converted_data": "ConvertedCurrencyData"},
        description="Mapping for the single output channel"
    )

    class Config:
        json_encoders = {Path: str}

    @field_validator('job_type')
    def _validate_data_type(cls, v: str) -> str:
        allowed = {"training","validation","testing","calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got {v!r}")
        return v

    @field_validator('mode')
    def _validate_mode(cls, v: str) -> str:
        allowed = {"per_split","split_after_conversion"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}, got {v!r}")
        return v

    @field_validator('currency_conversion_dict')
    def _validate_dict(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("currency_conversion_dict cannot be empty")
        if 1.0 not in v.values():
            raise ValueError("currency_conversion_dict must include a rate of 1.0")
        for k, rate in v.items():
            if rate <= 0:
                raise ValueError(f"Rate for {k} must be positive; got {rate}")
        return v

    @field_validator('currency_conversion_var_list')
    def _validate_vars(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            dup = [x for x in v if v.count(x) > 1]
            raise ValueError(f"Duplicate vars in currency_conversion_var_list: {dup}")
        return v

    @model_validator(mode='after')
    def _validate_full_config(self) -> "CurrencyConversionConfig":
        if self.enable_currency_conversion:
            if not self.marketplace_id_col:
                raise ValueError("marketplace_id_col required when conversion enabled")
            if not self.currency_conversion_var_list:
                raise ValueError("currency_conversion_var_list cannot be empty")
            if not self.marketplace_info:
                raise ValueError("marketplace_info must be provided")
            if self.mode == "split_after_conversion":
                # require label_field for stratification
                if not self.label_field:
                    raise ValueError("label_field required for split_after_conversion")
        return self

    def get_script_arguments(self) -> List[str]:
        """Flags for the argparse in currency_conversion.py"""
        args = [
            "--data-type", self.data_type,
            "--mode", self.mode,
            "--marketplace-id-col", self.marketplace_id_col,
            "--default-currency", self.default_currency,
            "--enable-conversion", str(self.enable_currency_conversion).lower(),
        ]
        if self.currency_col:
            args += ["--currency-col", self.currency_col]
        if self.skip_invalid_currencies:
            args.append("--skip-invalid-currencies")
        return args

    def get_environment_variables(self) -> Dict[str, str]:
        """Env vars so script knows splits and split‐ratios and label field"""
        env = {
            "CURRENCY_CONVERSION_VARS":   json.dumps(self.currency_conversion_var_list),
            "CURRENCY_CONVERSION_DICT":   json.dumps(self.currency_conversion_dict),
            "MARKETPLACE_INFO":           json.dumps(self.marketplace_info),
            "LABEL_FIELD":                self.label_field,
            "TRAIN_RATIO":                str(self.train_ratio),
            "TEST_VAL_RATIO":             str(self.test_val_ratio),
        }
        return env
