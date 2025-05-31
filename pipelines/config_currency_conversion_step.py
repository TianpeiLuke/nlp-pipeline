from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import json

from .config_processing_step_base import ProcessingStepConfigBase


class CurrencyConversionConfig(ProcessingStepConfigBase):
    """Configuration for currency conversion processing step."""
    
    # Processing configuration
    processing_entry_point: str = Field(
        default="currency_conversion.py",
        description="Entry point script for currency conversion processing."
    )
    
    use_large_processing_instance: bool = Field(
        default=False,
        description="Whether to use large instance type for processing."
    )

    # Currency conversion specific fields
    marketplace_id_col: str = Field(
        description="Column name containing marketplace IDs"
    )
    
    currency_col: Optional[str] = Field(
        default=None,
        description="Optional column name containing currency codes. If not provided, will use currency from marketplace_info."
    )
    
    currency_conversion_var_list: List[str] = Field(
        default_factory=list,
        description="List of variables to convert currencies"
    )
    
    currency_conversion_dict: Dict[str, float] = Field(
        description="Dictionary mapping currency codes to conversion rates (e.g., {'USD': 1.0, 'EUR': 1.2})"
    )
    
    marketplace_info: Dict[str, Dict[str, str]] = Field(
        description="Dictionary containing marketplace information including currency codes"
    )
    
    enable_currency_conversion: bool = Field(
        default=True,
        description="Whether to enable currency conversion"
    )
    
    # Default values for error handling
    default_currency: str = Field(
        default="USD",
        description="Default currency to use when currency code is missing"
    )
    
    skip_invalid_currencies: bool = Field(
        default=False,
        description="Whether to skip rows with invalid currency codes instead of failing"
    )
    
    # Input/Output configuration
    input_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "data_input": "Input data for currency conversion"
        },
        description="Dictionary mapping input names to their descriptions"
    )
    
    output_names: Dict[str, str] = Field(
        default_factory=lambda: {
            "converted_data": "Data with converted currencies"
        },
        description="Dictionary mapping output names to their descriptions"
    )

    class Config:
        json_encoders = {
            Path: str
        }

    @field_validator('currency_conversion_dict')
    def validate_currency_conversion_dict(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Validate currency conversion dictionary"""
        if not v:
            raise ValueError("currency_conversion_dict cannot be empty")
        
        # Ensure base currency (usually USD) has rate 1.0
        base_currency = next((k for k, v in v.items() if v == 1.0), None)
        if not base_currency:
            raise ValueError("currency_conversion_dict must have one currency with rate 1.0")
            
        # Validate rates
        for currency, rate in v.items():
            if not isinstance(rate, (int, float)):
                raise ValueError(f"Invalid rate for currency {currency}: {rate}")
            if rate <= 0:
                raise ValueError(f"Rate must be positive for currency {currency}")
                
        return v

    @field_validator('currency_conversion_var_list')
    def validate_conversion_vars(cls, v: List[str]) -> List[str]:
        """Validate conversion variable list"""
        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(f"Duplicate variables in currency_conversion_var_list: {duplicates}")
        return v

    @model_validator(mode='after')
    def validate_currency_configs(self) -> 'CurrencyConversionConfig':
        """Validate currency conversion configuration"""
        if self.enable_currency_conversion:
            # Validate required fields
            if not self.marketplace_id_col:
                raise ValueError("marketplace_id_col must be provided when currency conversion is enabled")
            if not self.currency_conversion_var_list:
                raise ValueError("currency_conversion_var_list cannot be empty when currency conversion is enabled")
            if not self.currency_conversion_dict:
                raise ValueError("currency_conversion_dict must be provided when currency conversion is enabled")
            if not self.marketplace_info:
                raise ValueError("marketplace_info must be provided when currency conversion is enabled")

            # Validate currency consistency
            marketplace_currencies = {
                info.get("currency_code") 
                for info in self.marketplace_info.values() 
                if "currency_code" in info
            }
            conversion_currencies = set(self.currency_conversion_dict.keys())
            
            missing_currencies = marketplace_currencies - conversion_currencies
            if missing_currencies and not self.skip_invalid_currencies:
                raise ValueError(
                    f"Missing conversion rates for currencies: {missing_currencies}. "
                    "Either add rates or set skip_invalid_currencies=True"
                )

            # Validate default currency
            if self.default_currency not in self.currency_conversion_dict:
                raise ValueError(f"Default currency {self.default_currency} not found in currency_conversion_dict")

        return self

    def get_script_arguments(self) -> List[str]:
        """Get script arguments for processing step"""
        args = [
            "--marketplace-id-col", self.marketplace_id_col,
            "--default-currency", self.default_currency,
            "--enable-conversion", str(self.enable_currency_conversion).lower()
        ]
        
        if self.currency_col:
            args.extend(["--currency-col", self.currency_col])
            
        if self.skip_invalid_currencies:
            args.append("--skip-invalid-currencies")
            
        return args

    def get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for processing step"""
        return {
            "CURRENCY_CONVERSION_VARS": json.dumps(self.currency_conversion_var_list),
            "CURRENCY_CONVERSION_DICT": json.dumps(self.currency_conversion_dict),
            "MARKETPLACE_INFO": json.dumps(self.marketplace_info)
        }

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization"""
        data = super().model_dump(**kwargs)
        # Add any custom serialization logic here if needed
        return data
