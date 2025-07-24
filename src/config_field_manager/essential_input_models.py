"""
Essential Input Models module.

This module defines Pydantic models for the essential user inputs (Tier 1)
in the three-tier configuration architecture. These models represent the
three key areas of configuration: Data, Model, and Registration.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator


class DateRangePeriod(BaseModel):
    """
    Represents a date range period for training or calibration
    """
    start_date: datetime = Field(..., description="Start date of the period")
    end_date: datetime = Field(..., description="End date of the period")

    @validator('end_date')
    def end_date_must_be_after_start_date(cls, end_date, values):
        """Validate that end_date is after start_date"""
        if 'start_date' in values and end_date <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return end_date


class DataConfig(BaseModel):
    """
    Essential data configuration for the XGBoost evaluation pipeline.
    
    This model represents the core data-related inputs that users must provide.
    """
    region: str = Field(..., description="Region code (NA, EU, FE)")
    training_period: DateRangePeriod = Field(
        ..., description="Training data date range"
    )
    calibration_period: DateRangePeriod = Field(
        ..., description="Calibration data date range"
    )
    feature_groups: Dict[str, bool] = Field(
        ..., description="Feature groups to include (group_name -> include_bool)"
    )
    custom_fields: List[str] = Field(
        default_factory=list, 
        description="Additional custom fields to include beyond feature groups"
    )
    
    @property
    def tab_field_list(self) -> List[str]:
        """
        Generate tabular field list from selected feature groups and custom fields
        
        Returns:
            List[str]: Combined list of tabular fields
        """
        # This is a simplified implementation - in a real system, we would
        # need to map feature groups to actual field names based on region
        # and potentially other factors.
        fields = []
        if self.custom_fields:
            fields.extend(self.custom_fields)
        return fields
    
    @property
    def cat_field_list(self) -> List[str]:
        """
        Generate categorical field list from selected feature groups
        
        Returns:
            List[str]: List of categorical fields
        """
        # This is a simplified implementation
        return []
    
    @property
    def full_field_list(self) -> List[str]:
        """
        Generate the full field list (combination of tabular and categorical)
        
        Returns:
            List[str]: Combined list of all fields
        """
        return sorted(set(self.tab_field_list + self.cat_field_list))


class ModelConfig(BaseModel):
    """
    Essential model configuration for the XGBoost evaluation pipeline.
    
    This model represents the core model-related inputs that users must provide.
    """
    is_binary: bool = Field(
        True, description="Whether this is a binary classification model"
    )
    label_name: str = Field(
        "is_abuse", description="Name of the target/label field"
    )
    id_name: str = Field(
        "order_id", description="Name of the ID field"
    )
    marketplace_id_col: str = Field(
        "marketplace_id", description="Name of the marketplace ID column"
    )
    num_round: int = Field(
        300, description="Number of boosting rounds"
    )
    max_depth: int = Field(
        10, description="Maximum depth of a tree"
    )
    min_child_weight: int = Field(
        1, description="Minimum sum of instance weight needed in a child"
    )
    
    @validator('num_round')
    def validate_num_round(cls, v):
        """Validate that num_round is positive"""
        if v <= 0:
            raise ValueError('num_round must be positive')
        return v
    
    @validator('max_depth')
    def validate_max_depth(cls, v):
        """Validate that max_depth is positive"""
        if v <= 0:
            raise ValueError('max_depth must be positive')
        return v


class RegistrationConfig(BaseModel):
    """
    Essential registration configuration for the XGBoost evaluation pipeline.
    
    This model represents the core model registration inputs that users must provide.
    """
    model_owner: str = Field(
        ..., description="Owner of the model (e.g., team name)"
    )
    model_registration_domain: str = Field(
        ..., description="Domain for model registration"
    )
    expected_tps: int = Field(
        2, description="Expected transactions per second"
    )
    max_latency_ms: int = Field(
        800, description="Maximum allowed latency in milliseconds"
    )
    max_error_rate: float = Field(
        0.2, description="Maximum acceptable error rate"
    )
    
    @validator('expected_tps')
    def validate_expected_tps(cls, v):
        """Validate that expected_tps is positive"""
        if v <= 0:
            raise ValueError('expected_tps must be positive')
        return v
    
    @validator('max_latency_ms')
    def validate_max_latency_ms(cls, v):
        """Validate that max_latency_ms is positive"""
        if v <= 0:
            raise ValueError('max_latency_ms must be positive')
        return v
    
    @validator('max_error_rate')
    def validate_max_error_rate(cls, v):
        """Validate that max_error_rate is between 0 and 1"""
        if v < 0 or v > 1:
            raise ValueError('max_error_rate must be between 0 and 1')
        return v


class EssentialInputs(BaseModel):
    """
    Complete essential inputs for the XGBoost evaluation pipeline.
    
    This model combines all essential user inputs across data, model, and registration.
    """
    data: DataConfig = Field(..., description="Data configuration")
    model: ModelConfig = Field(..., description="Model configuration")
    registration: RegistrationConfig = Field(..., description="Registration configuration")
    
    def expand(self) -> Dict[str, Any]:
        """
        Expand essential inputs into a flattened dictionary of configuration values
        
        Returns:
            Dict[str, Any]: Flattened configuration dictionary
        """
        # Start with an empty dictionary
        expanded = {}
        
        # Add data configuration values
        expanded["region"] = self.data.region
        expanded["training_start_datetime"] = self.data.training_period.start_date
        expanded["training_end_datetime"] = self.data.training_period.end_date
        expanded["calibration_start_datetime"] = self.data.calibration_period.start_date
        expanded["calibration_end_datetime"] = self.data.calibration_period.end_date
        expanded["tab_field_list"] = self.data.tab_field_list
        expanded["cat_field_list"] = self.data.cat_field_list
        expanded["full_field_list"] = self.data.full_field_list
        
        # Add model configuration values
        expanded["is_binary"] = self.model.is_binary
        expanded["label_name"] = self.model.label_name
        expanded["id_name"] = self.model.id_name
        expanded["marketplace_id_col"] = self.model.marketplace_id_col
        expanded["num_round"] = self.model.num_round
        expanded["max_depth"] = self.model.max_depth
        expanded["min_child_weight"] = self.model.min_child_weight
        
        # Add registration configuration values
        expanded["model_owner"] = self.registration.model_owner
        expanded["model_registration_domain"] = self.registration.model_registration_domain
        expanded["expected_tps"] = self.registration.expected_tps
        expanded["max_latency_ms"] = self.registration.max_latency_ms
        expanded["max_error_rate"] = self.registration.max_error_rate
        
        return expanded
