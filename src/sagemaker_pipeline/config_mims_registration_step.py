from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal, Optional, Dict
from enum import Enum

from .config_processing_step_base import ProcessingStepConfigBase

class VariableType(str, Enum):
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"
    
    
class ModelRegistrationConfig(ProcessingStepConfigBase):
    """Configuration for model registration step, extending ProcessingStepConfigBase."""
    
    # Model registration specific fields
    model_owner: str = Field(
        default="team_id",
        description="Team ID of model owner"
    )
    model_registration_domain: str = Field(
        default="BuyerSellerMessaging",
        description="Domain for model registration"
    )
    model_registration_objective: Optional[str] = Field(
        default=None,
        description="Objective of model registration"
    )

    # Content and response types with Literal type
    source_model_inference_content_types: Literal[["text/csv"], ["application/json"]] = Field(
        default=["text/csv"],
        description="Content type for model inference input. Must be exactly ['text/csv'] or ['application/json']"
    )
    source_model_inference_response_types: Literal[["text/csv"], ["application/json"]] = Field(
        default=["application/json"],
        description="Response type for model inference output. Must be exactly ['text/csv'] or ['application/json']"
    )

    # Variable lists for input and output
    source_model_inference_output_variable_list: Dict[str, VariableType] = Field(
        default={
            'legacy-score': VariableType.NUMERIC
        },
        description="Dictionary of output variables and their types (NUMERIC or TEXT)"
    )
    
    source_model_inference_input_variable_list: Dict[str, VariableType] = Field(
        default_factory=dict,
        description="Dictionary of input variables and their types (NUMERIC or TEXT)"
    )

    # Override or specify which processing instance type to use
    use_large_instance: bool = Field(
        default=False,
        description="Whether to use large instance type for processing"
    )

    @property
    def processing_instance_type(self) -> str:
        """Get the appropriate instance type based on configuration"""
        return self.processing_instance_type_large if self.use_large_instance else self.processing_instance_type_small

    @field_validator('source_model_inference_output_variable_list', 'source_model_inference_input_variable_list')
    @classmethod
    def validate_variable_list(cls, v: Dict[str, VariableType]) -> Dict[str, VariableType]:
        """Validate variable lists to ensure proper format and types"""
        if not v:  # If empty dictionary
            return v
            
        for key, value in v.items():
            # Validate key is a string
            if not isinstance(key, str):
                raise ValueError(f"Key must be string, got {type(key)} for key: {key}")
            
            # Validate value is correct type
            if not isinstance(value, VariableType):
                raise ValueError(f"Value must be either 'NUMERIC' or 'TEXT', got: {value}")
        
        return v

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'