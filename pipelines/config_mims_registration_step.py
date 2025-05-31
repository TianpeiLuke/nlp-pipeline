from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Union, Optional, Dict, List, Any
from enum import Enum
from datetime import datetime

from .config_base import BasePipelineConfig


class VariableType(str, Enum):
    NUMERIC = "NUMERIC"
    TEXT = "TEXT"
    
    @classmethod
    def _missing_(cls, value: str) -> Optional['VariableType']:
        """Handle string values"""
        try:
            return cls(value.upper())
        except ValueError:
            return None

    def __str__(self) -> str:
        """String representation"""
        return self.value
    
    
class ModelRegistrationConfig(BasePipelineConfig):
    """Configuration for model registration step."""
    
    # Model registration specific fields
    model_owner: str = Field(
        default="team id",
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

    # Content and response types
    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Content type for model inference input. Must be exactly ['text/csv'] or ['application/json']"
    )
    source_model_inference_response_types: List[str] = Field(
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

    class Config(BasePipelineConfig.Config):
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'
        json_encoders = {
            VariableType: lambda v: v.value
        }

    @field_validator('source_model_inference_content_types', 'source_model_inference_response_types')
    @classmethod
    def validate_content_types(cls, v: List[str]) -> List[str]:
        """Validate content and response types"""
        valid_types = [["text/csv"], ["application/json"]]
        if v not in valid_types:
            raise ValueError(f"Content/Response types must be one of {valid_types}")
        return v

    @field_validator('source_model_inference_output_variable_list', 'source_model_inference_input_variable_list')
    @classmethod
    def validate_variable_list(cls, v: Dict[str, Union[VariableType, str]]) -> Dict[str, str]:
        """Validate variable lists and convert to string values"""
        if not v:  # If empty dictionary
            return v
        
        result = {}
        for key, value in v.items():
            # Validate key is a string
            if not isinstance(key, str):
                raise ValueError(f"Key must be string, got {type(key)} for key: {key}")
        
            # Convert VariableType to string or validate string value
            if isinstance(value, VariableType):
                result[key] = value.value
            elif isinstance(value, str) and value in [vt.value for vt in VariableType]:
                result[key] = value
            else:
                raise ValueError(f"Value must be either 'NUMERIC' or 'TEXT', got: {value}")
    
        return result

    @model_validator(mode='after')
    def validate_registration_configs(self) -> 'ModelRegistrationConfig':
        """Validate registration-specific configurations"""
        if not self.model_registration_objective:
            raise ValueError("model_registration_objective must be provided")
        
        return self
        
    def get_registration_job_name(self) -> str:
        """Generate a unique registration job name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.pipeline_name}-register-{timestamp}"

    def get_model_package_group_name(self) -> str:
        """Generate model package group name"""
        return f"{self.pipeline_name}-{self.model_registration_objective}"

    def get_model_package_description(self) -> str:
        """Generate model package description"""
        return (f"Model package for {self.model_registration_objective} "
                f"in {self.model_registration_domain} domain")
        
    def get_variable_schema(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """Generate variable schema for model registration"""
        schema = {
            "input": {"variables": []},
            "output": {"variables": []}
        }
        
        # Add input variables
        for name, var_type in self.source_model_inference_input_variable_list.items():
            schema["input"]["variables"].append({
                "name": name,
                "type": var_type if isinstance(var_type, str) else var_type.value
            })
            
        # Add output variables
        for name, var_type in self.source_model_inference_output_variable_list.items():
            schema["output"]["variables"].append({
                "name": name,
                "type": var_type if isinstance(var_type, str) else var_type.value
            })
            
        return schema
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization"""
        data = super().model_dump(**kwargs)
        # Convert enums to strings in variable lists
        if 'source_model_inference_input_variable_list' in data:
            data['source_model_inference_input_variable_list'] = {
                k: v.value if isinstance(v, VariableType) else v
                for k, v in data['source_model_inference_input_variable_list'].items()
            }
        if 'source_model_inference_output_variable_list' in data:
            data['source_model_inference_output_variable_list'] = {
                k: v.value if isinstance(v, VariableType) else v
                for k, v in data['source_model_inference_output_variable_list'].items()
            }
        return data
