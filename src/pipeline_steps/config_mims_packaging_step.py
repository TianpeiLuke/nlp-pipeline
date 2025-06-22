from pydantic import Field, model_validator
from typing import Dict, Optional

from .config_processing_step_base import ProcessingStepConfigBase


class PackageStepConfig(ProcessingStepConfigBase):
    """Configuration for a model packaging step."""
    
    processing_entry_point: str = Field(
        default="mims_package.py",
        description="Entry point script for packaging."
    )

    # Input/output names for packaging with defaults
    input_names: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping input names to their descriptions. If None, defaults will be used."
    )
    
    output_names: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping output names to their descriptions. If None, defaults will be used."
    )

    class Config(ProcessingStepConfigBase.Config):
        pass

    @model_validator(mode='after')
    def validate_package_config(self) -> 'PackageStepConfig':
        """Additional validation specific to packaging configuration"""
        if not self.processing_entry_point:
            raise ValueError("packaging step requires a processing_entry_point")
        
        # Set default input names if not provided
        if self.input_names is None:
            self.input_names = {
                "model_input": "Input name for model artifacts",
                "inference_scripts_input": "Input name for inference scripts"
            }
        
        # Set default output names if not provided
        if self.output_names is None:
            self.output_names = {
                "packaged_model_output": "Output name for the packaged model"
            }
        
        # Validate required input/output names
        required_inputs = {"model_input", "inference_scripts_input"}
        required_outputs = {"packaged_model_output"}
        
        if not all(name in self.input_names for name in required_inputs):
            raise ValueError(f"Missing required input names: {required_inputs - set(self.input_names.keys())}")
        
        if not all(name in self.output_names for name in required_outputs):
            raise ValueError(f"Missing required output names: {required_outputs - set(self.output_names.keys())}")
            
        return self

    def get_input_names(self) -> Dict[str, str]:
        """
        Get the input names, using defaults if not set.
        """
        return self.input_names or {
            "model_input": "Input name for model artifacts",
            "inference_scripts_input": "Input name for inference scripts"
        }

    def get_output_names(self) -> Dict[str, str]:
        """
        Get the output names, using defaults if not set.
        """
        return self.output_names or {
            "packaged_model_output": "Output name for the packaged model"
        }


