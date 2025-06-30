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
        default_factory=lambda: {
            "model_input": "ModelArtifacts",              # KEY: logical name, VALUE: script input name
            "inference_scripts_input": "InferenceScripts"  # KEY: logical name, VALUE: script input name
        },
        description="Mapping of logical input names (keys) to script input names (values)."
    )
    
    output_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "packaged_model_output": "PackagedModel"  # KEY: logical name, VALUE: output descriptor
        },
        description="Mapping of logical output names (keys) to output descriptors (values)."
    )

    class Config(ProcessingStepConfigBase.Config):
        pass

    @model_validator(mode='after')
    def validate_config(self) -> 'PackageStepConfig':
        """Validate configuration and ensure defaults are set."""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("packaging step requires a processing_entry_point")

        # Set defaults if needed
        if self.input_names is None or len(self.input_names) == 0:
            self.input_names = {
                "model_input": "ModelArtifacts",              # KEY: logical name, VALUE: script input name
                "inference_scripts_input": "InferenceScripts"  # KEY: logical name, VALUE: script input name
            }
        
        if self.output_names is None or len(self.output_names) == 0:
            self.output_names = {
                "packaged_model_output": "PackagedModel"  # KEY: logical name, VALUE: output descriptor
            }

        # Validate required channels
        required_inputs = {"model_input", "inference_scripts_input"}
        required_outputs = {"packaged_model_output"}
        
        if not all(name in self.input_names for name in required_inputs):
            raise ValueError(
                f"Missing required input names: {required_inputs - set(self.input_names.keys())}"
            )
        
        if not all(name in self.output_names for name in required_outputs):
            raise ValueError(
                f"Missing required output names: {required_outputs - set(self.output_names.keys())}"
            )
            
        return self
