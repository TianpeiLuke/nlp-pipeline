from pydantic import Field, model_validator
from typing import TYPE_CHECKING
from pathlib import Path

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..pipeline_script_contracts.mims_package_contract import MIMS_PACKAGE_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract


class PackageStepConfig(ProcessingStepConfigBase):
    """Configuration for a model packaging step."""
    
    processing_entry_point: str = Field(
        default="mims_package.py",
        description="Entry point script for packaging."
    )

    class Config(ProcessingStepConfigBase.Config):
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'allow'  # Allow extra fields like __model_type__ and __model_module__ for type-aware serialization

    @model_validator(mode='after')
    def validate_config(self) -> 'PackageStepConfig':
        """Validate configuration and ensure defaults are set."""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("packaging step requires a processing_entry_point")

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")
        
        if "model_input" not in contract.expected_input_paths:
            raise ValueError("Script contract missing required input path: model_input")
        
        if "inference_scripts_input" not in contract.expected_input_paths:
            raise ValueError("Script contract missing required input path: inference_scripts_input")
            
        return self
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The MIMS package script contract
        """
        return MIMS_PACKAGE_CONTRACT
        
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
        else:
            contract = self.get_script_contract()
            if contract and hasattr(contract, 'entry_point'):
                entry_point = contract.entry_point
        
        if not entry_point:
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
