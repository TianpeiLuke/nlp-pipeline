from pydantic import Field, model_validator
from typing import TYPE_CHECKING

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
        pass

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
        Get script path from contract.
        
        Returns:
            Script path
        """
        # Use the entry_point from the contract
        contract = self.get_script_contract()
        if contract and contract.entry_point:
            return contract.entry_point
            
        # Fall back to processing_entry_point if contract doesn't specify
        return self.processing_entry_point
