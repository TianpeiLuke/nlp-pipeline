"""
Configuration for DummyTraining.

This module defines the configuration class for the DummyTraining step,
which is responsible for taking a pretrained model.tar.gz file and making
it available for downstream packaging and registration steps.
"""

from pydantic import Field, model_validator
from typing import TYPE_CHECKING, Optional
from pathlib import Path

from .config_processing_step_base import ProcessingStepConfigBase

# Import the script contract
from ..pipeline_script_contracts.dummy_training_contract import DUMMY_TRAINING_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract


class DummyTrainingConfig(ProcessingStepConfigBase):
    """
    Configuration for DummyTraining step.
    
    This step takes a pretrained model.tar.gz file and makes it available 
    for downstream packaging and registration steps, bypassing the actual 
    training process.
    """
    
    # Override with specific default for this step
    processing_entry_point: str = Field(
        default="dummy_training.py",
        description="Entry point script for dummy training."
    )
    
    # Unique to this step
    pretrained_model_path: str = Field(
        default="",
        description="Local path to pretrained model.tar.gz file."
    )

    class Config(ProcessingStepConfigBase.Config):
        pass

    @model_validator(mode='after')
    def validate_config(self) -> 'DummyTrainingConfig':
        """
        Validate configuration and ensure defaults are set.
        
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If pretrained_model_path is provided but file doesn't exist
            
        Returns:
            Self with validated configuration
        """
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("DummyTraining step requires a processing_entry_point")
            
        # Check for pretrained model path
        if not self.pretrained_model_path:
            raise ValueError("pretrained_model_path is required in DummyTrainingConfig")
            
        # Check if file exists (if path is concrete and not a variable)
        if not hasattr(self.pretrained_model_path, 'expr'):
            model_path = Path(self.pretrained_model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Pretrained model not found at {model_path}")
                
            # Check file extension
            if not model_path.suffix == '.tar.gz' and not str(model_path).endswith('.tar.gz'):
                self._logger.warning(f"Model file {model_path} does not have .tar.gz extension")

        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")
        
        if "pretrained_model_path" not in contract.expected_input_paths:
            raise ValueError("Script contract missing required input path: pretrained_model_path")
        
        if "model_input" not in contract.expected_output_paths:
            raise ValueError("Script contract missing required output path: model_input")
            
        return self
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The DummyTraining script contract
        """
        return DUMMY_TRAINING_CONTRACT
        
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
