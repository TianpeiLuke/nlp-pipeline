from pydantic import Field, model_validator
from typing import Optional, Dict, TYPE_CHECKING
from pathlib import Path
import logging

from .config_processing_step_base import ProcessingStepConfigBase
from .hyperparameters_xgboost import XGBoostModelHyperparameters

# Import the script contract
from ..pipeline_script_contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract

logger = logging.getLogger(__name__)


class XGBoostModelEvalConfig(ProcessingStepConfigBase):
    """
    Configuration for XGBoost model evaluation step.
    Inherits from ProcessingStepConfigBase.
    """
    processing_entry_point: str = Field(
        default="model_evaluation_xgb.py",
        description="Entry point script for model evaluation."
    )

    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test')."
    )

    hyperparameters: XGBoostModelHyperparameters = Field(
        ...,
        description="XGBoost model hyperparameters config, including id_name, label_name, field lists, etc."
    )

    eval_metric_choices: Optional[list] = Field(
        default_factory=lambda: ["auc", "average_precision", "f1_score"],
        description="List of evaluation metrics to compute"
    )

    # XGBoost specific fields
    xgboost_framework_version: str = Field(
        default="1.5-1",
        description="XGBoost framework version for processing"
    )

    class Config(ProcessingStepConfigBase.Config):
        pass

    @model_validator(mode='after')
    def validate_eval_config(self) -> 'XGBoostModelEvalConfig':
        """Additional validation specific to evaluation configuration"""
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("evaluation step requires a processing_entry_point")
            
        valid_job_types = {"training", "calibration", "validation", "test"}
        if self.job_type not in valid_job_types:
            raise ValueError(f"job_type must be one of {valid_job_types}, got '{self.job_type}'")
        
        if not isinstance(self.hyperparameters, XGBoostModelHyperparameters):
            raise ValueError("hyperparameters must be an instance of XGBoostModelHyperparameters")
        
        # Validate required fields from script contract
        contract = self.get_script_contract()
        
        # Check required environment variables from contract
        if not self.hyperparameters.id_name:
            raise ValueError("hyperparameters.id_name must be provided (required by model evaluation contract)")
            
        if not self.hyperparameters.label_name:
            raise ValueError("hyperparameters.label_name must be provided (required by model evaluation contract)")
            
        return self

    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the model evaluation script.
        
        Returns:
            Dict[str, str]: Dictionary mapping environment variable names to values
        """
        env_vars = {
            "ID_FIELD": self.hyperparameters.id_name,
            "LABEL_FIELD": self.hyperparameters.label_name
        }
        
        # Add any other environment variables needed
        if hasattr(self.hyperparameters, 'eval_metric_list') and self.hyperparameters.eval_metric_list:
            env_vars["EVAL_METRICS"] = ",".join(self.hyperparameters.eval_metric_list)
        
        return env_vars
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The model evaluation script contract
        """
        return MODEL_EVALUATION_CONTRACT
        
    def get_script_path(self) -> str:
        """
        Get script path for XGBoost model evaluation.
        
        SPECIAL CASE: Unlike other step configs, XGBoostModelEvalStepBuilder provides 
        processing_source_dir and processing_entry_point directly to the processor.run() 
        method separately. Therefore, this method should return only the entry point name 
        without combining with source_dir.
        
        Returns:
            Script entry point name (without source_dir)
        """
        # Determine which entry point to use
        entry_point = None
        
        # First priority: Use processing_entry_point if provided
        if self.processing_entry_point:
            entry_point = self.processing_entry_point
        # Second priority: Use contract entry point
        elif hasattr(self, 'script_contract') and self.script_contract and hasattr(self.script_contract, 'entry_point'):
            entry_point = self.script_contract.entry_point
        
        # Return just the entry point name without combining with source directory
        # This is important for XGBoostModelEvalStepBuilder which handles paths differently
        return entry_point
