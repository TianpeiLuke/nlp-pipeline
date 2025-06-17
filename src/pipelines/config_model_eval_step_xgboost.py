from pydantic import Field, model_validator
from typing import Optional, Dict, ClassVar

from .config_processing_step_base import ProcessingStepConfigBase
from .hyperparameters_base import ModelHyperparameters


class XGBoostModelEvalConfig(ProcessingStepConfigBase):
    """
    Configuration for XGBoost model evaluation step.
    Inherits from ProcessingStepConfigBase.
    """
    processing_entry_point: str = Field(
        default="model_evaluation_xgboost.py",
        description="Entry point script for model evaluation."
    )

    # Input/output names for evaluation with defaults
    INPUT_CHANNELS: ClassVar[Dict[str, str]] = {
        "model_input": "Model artifacts input",
        "eval_data_input": "Evaluation data input",
        "code_input": "Processing code input"
    }

    OUTPUT_CHANNELS: ClassVar[Dict[str, str]] = {
        "eval_output": "Output name for evaluation predictions",
        "metrics_output": "Output name for evaluation metrics"
    }

    # Add job_type to allow evaluation on different splits (e.g., 'training', 'calibration')
    job_type: str = Field(
        default="calibration",
        description="Which split to evaluate on (e.g., 'training', 'calibration', 'validation', 'test')."
    )

    # Use the base hyperparameters config for all label/field info
    hyperparameters: ModelHyperparameters = Field(
        ...,
        description="Model hyperparameters config, including id_name, label_name, field lists, etc."
    )

    eval_metric_choices: Optional[list] = Field(
        default_factory=lambda: ["auc", "average_precision", "f1_score"],
        description="List of evaluation metrics to compute"
    )

    class Config(ProcessingStepConfigBase.Config):
        pass

    @model_validator(mode='after')
    def validate_eval_config(self) -> 'XGBoostModelEvalConfig':
        """Additional validation specific to evaluation configuration"""
        if not self.processing_entry_point:
            raise ValueError("evaluation step requires a processing_entry_point")
            
        # Validate job_type
        valid_job_types = {"training", "calibration", "validation", "test"}
        if self.job_type not in valid_job_types:
            raise ValueError(f"job_type must be one of {valid_job_types}, got '{self.job_type}'")
        
        # Validate hyperparameters
        if not isinstance(self.hyperparameters, ModelHyperparameters):
            raise ValueError("hyperparameters must be an instance of ModelHyperparameters")
            
        return self

    def get_input_names(self) -> Dict[str, str]:
        """
        Get the fixed input channel names and descriptions.
        """
        return self.INPUT_CHANNELS

    def get_output_names(self) -> Dict[str, str]:
        """
        Get the fixed output channel names and descriptions.
        """
        return self.OUTPUT_CHANNELS

    def get_script_path(self) -> str:
        """
        Get the full path to the processing script.
        """
        return super().get_script_path() or self.processing_entry_point

