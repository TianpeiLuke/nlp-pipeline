from pydantic import Field, model_validator
from typing import Optional, Dict

from .config_processing_step_base import ProcessingStepConfigBase
from .hyperparameters_xgboost import XGBoostModelHyperparameters


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
    input_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "model_input": "ModelInput",        # KEY: logical name, VALUE: script input name
            "eval_data_input": "EvaluationData" # KEY: logical name, VALUE: script input name
        },
        description="Mapping of logical input names (keys) to script input names (values)."
    )
    
    output_names: Optional[Dict[str, str]] = Field(
        default_factory=lambda: {
            "eval_output": "EvaluationResults",    # KEY: logical name, VALUE: output descriptor
            "metrics_output": "EvaluationMetrics"  # KEY: logical name, VALUE: output descriptor
        },
        description="Mapping of logical output names (keys) to output descriptors (values)."
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
        if not self.processing_entry_point:
            raise ValueError("evaluation step requires a processing_entry_point")
            
        valid_job_types = {"training", "calibration", "validation", "test"}
        if self.job_type not in valid_job_types:
            raise ValueError(f"job_type must be one of {valid_job_types}, got '{self.job_type}'")
        
        if not isinstance(self.hyperparameters, XGBoostModelHyperparameters):
            raise ValueError("hyperparameters must be an instance of XGBoostModelHyperparameters")
        
        return self


    @model_validator(mode='after')
    def set_default_names(self) -> 'XGBoostModelEvalConfig':
        """Ensure default input and output names are set if not provided."""
        if self.input_names is None or len(self.input_names) == 0:
            self.input_names = {
                "model_input": "ModelInput",        # KEY: logical name, VALUE: script input name
                "eval_data_input": "EvaluationData" # KEY: logical name, VALUE: script input name
            }
        
        if self.output_names is None or len(self.output_names) == 0:
            self.output_names = {
                "eval_output": "EvaluationResults",    # KEY: logical name, VALUE: output descriptor
                "metrics_output": "EvaluationMetrics"  # KEY: logical name, VALUE: output descriptor
            }
        
        return self


    def get_script_path(self) -> str:
        return super().get_script_path() or self.processing_entry_point
