from pydantic import Field, model_validator
from typing import Optional, Dict

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
    input_names: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping input names to their descriptions. If None, defaults will be used."
    )
    output_names: Optional[Dict[str, str]] = Field(
        default=None,
        description="Dictionary mapping output names to their descriptions. If None, defaults will be used."
    )

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
        
        # Set default input names if not provided
        if self.input_names is None:
            self.input_names = {
                "model_input": "Input name for model artifacts",
                "eval_data_input": "Input name for evaluation data"
            }
        
        # Set default output names if not provided
        if self.output_names is None:
            self.output_names = {
                "eval_output": "Output name for evaluation predictions",
                "metrics_output": "Output name for evaluation metrics"
            }
        
        # Validate required input/output names
        required_inputs = {"model_input", "eval_data_input"}
        required_outputs = {"eval_output", "metrics_output"}
        
        if not all(name in self.input_names for name in required_inputs):
            raise ValueError(f"Missing required input names: {required_inputs - set(self.input_names.keys())}")
        
        if not all(name in self.output_names for name in required_outputs):
            raise ValueError(f"Missing required output names: {required_outputs - set(self.output_names.keys())}")
            
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
        Get the input names, using defaults if not set.
        """
        return self.input_names or {
            "model_input": "Input name for model artifacts",
            "eval_data_input": "Input name for evaluation data"
        }

    def get_output_names(self) -> Dict[str, str]:
        """
        Get the output names, using defaults if not set.
        """
        return self.output_names or {
            "eval_output": "Output name for evaluation predictions",
            "metrics_output": "Output name for evaluation metrics"
        }
