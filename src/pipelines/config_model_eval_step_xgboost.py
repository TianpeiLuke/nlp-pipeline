from .config_processing_step_base import ProcessingStepConfigBase
from pydantic import Field
from typing import Optional, Dict

class XGBoostModelEvalConfig(ProcessingStepConfigBase):
    """
    Configuration for XGBoost model evaluation step.
    Inherits from ProcessingStepConfigBase.
    """
    # Optionally add more fields specific to evaluation if needed
    eval_metric_choices: Optional[list] = Field(
        default_factory=lambda: ["auc", "average_precision", "f1_score"],
        description="List of evaluation metrics to compute"
    )
    id_name: Optional[str] = Field(
        default="id",
        description="Name of the ID column in the data"
    )
    label_name: Optional[str] = Field(
        default="label",
        description="Name of the label column in the data"
    )
    # Add more fields as needed
