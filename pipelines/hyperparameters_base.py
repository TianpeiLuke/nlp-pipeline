from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class ModelHyperparameters(BaseModel):
    # Field lists
    full_field_list: List[str] = Field(default=[
        'order_id', 'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag',
        'dialogue', 'llm_reversal_flag'
    ],  description="Full field list")
    
    cat_field_list: List[str] = Field(default=['dialogue'], description="Categorical fields")
    
    tab_field_list: List[str] = Field(default=[
        'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag'
    ], description="Tabular fields")

    # Identifier and label fields
    id_name: str = Field(default='order_id', description="ID field name")    
    label_name: str = Field(default='llm_reversal_flag', description="Label field name")

    # Classification parameters
    is_binary: bool = Field(default=True, description="Binary classification flag")
    num_classes: int = Field(default=2, description="Number of classes for classification")
    multiclass_categories: List[int] = Field(default=[0, 1], description="Multiclass categories")
    class_weights: List[float] = Field(default=[1.0, 1.0], description="Class weights for loss function")
    device: int = Field(default=-1, description="Device ID for training")
    
    # Model configuration
    model_class: str = Field(default='multimodal_bert', description="Model class name")

    # Input configuration
    header: int = Field(default=0,  description="Header row for CSV files")
    input_tab_dim: int  = Field(default=6, description="Input tabular dimension")

    # Training parameters
    learning_rate: float = Field(default=3e-05, description="Learning rate")
    batch_size: int = Field(default=2, gt=0, le=256, description="Batch size for training")
    max_epochs: int = Field(default=3, gt=0, le=10, description="Maximum epochs for training")
    metric_choices: List[str] = Field(default=['f1_score', 'auroc'], description="Metric choices for evaluation")

    # Optimizer and model settings
    optimizer: str = Field(default='SGD', description="Optimizer type")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'
        protected_namespaces = ()

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'ModelHyperparameters':
        """Validate model dimensions and configurations"""
        # Validate class weights match number of classes
        if len(self.class_weights) != len(self.multiclass_categories):
            raise ValueError(
                f"The dimension of class_weights ({len(self.class_weights)}) "
                f"does not match with number of classes ({len(self.multiclass_categories)})"
            )
        
        # Validate input tab dimensions
        if self.input_tab_dim != len(self.tab_field_list):
            raise ValueError(
                f"input_tab_dim ({self.input_tab_dim}) does not match "
                f"length of tab_field_list ({len(self.tab_field_list)})"
            )
        
        # Validate binary classification setting
        if self.is_binary and len(self.multiclass_categories) != 2:
            raise ValueError(
                f"is_binary is True but number of classes is {len(self.multiclass_categories)}"
            )
        
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        return self.model_dump()

    def serialize_config(self) -> Dict[str, str]:
        """Serialize configuration for SageMaker"""
        config = self.get_config()
        return {
            k: json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)
            for k, v in config.items()
        }