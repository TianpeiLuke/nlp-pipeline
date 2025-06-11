from pydantic import BaseModel, Field, model_validator, PrivateAttr
from typing import List, Union, Dict, Any
import json

class ModelHyperparameters(BaseModel):
    """Base model hyperparameters for training tasks, with automatic aliasing for long field names."""

    # --- Private Attributes for Internal State ---
    # These will not be part of the model's schema but can be used in methods.
    _alias_map: Dict[str, str] = PrivateAttr(default_factory=dict)
    _aliased_full_field_list: List[str] = PrivateAttr(default_factory=list)
    _aliased_cat_field_list: List[str] = PrivateAttr(default_factory=list)
    _aliased_tab_field_list: List[str] = PrivateAttr(default_factory=list)

    # --- Field lists ---
    full_field_list: List[str] = Field(default=[
        'order_id', 'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag',
        'dialogue', 'llm_reversal_flag'
    ], description="Full list of original, potentially long field names.")
    
    cat_field_list: List[str] = Field(default=['dialogue'], description="Categorical fields using original names.")
    
    tab_field_list: List[str] = Field(default=[
        'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag'
    ], description="Tabular fields using original names.")

    categorical_features_to_encode: List[str] = Field(default_factory=list, description="List of categorical fields that require specific encoding.")

    # --- Identifier and label fields ---
    id_name: str = Field(default='order_id', description="ID field name.")
    label_name: str = Field(default='llm_reversal_flag', description="Label field name.")

    # --- Classification parameters ---
    is_binary: bool = Field(default=True, description="Binary classification flag.")
    num_classes: int = Field(default=2, description="Number of classes for classification.")
    multiclass_categories: List[Union[int, str]] = Field(default=[0, 1], description="List of unique category labels.")
    class_weights: List[float] = Field(default=[1.0, 1.0], description="Class weights for loss function.")
    device: int = Field(default=-1, description="Device ID for training.")
    
    # --- Model and Training Parameters ---
    model_class: str = Field(default='base_model', description="Model class name.")
    header: int = Field(default=0, description="Header row for CSV files.")
    input_tab_dim: int = Field(default=6, description="Input tabular dimension.")
    lr: float = Field(default=3e-05, description="Learning rate.")
    batch_size: int = Field(default=2, gt=0, le=256, description="Batch size for training.")
    max_epochs: int = Field(default=3, gt=0, le=10, description="Maximum epochs for training.")
    metric_choices: List[str] = Field(default=['f1_score', 'auroc'], description="Metric choices for evaluation.")
    optimizer: str = Field(default='SGD', description="Optimizer type.")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'
        protected_namespaces = ()

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization hook to create aliases for long field names.
        This runs after the model is fully created and validated.
        """
        self._create_aliases()

    def _create_alias(self, field_name: str) -> str:
        """
        Creates a short alias for a field name if it's long and contains dots.
        Example: 'Abuse.bsm_stats.n_max_time_gap' -> 'n_max_time_gap'
        """
        if field_name.count('.') >= 2:
            return field_name.split('.')[-1]
        return field_name

    def _create_aliases(self) -> None:
        """
        Populates the internal alias map and aliased field lists.
        """
        # Reset internal state in case of re-validation
        self._alias_map = {}
        self._aliased_full_field_list = []

        for field in self.full_field_list:
            alias = self._create_alias(field)
            if alias in self._alias_map and self._alias_map[alias] != field:
                raise ValueError(f"Alias collision detected. Both '{self._alias_map[alias]}' and '{field}' create the alias '{alias}'.")
            
            self._alias_map[alias] = field
            self._aliased_full_field_list.append(alias)
            
        # Create aliased versions of the other lists using the generated map
        original_to_alias = {v: k for k, v in self._alias_map.items()}
        self._aliased_cat_field_list = [original_to_alias[f] for f in self.cat_field_list]
        self._aliased_tab_field_list = [original_to_alias[f] for f in self.tab_field_list]

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'ModelHyperparameters':
        """Validate model dimensions and configurations"""
        if len(self.class_weights) != len(self.multiclass_categories):
            raise ValueError(f"class_weights length ({len(self.class_weights)}) must match multiclass_categories length ({len(self.multiclass_categories)}).")
        if self.input_tab_dim != len(self.tab_field_list):
            raise ValueError(f"input_tab_dim ({self.input_tab_dim}) must match length of tab_field_list ({len(self.tab_field_list)}).")
        if self.is_binary and (len(self.multiclass_categories) != 2 or self.num_classes != 2):
            raise ValueError("For binary classification, num_classes and multiclass_categories length must be 2.")
        if not self.is_binary and (self.num_classes != len(self.multiclass_categories)):
            raise ValueError("For multiclass, num_classes must match the length of multiclass_categories.")
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.model_dump()

    def get_alias_map(self) -> Dict[str, str]:
        """
        Returns the mapping from short alias names to their original long field names.
        """
        return self._alias_map

    def serialize_config(self) -> Dict[str, str]:
        """
        Serialize configuration for SageMaker.
        This version replaces the long field lists with their aliased versions
        to avoid exceeding hyperparameter size limits.
        """
        # Start with the full model configuration
        config = self.get_config()

        # Remove the original, long field lists and the alias map
        config.pop('full_field_list', None)
        config.pop('cat_field_list', None)
        config.pop('tab_field_list', None)
        config.pop('alias_map', None) # This is handled by get_alias_map()

        # Add the aliased lists back in under the original keys
        config['full_field_list'] = self._aliased_full_field_list
        config['cat_field_list'] = self._aliased_cat_field_list
        config['tab_field_list'] = self._aliased_tab_field_list
        
        # Serialize all values to strings for SageMaker
        return {
            k: json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)
            for k, v in config.items()
        }
