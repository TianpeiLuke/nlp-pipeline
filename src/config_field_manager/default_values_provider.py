"""
DefaultValuesProvider module.

This module defines the DefaultValuesProvider class which provides default values
for system inputs (Tier 2) in the three-tier configuration architecture.
"""

import logging
from typing import Dict, Any, Optional, Callable, List, Union

logger = logging.getLogger(__name__)


class DefaultValuesProvider:
    """
    Provides default values for system inputs (Tier 2).
    
    This class manages the standardized default values for all configuration fields
    that don't require direct user input but may need administrative customization.
    """
    
    # Default values for system inputs, organized by category
    DEFAULT_VALUES = {
        # Base Model Hyperparameters
        "metric_choices": lambda config: ['f1_score', 'auroc'] if getattr(config, 'is_binary', True) else ['accuracy', 'f1_score'],
        "device": -1,
        "header": "true",
        "batch_size": 32,
        "lr": 0.01,
        "max_epochs": 100,
        "optimizer": "adam",
        
        # Framework Settings
        "py_version": "py3",
        "processing_framework_version": "1.2-1",
        
        # Processing Resources
        "processing_instance_type_large": "ml.m5.4xlarge",
        "processing_instance_type_small": "ml.m5.xlarge",
        "processing_instance_count": 1,
        "processing_volume_size": 500,
        "test_val_ratio": 0.5,
        
        # Training Resources
        "training_instance_count": 1,
        "training_volume_size": 800,
        "training_instance_type": "ml.m5.4xlarge",
        
        # Inference Resources
        "inference_instance_type": "ml.m5.4xlarge",
        
        # Processing Entry Points
        "processing_entry_point": lambda config: DefaultValuesProvider._get_entry_point_by_config_type(config),
        "model_eval_processing_entry_point": "model_eval_xgb.py",
        "model_eval_job_type": "training",
        "packaging_entry_point": "mims_package.py",
        "training_entry_point": "train_xgb.py",
        
        # Calibration Settings
        "calibration_method": "gam",
        "score_field": "prob_class_1",
        "score_field_prefix": "prob_class_",
        
        # Model Evaluation Settings
        "use_large_processing_instance": True,
        "eval_metric_choices": lambda config: ["auc", "average_precision", "f1_score"] 
                                if getattr(config, "is_binary", True) 
                                else ["accuracy", "f1_score"],
        
        # Payload Configuration
        "max_acceptable_error_rate": 0.2,
        "default_numeric_value": 0.0,
        "default_text_value": "Default",
        "special_field_values": None,
        
        # Integration Settings
        "source_model_inference_content_types": ["text/csv"],
        "source_model_inference_response_types": ["application/json"],
    }
    
    # Lookup table for determining entry points based on config type
    CONFIG_TYPE_ENTRY_POINTS = {
        "TabularPreprocessingConfig": "tabular_preprocess.py",
        "ModelCalibrationConfig": "model_calibration.py",
        "ModelEvaluationConfig": "model_eval_xgb.py",
        "PayloadConfig": "mims_payload.py",
        "XGBoostTrainingConfig": "train_xgb.py"
    }
    
    @classmethod
    def apply_defaults(cls, config: Any, override_values: Optional[Dict[str, Any]] = None, 
                      logger: Optional[logging.Logger] = None) -> Any:
        """
        Apply default values to a configuration object
        
        Args:
            config: Configuration object to apply defaults to
            override_values: Optional dictionary of values to override defaults
            logger: Optional logger for reporting applied defaults
            
        Returns:
            The modified configuration object
        """
        # Create merged defaults dictionary with any overrides
        defaults = cls.DEFAULT_VALUES.copy()
        if override_values:
            defaults.update(override_values)
            
        # Track changes for reporting
        applied_defaults = {}
            
        # Apply each default if the field is not already set
        for field_name, default_value in defaults.items():
            # Skip if field is already set to a non-None value
            if hasattr(config, field_name) and getattr(config, field_name) is not None:
                continue
                
            # Apply default (either value or callable)
            if callable(default_value):
                try:
                    value = default_value(config)
                except Exception as e:
                    if logger:
                        logger.warning(f"Could not apply callable default for {field_name}: {str(e)}")
                    continue
            else:
                value = default_value
                
            # Set the default value on the config
            setattr(config, field_name, value)
            applied_defaults[field_name] = value
                
        # Log applied defaults if requested
        if logger and applied_defaults:
            logger.info(f"Applied {len(applied_defaults)} defaults to {config.__class__.__name__}: {applied_defaults}")
            
        return config
    
    @classmethod
    def apply_defaults_to_multiple(cls, configs: List[Any], 
                                  override_values: Optional[Dict[str, Any]] = None,
                                  logger: Optional[logging.Logger] = None) -> List[Any]:
        """
        Apply defaults to multiple configuration objects
        
        Args:
            configs: List of configuration objects
            override_values: Optional dictionary of values to override defaults
            logger: Optional logger for reporting applied defaults
            
        Returns:
            The list of modified configuration objects
        """
        return [cls.apply_defaults(config, override_values, logger) for config in configs]
    
    @staticmethod
    def _get_entry_point_by_config_type(config: Any) -> Optional[str]:
        """
        Determine the appropriate processing entry point based on config type
        
        Args:
            config: Configuration object
            
        Returns:
            str: Entry point script name or None if not found
        """
        config_type = config.__class__.__name__
        return DefaultValuesProvider.CONFIG_TYPE_ENTRY_POINTS.get(
            config_type, "processing.py"  # Default fallback
        )
        
    @classmethod
    def get_defaults_for_config_type(cls, config_class: type) -> Dict[str, Any]:
        """
        Get all applicable defaults for a specific configuration class
        
        Args:
            config_class: The configuration class
            
        Returns:
            dict: Defaults applicable to this configuration type
        """
        # Create a minimal instance to use for callable defaults
        try:
            instance = config_class()
        except Exception:
            # If we can't create an instance, return non-callable defaults only
            return {k: v for k, v in cls.DEFAULT_VALUES.items() 
                   if not callable(v) and k in config_class.__annotations__}
        
        # Apply all defaults that match the class's fields
        result = {}
        for field_name, default_value in cls.DEFAULT_VALUES.items():
            if field_name in config_class.__annotations__:
                if callable(default_value):
                    try:
                        result[field_name] = default_value(instance)
                    except Exception:
                        pass  # Skip if callable fails
                else:
                    result[field_name] = default_value
                    
        return result
