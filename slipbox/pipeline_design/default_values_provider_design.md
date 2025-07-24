# DefaultValuesProvider Design

## Overview

The `DefaultValuesProvider` is a core component of the three-tier configuration architecture that manages all system inputs (Tier 2). This document outlines the design and implementation of this class, which is responsible for providing standardized default values for configuration fields that don't require direct user input but may need administrative customization.

## Purpose and Responsibilities

The `DefaultValuesProvider` serves several key purposes:

1. Provide standardized default values for system inputs (Tier 2)
2. Ensure consistent settings across pipeline configurations
3. Allow centralized administration of default values
4. Support conditional defaults based on other configuration values
5. Maintain backward compatibility with existing pipelines

## Core Design Principles

### 1. Single Source of Truth

The `DefaultValuesProvider` implements the Single Source of Truth principle by centralizing all default values in a single location. This ensures that:

- All system inputs have consistent default values
- Changes to defaults can be made in one place
- System behavior is predictable and reproducible

### 2. Declarative Over Imperative

The design favors a declarative approach to defaults:

- Defaults are defined as a static mapping rather than procedural code
- Conditional defaults are expressed as lambda functions that declare the logic
- This makes the defaults easier to understand, review, and maintain

### 3. Explicit Over Implicit

All default values and their application logic are explicit:

- Each default has a clear origin in the `DEFAULT_VALUES` dictionary
- The application process is transparent and logged
- Overrides are explicitly tracked and reported

### 4. Maintainability and Extensibility

The design prioritizes maintainability:

- Adding new defaults requires minimal code changes
- The system adapts to new configuration fields automatically
- Dependencies between defaults are clearly documented

## Class Structure

```python
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
        "eval_metric_choices": lambda config: ["auc", "average_precision", "f1_score"] if getattr(config, "is_binary", True) else ["accuracy", "f1_score"],
        
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
    def apply_defaults(cls, config, override_values=None, logger=None):
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
    def apply_defaults_to_multiple(cls, configs, override_values=None, logger=None):
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
    def _get_entry_point_by_config_type(config):
        """
        Determine the appropriate processing entry point based on config type
        
        Args:
            config: Configuration object
            
        Returns:
            str: Entry point script name
        """
        config_type = config.__class__.__name__
        return DefaultValuesProvider.CONFIG_TYPE_ENTRY_POINTS.get(
            config_type, "processing.py"  # Default fallback
        )
        
    @classmethod
    def get_defaults_for_config_type(cls, config_class):
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
```

## Default Value Types

The `DefaultValuesProvider` supports two types of default values:

### 1. Static Values

Simple, fixed values that are applied directly:

```python
"device": -1,
"header": "true",
"batch_size": 32
```

### 2. Dynamic Values (Lambda Functions)

Values that depend on other configuration settings:

```python
"metric_choices": lambda config: ['f1_score', 'auroc'] if getattr(config, 'is_binary', True) else ['accuracy', 'f1_score']
```

These lambda functions:
- Take the configuration object as an input
- Can access any of its attributes
- Return the appropriate default value based on the configuration context

## Default Application Process

When defaults are applied to a configuration:

1. The system checks if the field already has a non-None value
2. If not, it applies the default (evaluating lambdas if necessary)
3. The value is set on the configuration object
4. The system tracks which defaults were applied

## Integration with Three-Tier Architecture

Within the three-tier architecture, the `DefaultValuesProvider` serves as the implementation of Tier 2 (System Inputs). It works alongside:

1. **Essential User Interface (Tier 1)**: Collects user inputs before defaults are applied
2. **Field Derivation Engine (Tier 3)**: Runs after defaults are applied to derive dependent fields

The typical workflow is:

```python
# 1. Collect essential user inputs (Tier 1)
essential_config = collect_user_inputs()

# 2. Create config objects from essential inputs
config_objects = create_config_objects(essential_config)

# 3. Apply system defaults (Tier 2)
DefaultValuesProvider.apply_defaults_to_multiple(config_objects)

# 4. Derive dependent fields (Tier 3)
FieldDerivationEngine.derive_fields_for_multiple(config_objects)

# 5. Generate final configuration
final_config = merge_configs(config_objects)
```

## Configuration and Customization

The `DefaultValuesProvider` can be customized in several ways:

### 1. Override Default Values

Administrators can override default values by providing an `override_values` dictionary:

```python
custom_defaults = {
    "processing_instance_type_large": "ml.m5.12xlarge",
    "processing_volume_size": 1000
}

DefaultValuesProvider.apply_defaults(config, override_values=custom_defaults)
```

### 2. Add New Default Values

To add new defaults, extend the `DEFAULT_VALUES` dictionary:

```python
# Add support for a new field
DefaultValuesProvider.DEFAULT_VALUES["new_field_name"] = "default_value"
```

### 3. Custom Config Type Handling

For new configuration types, extend the `CONFIG_TYPE_ENTRY_POINTS` dictionary:

```python
# Add entry point for a new config type
DefaultValuesProvider.CONFIG_TYPE_ENTRY_POINTS["NewConfigType"] = "new_script.py"
```

## Compatibility with Existing Pipeline

The `DefaultValuesProvider` is designed to be compatible with the existing pipeline system:

1. It preserves the same field names used in current configurations
2. It generates values consistent with current defaults
3. It can be used with existing configuration classes without modification

## Logging and Diagnostics

The `DefaultValuesProvider` includes comprehensive logging:

- Records which defaults are applied to each configuration
- Tracks any failures in applying callable defaults
- Provides methods for inspecting applicable defaults by configuration type

## Implementation Considerations

### Performance

The default application process is optimized for performance:

- Non-applicable defaults are quickly skipped
- Configuration instances are modified in-place
- Callable defaults are only evaluated when needed

### Error Handling

The system includes robust error handling:

- Gracefully handles failures in callable defaults
- Provides informative log messages for troubleshooting
- Falls back to safe alternatives when appropriate

### Testing

The design facilitates comprehensive testing:

- Default values are clearly separated from application logic
- Each default can be tested independently
- The application process can be verified with mock configurations

## Conclusion

The `DefaultValuesProvider` is a key component of the three-tier architecture that manages all system inputs (Tier 2). It provides a centralized, maintainable, and extensible system for applying default values to configurations, ensuring consistency and reducing the burden on users.

By implementing core design principles such as Single Source of Truth and Explicit Over Implicit, the `DefaultValuesProvider` creates a robust foundation for the configuration system that simplifies the user experience while maintaining compatibility with existing pipeline components.
