# Type-Aware Serializer

## Overview

The TypeAwareSerializer provides a robust system for serializing and deserializing complex types with complete type information preservation. This component is essential for maintaining type safety throughout the configuration lifecycle, ensuring that configurations can be correctly reconstructed during loading.

## Purpose

The purpose of TypeAwareSerializer is to:

1. **Maintain type information** during serialization to JSON
2. **Support complex nested types** including Pydantic models
3. **Enable accurate deserialization** of complex objects
4. **Handle special types** like datetime, Enum, and Path
5. **Provide graceful error handling** when serialization fails

## Key Components

### 1. Type Metadata Fields

TypeAwareSerializer adds type metadata to serialized objects:

```python
# Constants for metadata fields
MODEL_TYPE_FIELD = "__model_type__"    # Stores the class name
MODEL_MODULE_FIELD = "__model_module__" # Stores the module name
```

These fields are embedded in serialized dictionaries to preserve type information:

```json
{
  "__model_type__": "CradleDataLoadConfig",
  "__model_module__": "src.pipeline_steps.config_cradle_data_load",
  "job_type": "training",
  "region": "us-west-2",
  "data_source_type": "s3"
}
```

### 2. Serialization Methods

The serializer handles various types with specialized processing:

```python
def serialize(self, val):
    """
    Serialize a value with type information when needed.
    
    Args:
        val: The value to serialize
        
    Returns:
        Serialized value suitable for JSON
    """
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, BaseModel):  # Handle Pydantic models
        try:
            # Get class details
            cls = val.__class__
            module_name = cls.__module__
            cls_name = cls.__name__
            
            # Create serialized dict with type metadata
            result = {
                self.MODEL_TYPE_FIELD: cls_name,
                self.MODEL_MODULE_FIELD: module_name,
                **{k: self.serialize(v) for k, v in val.model_dump().items()}
            }
            return result
        except Exception as e:
            self.logger.warning(f"Error serializing {val.__class__.__name__}: {str(e)}")
            return f"<Serialization error: {str(e)}>"
    if isinstance(val, dict):
        return {k: self.serialize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [self.serialize(v) for v in val]
    return val
```

### 3. Deserialization Methods

The deserializer reconstructs objects based on type metadata:

```python
def deserialize(self, field_data, field_name=None, expected_type=None):
    """
    Deserialize data with proper type handling.
    
    Args:
        field_data: The serialized data
        field_name: Optional name of the field (for logging)
        expected_type: Optional expected type
        
    Returns:
        Deserialized value
    """
    # Skip if not a dict or no type info needed
    if not isinstance(field_data, dict):
        return field_data
        
    # Check for type metadata
    type_name = field_data.get(self.MODEL_TYPE_FIELD)
    module_name = field_data.get(self.MODEL_MODULE_FIELD)
    
    if not type_name:
        # No type information, use the expected_type if applicable
        if expected_type and isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
            return self._deserialize_model(field_data, expected_type)
        return field_data
        
    # Get the actual class to use
    actual_class = self._get_class_by_name(type_name, module_name)
    
    # If we couldn't find the class, log warning and use expected_type
    if not actual_class:
        self.logger.warning(
            f"Could not find class {type_name} for field {field_name or 'unknown'}, "
            f"using {expected_type.__name__ if expected_type else 'dict'}"
        )
        actual_class = expected_type
        
    # If still no class, return as is
    if not actual_class:
        return field_data
        
    return self._deserialize_model(field_data, actual_class)
```

### 4. Model Deserialization

Special handling for Pydantic model deserialization:

```python
def _deserialize_model(self, field_data, model_class):
    """
    Deserialize a model instance.
    
    Args:
        field_data: Serialized model data
        model_class: Class to instantiate
        
    Returns:
        Model instance
    """
    # Remove metadata fields
    filtered_data = {k: v for k, v in field_data.items() 
                   if k not in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD)}
                   
    # Recursively deserialize nested models
    for k, v in list(filtered_data.items()):
        if isinstance(v, dict) and self.MODEL_TYPE_FIELD in v:
            # Get nested field type if available
            nested_type = None
            if hasattr(model_class, 'model_fields') and k in model_class.model_fields:
                nested_type = model_class.model_fields[k].annotation
            filtered_data[k] = self.deserialize(v, k, nested_type)
    
    try:
        return model_class(**filtered_data)
    except Exception as e:
        self.logger.error(f"Failed to instantiate {model_class.__name__}: {str(e)}")
        # Return as plain dict if instantiation fails
        return filtered_data
```

### 5. Class Resolution

Resolves classes by name using the config registry:

```python
def _get_class_by_name(self, class_name, module_name=None):
    """
    Get a class by name, from config_classes or by importing.
    
    Args:
        class_name: Name of the class
        module_name: Optional module to import from
        
    Returns:
        Class or None if not found
    """
    # First check registered classes
    if class_name in self.config_classes:
        return self.config_classes[class_name]
        
    # Try to import from module if provided
    if module_name:
        try:
            self.logger.debug(f"Attempting to import {class_name} from {module_name}")
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError as e:
            self.logger.warning(f"Failed to import {class_name} from {module_name}: {str(e)}")
    
    self.logger.warning(f"Class {class_name} not found")
    return None
```

## Usage Examples

### 1. Serializing a Config Object

```python
# Initialize serializer
serializer = TypeAwareSerializer()

# Create a config object
config = CradleDataLoadConfig(
    job_type="training",
    region="us-west-2",
    data_source_type="s3"
)

# Serialize with type information
serialized = serializer.serialize(config)
# Result:
# {
#   "__model_type__": "CradleDataLoadConfig",
#   "__model_module__": "src.pipeline_steps.config_cradle_data_load",
#   "job_type": "training",
#   "region": "us-west-2",
#   "data_source_type": "s3"
# }
```

### 2. Deserializing a Config Object

```python
# Deserialize back to a config object
config_classes = {"CradleDataLoadConfig": CradleDataLoadConfig}
serializer = TypeAwareSerializer(config_classes)

deserialized = serializer.deserialize(serialized)
# Result: CradleDataLoadConfig instance with all fields

assert isinstance(deserialized, CradleDataLoadConfig)
assert deserialized.job_type == "training"
```

### 3. Handling Nested Objects

```python
# Config with nested Pydantic model
config = XGBoostTrainingConfig(
    hyperparameters=XGBoostHyperparameters(
        max_depth=6,
        eta=0.3
    )
)

# Serialize with nested type information
serialized = serializer.serialize(config)
# Result includes nested type information:
# {
#   "__model_type__": "XGBoostTrainingConfig",
#   "__model_module__": "src.pipeline_steps.config_xgboost_training",
#   "hyperparameters": {
#     "__model_type__": "XGBoostHyperparameters",
#     "__model_module__": "src.pipeline_steps.hyperparameters_xgboost",
#     "max_depth": 6,
#     "eta": 0.3
#   }
# }

# Deserialize with nested objects
deserialized = serializer.deserialize(serialized)
assert isinstance(deserialized.hyperparameters, XGBoostHyperparameters)
assert deserialized.hyperparameters.max_depth == 6
```

## Benefits

1. **Complete Type Preservation**: Maintains full type information for complex objects
2. **Deep Nesting Support**: Handles arbitrarily nested models and collections
3. **Robust Error Handling**: Gracefully handles serialization and instantiation errors
4. **Flexible Class Resolution**: Uses registry and dynamic importing for class resolution
5. **Automatic Type Recognition**: Recognizes and handles special types automatically
6. **Recursive Processing**: Properly handles nested collections and models

## Implementation Details

### 1. Class Registration

The serializer can work with a provided class dictionary or build one on demand:

```python
def __init__(self, config_classes=None):
    """
    Initialize with optional config classes.
    
    Args:
        config_classes: Optional dictionary mapping class names to class objects
    """
    self.config_classes = config_classes or build_complete_config_classes()
    self.logger = logging.getLogger(__name__)
```

### 2. Special Type Handling

- **datetime**: Converted to ISO 8601 string
- **Enum**: Stored as the enum value
- **Path**: Converted to string
- **Pydantic models**: Serialized with type information
- **Collections**: Recursively processed

### 3. Deserialization Strategy

1. Check if input is a dictionary
2. Look for type metadata fields
3. Resolve the class using name and module
4. Remove metadata fields from data
5. Recursively process nested fields
6. Instantiate the class with processed fields
7. Fall back to dict if instantiation fails

## Error Handling

1. **Missing Classes**: Logs warning and falls back to expected type or dict
2. **Serialization Errors**: Catches exceptions and returns error message
3. **Instantiation Errors**: Logs error and returns dict with fields
4. **Import Errors**: Logs warning and attempts alternative resolution

## Future Improvements

1. **Schema Validation**: Add schema validation during deserialization
2. **Versioning Support**: Add versioning for backward compatibility
3. **Custom Type Handlers**: Support for registering custom type serializers
4. **Performance Optimization**: Caching for frequently used classes
5. **Format Options**: Support for alternative formats (YAML, TOML)

## References

- [Config Field Categorization](./config_field_categorization.md)
- [Config Registry](./config_registry.md)
- [Pydantic Documentation](https://docs.pydantic.dev/latest/)
