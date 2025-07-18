# Configuration Field Categorization and Placement Logic

## Overview

This document provides a detailed explanation of how the configuration system categorizes and places fields in various sections of the output JSON file. It includes the decision tree for determining where each field should be placed, special handling for complex fields like hyperparameters, and the implementation details of the `merge_and_save_configs` function.

## Configuration Structure

The configuration system organizes fields into a nested structure:

```
{
  "shared": { ... },              // Fields common across ALL configs with static values
  "processing": {
    "processing_shared": { ... }, // Fields common across processing configs only
    "processing_specific": {      // Fields specific to individual processing configs
      "Step1": { ... },
      "Step2": { ... }
    }
  },
  "specific": {                   // Fields specific to individual non-processing configs
    "Step3": { ... },
    "Step4": { ... }
  }
}
```

## Field Categorization Rules

### 1. Special Fields

Special fields are always kept in specific sections, regardless of their values or other categorization rules.

```python
SPECIAL_FIELDS_TO_KEEP_SPECIFIC = {
    "hyperparameters", 
    "data_sources_spec", 
    "transform_spec", 
    "output_spec", 
    "output_schema"
}
```

These fields are typically complex Pydantic models that contain nested configuration data. They're always kept specific to ensure that:

- Each configuration maintains its own distinct set of these fields
- Changes to one configuration's special fields don't affect others
- The fields appear in the expected location for downstream consumers

### 2. Cross-Type Fields vs. Type-Specific Fields

The system distinguishes between:
- **Cross-type fields**: Fields that appear in both processing and non-processing configs
- **Type-specific fields**: Fields that appear only in processing configs or only in non-processing configs

### 3. General Field Placement Decision Tree

For non-special fields, the system uses the following logic:

```
IF is_special_field(field)
  → Always put in specific/processing_specific

ELSE IF field has different values across configs OR appears in only one config OR is non-static
  → Put in specific/processing_specific

ELSE IF is_cross_type_field(field) AND field is in ALL configs
  → Put in shared 

ELSE IF field is in processing configs only AND has identical values AND is in ALL processing configs
  → Put in processing_shared

ELSE IF field is in non-processing configs only AND has identical values
  → Put in shared

ELSE
  → Put in specific/processing_specific
```

### 4. Static vs. Non-Static Fields

Only static fields are eligible for the shared section. A field is considered static if:

- It doesn't have a name pattern suggesting runtime values (`_names`, `input_`, `output_`, etc.)
- It's not a complex type (large dictionaries, lists, or Pydantic models)
- It's not explicitly marked as a special field

## Config Step Name Generation and Uniqueness

### Step Name Generation Logic

Each configuration is assigned a unique step name that becomes the key in the specific sections. This step name is generated based on:

1. Base step name from the configuration class name
2. Additional distinguishing attributes if present

```python
# Base step name from registry
base_step = BasePipelineConfig.get_step_name(config.__class__.__name__)
step_name = base_step

# Append distinguishing attributes
for attr in ("job_type", "data_type", "mode"):
    if hasattr(config, attr):
        val = getattr(config, attr)
        if val is not None:
            step_name = f"{step_name}_{val}"
```

### Important Config Uniqueness Assumption

**The configuration system assumes that configs of the same class type will be distinguished by at least one of these attributes: "job_type", "data_type", or "mode".** If multiple configs of the same class exist without these distinguishing attributes, they will have the same step_name and only one will appear in the output structure (the later one will overwrite earlier ones).

This is an intentional design decision to enforce logical separation of configurations. Different configs should either:
- Be different classes (with unique class names)
- Or have different purposes indicated by job_type, data_type, or mode attributes

For proper handling of multiple similar configurations, ensure they have distinct values for at least one of these attributes.

## Decision Process Flow

The full decision process for each field follows these steps:

1. **Collection Phase**
   - Serialize all configs to collect field values and metadata
   - Determine which configs have each field
   - Categorize fields as processing, non-processing, or cross-type
   - Identify which fields are likely static

2. **Special Field Check**
   - Check if any field is in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
   - Mark fields as special if they are Pydantic models

3. **Processing Fields Categorization**
   - Check if processing fields have identical values across all processing configs
   - If identical and not cross-type → processing_shared
   - Otherwise → processing_specific

4. **Non-Processing Fields Categorization**
   - Check if non-processing fields have identical values
   - If identical, static, and not special → shared
   - Otherwise → specific

5. **Cross-Type Field Handling**
   - For fields in both processing and non-processing configs:
     - If identical across ALL configs → shared
     - Otherwise → put in appropriate specific section

6. **Special Field Recovery**
   - For any special field that ended up in shared → move to specific
   - For any special field that ended up in processing_shared → move to processing_specific

7. **Validation**
   - Ensure mutual exclusivity between shared/specific and processing_shared/processing_specific
   - Check for any fields that might have been missed and force add them to the correct section

## Implementation Challenges

### Handling Pydantic Models

Complex Pydantic models like `hyperparameters` require special handling:

1. **Serialization**: The `_serialize` function recursively handles Pydantic models:
   ```python
   if isinstance(val, BaseModel):  # Handle Pydantic models
       result = {k: _serialize(v) for k, v in val.model_dump().items()}
       return result
   ```

2. **Detecting Special Fields**: The system identifies Pydantic models as special:
   ```python
   # Check if this field is a Pydantic model
   value = getattr(config, field_name, None)
   if isinstance(value, BaseModel):
       # Complex Pydantic models should be kept specific
       return True
   ```

### The Special Field Gap Issue

During testing, we discovered a gap in the workflow that could cause special fields not to appear in the output JSON, despite being correctly identified. The issue was:

1. The field would be correctly identified as special
2. The logic to move special fields from shared to specific sections worked correctly
3. But if the field wasn't being processed in either flow, it would not be added to the output

Our solution adds a final verification step that:
1. Checks each processing config for special fields
2. Verifies if each special field exists in its appropriate specific section
3. Force adds any missing special fields:

```python
# Check all processing configs to see if they have special fields
for cfg in processing_configs:
    step = serialize_config(cfg)["_metadata"]["step_name"]
    
    for field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        if hasattr(cfg, field_name):
            if step not in merged['processing']['processing_specific'] or \
               field_name not in merged['processing']['processing_specific'][step]:
                # Field missing - force add it
                value = getattr(cfg, field_name)
                serialized = _serialize(value)
                
                if step not in merged['processing']['processing_specific']:
                    merged['processing']['processing_specific'][step] = {}
                    
                merged['processing']['processing_specific'][step][field_name] = serialized
```

## Common Pitfalls and Solutions

### 1. Serialization Failures

If a field can't be serialized to JSON, it won't be included in the output. The system handles this by:
- Using a robust `_serialize` function that handles common Python types
- Adding error handling and fallbacks for problematic types
- Tracking serialization failures in logs

### 2. Processing vs. Non-Processing Confusion

Fields can appear in both processing and non-processing configs, which can complicate placement decisions. The solution:
- Explicitly track which fields appear in which types of configs
- Use the concept of "cross-type fields" for fields that appear in both
- Apply more stringent rules for these fields to prevent them from being incorrectly shared

### 3. Special Fields Not Appearing

As we found with the hyperparameters field, special fields might be identified correctly but still not appear in the output JSON. Our solution:
- Add a final verification pass that ensures all special fields are present
- Force add any missing special fields to their appropriate sections
- Add extra logging to track the handling of special fields

### 4. Multiple Configs of the Same Class

If multiple configs of the same class exist without distinguishing attributes (job_type, data_type, mode), they will have the same step_name and only one will appear in the output. Solutions:
- Always provide distinguishing attributes for configs of the same class
- Use different class types if the configurations serve fundamentally different purposes
- Check the output to ensure all expected configs appear with their own sections

## Recommendations for Adding New Special Fields

When adding new fields that should always be kept specific:

1. Add the field to the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
2. Ensure the field is a well-defined class, preferably a Pydantic model
3. Add appropriate validation rules to the model
4. Test the serialization and deserialization of the field

## Testing Field Categorization

To verify field categorization is working correctly:

1. Create configs with known fields and values
2. Run `merge_and_save_configs` to generate the output JSON
3. Verify special fields appear in their specific sections
4. Check that shared fields have identical values across configs
5. Verify that no field appears in both shared and specific sections
6. If testing multiple configs of the same class, ensure they have distinguishing attributes

---

This design document provides a comprehensive overview of the configuration field categorization system, including the decision logic, implementation details, and solutions to common issues. The system is designed to be robust and flexible, handling a wide variety of field types and ensuring that special fields like hyperparameters are always placed in the appropriate sections.

## Type-Aware Model Serialization and Deserialization

### The Derived Class Challenge

A significant challenge in the configuration system is handling derived model classes, particularly with hyperparameters. For example:

1. `DummyTrainingConfig` accepts a base `ModelHyperparameters` field:
   ```python
   hyperparameters: ModelHyperparameters = Field(...)
   ```

2. In practice, derived classes like `BSMModelHyperparameters` or `XGBoostHyperparameters` are used, which contain additional fields not present in the base class.

3. When saving, all fields (including derived class fields) are serialized.

4. When loading, the system attempts to reconstruct a `ModelHyperparameters` instance with all these additional fields, causing validation errors since the base class has `extra='forbid'` in its configuration.

### Type-Aware Serialization Solution

To solve this issue, we've implemented type-aware serialization and deserialization:

1. **Track Type Information During Serialization**:
   ```python
   # Constants for metadata fields
   MODEL_TYPE_FIELD = "__model_type__"
   MODEL_MODULE_FIELD = "__model_module__"
   
   # In _serialize function
   if isinstance(val, BaseModel):
       result = {
           MODEL_TYPE_FIELD: val.__class__.__name__,
           MODEL_MODULE_FIELD: val.__class__.__module__,
           **{k: _serialize(v) for k, v in val.model_dump().items()}
       }
   ```

2. **Use Correct Type During Deserialization**:
   ```python
   # In load_configs function, when processing fields
   if field_name in fields and isinstance(fields[field_name], dict):
       if MODEL_TYPE_FIELD in fields[field_name]:
           model_type = fields[field_name][MODEL_TYPE_FIELD]
           if model_type in config_classes:
               hyper_cls = config_classes[model_type]
               fields[field_name] = hyper_cls(**{k: v for k, v in fields[field_name].items() 
                                                if k not in (MODEL_TYPE_FIELD, MODEL_MODULE_FIELD)})
   ```

### Implementation Details

The system uses these key components:

1. **Type Tracking**: When serializing Pydantic models, the class name and module are stored as metadata.

2. **Class Registry**: A comprehensive registry of all model classes (`CONFIG_CLASSES`) includes both configuration classes and model classes like hyperparameter classes.

3. **Type-Aware Deserialization**: When deserializing fields, the system checks for type metadata and instantiates the correct class.

4. **Fallback Mechanism**: If the exact class can't be found, the system falls back to the field's declared type.

5. **Recursive Handling**: The system handles nested models with different types at any depth.

### Best Practices for Derived Model Classes

When working with derived model classes:

1. **Register All Classes**: Ensure all derived model classes are registered in the `CONFIG_CLASSES` dictionary.

2. **Maintain Base Class Minimalism**: Keep the base class minimal and focused on core functionality.

3. **Use Strict Validation**: Continue to use `extra='forbid'` in base classes to ensure strict validation during normal usage.

4. **Document Derived Fields**: Clearly document fields added in derived classes to assist in troubleshooting.

5. **Test Serialization Cycle**: Always test the full serialization/deserialization cycle when adding new derived classes.

This approach maintains the design principle of having minimal base classes with strict validation while still allowing for specialized derived classes with additional fields. It ensures that configurations can be properly saved and loaded regardless of which hyperparameter class is used.
