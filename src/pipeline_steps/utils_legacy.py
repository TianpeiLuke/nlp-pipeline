"""
Legacy utility functions for configuration management.

This module contains the original implementation of configuration utilities
before the refactoring to src.config_field_manager.
"""

from typing import List, Dict, Any, Type, Set, Optional, Tuple, Union
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

from .config_base import BasePipelineConfig
from .config_processing_step_base import ProcessingStepConfigBase

logger = logging.getLogger(__name__)

# Define fields that should always be kept specific
SPECIAL_FIELDS_TO_KEEP_SPECIFIC = {"hyperparameters", "data_sources_spec", "transform_spec", "output_spec", "output_schema"}

# Constants for type-aware serialization
MODEL_TYPE_FIELD = "__model_type__"
MODEL_MODULE_FIELD = "__model_module__"

# Recursive serializer for complex types
def _serialize(val: Any) -> Any:
    """Convert complex types including Pydantic models to JSON-serializable values."""
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
                MODEL_TYPE_FIELD: cls_name,
                MODEL_MODULE_FIELD: module_name,
                **{k: _serialize(v) for k, v in val.model_dump().items()}
            }
            return result
        except Exception as e:
            logger.warning(f"Error serializing {val.__class__.__name__}: {str(e)}")
            return f"<Serialization error: {str(e)}>"
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_serialize(v) for v in val]
    return val

def get_class_by_name(class_name: str, config_classes: Dict[str, Type[BaseModel]], module_name: Optional[str] = None) -> Optional[Type[BaseModel]]:
    """
    Get a class from config_classes by name or attempt to import it from module.
    
    Args:
        class_name: Name of the class to find
        config_classes: Dictionary of registered classes
        module_name: Optional module name to import from
    
    Returns:
        The class type or None if not found
    """
    # First check registered classes
    if class_name in config_classes:
        return config_classes[class_name]
        
    # Try to import from module if provided
    if module_name:
        try:
            logger.debug(f"Attempting to import {class_name} from {module_name}")
            module = __import__(module_name, fromlist=[class_name])
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError as e:
            logger.warning(f"Failed to import {class_name} from {module_name}: {str(e)}")
    
    logger.warning(f"Class {class_name} not found in config_classes or {module_name}")
    return None

def deserialize_complex_field(field_data: Dict[str, Any], 
                             field_name: str,
                             field_type: Type,
                             config_classes: Dict[str, Type[BaseModel]]) -> Any:
    """
    Deserialize a complex field (like a nested Pydantic model) with proper type handling.
    
    Args:
        field_data: The serialized field data
        field_name: The name of the field
        field_type: The expected type of the field
        config_classes: Dictionary of registered classes
        
    Returns:
        Deserialized field value
    """
    # Skip if field_data isn't a dict
    if not isinstance(field_data, dict):
        return field_data
        
    # Check if we have type metadata
    type_name = field_data.get(MODEL_TYPE_FIELD)
    module_name = field_data.get(MODEL_MODULE_FIELD)
    
    if not type_name:
        # No type information, use the field_type
        if not isinstance(field_type, type) or not issubclass(field_type, BaseModel):
            # Not a model, return as is
            return field_data
        return field_type(**{k: v for k, v in field_data.items() 
                          if k not in (MODEL_TYPE_FIELD, MODEL_MODULE_FIELD)})
    
    # Get the actual class to use
    actual_class = get_class_by_name(type_name, config_classes, module_name)
    
    # If we couldn't find the class, log a warning and use field_type
    if not actual_class:
        logger.warning(f"Could not find class {type_name} for field {field_name}, using {field_type.__name__}")
        actual_class = field_type
    
    # Create instance with filtered data (removing metadata fields)
    filtered_data = {k: v for k, v in field_data.items() 
                    if k not in (MODEL_TYPE_FIELD, MODEL_MODULE_FIELD)}
                    
    # Recursively deserialize nested models
    for k, v in list(filtered_data.items()):
        if isinstance(v, dict) and MODEL_TYPE_FIELD in v:
            # This is a nested model
            nested_type = actual_class.model_fields[k].annotation if k in actual_class.model_fields else dict
            filtered_data[k] = deserialize_complex_field(v, k, nested_type, config_classes)
    
    try:
        return actual_class(**filtered_data)
    except Exception as e:
        logger.error(f"Failed to instantiate {actual_class.__name__} for field {field_name}: {str(e)}")
        # Last resort: try to use the base field type
        try:
            return field_type(**filtered_data)
        except Exception as nested_e:
            logger.error(f"Failed to instantiate {field_type.__name__} as fallback: {str(nested_e)}")
            # Return as plain dict if all else fails
            return filtered_data

def get_field_default(cls, field_name: str) -> Optional[Any]:
    """
    Get default value for a field if it exists, without creating an instance.
    This avoids validation errors with required fields.
    """
    if not hasattr(cls, "model_fields"):
        return None
        
    if field_name not in cls.model_fields:
        return None
        
    field = cls.model_fields[field_name]
    
    # Check if field has a default value
    if hasattr(field, "default") and field.default is not None and field.default != ...:
        return field.default
    
    # Check if field has a default_factory
    if hasattr(field, "default_factory") and field.default_factory is not None:
        try:
            return field.default_factory()
        except Exception as e:
            logger.debug(f"Error calling default_factory for {cls.__name__}.{field_name}: {e}")
            return None
            
    return None

# These field patterns get default values based on naming pattern
DICT_FIELD_PATTERNS = {"_names", "channel", "_values"}
PATH_FIELD_PATTERNS = {"_dir", "_path"}

def should_keep_specific(config, field_name, all_configs=None):
    """
    Determine if a field should be kept specific to this config based STRICTLY on
    whether the field has different values across configs.

    Categorization model:
    1. "shared" section: Fields with identical values across all configs
       - Static configuration values that apply to all configs
       - Example: author, bucket, pipeline_name that are the same across configs

    2. "specific" section: Fields that either:
       - Have different values across configs (e.g., input_names that differ)
       - Are unique to just one config (e.g., derived1_field)

    Args:
        config: The config instance being checked
        field_name: The name of the field to check
        all_configs: Optional list of all config instances for comparison
        
    Returns:
        bool: True if the field should be kept specific to this config
    """
    # Check for special fields that should always be kept specific
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return True
        
    # Check if this field is a Pydantic model
    value = getattr(config, field_name, None)
    if isinstance(value, BaseModel):
        # Complex Pydantic models should be kept specific
        return True
        
    # If we don't have all configs for comparison, we can't make a proper decision
    # So default to keeping it specific for safety
    if all_configs is None:
        return True
        
    # Count how many configs have this field
    configs_with_field = 0
    for cfg in all_configs:
        if hasattr(cfg, field_name):
            configs_with_field += 1
    
    # If only one config has this field, it's definitely specific
    if configs_with_field <= 1:
        return True
        
    # For fields present in multiple configs, check if values differ
    values = set()
    for cfg in all_configs:
        if hasattr(cfg, field_name):
            # Use json.dumps to handle complex objects like dicts/lists
            value = getattr(cfg, field_name, None)
            if value is not None:
                try:
                    value_str = json.dumps(value, sort_keys=True)
                    values.add(value_str)
                except (TypeError, ValueError):
                    # If we can't serialize for comparison, be safe and keep specific
                    return True
    
    # If we have multiple different values, keep field specific
    # Otherwise (all values identical), it can be shared
    return len(values) > 1


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """
    Serialize a single Pydantic config to a JSON‐serializable dict,
    embedding metadata including a unique 'step_name'.
    Enhanced to include default values from Pydantic model definitions.
    """
    # Dump model to plain dict using Pydantic v2's model_dump() method
    config_dict = config.model_dump()
    
    # Add default values for fields that are None or missing in config_dict
    # Use the field definitions directly instead of creating a temp instance
    cls = config.__class__
    if hasattr(cls, "model_fields"):
        for field_name in cls.model_fields.keys():
            if field_name not in config_dict or config_dict[field_name] is None:
                # Try to get default from field definition
                default_value = get_field_default(cls, field_name)
                if default_value is not None:
                    config_dict[field_name] = default_value
    
    # Base step name from registry
    base_step = BasePipelineConfig.get_step_name(config.__class__.__name__)
    step_name = base_step
    # Append distinguishing attributes
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"

    # Inject metadata
    config_dict["_metadata"] = {
        "step_name": step_name,
        "config_type": config.__class__.__name__,
    }
    
    # Use pattern matching to handle default values for different field types
    for field_name in cls.model_fields.keys():
        if hasattr(config, field_name) and field_name not in config_dict:
            if field_name.endswith("_names") or "channel" in field_name or field_name.endswith("_values"):
                # Dictionary-like fields get empty dict defaults
                config_dict[field_name] = getattr(config, field_name) or {}
            elif field_name.endswith("_dir") or field_name.endswith("_path"):
                # Path-like fields get empty string defaults
                config_dict[field_name] = getattr(config, field_name) or ""

    return {k: _serialize(v) for k, v in config_dict.items()}


def _is_likely_static(field_name: str, value: Any) -> bool:
    """
    Determine if a field is likely to be static based on its name and value.
    Static fields are those where:
    1. The value doesn't change during runtime
    2. The field represents configuration rather than dynamic data
o
    Returns:
        bool: True if the field is likely static, False otherwise
    """
    # Special fields that should never be considered static
    if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        return False
        
    # Pydantic models are typically not static configuration values
    if isinstance(value, BaseModel):
        return False
    
    # Fields that are generally not static
    non_static_patterns = {"_names", "input_", "output_", "_specific", "_count"}
    
    # Check if field name indicates it's not static
    if any(pattern in field_name for pattern in non_static_patterns):
        return False
        
    # Check if value type indicates it's not static
    if isinstance(value, (dict, list)):
        # Complex types are less likely to be static unless they're small
        if isinstance(value, dict) and len(value) > 3:
            return False
        if isinstance(value, list) and len(value) > 5:
            return False
            
    # Default to considering it static
    return True

def verify_configs(config_list: List[BaseModel]) -> None:
    """
    Verify that the configurations are valid.
    
    Args:
        config_list: List of configurations to verify
    """
    # Ensure unique step names
    step_names = set()
    for config in config_list:
        serialized = serialize_config(config)
        step_name = serialized["_metadata"]["step_name"]
        if step_name in step_names:
            raise ValueError(f"Duplicate step name: {step_name}")
        step_names.add(step_name)

def merge_and_save_configs_legacy(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """
    LEGACY IMPLEMENTATION
    
    Merge and save multiple configs to JSON. Handles multiple instantiations with unique step_name.
    Better handles class hierarchy for fields like input_names that should be kept specific.

    Field Categorization Rules:
    ---------------------------
    1. For fields that appear in BOTH processing and non-processing configs (cross-type fields):
       - If the field has identical values across ALL configs (both types): place in "shared"
       - Otherwise: place in appropriate specific sections ("processing_specific" or "specific")

    2. For fields EXCLUSIVE to processing configs:
       - If the field exists in ALL processing configs AND has identical values: place in "processing_shared"
       - Otherwise: place in "processing_specific" for each config

    3. For fields EXCLUSIVE to non-processing configs:
       - If the field exists in multiple configs AND has identical values: place in "shared"
       - Otherwise: place in "specific" for each config

    We build a nested structure:
      - "shared": fields that appear (with identical values) in two or more configs; and the values of these fields are static
      - "processing": configuration for ProcessingStepConfigBase subclasses
          - "processing_shared": fields common across all processing configs with identical values
          - "processing_specific": 1) fields unique to specific processing configs; 2) fields that are shared across multiple processing configs, but the values of them are different for different processing config; grouped by step name
      - "specific": 1) fields unique to specific configs; 2) fields that are shared across multiple configs, but the values of them are different for different config; grouped by step name

    The following categories are mutually exclusive (fields appear in only one location):
      - "shared" and "specific" sections have no overlapping fields
      - "processing_shared" and "processing_specific" sections have no overlapping fields
      
    Finally, under "metadata" → "config_types" we map each unique step_name → config class name.
    """
    
    merged = {
        "shared": {}, 
        "processing": {
            "processing_shared": {},
            "processing_specific": defaultdict(dict)
        }, 
        "specific": defaultdict(dict)
    }
    field_values = defaultdict(set)
    field_sources = defaultdict(lambda: defaultdict(list))
    field_static = defaultdict(bool)  # Track if fields are considered static
    cross_type_fields = set()  # Track fields that appear in both processing and non-processing configs

    # Collect all values and sources
    for cfg in config_list:
        # Use serialize_config to get all values including defaults
        d = serialize_config(cfg)
        step = d["_metadata"]["step_name"]
        valid = set(cfg.__class__.model_fields.keys())
        
        # Handle all fields in the serialized dict
        for k, v in d.items():
            if k == "_metadata":
                continue
            
            # Skip invalid fields (not in model_fields)
            if k not in valid:
                continue
            
            try:
                txt = json.dumps(v, sort_keys=True)
                field_values[k].add(txt)
                field_sources['all'][k].append(step)
            except Exception as e:
                logger.debug(f"Failed to serialize '{k}' in {cfg.__class__.__name__}: {str(e)}")
                # Continue without this field value
            
            # Properly categorize processing fields vs specific fields
            if isinstance(cfg, ProcessingStepConfigBase) and k in ProcessingStepConfigBase.model_fields:
                field_sources['processing'][k].append(step)
            else:
                field_sources['specific'][k].append(step)
            
            # Track cross-type fields (appearing in both processing and non-processing configs)
            # We'll accumulate this information across all fields and configs
                
            # Determine if field is likely static (can only be a candidate for shared section)
            # Static fields are those that don't vary across configs and are generally simple types
            field_static[k] = _is_likely_static(k, v)
        
        # Add a special check for ProcessingStepConfigBase fields
        if isinstance(cfg, ProcessingStepConfigBase):
            print(f"Special check for ProcessingStepConfigBase fields in {cfg.__class__.__name__}:")
            for field_name, field_info in ProcessingStepConfigBase.model_fields.items():
                if field_name not in d:  # Field wasn't in serialized dict
                    print(f"  Found missing field: {field_name}")
                    # Get the value directly from the instance
                    val = getattr(cfg, field_name)
                    try:
                        txt = json.dumps(val, sort_keys=True)
                        print(f"  Adding {field_name} = {txt}")
                        field_values[field_name].add(txt)
                        field_sources['all'][field_name].append(step)
                        field_sources['processing'][field_name].append(step)
                    except (TypeError, ValueError):
                        print(f"  Couldn't serialize {field_name}")

    # Get list of processing configs and non-processing configs
    processing_configs = [cfg for cfg in config_list if isinstance(cfg, ProcessingStepConfigBase)]
    non_processing_configs = [cfg for cfg in config_list if not isinstance(cfg, ProcessingStepConfigBase)]
    
    # Identify cross-type fields (fields present in both processing and non-processing configs)
    for field_name in field_sources['all']:
        # Check how many processing and non-processing configs have this field
        processing_configs_with_field = [cfg for cfg in processing_configs if hasattr(cfg, field_name)]
        non_processing_configs_with_field = [cfg for cfg in non_processing_configs if hasattr(cfg, field_name)]
        
        processing_has_field = len(processing_configs_with_field) > 0
        non_processing_has_field = len(non_processing_configs_with_field) > 0
        
        # Only consider it a cross-type field if it appears in BOTH types of configs
        if processing_has_field and non_processing_has_field:
            cross_type_fields.add(field_name)
            print(f"Cross-type field detected: {field_name} (appears in both processing and non-processing configs)")
    
    # Debug: Print processing configs info
    print(f"Found {len(processing_configs)} processing configs:")
    for cfg in processing_configs:
        print(f" - {cfg.__class__.__name__}")
    
    # Debug processing fields
    if len(processing_configs) > 0:
        first_proc_config = processing_configs[0]
        print(f"ProcessingStepConfigBase fields: {list(ProcessingStepConfigBase.model_fields.keys())}")
        print(f"First processing config fields: {list(first_proc_config.__class__.model_fields.keys())}")
        for field_name in dir(first_proc_config):
            if not field_name.startswith('_') and not callable(getattr(first_proc_config, field_name)):
                print(f"Attribute: {field_name} = {getattr(first_proc_config, field_name)}")
    
    # Distribute into shared/processing/specific
    for k, vals in field_values.items():
        sources = field_sources['all'][k]
        
        # Check each config to see if this field should be kept specific for any of them
        is_special_field = any(should_keep_specific(cfg, k, config_list) for cfg in config_list if hasattr(cfg, k))
        is_cross_type = k in cross_type_fields
        
        # Count how many processing and non-processing configs have this field
        processing_configs_with_field = [cfg for cfg in processing_configs if hasattr(cfg, k)]
        non_processing_configs_with_field = [cfg for cfg in non_processing_configs if hasattr(cfg, k)]
        
        # First check: fields that should NEVER be shared
        # 1. Fields that only appear in one config - these are clearly specific to that config
        # 2. Fields with different values across configs - these can't be shared
        # 3. Fields that are already identified as special via should_keep_specific
        # 4. Non-static fields
        never_share = (
            len(sources) <= 1 or           # Only in one config
            len(vals) > 1 or               # Different values
            is_special_field or            # Already deemed special
            not field_static[k]            # Not a static field
        )
        
        # Then identify fields that only exist in processing or non-processing configs,
        # but not both (these are not cross-type fields, but "type-specific" fields)
        is_type_specific = (
            (len(processing_configs_with_field) > 0 and len(non_processing_configs_with_field) == 0) or
            (len(processing_configs_with_field) == 0 and len(non_processing_configs_with_field) > 0)
        )
        
        # For type-specific fields to be eligible for shared, they must be in
        # ALL configs of that type
        configs_of_relevant_type = None
        if is_type_specific:
            configs_of_relevant_type = processing_configs if len(processing_configs_with_field) > 0 else non_processing_configs
            if len(sources) != len(configs_of_relevant_type):
                # Not in all configs of its type, keep it in specifics
                never_share = True
        
        # Special handling for fields with names suggesting they are specific to a type
        # or fields that have "varying" in their name (indicating they vary between configs)
        type_specific_patterns = ["_only", "only_", "standard_", "processing_", "varying"]
        if any(pattern in k for pattern in type_specific_patterns):
            never_share = True
        
        # Enhanced logic for shared fields:
        # 1) Must have identical values across all configs where it appears
        # 2) Must be present in multiple configs
        # 3) Must be considered static
        # 4) Must not be determined as special field
        # 5) For cross-type fields (appearing in both processing and non-processing), 
        #    must be in ALL configs to be shared
        if not never_share and (not is_cross_type or len(sources) == len(config_list)):
            # Shared fields (identical values across configs, considered static)
            merged['shared'][k] = json.loads(next(iter(vals)))
        elif k in field_sources['processing']:
            # Handle processing fields differently
            # Check if this processing field is common across all processing configs
            processing_values = set()
            processing_configs_with_field = []
            
            print(f"Processing field: {k}")
            
            for cfg in processing_configs:
                if hasattr(cfg, k):
                    processing_configs_with_field.append(cfg)
                    val = getattr(cfg, k)
                    try:
                        value_str = json.dumps(val, sort_keys=True)
                        print(f"  {cfg.__class__.__name__}.{k} = {value_str}")
                        processing_values.add(value_str)
                    except (TypeError, ValueError):
                        # If we can't serialize, treat it as specific
                        print(f"  {cfg.__class__.__name__}.{k} is not serializable")
                        processing_values.add(id(val))
            
            print(f"  Processing field {k}:")
            print(f"  - Values count: {len(processing_values)}")
            print(f"  - Configs with field count: {len(processing_configs_with_field)}")
            
            # For processing_shared - must have identical values across ALL processing configs
            # AND must not be a cross-type field (otherwise it belongs in specific sections)
            if len(processing_values) == 1 and len(processing_configs_with_field) == len(processing_configs) and not is_cross_type:
                # Common value across ALL processing configs - put in processing_shared
                value_str = next(iter(processing_values))
                print(f"  ✓ Adding {k} to processing_shared with value {value_str}")
                merged['processing']['processing_shared'][k] = json.loads(value_str)
            else:
                # Different values or not in all processing configs - put in processing_specific
                # This covers fields unique to specific processing configs or shared with different values
                # Log why field is not eligible for processing_shared
                if logger.isEnabledFor(logging.DEBUG):
                    reasons = []
                    if len(processing_values) > 1:
                        reasons.append(f"Values differ across configs ({len(processing_values)} different values)")
                    if len(processing_configs_with_field) < len(processing_configs):
                        reasons.append(f"Not present in all processing configs ({len(processing_configs_with_field)} of {len(processing_configs)})")
                    if is_cross_type:
                        reasons.append("It's a cross-type field (appears in both processing and non-processing configs)")
                    logger.debug(f"Field '{k}' not eligible for processing_shared because: {', '.join(reasons)}")
                    
                # Process the field for all processing configs that have it
                for cfg in processing_configs:
                    if hasattr(cfg, k):
                        d = serialize_config(cfg)
                        step = d['_metadata']['step_name']
                        val = getattr(cfg, k)
                        
                        # Use the recursive serializer for potentially complex values
                        try:
                            serialized_val = _serialize(val)
                            
                            # Check if step exists in processing_specific
                            if step not in merged['processing']['processing_specific']:
                                merged['processing']['processing_specific'][step] = {}
                                
                            # Add the field to processing_specific
                            merged['processing']['processing_specific'][step][k] = serialized_val
                        except Exception as e:
                            logger.warning(f"Failed to serialize '{k}' for {step}: {str(e)}")
                        
        # Special handling for cross-type fields
        if is_cross_type and k not in merged['shared']:
            # For processing configs - add to processing_specific only if not in processing_shared
            for cfg in processing_configs:
                if (hasattr(cfg, k) and 
                    isinstance(cfg, ProcessingStepConfigBase) and 
                    k not in merged['processing']['processing_shared']):
                    d = serialize_config(cfg)
                    step = d['_metadata']['step_name']
                    val = getattr(cfg, k)
                    # Use the recursive serializer for potentially complex values
                    if step not in merged['processing']['processing_specific']:
                        merged['processing']['processing_specific'][step] = {}
                    merged['processing']['processing_specific'][step][k] = _serialize(val)
                    
            # For non-processing configs - add to standard specific only if not in shared
            for cfg in non_processing_configs:
                if hasattr(cfg, k) and k not in merged['shared']:
                    d = serialize_config(cfg)
                    step = d['_metadata']['step_name']
                    val = getattr(cfg, k)
                    # Use the recursive serializer for potentially complex values
                    merged['specific'][step][k] = _serialize(val)
        else:
            # Regular specific fields - only add if not already in shared section
            # This covers fields unique to specific configs or shared with different values
            for cfg in config_list:
                if (not isinstance(cfg, ProcessingStepConfigBase) and 
                    hasattr(cfg, k) and 
                    k not in merged['shared']):
                    d = serialize_config(cfg)
                    step = d['_metadata']['step_name']
                    val = getattr(cfg, k)
                    # Use the recursive serializer for potentially complex values
                    merged['specific'][step][k] = _serialize(val)
    
    # Double-check that special fields are never in shared section
    # Get all fields that any config needs to keep specific
    all_special_fields = set()
    for cfg in config_list:
        if hasattr(cfg.__class__, 'model_fields'):
            for field_name in cfg.__class__.model_fields.keys():
                if should_keep_specific(cfg, field_name, config_list):
                    all_special_fields.add(field_name)
    
    # Move any special fields from shared to their specific configs
    for field_name in all_special_fields:
        if field_name in merged['shared']:
            # Move this field from shared to all specific configs
            shared_value = merged['shared'].pop(field_name)
            
            # Add to all configs that need it
            for cfg in config_list:
                if hasattr(cfg, field_name):
                    step = serialize_config(cfg)["_metadata"]["step_name"]
                    # Either use the config's specific value or the shared value
                    value = getattr(cfg, field_name, shared_value)
                    
                    # Add to the right section based on config type
                    if isinstance(cfg, ProcessingStepConfigBase) and field_name in ProcessingStepConfigBase.model_fields:
                        # Put in processing_specific for this specific processing step
                        merged['processing']['processing_specific'][step][field_name] = _serialize(value)
                    else:
                        # Put in regular specific section
                        merged['specific'][step][field_name] = _serialize(value)

    # Special handling for hyperparameters - always keep them in their specific sections
    for field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
        # If it's in shared, move it to specific configs
        if field_name in merged['shared']:
            logger.info(f"Moving special field '{field_name}' from shared section to specific sections")
            shared_value = merged['shared'].pop(field_name)
            # Add to all configs that had this field
            for cfg in config_list:
                if hasattr(cfg, field_name):
                    step = serialize_config(cfg)["_metadata"]["step_name"]
                    value = getattr(cfg, field_name, shared_value)
                    
                    # Add to the right section based on config type
                    if isinstance(cfg, ProcessingStepConfigBase):
                        # Put in processing_specific for this processing step
                        merged['processing']['processing_specific'][step][field_name] = _serialize(value)
                    else:
                        # Put in regular specific section
                        merged['specific'][step][field_name] = _serialize(value)
                        
    # Now remove empty sections to keep the output clean
    if len(merged['processing']['processing_shared']) == 0:
        del merged['processing']['processing_shared']
        
    if len(merged['processing']['processing_specific']) == 0:
        del merged['processing']['processing_specific']
        
    if len(merged['processing']) == 0:
        del merged['processing']
    
    if len(merged['specific']) == 0:
        del merged['specific']
    
    # Create step_name -> class_type mapping for the metadata
    config_types = {}
    for config in config_list:
        serialized = serialize_config(config)
        step_name = serialized["_metadata"]["step_name"]
        class_name = config.__class__.__name__
        config_types[step_name] = class_name
    
    # Create final output dict with metadata
    output_dict = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "config_types": config_types
        },
        "configuration": merged
    }
    
    # Save to file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_dict, f, indent=2, sort_keys=True)
    
    logger.info(f"Saved merged configuration to {output_file}")
    
    return merged
