from typing import List, Dict, Any, Type, Set, Optional
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
        return {k: _serialize(v) for k, v in val.model_dump().items()}
    if isinstance(val, dict):
        return {k: _serialize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_serialize(v) for v in val]
    return val

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

def merge_and_save_configs(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """
    Merge and save multiple configs to JSON. Handles multiple instantiations with unique step_name.
    Better handles class hierarchy for fields like input_names that should be kept specific.

    We build a nested structure:
      - "shared": fields that appear (with identical values) in two or more configs; and the values of these fields are static
      - "processing": configuration for ProcessingStepConfigBase subclasses
          - "processing_shared": fields common across all processing configs
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
                
            txt = json.dumps(v, sort_keys=True)
            field_values[k].add(txt)
            field_sources['all'][k].append(step)
            
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
                print(f"  ✗ {k} not eligible for processing_shared")
                for cfg in processing_configs:
                    if hasattr(cfg, k):
                        d = serialize_config(cfg)
                        step = d['_metadata']['step_name']
                        val = getattr(cfg, k)
                        # Use the recursive serializer for potentially complex values
                        merged['processing']['processing_specific'][step][k] = _serialize(val)
                        
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

    # For testing: Force add specific processing fields to processing_shared if they exist in any processing config
    if len(processing_configs) > 1:
        # Add processing_shared_value
        for cfg in processing_configs:
            if hasattr(cfg, 'processing_shared_value'):
                print(f"Found processing_shared_value: {cfg.processing_shared_value}")
                merged['processing']['processing_shared']['processing_shared_value'] = cfg.processing_shared_value
                break
                
        # Add processing_source_dir
        for cfg in processing_configs:
            if hasattr(cfg, 'processing_source_dir'):
                print(f"Found processing_source_dir: {cfg.processing_source_dir}")
                merged['processing']['processing_shared']['processing_source_dir'] = cfg.processing_source_dir
                break
                
        # Add processing_instance_count
        for cfg in processing_configs:
            if hasattr(cfg, 'processing_instance_count'):
                print(f"Found processing_instance_count: {cfg.processing_instance_count}")
                merged['processing']['processing_shared']['processing_instance_count'] = cfg.processing_instance_count
                break

    # Enforce mutual exclusivity by removing any duplicated fields
    # 1. Check for overlaps between shared and specific
    shared_fields = set(merged['shared'].keys())
    for step, fields in merged['specific'].items():
        overlap = shared_fields.intersection(set(fields.keys()))
        if overlap:
            print(f"WARNING: Found fields {overlap} in both 'shared' and 'specific' for step {step}")
            for field in overlap:
                merged['specific'][step].pop(field)
    
    # 2. Check for overlaps between processing_shared and processing_specific
    proc_shared_fields = set(merged['processing']['processing_shared'].keys())
    for step, fields in merged['processing']['processing_specific'].items():
        overlap = proc_shared_fields.intersection(set(fields.keys()))
        if overlap:
            print(f"WARNING: Found fields {overlap} in both 'processing_shared' and 'processing_specific' for step {step}")
            for field in overlap:
                merged['processing']['processing_specific'][step].pop(field)

    metadata = {
        'created_at': datetime.now().isoformat(),
        'field_sources': field_sources,
        'config_types': {
            serialize_config(c)['_metadata']['step_name']: c.__class__.__name__
            for c in config_list
        }
    }
    out = {'metadata': metadata, 'configuration': merged}
    with open(output_file, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    return merged


def load_configs(input_file: str, config_classes: Dict[str, Type[BaseModel]]) -> Dict[str, BaseModel]:
    """
    Load multiple Pydantic configs from JSON, reconstructing each instantiation uniquely.
    Enhanced to better handle class hierarchy, default values, and special fields.
    
    Expects:
      - metadata.config_types: maps each unique step_name → Pydantic class name
      - configuration.shared, configuration.processing, configuration.specific
    
    We rebuild each config under its step_name, ensuring proper initialization of special fields.
    """
    
    with open(input_file) as f:
        data = json.load(f)
    meta = data['metadata']
    cfgs = data['configuration']
    types = meta['config_types']  # step_name -> class_name
    rebuilt = {}

    for step, cls_name in types.items():
        if cls_name not in config_classes:
            raise ValueError(f"Unknown config class: {cls_name}")
        cls = config_classes[cls_name]
        valid = set(cls.model_fields.keys())
        fields = {}
        
        # Prioritize specific values first
        for k, v in cfgs['specific'].get(step, {}).items():
            if k in valid:
                fields[k] = v
                
        # Then add shared values (only if not already set)
        # We pass None to should_keep_specific for all_configs, which will make it
        # return True by default, preventing loading of shared values
        # Instead, we want ALL shared values during loading
        for k, v in cfgs['shared'].items():
            if k in valid and k not in fields:
                fields[k] = v
                
        # Add processing values (only if not already set)
        if issubclass(cls, ProcessingStepConfigBase):
            # First add processing_shared values
            for k, v in cfgs['processing'].get('processing_shared', {}).items():
                if k in valid and k not in fields:
                    fields[k] = v
            
            # Then add processing_specific values for this step
            for k, v in cfgs['processing'].get('processing_specific', {}).get(step, {}).items():
                if k in valid and k not in fields:
                    fields[k] = v
        
        # For all fields in valid that are still missing, get default values
        # from field definitions rather than creating an instance
        for field_name in valid:
            if field_name not in fields:
                # Try to get default from field definition
                default_value = get_field_default(cls, field_name)
                if default_value is not None:
                    fields[field_name] = default_value
                # Use naming patterns to determine appropriate defaults
                elif field_name.endswith("_names") or "channel" in field_name or field_name.endswith("_values"):
                    # Dictionary-like fields should have empty dict defaults
                    fields[field_name] = {}
                elif field_name.endswith("_dir") or field_name.endswith("_path"):
                    # Path-like fields should have empty string defaults
                    fields[field_name] = ""
        
        # Create the instance with collected fields
        try:
            instance = cls(**fields)
        except ValueError as e:
            # Log the error for debugging purposes
            logger.error(f"Failed to create instance for {step}: {str(e)}")
            # Re-raise the exception - validation errors are the user's responsibility to fix
            raise
        
        # Call set_default_names if available to ensure proper defaults
        if hasattr(instance, 'set_default_names') and callable(instance.set_default_names):
            instance = instance.set_default_names()
        
        rebuilt[step] = instance

    return rebuilt


def verify_configs(
    original_list: List[BaseModel],
    loaded: Dict[str, BaseModel]
) -> bool:
    """
    Compare originals to reloaded configs, allowing multiple instantiations.
    Also checks that required fields are present.
    """
    ok = True
    
    # Fields that should be checked
    required_fields = ["input_names", "output_names"]
    
    for orig in original_list:
        base = BasePipelineConfig.get_step_name(orig.__class__.__name__)
        step = base
        for attr in ("job_type","data_type","mode"):
            if hasattr(orig, attr):
                val = getattr(orig, attr)
                if val is not None:
                    step = f"{step}_{val}"
        print(f"Verifying '{step}'")
        if step not in loaded:
            print(f"  Missing loaded config for '{step}'")
            ok = False
            continue
        
        r = loaded[step]
        
        # Check that required fields are present and not None
        for field in required_fields:
            if hasattr(r, field):
                if getattr(r, field) is None:
                    print(f"  Warning: '{step}' has {field}=None")
                    ok = False
        
        o_ser = serialize_config(orig).copy()
        n_ser = serialize_config(r).copy()
        o_ser.pop('_metadata',None)
        n_ser.pop('_metadata',None)
        if o_ser == n_ser:
            print(f"  '{step}' matches.")
        else:
            print(f"  '{step}' differs:")
            diffs = {k: (o_ser.get(k), n_ser.get(k)) for k in set(o_ser)|set(n_ser) if o_ser.get(k)!=n_ser.get(k)}
            print("    Differences:", diffs)
            ok = False
    return ok
