from typing import List, Dict, Any, Type, Set
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

from .config_base import BasePipelineConfig
from .config_processing_step_base import ProcessingStepConfigBase

# Dictionary fields should have empty dict defaults
DICT_FIELDS = {
    "input_names", 
    "output_names", 
    "training_input_channels", 
    "eval_input_channels",
    "special_field_values"
}

# String fields should have empty string defaults
STRING_FIELDS = {
    "processing_source_dir", 
    "payload_source_dir"
}

# All fields that should be kept specific to each config
ALWAYS_KEEP_SPECIFIC = DICT_FIELDS.union(STRING_FIELDS)

def should_keep_specific(config, field_name):
    """
    Determine if a field should be kept specific to this config.
    """
    # Dictionary fields should always be kept specific if they belong to the config
    if field_name in DICT_FIELDS and hasattr(config, field_name):
        return True
    
    # String path fields that some configs use
    if field_name in STRING_FIELDS and hasattr(config, field_name):
        return True
        
    # Everything else can be shared if values match
    return False


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """
    Serialize a single Pydantic config to a JSON‐serializable dict,
    embedding metadata including a unique 'step_name'.
    Supports multiple instantiations distinguished by job_type, data_type, or mode.
    Ensures critical fields are preserved with proper default values.
    """
    # Dump model to plain dict
    config_dict = config.model_dump() if hasattr(config, "model_dump") else config.dict()

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
    
    # Handle dictionary fields
    for field_name in DICT_FIELDS:
        if hasattr(config, field_name) and field_name not in config_dict:
            config_dict[field_name] = getattr(config, field_name) or {}
            
    # Handle string fields
    for field_name in STRING_FIELDS:
        if hasattr(config, field_name) and field_name not in config_dict:
            config_dict[field_name] = getattr(config, field_name) or ""

    # Recursive serializer for complex types
    def _serialize(val: Any) -> Any:
        if isinstance(val, datetime):
            return val.isoformat()
        if isinstance(val, Enum):
            return val.value
        if isinstance(val, Path):
            return str(val)
        if isinstance(val, dict):
            return {k: _serialize(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_serialize(v) for v in val]
        return val

    return {k: _serialize(v) for k, v in config_dict.items()}


def merge_and_save_configs(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """
    Merge and save multiple configs to JSON. Handles multiple instantiations with unique step_name.
    Better handles class hierarchy for fields like input_names that should be kept specific.

    We build three sections:
      - "shared": fields that appear (with identical values) in two or more configs
      - "processing": fields that belong to any ProcessingStepConfigBase subclass,
          grouped by each unique step_name
      - "specific": all other fields, grouped by each unique step_name

    Finally, under "metadata" → "config_types" we map each unique step_name → config class name.
    """
    
    merged = {"shared": {}, "processing": defaultdict(dict), "specific": defaultdict(dict)}
    field_values = defaultdict(set)
    field_sources = defaultdict(lambda: defaultdict(list))

    # Collect all values and sources
    for cfg in config_list:
        d = serialize_config(cfg)
        step = d["_metadata"]["step_name"]
        valid = set(cfg.__class__.model_fields.keys())
        for k, v in d.items():
            if k == "_metadata" or k not in valid:
                continue
            txt = json.dumps(v, sort_keys=True)
            field_values[k].add(txt)
            field_sources['all'][k].append(step)
            if isinstance(cfg, ProcessingStepConfigBase) and k in ProcessingStepConfigBase.model_fields:
                field_sources['processing'][k].append(step)
            else:
                field_sources['specific'][k].append(step)

    # Distribute into shared/processing/specific
    for k, vals in field_values.items():
        sources = field_sources['all'][k]
        
        # Check each config to see if this field should be kept specific for any of them
        is_special_field = any(should_keep_specific(cfg, k) for cfg in config_list if hasattr(cfg, k))
        
        if len(vals) == 1 and len(sources) > 1 and not is_special_field:
            # Shared fields (identical values across configs and not a special field)
            merged['shared'][k] = json.loads(next(iter(vals)))
        elif k in field_sources['processing']:
            for cfg in config_list:
                if isinstance(cfg, ProcessingStepConfigBase):
                    d = serialize_config(cfg)
                    step = d['_metadata']['step_name']
                    if k in d:
                        merged['processing'][step][k] = d[k]
        else:
            for cfg in config_list:
                d = serialize_config(cfg)
                step = d['_metadata']['step_name']
                if k in d:
                    merged['specific'][step][k] = d[k]
    
    # Double-check that special fields are never in shared section
    # Get all fields that any config needs to keep specific
    all_special_fields = set()
    for cfg in config_list:
        if hasattr(cfg.__class__, 'model_fields'):
            for field_name in cfg.__class__.model_fields.keys():
                if should_keep_specific(cfg, field_name):
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
                        merged['processing'][step][field_name] = value
                    else:
                        merged['specific'][step][field_name] = value

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
    Enhanced to better handle class hierarchy and special fields that need proper default values.
    
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
                
        # Then add shared values (only if not already set and not one that should remain specific)
        for k, v in cfgs['shared'].items():
            if k in valid and k not in fields and not should_keep_specific(cls, k):
                fields[k] = v
                
        # Add processing values (only if not already set)
        if issubclass(cls, ProcessingStepConfigBase):
            for k, v in cfgs['processing'].get(step, {}).items():
                if k in valid and k not in fields:
                    fields[k] = v
        
        # Add appropriate defaults for required fields if missing
        # Dictionary fields get empty dict defaults
        for field_name in DICT_FIELDS:
            if field_name in valid and field_name not in fields:
                fields[field_name] = {}
                
        # String fields get empty string defaults
        for field_name in STRING_FIELDS:
            if field_name in valid and field_name not in fields:
                fields[field_name] = ""
        
        # Create the instance with collected fields
        instance = cls(**fields)
        
        # Call set_default_names if available to ensure proper defaults
        if hasattr(instance, 'set_default_names') and callable(instance.set_default_names):
            instance = instance.set_default_names()
            
        # Special handling for ModelRegistrationConfig and PayloadConfig
        is_registration_class = False
        is_payload_class = False
        
        # Check MRO (inheritance chain) for ModelRegistrationConfig
        for base_cls in cls.__mro__:
            if base_cls.__name__ == 'ModelRegistrationConfig':
                is_registration_class = True
            if base_cls.__name__ == 'PayloadConfig':
                is_payload_class = True
                
        # For ModelRegistrationConfig, ensure input_names has required values
        if is_registration_class:
            if not hasattr(instance, 'input_names') or not getattr(instance, 'input_names'):
                instance.input_names = {
                    "packaged_model_output": "Output from packaging step (S3 path or Properties object)",
                    "payload_s3_key": "S3 key for payload data",
                    "payload_s3_uri": "S3 URI for payload data"
                }
                
        # For PayloadConfig, ensure proper output_names
        if is_payload_class:
            if not hasattr(instance, 'output_names') or not getattr(instance, 'output_names'):
                instance.output_names = {
                    "payload_sample": "Directory containing the generated payload samples",
                    "payload_metadata": "Directory containing the payload metadata"
                }
        
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
