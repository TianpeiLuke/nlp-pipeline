from typing import List, Dict, Any, Type, Union, Set
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

from .config_base import BasePipelineConfig
from .config_processing_step_base import ProcessingStepConfigBase


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """Serialize a single config to dictionary, handling special types"""
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
    
    # Add metadata
    config_dict['_metadata'] = {
        'step_name': BasePipelineConfig.get_step_name(config.__class__.__name__),
        'config_type': config.__class__.__name__
    }
    
    def serialize_value(value: Any) -> Any:
        """Recursively serialize values"""
        if isinstance(value, datetime):
            return value.isoformat()
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, Path):
            return str(value)
        elif isinstance(value, dict):
            return {k: serialize_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [serialize_value(item) for item in value]
        return value
    
    return {key: serialize_value(value) for key, value in config_dict.items()}


def merge_and_save_configs(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """
    Merge multiple configs and save to JSON, handling overlapping processing fields
    """
    merged_config: Dict[str, Dict[str, Any]] = {
        "shared": {},  # For fields with same values across configs
        "processing": defaultdict(dict),  # For processing-specific fields
        "specific": defaultdict(dict)  # For other config-specific fields
    }
    
    field_values: Dict[str, Set[Any]] = defaultdict(set)
    field_sources: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    # First pass: collect all field values and their sources
    for config in config_list:
        config_dict = serialize_config(config)
        config_type = config_dict['_metadata']['config_type']
        step_name = config_dict['_metadata']['step_name']
        
        # Get valid fields for this config type
        valid_fields = set(config.__class__.model_fields.keys())
        
        for key, value in config_dict.items():
            if key != '_metadata' and key in valid_fields:
                # Store serializable value for comparison
                serialized_value = json.dumps(value, sort_keys=True)
                field_values[key].add(serialized_value)
                field_sources["all"][key].append(step_name)
                
                # Categorize the field
                if isinstance(config, ProcessingStepConfigBase):
                    if key in ProcessingStepConfigBase.model_fields:
                        field_sources["processing"][key].append(step_name)
                    else:
                        field_sources["specific"][key].append(step_name)
                else:
                    field_sources["specific"][key].append(step_name)

    # Second pass: categorize and store fields
    for key, values in field_values.items():
        if len(values) == 1 and len(field_sources["all"][key]) > 1:
            # Shared field (same value across configs)
            merged_config["shared"][key] = json.loads(next(iter(values)))
        elif key in field_sources["processing"]:
            # Processing-specific field
            for config in config_list:
                if isinstance(config, ProcessingStepConfigBase):
                    config_dict = serialize_config(config)
                    if key in config_dict and key in config.__class__.model_fields:
                        step_name = config_dict['_metadata']['step_name']
                        merged_config["processing"][step_name][key] = config_dict[key]
        else:
            # Config-specific field
            for config in config_list:
                config_dict = serialize_config(config)
                if key in config_dict and key in config.__class__.model_fields:
                    step_name = config_dict['_metadata']['step_name']
                    merged_config["specific"][step_name][key] = config_dict[key]

    metadata = {
        "created_at": datetime.now().isoformat(),
        "field_sources": field_sources,
        "config_types": {
            config.__class__.__name__: BasePipelineConfig.get_step_name(config.__class__.__name__)
            for config in config_list
        }
    }

    output_data = {
        "metadata": metadata,
        "configuration": merged_config
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    return merged_config


def load_configs(input_file: str, config_classes: Dict[str, Type[BaseModel]]) -> Dict[str, BaseModel]:
    """Load configurations from JSON file"""
    with open(input_file, 'r') as f:
        data = json.load(f)

    metadata = data["metadata"]
    config_data = data["configuration"]
    config_types = metadata["config_types"]
    field_sources = metadata["field_sources"]

    configs = {}
    for config_name, step_name in config_types.items():
        if config_name not in config_classes:
            raise ValueError(f"Unknown config type: {config_name}")

        config_class = config_classes[config_name]
        
        # Get the fields that are valid for this config class
        valid_fields = set(config_class.model_fields.keys())
        
        # Combine fields for this config
        fields = {}
        
        # Add shared fields (only if they're valid for this config)
        for key, value in config_data["shared"].items():
            if key in valid_fields:
                fields[key] = value
        
        # Add processing-specific fields if applicable
        if issubclass(config_class, ProcessingStepConfigBase):
            if step_name in config_data["processing"]:
                processing_fields = config_data["processing"][step_name]
                fields.update({
                    k: v for k, v in processing_fields.items()
                    if k in valid_fields
                })
        
        # Add config-specific fields
        if step_name in config_data["specific"]:
            specific_fields = config_data["specific"][step_name]
            fields.update({
                k: v for k, v in specific_fields.items()
                if k in valid_fields
            })

        try:
            configs[step_name] = config_class(**fields)
        except Exception as e:
            print(f"Error creating {step_name} config: {str(e)}")
            print(f"Attempted to create with fields: {fields}")
            print(f"Valid fields for {config_name}: {valid_fields}")
            raise

    return configs


def verify_configs(
    original_configs: List[BaseModel],
    loaded_configs: Dict[str, BaseModel]
) -> bool:
    """Verify that loaded configs match the originals"""
    all_match = True
    
    for original in original_configs:
        step_name = BasePipelineConfig.get_step_name(original.__class__.__name__)
        print(f"\nVerifying {step_name}:")
        
        if step_name not in loaded_configs:
            print(f"⚠ No loaded config found for {step_name}")
            all_match = False
            continue

        loaded_config = loaded_configs[step_name]
        
        # Compare serialized versions
        original_dict = serialize_config(original)
        loaded_dict = serialize_config(loaded_config)
        
        # Remove metadata for comparison
        original_dict.pop('_metadata', None)
        loaded_dict.pop('_metadata', None)
        
        if original_dict == loaded_dict:
            print(f"✓ Configuration matches original for {step_name}")
        else:
            print(f"⚠ Configuration differs for {step_name}")
            diffs = {
                k: (original_dict.get(k), loaded_dict.get(k))
                for k in set(original_dict) | set(loaded_dict)
                if original_dict.get(k) != loaded_dict.get(k)
            }
            print("Differences:", diffs)
            all_match = False

    return all_match
