from typing import List, Dict, Any, Type, Union
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

from .config_base import BasePipelineConfig


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """Serialize a single config to dictionary, handling special types"""
    config_dict = config.model_dump() if hasattr(config, 'model_dump') else config.dict()
    
    # Add step name to metadata
    config_dict['_step_name'] = BasePipelineConfig.get_step_name(config.__class__.__name__)
    
    # Handle special types
    for key, value in config_dict.items():
        if isinstance(value, datetime):
            config_dict[key] = value.isoformat()
        elif isinstance(value, Enum):
            config_dict[key] = value.value
        elif isinstance(value, Path):
            config_dict[key] = str(value)
    
    return config_dict


def merge_and_save_configs(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """Merge multiple configs and save to JSON"""
    merged_config: Dict[str, Any] = {}
    field_sources: Dict[str, List[str]] = {}
    step_names: Dict[str, str] = {}

    for config in config_list:
        config_dict = serialize_config(config)
        config_class_name = config.__class__.__name__
        step_name = BasePipelineConfig.get_step_name(config_class_name)
        step_names[config_class_name] = step_name
        
        for key, value in config_dict.items():
            if key != '_step_name':  # Skip the metadata field
                merged_config[key] = value
                if key not in field_sources:
                    field_sources[key] = []
                field_sources[key].append(step_name)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "config_sources": field_sources,
        "step_names": step_names
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
    step_names = metadata["step_names"]
    config_sources = metadata["config_sources"]

    # Create reverse mapping from step names to config class names
    reverse_mapping = {v: k for k, v in step_names.items()}

    configs = {}
    for step_name, config_class_name in reverse_mapping.items():
        if config_class_name not in config_classes:
            raise ValueError(f"Unknown config type: {config_class_name}")

        config_class = config_classes[config_class_name]
        
        # Get fields for this config type
        fields = {
            k: v for k, v in config_data.items()
            if step_name in config_sources.get(k, [])
        }

        try:
            configs[step_name] = config_class(**fields)
        except Exception as e:
            print(f"Error creating {step_name} config: {str(e)}")
            raise

    return configs


def verify_configs(
    original_configs: List[BaseModel],
    loaded_configs: Dict[str, BaseModel]
) -> bool:
    """
    Verify that loaded configs match the originals.
    Returns True if all configs match.
    """
    all_match = True
    
    for config_name, loaded_config in loaded_configs.items():
        print(f"\nVerifying {config_name}:")
        
        # Find original config
        original = next(
            (c for c in original_configs if c.__class__.__name__ == config_name),
            None
        )
        
        if original is None:
            print(f"⚠ No original config found for {config_name}")
            all_match = False
            continue

        # Compare serialized versions
        original_dict = serialize_config(original)
        loaded_dict = serialize_config(loaded_config)
        
        if original_dict == loaded_dict:
            print("✓ Configuration matches original")
        else:
            print("⚠ Configuration differs from original")
            # Find differences
            diffs = {
                k: (original_dict.get(k), loaded_dict.get(k))
                for k in set(original_dict) | set(loaded_dict)
                if original_dict.get(k) != loaded_dict.get(k)
            }
            print("Differences:", diffs)
            all_match = False

    return all_match