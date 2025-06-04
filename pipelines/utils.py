from typing import List, Dict, Any, Type, Set
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from collections import defaultdict

from .config_base import BasePipelineConfig
from .config_processing_step_base import ProcessingStepConfigBase


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """
    Serialize a single Pydantic config to a JSON‐serializable dict,
    embedding a metadata block that includes a unique 'step_name'.

    If the config has a 'job_type' attribute, we append that to the
    standard step name so that multiple CradleDataLoadConfig instances
    won't conflict.
    """
    # 1) Dump the model to a plain dict (handles nested submodels)
    config_dict = config.model_dump() if hasattr(config, "model_dump") else config.dict()

    # 2) Compute the base step name
    base_step = BasePipelineConfig.get_step_name(config.__class__.__name__)

    # 3) If this config has 'job_type', append it to make the step name unique
    if hasattr(config, "job_type"):
        job_type_val = getattr(config, "job_type").capitalize()
        # CamelCase‐style: e.g. "CradleDataLoadingStep_training" → "CradleDataLoadingStep_training"
        step_name = f"{base_step}_{job_type_val}"
    else:
        step_name = base_step

    # 4) Inject metadata
    config_dict["_metadata"] = {
        "step_name": step_name,
        "config_type": config.__class__.__name__,
    }

    # 5) Recursively serialize any datetime, Path, Enum, nested dict, or list
    def serialize_value(value: Any) -> Any:
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

    return {key: serialize_value(val) for key, val in config_dict.items()}


def merge_and_save_configs(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """
    Merge multiple configs of potentially the same class (e.g. CradleDataLoadConfig for different job_types),
    then save them to a single JSON. Returns the "merged" structure (so you can inspect it if desired).

    We build three sections:
      - "shared": fields that appear (with identical values) in two or more configs
      - "processing": fields that belong to any ProcessingStepConfigBase subclass,
          grouped by each unique step_name
      - "specific": all other fields, grouped by each unique step_name

    Finally, under "metadata" → "config_types" we map each unique step_name → config class name.
    """
    merged_config: Dict[str, Dict[str, Any]] = {
        "shared": {},
        "processing": defaultdict(dict),
        "specific": defaultdict(dict),
    }

    # Track all encountered JSON‐stringified values for each field
    field_values: Dict[str, Set[str]] = defaultdict(set)
    # Track which step_names contributed to which field
    field_sources: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    # 1) First pass: collect field‐values and where they came from
    for config in config_list:
        config_dict = serialize_config(config)
        step_name = config_dict["_metadata"]["step_name"]

        # All valid top‐level fields for this Pydantic class
        valid_fields = set(config.__class__.model_fields.keys())

        for key, val in config_dict.items():
            if key == "_metadata" or key not in valid_fields:
                continue

            # JSON‐serialize the candidate value (stable sort_keys)
            json_text = json.dumps(val, sort_keys=True)
            field_values[key].add(json_text)
            field_sources["all"][key].append(step_name)

            # If this config is a ProcessingStepConfigBase, mark accordingly
            if isinstance(config, ProcessingStepConfigBase):
                if key in ProcessingStepConfigBase.model_fields:
                    field_sources["processing"][key].append(step_name)
                else:
                    field_sources["specific"][key].append(step_name)
            else:
                field_sources["specific"][key].append(step_name)

    # 2) Second pass: divide into shared vs. processing vs. specific
    for key, jsons in field_values.items():
        # If exactly one unique JSON value but used by multiple step_names → shared
        if len(jsons) == 1 and len(field_sources["all"][key]) > 1:
            merged_config["shared"][key] = json.loads(next(iter(jsons)))
        # Otherwise, if this key ever appeared under ProcessingStepConfigBase, put under "processing"
        elif key in field_sources["processing"]:
            for config in config_list:
                if isinstance(config, ProcessingStepConfigBase):
                    config_dict = serialize_config(config)
                    step_name = config_dict["_metadata"]["step_name"]
                    if key in config_dict and key in config.__class__.model_fields:
                        merged_config["processing"][step_name][key] = config_dict[key]
        # Finally, any leftover goes under "specific"
        else:
            for config in config_list:
                config_dict = serialize_config(config)
                step_name = config_dict["_metadata"]["step_name"]
                if key in config_dict and key in config.__class__.model_fields:
                    merged_config["specific"][step_name][key] = config_dict[key]

    # 3) metadata.section: record exact timestamp & field_sources & config_types
    metadata = {
        "created_at": datetime.now().isoformat(),
        "field_sources": field_sources,
        # Instead of class_name → step_name, we do step_name → class_name (allows duplicates of same class)
        "config_types": {
            serialize_config(cfg)["_metadata"]["step_name"]: cfg.__class__.__name__
            for cfg in config_list
        },
    }

    output_data = {"metadata": metadata, "configuration": merged_config}

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, sort_keys=True)

    return merged_config


def load_configs(input_file: str, config_classes: Dict[str, Type[BaseModel]]) -> Dict[str, BaseModel]:
    """
    Load Pydantic configs back from a JSON file produced by merge_and_save_configs.
    Expects:
      - metadata.config_types: maps each unique step_name → Pydantic class name
      - configuration.shared, configuration.processing, configuration.specific
    We rebuild each config under its step_name.
    """
    with open(input_file, "r") as f:
        data = json.load(f)

    metadata = data["metadata"]
    config_data = data["configuration"]
    config_types: Dict[str, str] = metadata["config_types"]

    rebuilt: Dict[str, BaseModel] = {}

    for step_name, class_name in config_types.items():
        if class_name not in config_classes:
            raise ValueError(f"Unknown config type: {class_name}")

        cls = config_classes[class_name]
        valid_fields = set(cls.model_fields.keys())

        # 1) start with shared fields if they belong to this class
        fields: Dict[str, Any] = {}
        for key, val in config_data["shared"].items():
            if key in valid_fields:
                fields[key] = val

        # 2) if cls is a ProcessingStepConfigBase, merge its portion
        if issubclass(cls, ProcessingStepConfigBase):
            proc_map = config_data["processing"].get(step_name, {})
            for key, val in proc_map.items():
                if key in valid_fields:
                    fields[key] = val

        # 3) finally, merge any class-specific portion
        spec_map = config_data["specific"].get(step_name, {})
        for key, val in spec_map.items():
            if key in valid_fields:
                fields[key] = val

        # Instantiate
        try:
            rebuilt[step_name] = cls(**fields)
        except Exception as e:
            print(f"Error recreating config for step_name='{step_name}': {e}")
            print("Attempted fields:", fields)
            raise

    return rebuilt


def verify_configs(
    original_configs: List[BaseModel],
    loaded_configs: Dict[str, BaseModel]
) -> bool:
    """
    Compare each original Pydantic config to its reloaded version.
    Returns True if all match; False otherwise. Prints differences for debugging.
    """
    all_match = True

    for original in original_configs:
        # Recompute the unique step_name (same logic as serialize_config)
        base_step = BasePipelineConfig.get_step_name(original.__class__.__name__)
        if hasattr(original, "job_type"):
            step_name = f"{base_step}_{getattr(original, 'job_type')}"
        else:
            step_name = base_step

        print(f"\nVerifying step_name='{step_name}':")

        if step_name not in loaded_configs:
            print(f"  ⚠ Missing reloaded config for '{step_name}'")
            all_match = False
            continue

        reloaded = loaded_configs[step_name]

        orig_serial = serialize_config(original)
        new_serial = serialize_config(reloaded)
        orig_serial.pop("_metadata", None)
        new_serial.pop("_metadata", None)

        if orig_serial == new_serial:
            print(f"  ✓ '{step_name}' matches exactly.")
        else:
            print(f"  ⚠ '{step_name}' differs:")
            diffs = {
                k: (orig_serial.get(k), new_serial.get(k))
                for k in set(orig_serial) | set(new_serial)
                if orig_serial.get(k) != new_serial.get(k)
            }
            print("    Differences:", diffs)
            all_match = False

    return all_match
