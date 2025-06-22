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
    embedding metadata including a unique 'step_name'.
    Supports multiple instantiations distinguished by job_type, data_type, or mode.
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
        if len(vals) == 1 and len(sources) > 1:
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
    Expects:
      - metadata.config_types: maps each unique step_name → Pydantic class name
      - configuration.shared, configuration.processing, configuration.specific
    We rebuild each config under its step_name.
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
        # shared
        for k, v in cfgs['shared'].items():
            if k in valid:
                fields[k] = v
        # processing
        if issubclass(cls, ProcessingStepConfigBase):
            for k, v in cfgs['processing'].get(step, {}).items():
                if k in valid:
                    fields[k] = v
        # specific
        for k, v in cfgs['specific'].get(step, {}).items():
            if k in valid:
                fields[k] = v
        rebuilt[step] = cls(**fields)

    return rebuilt


def verify_configs(
    original_list: List[BaseModel],
    loaded: Dict[str, BaseModel]
) -> bool:
    """
    Compare originals to reloaded configs, allowing multiple instantiations.
    """
    ok = True
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