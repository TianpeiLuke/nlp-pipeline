"""
Configuration utility functions for merging, saving, and loading multiple Pydantic configs.

IMPORTANT: This module is maintained for backward compatibility.
For new code, please import directly from src.config_field_manager:

    from src.config_field_manager import merge_and_save_configs, load_configs

This module provides a high-level API for configuration management, leveraging
the optimized implementation in src.config_field_manager while maintaining
backward compatibility with existing code.
"""

from typing import List, Dict, Any, Type, Set, Optional, Tuple, Union
import logging
import os
import json
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

from .config_base import BasePipelineConfig
from .config_processing_step_base import ProcessingStepConfigBase

# Import from the advanced implementation
# RECOMMENDED: Use these imports directly in your code:
#     from src.config_field_manager import merge_and_save_configs, load_configs
from src.config_field_manager import (
    merge_and_save_configs as new_merge_and_save_configs,
    load_configs as new_load_configs,
    serialize_config as new_serialize_config,
    deserialize_config as new_deserialize_config,
    ConfigClassStore,
    register_config_class
)

# Constants for the simplified categorization model
from enum import Enum, auto
class CategoryType(Enum):
    SHARED = auto()
    SPECIFIC = auto()
from src.config_field_manager.type_aware_config_serializer import (
    serialize_config as new_serialize_config,
    deserialize_config
)

# Constants required for backward compatibility
MODEL_TYPE_FIELD = "__model_type__"
MODEL_MODULE_FIELD = "__model_module__"

logger = logging.getLogger(__name__)


def serialize_config(config: BaseModel) -> Dict[str, Any]:
    """
    Serialize a single Pydantic config to a JSON‐serializable dict,
    embedding metadata including a unique 'step_name'.
    Enhanced to include default values from Pydantic model definitions.
    
    This function maintains backward compatibility while using the new implementation.
    """
    # Get the serialized dict from the new implementation
    serialized = new_serialize_config(config)
    
    # Ensure backward compatibility for step_name in metadata
    if "_metadata" not in serialized:
        # Base step name from registry
        base_step = BasePipelineConfig.get_step_name(config.__class__.__name__)
        step_name = base_step
        
        # Append distinguishing attributes
        for attr in ("job_type", "data_type", "mode"):
            if hasattr(config, attr):
                val = getattr(config, attr)
                if val is not None:
                    step_name = f"{step_name}_{val}"
        
        # Add the metadata
        serialized["_metadata"] = {
            "step_name": step_name,
            "config_type": config.__class__.__name__,
        }
    
    # Remove model type fields for backward compatibility
    if MODEL_TYPE_FIELD in serialized:
        del serialized[MODEL_TYPE_FIELD]
    if MODEL_MODULE_FIELD in serialized:
        del serialized[MODEL_MODULE_FIELD]
    
    return serialized


def verify_configs(config_list: List[BaseModel]) -> None:
    """
    Verify that the configurations are valid.
    
    Args:
        config_list: List of configurations to verify
        
    Raises:
        ValueError: If configurations are invalid (e.g., duplicate step names)
    """
    # Ensure unique step names
    step_names = set()
    for config in config_list:
        serialized = serialize_config(config)
        step_name = serialized["_metadata"]["step_name"]
        if step_name in step_names:
            raise ValueError(f"Duplicate step name: {step_name}")
        step_names.add(step_name)
    
    # Add more validation logic as needed
    # For example, ensure required fields are present
    for config in config_list:
        if not hasattr(config, 'pipeline_name'):
            raise ValueError(f"Config of type {config.__class__.__name__} missing pipeline_name")
            
    # Log validation success
    logger.info(f"Verified {len(config_list)} configurations successfully")


def merge_and_save_configs(config_list: List[BaseModel], output_file: str) -> Dict[str, Any]:
    """
    Merge and save multiple configs to JSON. Handles multiple instantiations with unique step_name.
    Better handles class hierarchy for fields like input_names that should be kept specific.
    
    This is a wrapper for the new implementation in src.config_field_manager.
    
    Simplified Field Categorization Rules:
    -------------------------------------
    1. **Field is special** → Place in `specific`
       - Special fields include those in the `SPECIAL_FIELDS_TO_KEEP_SPECIFIC` list
       - Pydantic models are considered special fields
       - Complex nested structures are considered special fields

    2. **Field appears only in one config** → Place in `specific`
       - If a field exists in only one configuration instance, it belongs in that instance's specific section

    3. **Field has different values across configs** → Place in `specific`
       - If a field has the same name but different values across multiple configs, each instance goes in specific

    4. **Field is non-static** → Place in `specific`
       - Fields identified as non-static (runtime values, input/output fields, etc.) go in specific

    5. **Field has identical value across all configs** → Place in `shared`
       - If a field has the same value across all configs and is not caught by the above rules, it belongs in shared

    6. **Default case** → Place in `specific`
       - When in doubt, place in specific to ensure proper functioning
    
    We build a simplified structure:
      - "shared": fields that appear with identical values across all configs and are static
      - "specific": fields that are unique to specific configs or have different values across configs
      
    The following categories are mutually exclusive:
      - "shared" and "specific" sections have no overlapping fields
    
    Under "metadata" → "config_types" we map each unique step_name → config class name.
    """
    # Simply delegate to the new implementation
    return new_merge_and_save_configs(config_list, output_file)


def load_configs(input_file: str, config_classes: Dict[str, Type[BaseModel]]) -> Dict[str, BaseModel]:
    """
    Load multiple Pydantic configs from JSON, reconstructing each instantiation uniquely.
    Mirrors the saving algorithm's logic for where fields should come from.
    
    This is a wrapper for the new implementation in src.config_field_manager.
    
    Config fields are loaded with the following simplified priority order:
    1. Specific values for this exact config (highest priority) 
    2. Shared values (lowest priority)
    
    This simplified approach makes it easy to understand where each field's value 
    comes from, eliminating the complexity of the nested processing hierarchy.
    """
    # Use ConfigClassStore to ensure we have all classes registered
    for _, cls in config_classes.items():
        ConfigClassStore.register(cls)
        
    # Load configs from file - this will give us a dict with only step names to config instances
    loaded_configs_dict = new_load_configs(input_file, config_classes)
    
    # For backward compatibility, we may need to process some special fields 
    # or ensure certain config objects are properly reconstructed
    result_configs = {}
    
    with open(input_file, 'r') as f:
        file_data = json.load(f)
    
    # Extract metadata for proper config reconstruction
    if "metadata" in file_data and "config_types" in file_data["metadata"]:
        config_types = file_data["metadata"]["config_types"]
        
        # Make sure all configs in the metadata are properly loaded
        for step_name, class_name in config_types.items():
            if step_name in loaded_configs_dict:
                result_configs[step_name] = loaded_configs_dict[step_name]
            elif class_name in config_classes:
                # Create an instance using the appropriate class
                logger.info(f"Creating additional config instance for {step_name} ({class_name})")
                try:
                    # Get shared data from file_data
                    shared_data = {}
                    specific_data = {}
                    
                    # Get from the correct location based on structure
                    if "configuration" in file_data:
                        config_data = file_data["configuration"]
                        if "shared" in config_data:
                            shared_data = config_data["shared"]
                        if "specific" in config_data and step_name in config_data["specific"]:
                            specific_data = config_data["specific"][step_name]
                    
                    # Combine data with specific overriding shared
                    combined_data = {**shared_data, **specific_data}
                    
                    # Create the config instance
                    config_class = config_classes[class_name]
                    result_configs[step_name] = config_class(**combined_data)
                except Exception as e:
                    logger.warning(f"Failed to create config for {step_name}: {str(e)}")
    else:
        # Just use the loaded configs as is
        result_configs = loaded_configs_dict
    
    return result_configs


def build_complete_config_classes() -> Dict[str, Type[BaseModel]]:
    """
    Build a complete dictionary of all relevant config classes using
    both step and hyperparameter registries as the single source of truth.
    
    IMPORTANT: Consider using ConfigClassStore to register your config classes instead:
    
        from src.config_field_manager import ConfigClassStore, register_config_class
        
        # Register a class
        @ConfigClassStore.register
        class MyConfig:
            ...
            
        # Or use the register_config_class alias
        @register_config_class
        class AnotherConfig:
            ...
    
    Returns:
        Dictionary mapping class names to class types
    """
    from src.pipeline_registry import STEP_NAMES, HYPERPARAMETER_REGISTRY
    
    # Initialize an empty dictionary to store the classes
    config_classes = {}
    
    # Import step config classes from registry
    for step_name, info in STEP_NAMES.items():
        class_name = info["config_class"]
        try:
            # Most config classes follow a naming pattern of config_<step_name_lowercase>.py
            module_name = f"config_{step_name.lower()}"
            # Try to import from pipeline_steps package
            try:
                # First try as a relative import within the package
                module = __import__(f".{module_name}", globals(), locals(), [class_name], 1)
                if hasattr(module, class_name):
                    config_classes[class_name] = getattr(module, class_name)
                    logger.debug(f"Registered {class_name} from relative import")
                    continue
            except (ImportError, AttributeError):
                # Fall back to an absolute import
                try:
                    module = __import__(f"src.pipeline_steps.{module_name}", fromlist=[class_name])
                    if hasattr(module, class_name):
                        config_classes[class_name] = getattr(module, class_name)
                        logger.debug(f"Registered {class_name} from absolute import")
                        continue
                except (ImportError, AttributeError):
                    pass
            
            # If still not found, import base config classes directly
            if class_name in ["BasePipelineConfig", "ProcessingStepConfigBase"]:
                module_name = class_name.lower()
                try:
                    module = __import__(f".{module_name}", globals(), locals(), [class_name], 1)
                    if hasattr(module, class_name):
                        config_classes[class_name] = getattr(module, class_name)
                        logger.debug(f"Registered {class_name} from base config")
                except (ImportError, AttributeError):
                    logger.debug(f"Could not import {class_name} from any location")
        except Exception as e:
            logger.debug(f"Error importing {class_name}: {str(e)}")
    
    # Import hyperparameter classes from registry
    for class_name, info in HYPERPARAMETER_REGISTRY.items():
        try:
            module_path = info["module_path"]
            module_parts = module_path.split(".")
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                config_classes[class_name] = getattr(module, class_name)
                logger.debug(f"Registered hyperparameter class {class_name}")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not import {class_name}: {str(e)}")
    
    # Basic fallback for core classes in case the dynamic imports failed
    try:
        from .config_base import BasePipelineConfig
        config_classes.setdefault('BasePipelineConfig', BasePipelineConfig)
        
        from .config_processing_step_base import ProcessingStepConfigBase
        config_classes.setdefault('ProcessingStepConfigBase', ProcessingStepConfigBase)
        
        from .hyperparameters_base import ModelHyperparameters
        config_classes.setdefault('ModelHyperparameters', ModelHyperparameters)
    except ImportError as e:
        logger.warning(f"Could not import core classes: {str(e)}")
    
    # Register all classes with the ConfigClassStore
    for class_name, cls in config_classes.items():
        ConfigClassStore.register(cls)
        logger.debug(f"Registered with ConfigClassStore: {class_name}")
        
    return config_classes
