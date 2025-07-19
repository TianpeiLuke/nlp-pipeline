"""
Type-aware serializer for configuration objects.

This module provides a serializer that preserves type information during serialization,
allowing for proper reconstruction of objects during deserialization.
Implements the Type-Safe Specifications principle.
"""

import json
import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Type, List, Set, Union, Tuple

from pydantic import BaseModel

from src.config_field_manager.config_class_store import build_complete_config_classes
from src.config_field_manager.constants import SerializationMode, TYPE_MAPPING
from src.config_field_manager.circular_reference_tracker import CircularReferenceTracker


class TypeAwareConfigSerializer:
    """
    Handles serialization and deserialization of complex types with type information.
    
    Maintains type information during serialization and uses it for correct
    instantiation during deserialization, implementing the Type-Safe Specifications principle.
    """
    
    # Constants for metadata fields - following Single Source of Truth principle
    MODEL_TYPE_FIELD = "__model_type__"
    MODEL_MODULE_FIELD = "__model_module__"
    TYPE_INFO_FIELD = "__type_info__"
    
    def __init__(self, config_classes: Optional[Dict[str, Type]] = None, 
                 mode: SerializationMode = SerializationMode.PRESERVE_TYPES):
        """
        Initialize with optional config classes.
        
        Args:
            config_classes: Optional dictionary mapping class names to class objects
            mode: Serialization mode controlling type preservation behavior
        """
        self.config_classes = config_classes or build_complete_config_classes()
        self.mode = mode
        self.logger = logging.getLogger(__name__)
        # Use the CircularReferenceTracker for advanced circular reference detection
        self.ref_tracker = CircularReferenceTracker(max_depth=100)
        
    def serialize(self, val: Any) -> Any:
        """
        Serialize a value with type information when needed.
        
        Args:
            val: The value to serialize
            
        Returns:
            Serialized value suitable for JSON
        """
        # Handle None
        if val is None:
            return None
            
        # Handle basic types that don't need special handling
        if isinstance(val, (str, int, float, bool)):
            return val
            
        # Handle datetime
        if isinstance(val, datetime):
            if self.mode == SerializationMode.PRESERVE_TYPES:
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["datetime"],
                    "value": val.isoformat()
                }
            return val.isoformat()
            
        # Handle Enum
        if isinstance(val, Enum):
            if self.mode == SerializationMode.PRESERVE_TYPES:
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["Enum"],
                    "enum_class": f"{val.__class__.__module__}.{val.__class__.__name__}",
                    "value": val.value
                }
            return val.value
            
        # Handle Path
        if isinstance(val, Path):
            if self.mode == SerializationMode.PRESERVE_TYPES:
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["Path"],
                    "value": str(val)
                }
            return str(val)
            
        # Handle Pydantic models
        if isinstance(val, BaseModel):
            try:
                # Get class details
                cls = val.__class__
                module_name = cls.__module__
                cls_name = cls.__name__
                
                # Create serialized dict with type metadata - implementing Type-Safe Specifications
                # Always add type metadata for Pydantic models
                result = {
                    self.MODEL_TYPE_FIELD: cls_name,
                    self.MODEL_MODULE_FIELD: module_name,
                }
                # Add fields with serialized values
                for k, v in val.model_dump().items():
                    result[k] = self.serialize(v)
                return result
            except Exception as e:
                self.logger.warning(f"Error serializing {val.__class__.__name__}: {str(e)}")
                return f"<Serialization error: {str(e)}>"
                
        # Handle dict
        if isinstance(val, dict):
            if self.mode == SerializationMode.PRESERVE_TYPES and any(
                isinstance(v, (BaseModel, Enum, datetime, Path, set, frozenset, tuple))
                for v in val.values()
            ):
                # Only add type info if there are complex values
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["dict"],
                    "value": {k: self.serialize(v) for k, v in val.items()}
                }
            return {k: self.serialize(v) for k, v in val.items()}
            
        # Handle list
        if isinstance(val, list):
            if self.mode == SerializationMode.PRESERVE_TYPES and any(
                isinstance(v, (BaseModel, Enum, datetime, Path, set, frozenset, tuple))
                for v in val
            ):
                # Only add type info if there are complex values
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["list"],
                    "value": [self.serialize(v) for v in val]
                }
            return [self.serialize(v) for v in val]
            
        # Handle tuple
        if isinstance(val, tuple):
            if self.mode == SerializationMode.PRESERVE_TYPES:
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["tuple"],
                    "value": [self.serialize(v) for v in val]
                }
            return [self.serialize(v) for v in val]
            
        # Handle set
        if isinstance(val, set):
            if self.mode == SerializationMode.PRESERVE_TYPES:
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["set"],
                    "value": [self.serialize(v) for v in val]
                }
            return [self.serialize(v) for v in val]
            
        # Handle frozenset
        if isinstance(val, frozenset):
            if self.mode == SerializationMode.PRESERVE_TYPES:
                return {
                    self.TYPE_INFO_FIELD: TYPE_MAPPING["frozenset"],
                    "value": [self.serialize(v) for v in val]
                }
            return [self.serialize(v) for v in val]
            
        # Fall back to string representation for unsupported types
        try:
            return str(val)
        except Exception:
            return f"<Unserializable object of type {type(val).__name__}>"
        
    def deserialize(self, field_data: Any, field_name: Optional[str] = None, 
                    expected_type: Optional[Type] = None) -> Any:
        """
        Deserialize data with proper type handling.
        
        Args:
            field_data: The serialized data
            field_name: Optional name of the field (for logging)
            expected_type: Optional expected type
            
        Returns:
            Deserialized value
        """
        # Skip non-dict objects (can't have circular refs)
        if not isinstance(field_data, dict):
            return field_data
            
        # Use the tracker to check for circular references
        context = {'expected_type': expected_type.__name__ if expected_type else None}
        is_circular, error = self.ref_tracker.enter_object(field_data, field_name, context)
        
        if is_circular:
            # Log the detailed error message
            self.logger.warning(error)
            # Return None instead of the circular reference
            return None
            
        try:
            # Handle None, primitives
            if field_data is None or isinstance(field_data, (str, int, float, bool)):
                return field_data
                
            # Handle type-info dict - from preserved types
            if isinstance(field_data, dict) and self.TYPE_INFO_FIELD in field_data:
                type_info = field_data[self.TYPE_INFO_FIELD]
                value = field_data.get("value")
                
                # Handle each preserved type
                if type_info == TYPE_MAPPING["datetime"]:
                    return datetime.fromisoformat(value)
                    
                elif type_info == TYPE_MAPPING["Enum"]:
                    # This requires dynamic import - error prone, consider alternatives
                    enum_class_path = field_data.get("enum_class")
                    if not enum_class_path:
                        return field_data  # Can't deserialize without class info
                        
                    try:
                        module_name, class_name = enum_class_path.rsplit(".", 1)
                        module = __import__(module_name, fromlist=[class_name])
                        enum_class = getattr(module, class_name)
                        return enum_class(field_data.get("value"))
                    except (ImportError, AttributeError, ValueError) as e:
                        self.logger.warning(f"Failed to deserialize enum: {str(e)}")
                        return field_data.get("value")  # Fall back to raw value
                        
                elif type_info == TYPE_MAPPING["Path"]:
                    return Path(value)
                    
                elif type_info == TYPE_MAPPING["dict"]:
                    return {k: self.deserialize(v) for k, v in value.items()}
                    
                elif type_info in [TYPE_MAPPING["list"], TYPE_MAPPING["tuple"], 
                                 TYPE_MAPPING["set"], TYPE_MAPPING["frozenset"]]:
                    deserialized_list = [self.deserialize(v) for v in value]
                    
                    # Convert to appropriate container type
                    if type_info == TYPE_MAPPING["tuple"]:
                        return tuple(deserialized_list)
                    elif type_info == TYPE_MAPPING["set"]:
                        return set(deserialized_list)
                    elif type_info == TYPE_MAPPING["frozenset"]:
                        return frozenset(deserialized_list)
                    return deserialized_list
            
            # Handle model data - fields with model type information
            if isinstance(field_data, dict) and self.MODEL_TYPE_FIELD in field_data:
                return self._deserialize_model(field_data, expected_type)
                
            # Handle dict
            if isinstance(field_data, dict):
                return {k: self.deserialize(v) for k, v in field_data.items()}
                
            # Handle list
            if isinstance(field_data, list):
                return [self.deserialize(v) for v in field_data]
                
            # Return as is for unhandled types
            return field_data
        finally:
            # Always exit the object when done, even if an exception occurred
            self.ref_tracker.exit_object()
        
    def _deserialize_model(self, field_data: Dict[str, Any], expected_type: Optional[Type] = None) -> Any:
        """
        Deserialize a model instance.
        
        Args:
            field_data: Serialized model data
            expected_type: Optional expected model type
            
        Returns:
            Model instance or dict if instantiation fails
        """
        # Note: Circular reference detection is now handled by CircularReferenceTracker
        # in the parent deserialize method, so we don't need to check for it here
            
        # Check for type metadata - implementing Explicit Over Implicit
        type_name = field_data.get(self.MODEL_TYPE_FIELD)
        module_name = field_data.get(self.MODEL_MODULE_FIELD)
        
        if not type_name:
            # No type information, use the expected_type if applicable
            if expected_type and isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                # Remove metadata fields
                filtered_data = {k: v for k, v in field_data.items() 
                               if k not in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD)}
                               
                # Recursively deserialize nested fields
                for k, v in list(filtered_data.items()):
                    filtered_data[k] = self.deserialize(v)
                    
                try:
                    return expected_type(**filtered_data)
                except Exception as e:
                    self.logger.error(f"Failed to instantiate {expected_type.__name__}: {str(e)}")
                    return filtered_data
            return field_data
            
        # Get the actual class to use - implementing Single Source of Truth
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
            return {k: self.deserialize(v) for k, v in field_data.items() 
                   if k not in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD)}
            
        # Remove metadata fields
        filtered_data = {k: v for k, v in field_data.items() 
                       if k not in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD)}
                       
        # Recursively deserialize nested models
        for k, v in list(filtered_data.items()):
            # Get nested field type if available
            nested_type = None
            if hasattr(actual_class, 'model_fields') and k in actual_class.model_fields:
                nested_type = actual_class.model_fields[k].annotation
                
            filtered_data[k] = self.deserialize(v, k, nested_type)
        
        try:
            result = actual_class(**filtered_data)
            return result
        except Exception as e:
            self.logger.error(f"Failed to instantiate {actual_class.__name__}: {str(e)}")
            # Return as plain dict if instantiation fails
            return filtered_data
            
    def _get_class_by_name(self, class_name: str, module_name: Optional[str] = None) -> Optional[Type]:
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
        
    def generate_step_name(self, config: Any) -> str:
        """
        Generate a step name for a config, including job type and other distinguishing attributes.
        
        This implements the job type variant handling described in the July 4, 2025 solution document.
        It creates distinct step names for different job types (e.g., "CradleDataLoading_training"),
        which is essential for proper dependency resolution and pipeline variant creation.
        
        Args:
            config: The configuration object
            
        Returns:
            str: Generated step name with job type and other variants included
        """
        # First check for step_name_override - highest priority
        if hasattr(config, "step_name_override") and config.step_name_override != config.__class__.__name__:
            return config.step_name_override
            
        # Get class name
        class_name = config.__class__.__name__
        
        # Look up the step name from the registry (primary source of truth)
        try:
            from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY
            if class_name in CONFIG_STEP_REGISTRY:
                base_step = CONFIG_STEP_REGISTRY[class_name]
            else:
                # Fall back to the old behavior if not in registry
                # Import here to avoid circular imports
                from src.pipeline_steps.config_base import BasePipelineConfig
                base_step = BasePipelineConfig.get_step_name(class_name)
        except (ImportError, AttributeError):
            # If registry not available, fall back to the old behavior
            # Import here to avoid circular imports
            from src.pipeline_steps.config_base import BasePipelineConfig
            base_step = BasePipelineConfig.get_step_name(class_name)
        
        step_name = base_step
        
        # Append distinguishing attributes - essential for job type variants
        for attr in ("job_type", "data_type", "mode"):
            if hasattr(config, attr):
                val = getattr(config, attr)
                if val is not None:
                    step_name = f"{step_name}_{val}"
                    
        return step_name


# Removed duplicate _generate_step_name function - now using TypeAwareConfigSerializer.generate_step_name instead


def serialize_config(config: Any) -> Dict[str, Any]:
    """
    Serialize a single config object with default settings.

    Preserves job type variant information in the step name, ensuring proper
    dependency resolution between job type variants (training, calibration, etc.).

    Args:
        config: Configuration object to serialize

    Returns:
        dict: Serialized configuration with proper metadata including step name
    """
    serializer = TypeAwareConfigSerializer()
    result = serializer.serialize(config)

    # If serialization resulted in a non-dict, wrap it in a dictionary
    if not isinstance(result, dict):
        step_name = serializer.generate_step_name(config) if hasattr(config, "__class__") else "unknown"
        model_type = config.__class__.__name__ if hasattr(config, "__class__") else "unknown"
        model_module = config.__class__.__module__ if hasattr(config, "__class__") else "unknown"
        
        return {
            "__model_type__": model_type,
            "__model_module__": model_module,
            "_metadata": {
                "step_name": step_name,
                "config_type": model_type,
                "serialization_note": "Object could not be fully serialized"
            },
            "value": result
        }

    # Ensure metadata with proper step name is present
    if "_metadata" not in result:
        step_name = serializer.generate_step_name(config)
        result["_metadata"] = {
            "step_name": step_name,
            "config_type": config.__class__.__name__,
        }

    return result


def deserialize_config(data: Dict[str, Any], expected_type: Optional[Type] = None) -> Any:
    """
    Deserialize a single config object with default settings.
    
    Args:
        data: Serialized configuration data
        expected_type: Optional expected type
        
    Returns:
        Configuration object
    """
    serializer = TypeAwareConfigSerializer()
    return serializer.deserialize(data, expected_type=expected_type)
