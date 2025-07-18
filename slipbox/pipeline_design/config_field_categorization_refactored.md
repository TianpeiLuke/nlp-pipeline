# Simplified Config Field Categorization System

## Overview

The Simplified Config Field Categorization System provides a **streamlined architecture for managing configuration fields** across multiple configurations. It improves upon the previous implementation by introducing a flattened structure, clearer categorization rules, and enhanced usability while maintaining type-safety and modularity.

## Core Purpose

The simplified system provides a **maintainable field categorization framework** that enables:

1. **Simpler Mental Model** - A flattened structure that's easier to understand and reason about
2. **Clear Rules** - Explicit, easy-to-understand rules for field categorization
3. **Modular Architecture** - Separation of concerns with dedicated classes for each responsibility
4. **Type Safety** - Enhanced type-aware serialization and deserialization
5. **Robust Error Handling** - Comprehensive error checking and reporting
6. **Improved Testability** - Isolated components that can be independently tested

## Alignment with Core Architectural Principles

The refactored design directly implements our core architectural principles:

### Single Source of Truth

- **Configuration Registry**: Centralized registry for all configuration classes eliminates redundant class lookups and references
- **Categorization Rules**: Rules defined once in `ConfigFieldCategorizer` provide a single authoritative source for categorization decisions
- **Field Information**: Comprehensive field information collected once and used throughout the system
- **Special Field Handling**: Special fields defined in one location (`SPECIAL_FIELDS_TO_KEEP_SPECIFIC`) for consistency

This principle ensures all components refer to the same canonical source for configuration classes and categorization decisions.

### Declarative Over Imperative

- **Rule-Based Categorization**: Fields are categorized based on declarative rules rather than imperative logic
- **Configuration-Driven**: The system works with the configuration's inherent structure rather than forcing a specific format
- **Explicit Categories**: Categories are explicitly defined and serve as a contract between components
- **Separation of Definition and Execution**: Field categorization rules are separate from their execution

By defining what makes a field belong to each category rather than procedural logic for categorization, we create a more maintainable and understandable system.

### Type-Safe Specifications

- **CategoryType Enum**: Strong typing for categories prevents incorrect category assignments
- **Type-Aware Serialization**: Maintains type information during serialization for correct reconstruction
- **Model Classes**: Uses Pydantic's strong typing to validate field values
- **Explicit Type Metadata**: Serialized objects include type information for proper deserialization

This principle helps prevent errors by catching type issues at definition time rather than runtime.

### Explicit Over Implicit

- **Explicit Categorization Rules**: Clear rules with defined precedence make categorization decisions transparent
- **Named Categories**: Categories have meaningful names that express their purpose
- **Logging of Decisions**: Category assignments and special cases are explicitly logged
- **Clear Class Responsibilities**: Each class has an explicitly defined role with clear interfaces

Making categorization decisions explicit improves maintainability and helps developers understand the system's behavior.

## Key Components

### 1. ConfigFieldCategorizer

A dedicated class responsible for applying categorization rules:

```python
class ConfigFieldCategorizer:
    """
    Responsible for categorizing configuration fields based on their characteristics.
    
    Analyzes field values and metadata across configs to determine proper placement.
    Uses explicit rules with clear precedence for categorization decisions.
    """
    
    def __init__(self, config_list):
        self.config_list = config_list
        self.processing_configs = [c for c in config_list if isinstance(c, ProcessingStepConfigBase)]
        self.non_processing_configs = [c for c in config_list if not isinstance(c, ProcessingStepConfigBase)]
        self.field_info = self._collect_field_info()
        self.categorization = self._categorize_fields()
        
    def _collect_field_info(self):
        """
        Collect comprehensive information about all fields across configs.
        
        Returns:
            dict: Field information including values, sources, types, etc.
        """
        field_info = {
            'values': defaultdict(set),            # field_name -> set of values (as JSON strings)
            'sources': defaultdict(list),          # field_name -> list of step names
            'processing_sources': defaultdict(list), # field_name -> list of processing step names
            'non_processing_sources': defaultdict(list), # field_name -> list of non-processing step names
            'is_static': defaultdict(bool),        # field_name -> bool (is this field likely static)
            'is_special': defaultdict(bool),       # field_name -> bool (is this a special field)
            'is_cross_type': defaultdict(bool),    # field_name -> bool (appears in both processing/non-processing)
            'raw_values': defaultdict(dict)        # field_name -> {step_name: actual value}
        }
        
        # Collect information from all configs
        for config in self.config_list:
            serialized = serialize_config(config)
            step_name = serialized["_metadata"]["step_name"]
            
            # Process each field
            for field_name, value in serialized.items():
                if field_name == "_metadata":
                    continue
                    
                # Track raw value
                field_info['raw_values'][field_name][step_name] = value
                
                # Track serialized value for comparison
                try:
                    value_str = json.dumps(value, sort_keys=True)
                    field_info['values'][field_name].add(value_str)
                except (TypeError, ValueError):
                    # If not JSON serializable, use object ID as placeholder
                    field_info['values'][field_name].add(f"__non_serializable_{id(value)}__")
                
                # Track sources
                field_info['sources'][field_name].append(step_name)
                
                # Track processing/non-processing sources
                if isinstance(config, ProcessingStepConfigBase):
                    field_info['processing_sources'][field_name].append(step_name)
                else:
                    field_info['non_processing_sources'][field_name].append(step_name)
                
                # Determine if cross-type
                is_processing = bool(field_info['processing_sources'][field_name])
                is_non_processing = bool(field_info['non_processing_sources'][field_name])
                field_info['is_cross_type'][field_name] = is_processing and is_non_processing
                
                # Check if special
                field_info['is_special'][field_name] = self._is_special_field(field_name, value, config)
                
                # Check if static
                field_info['is_static'][field_name] = self._is_likely_static(field_name, value)
                
        return field_info
    
    def _is_special_field(self, field_name, value, config):
        """
        Determine if a field should be treated as special.
        
        Special fields are always kept in specific sections.
        
        Args:
            field_name: Name of the field
            value: Value of the field
            config: The config containing this field
            
        Returns:
            bool: True if the field is special
        """
        # Check against known special fields
        if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            return True
            
        # Check if it's a Pydantic model
        if isinstance(value, BaseModel):
            return True
            
        return False
    
    def _is_likely_static(self, field_name, value):
        """
        Determine if a field is likely static based on name and value.
        
        Static fields are those that don't change at runtime.
        
        Args:
            field_name: Name of the field
            value: Value of the field
            
        Returns:
            bool: True if the field is likely static
        """
        # Special fields are never static
        if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
            return False
            
        # Pydantic models are never static
        if isinstance(value, BaseModel):
            return False
        
        # Check name patterns that suggest non-static fields
        non_static_patterns = {"_names", "input_", "output_", "_specific", "_count"}
        if any(pattern in field_name for pattern in non_static_patterns):
            return False
            
        # Check complex values
        if isinstance(value, dict) and len(value) > 3:
            return False
        if isinstance(value, list) and len(value) > 5:
            return False
            
        # Default to static
        return True
        
    def _categorize_fields(self):
        """
        Apply categorization rules to all fields.
        
        Returns:
            dict: Field categorization results
        """
        # Following the Declarative Over Imperative principle with clear structure
        categorization = {
            'shared': {},
            'processing': {
                'processing_shared': {},
                'processing_specific': defaultdict(dict)
            },
            'specific': defaultdict(dict)
        }
        
        # Apply categorization rules to each field
        for field_name in self.field_info['sources']:
            # Explicit categorization following Explicit Over Implicit principle
            category = self._categorize_field(field_name)
            
            # Place field in the appropriate category
            self._place_field(field_name, category, categorization)
            
        return categorization
    
    def _categorize_field(self, field_name):
        """
        Determine the category for a field based on explicit rules.
        
        Args:
            field_name: Name of the field to categorize
            
        Returns:
            str: Category name ('shared', 'processing_shared', 'processing_specific', 'specific')
        """
        info = self.field_info
        
        # Rule 1: Special fields always go to their specific sections
        if info['is_special'][field_name]:
            if field_name in info['processing_sources']:
                return 'processing_specific'
            else:
                return 'specific'
                
        # Rule 2: Fields that only appear in one config are specific
        if len(info['sources'][field_name]) <= 1:
            if field_name in info['processing_sources']:
                return 'processing_specific'
            else:
                return 'specific'
                
        # Rule 3: Fields with different values across configs are specific
        if len(info['values'][field_name]) > 1:
            if field_name in info['processing_sources']:
                return 'processing_specific'
            else:
                return 'specific'
                
        # Rule 4: Non-static fields are specific
        if not info['is_static'][field_name]:
            if field_name in info['processing_sources']:
                return 'processing_specific'
            else:
                return 'specific'
                
        # Rule 5: Cross-type fields with identical values go to shared
        if info['is_cross_type'][field_name] and len(info['values'][field_name]) == 1:
            # If in ALL configs, can be in shared
            if len(info['sources'][field_name]) == len(self.config_list):
                return 'shared'
            else:
                # Not in all configs, must be specific
                return 'specific' if not info['processing_sources'][field_name] else 'processing_specific'
        
        # Rule 6: Processing-only fields with identical values
        if field_name in info['processing_sources'] and not info['is_cross_type'][field_name]:
            # If in ALL processing configs, can be in processing_shared
            if len(info['processing_sources'][field_name]) == len(self.processing_configs):
                return 'processing_shared'
            else:
                return 'processing_specific'
                
        # Rule 7: Non-processing fields with identical values go to shared
        if field_name in info['non_processing_sources'] and not info['is_cross_type'][field_name]:
            return 'shared'
            
        # Default case: if we can't determine clearly, be safe and make it specific
        if field_name in info['processing_sources']:
            return 'processing_specific'
        else:
            return 'specific'
    
    def _place_field(self, field_name, category, categorization):
        """
        Place a field into the appropriate category in the categorization structure.
        
        Args:
            field_name: Name of the field
            category: Category to place the field in
            categorization: Categorization structure to update
        """
        info = self.field_info
        
        # Handle each category
        if category == 'shared':
            # Use the common value for all configs
            value_str = next(iter(info['values'][field_name]))
            categorization['shared'][field_name] = json.loads(value_str)
            
        elif category == 'processing_shared':
            # Use the common value for all processing configs
            value_str = next(iter(info['values'][field_name]))
            categorization['processing']['processing_shared'][field_name] = json.loads(value_str)
            
        elif category == 'processing_specific':
            # Add to each processing config that has this field
            for config in self.processing_configs:
                if hasattr(config, field_name):
                    step_name = serialize_config(config)["_metadata"]["step_name"]
                    value = getattr(config, field_name)
                    categorization['processing']['processing_specific'][step_name][field_name] = value
                    
        elif category == 'specific':
            # Add to each non-processing config that has this field
            for config in self.non_processing_configs:
                if hasattr(config, field_name):
                    step_name = serialize_config(config)["_metadata"]["step_name"]
                    value = getattr(config, field_name)
                    categorization['specific'][step_name][field_name] = value
    
    def get_category_for_field(self, field_name, config=None):
        """
        Get the category for a specific field, optionally in a specific config.
        
        Args:
            field_name: Name of the field
            config: Optional config instance
            
        Returns:
            str: Category name or None if field not found
        """
        if field_name not in self.field_info['sources']:
            return None
            
        if config is None:
            # Return general category
            return self._categorize_field(field_name)
        else:
            # Check if this config has this field
            if not hasattr(config, field_name):
                return None
                
            # Get category for this specific instance
            is_processing = isinstance(config, ProcessingStepConfigBase)
            category = self._categorize_field(field_name)
            
            # Adjust category based on config type
            if category == 'shared':
                return 'shared'
            elif category == 'processing_shared' and is_processing:
                return 'processing_shared'
            elif is_processing:
                return 'processing_specific'
            else:
                return 'specific'
```

### 2. Step Name Generation with Job Type Variants

A critical aspect of the configuration system is proper generation of step names, especially for configurations that represent job type variants (training, calibration, validation, testing). The legacy implementation ensured that attributes like `job_type`, `data_type`, and `mode` were appended to step names:

```python
# From legacy implementation
def serialize_config(config: BaseModel) -> Dict[str, Any]:
    # ...
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

This step name generation is essential for the job type variant solution, which relies on distinct step names for different job types. The refactored system preserves this crucial functionality in the serializer:

```python
# In TypeAwareConfigSerializer
def _generate_step_name(self, config: BaseModel) -> str:
    """
    Generate a step name for a config, including job type and other distinguishing attributes.
    
    Args:
        config: The configuration object
        
    Returns:
        str: Generated step name
    """
    # Base step name from registry
    base_step = BasePipelineConfig.get_step_name(config.__class__.__name__)
    step_name = base_step
    
    # Append distinguishing attributes - essential for job type variants
    for attr in ("job_type", "data_type", "mode"):
        if hasattr(config, attr):
            val = getattr(config, attr)
            if val is not None:
                step_name = f"{step_name}_{val}"
                
    return step_name
```

This ensures that distinct step names are generated for different job type variants (e.g., "CradleDataLoading_training", "CradleDataLoading_calibration"), which is essential for:

1. **Proper dependency resolution** between job type variants
2. **Pipeline variant creation** (training-only, evaluation-only, end-to-end)
3. **Semantic keyword matching** in the step specification system

### 3. TypeAwareSerializer

A dedicated class for robust type-aware serialization that implements the Type-Safe Specifications principle:

```python
class TypeAwareSerializer:
    """
    Handles serialization and deserialization of complex types with type information.
    
    Maintains type information during serialization and uses it for correct
    instantiation during deserialization.
    """
    
    # Constants for metadata fields - following Single Source of Truth principle
    MODEL_TYPE_FIELD = "__model_type__"
    MODEL_MODULE_FIELD = "__model_module__"
    
    def __init__(self, config_classes=None):
        """
        Initialize with optional config classes.
        
        Args:
            config_classes: Optional dictionary mapping class names to class objects
        """
        self.config_classes = config_classes or build_complete_config_classes()
        self.logger = logging.getLogger(__name__)
        
    def serialize(self, val):
        """
        Serialize a value with type information when needed.
        
        Args:
            val: The value to serialize
            
        Returns:
            Serialized value suitable for JSON
        """
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
                
                # Create serialized dict with type metadata - implementing Type-Safe Specifications
                result = {
                    self.MODEL_TYPE_FIELD: cls_name,
                    self.MODEL_MODULE_FIELD: module_name,
                    **{k: self.serialize(v) for k, v in val.model_dump().items()}
                }
                return result
            except Exception as e:
                self.logger.warning(f"Error serializing {val.__class__.__name__}: {str(e)}")
                return f"<Serialization error: {str(e)}>"
        if isinstance(val, dict):
            return {k: self.serialize(v) for k, v in val.items()}
        if isinstance(val, list):
            return [self.serialize(v) for v in val]
        return val
        
    def deserialize(self, field_data, field_name=None, expected_type=None):
        """
        Deserialize data with proper type handling.
        
        Args:
            field_data: The serialized data
            field_name: Optional name of the field (for logging)
            expected_type: Optional expected type
            
        Returns:
            Deserialized value
        """
        # Skip if not a dict or no type info needed
        if not isinstance(field_data, dict):
            return field_data
            
        # Check for type metadata - implementing Explicit Over Implicit
        type_name = field_data.get(self.MODEL_TYPE_FIELD)
        module_name = field_data.get(self.MODEL_MODULE_FIELD)
        
        if not type_name:
            # No type information, use the expected_type if applicable
            if expected_type and isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                return self._deserialize_model(field_data, expected_type)
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
            return field_data
            
        return self._deserialize_model(field_data, actual_class)
    
    def _deserialize_model(self, field_data, model_class):
        """
        Deserialize a model instance.
        
        Args:
            field_data: Serialized model data
            model_class: Class to instantiate
            
        Returns:
            Model instance
        """
        # Remove metadata fields
        filtered_data = {k: v for k, v in field_data.items() 
                       if k not in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD)}
                       
        # Recursively deserialize nested models
        for k, v in list(filtered_data.items()):
            if isinstance(v, dict) and self.MODEL_TYPE_FIELD in v:
                # Get nested field type if available
                nested_type = None
                if hasattr(model_class, 'model_fields') and k in model_class.model_fields:
                    nested_type = model_class.model_fields[k].annotation
                filtered_data[k] = self.deserialize(v, k, nested_type)
        
        try:
            return model_class(**filtered_data)
        except Exception as e:
            self.logger.error(f"Failed to instantiate {model_class.__name__}: {str(e)}")
            # Return as plain dict if instantiation fails
            return filtered_data
            
    def _get_class_by_name(self, class_name, module_name=None):
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
```

### 3. ConfigMerger

A class to handle the merging process, implementing the Separation of Concerns design principle:

```python
class ConfigMerger:
    """
    Handles the merging of multiple configs based on field categorization.
    
    Coordinates the categorization, serialization, and validation processes
    to produce a properly merged configuration.
    """
    
    def __init__(self, config_list, categorizer=None, serializer=None):
        """
        Initialize with configs and optional components.
        
        Args:
            config_list: List of config objects to merge
            categorizer: Optional ConfigFieldCategorizer instance
            serializer: Optional TypeAwareSerializer instance
        """
        self.config_list = config_list
        # Dependency injection following Separation of Concerns
        self.categorizer = categorizer or ConfigFieldCategorizer(config_list)
        self.serializer = serializer or TypeAwareSerializer()
        self.logger = logging.getLogger(__name__)
        
    def merge(self):
        """
        Merge configs based on field categorization.
        
        Returns:
            dict: Merged configuration
        """
        merged = self.categorizer.categorization
        
        # Serialize all values
        self._serialize_all_values(merged)
        
        # Handle special fields to ensure they're in the right place
        self._handle_special_fields(merged)
        
        # Ensure mutual exclusivity
        self._ensure_mutual_exclusivity(merged)
        
        return merged
        
    def _serialize_all_values(self, merged):
        """
        Serialize all values in the merged config.
        
        Args:
            merged: The merged configuration to update
        """
        # Serialize shared fields
        for k, v in list(merged['shared'].items()):
            merged['shared'][k] = self.serializer.serialize(v)
            
        # Serialize processing_shared fields
        for k, v in list(merged['processing']['processing_shared'].items()):
            merged['processing']['processing_shared'][k] = self.serializer.serialize(v)
            
        # Serialize processing_specific fields
        for step, fields in merged['processing']['processing_specific'].items():
            for k, v in list(fields.items()):
                merged['processing']['processing_specific'][step][k] = self.serializer.serialize(v)
                
        # Serialize specific fields
        for step, fields in merged['specific'].items():
            for k, v in list(fields.items()):
                merged['specific'][step][k] = self.serializer.serialize(v)
    
    def _handle_special_fields(self, merged):
        """
        Ensure special fields are in their appropriate sections.
        
        Args:
            merged: The merged configuration to update
        """
        # Handle special fields in shared - implementing Single Source of Truth
        for field_name in list(merged['shared'].keys()):
            if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
                self.logger.info(f"Moving special field '{field_name}' from shared")
                shared_value = merged['shared'].pop(field_name)
                
                # Add to specific configs that have this field
                for config in self.config_list:
                    if hasattr(config, field_name):
                        step = serialize_config(config)["_metadata"]["step_name"]
                        value = getattr(config, field_name)
                        serialized_value = self.serializer.serialize(value)
                        
                        if isinstance(config, ProcessingStepConfigBase):
                            if step not in merged['processing']['processing_specific']:
                                merged['processing']['processing_specific'][step] = {}
                            merged['processing']['processing_specific'][step][field_name] = serialized_value
                        else:
                            merged['specific'][step][field_name] = serialized_value
                            
        # Handle special fields in processing_shared
        for field_name in list(merged['processing']['processing_shared'].keys()):
            if field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
                self.logger.info(f"Moving special field '{field_name}' from processing_shared")
                shared_value = merged['processing']['processing_shared'].pop(field_name)
                
                # Add to processing_specific configs that have this field
                for config in self.config_list:
                    if isinstance(config, ProcessingStepConfigBase) and hasattr(config, field_name):
                        step = serialize_config(config)["_metadata"]["step_name"]
                        value = getattr(config, field_name)
                        serialized_value = self.serializer.serialize(value)
                        
                        if step not in merged['processing']['processing_specific']:
                            merged['processing']['processing_specific'][step] = {}
                        merged['processing']['processing_specific'][step][field_name] = serialized_value
                        
        # Final verification for special fields - implementing Build-Time Validation
        for config in self.config_list:
            step = serialize_config(config)["_metadata"]["step_name"]
            
            for field_name in SPECIAL_FIELDS_TO_KEEP_SPECIFIC:
                if hasattr(config, field_name):
                    if isinstance(config, ProcessingStepConfigBase):
                        # Check if field exists in processing_specific
                        if (step not in merged['processing']['processing_specific'] or
                           field_name not in merged['processing']['processing_specific'][step]):
                            # Add the field
                            value = getattr(config, field_name)
                            serialized_value = self.serializer.serialize(value)
                            
                            if step not in merged['processing']['processing_specific']:
                                merged['processing']['processing_specific'][step] = {}
                                
                            merged['processing']['processing_specific'][step][field_name] = serialized_value
                    else:
                        # Check if field exists in specific
                        if field_name not in merged['specific'].get(step, {}):
                            # Add the field
                            value = getattr(config, field_name)
                            serialized_value = self.serializer.serialize(value)
                            
                            if step not in merged['specific']:
                                merged['specific'][step] = {}
                                
                            merged['specific'][step][field_name] = serialized_value
    
    def _ensure_mutual_exclusivity(self, merged):
        """
        Ensure mutual exclusivity between shared/specific sections.
        
        Args:
            merged: The merged configuration to update
        """
        # Check shared vs specific - implementing Build-Time Validation
        shared_fields = set(merged['shared'].keys())
        for step, fields in merged['specific'].items():
            overlap = shared_fields.intersection(set(fields.keys()))
            if overlap:
                self.logger.warning(f"Found fields {overlap} in both 'shared' and 'specific' for step {step}")
                for field in overlap:
                    merged['specific'][step].pop(field)
        
        # Check processing_shared vs processing_specific
        proc_shared_fields = set(merged['processing']['processing_shared'].keys())
        for step, fields in merged['processing']['processing_specific'].items():
            overlap = proc_shared_fields.intersection(set(fields.keys()))
            if overlap:
                self.logger.warning(f"Found fields {overlap} in both 'processing_shared' and 'processing_specific'")
                for field in overlap:
                    merged['processing']['processing_specific'][step].pop(field)
    
    def save(self, output_file):
        """
        Save merged config to a file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            dict: The merged configuration
        """
        merged = self.merge()
        
        # Create metadata - implementing Explicit Over Implicit
        metadata = {
            'created_at': datetime.now().isoformat(),
            'config_types': {
                serialize_config(c)['_metadata']['step_name']: c.__class__.__name__
                for c in self.config_list
            }
        }
        
        output = {'metadata': metadata, 'configuration': merged}
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2, sort_keys=True)
            self.logger.info(f"Successfully wrote config to {output_file}")
        except Exception as e:
            self.logger.error(f"Error writing JSON: {str(e)}")
            
        return merged
```

### 4. ConfigRegistry

A registry for config classes that implements the Single Source of Truth principle:

```python
class ConfigRegistry:
    """
    Registry of configuration classes for serialization and deserialization.
    
    Maintains a centralized registry of config classes that can be easily extended.
    """
    
    # Single registry instance - implementing Single Source of Truth
    _registry = {}
    
    @classmethod
    def register(cls, config_class):
        """
        Register a config class.
        
        Can be used as a decorator:
        
        @ConfigRegistry.register
        class MyConfig(BasePipelineConfig):
            ...
        
        Args:
            config_class: The class to register
            
        Returns:
            The registered class (for decorator usage)
        """
        cls._registry[config_class.__name__] = config_class
        return config_class
        
    @classmethod
    def get_class(cls, class_name):
        """
        Get a registered class by name.
        
        Args:
            class_name: Name of the class
            
        Returns:
            The class or None if not found
        """
        return cls._registry.get(class_name)
        
    @classmethod
    def get_all_classes(cls):
        """
        Get all registered classes.
        
        Returns:
            dict: Mapping of class names to classes
        """
        return cls._registry.copy()
        
    @classmethod
    def register_many(cls, *config_classes):
        """
        Register multiple config classes at once.
        
        Args:
            *config_classes: Classes to register
        """
        for config_class in config_classes:
            cls.register(config_class)
```

## Field Sources Tracking

The refactored system includes a critical enhancement for field source tracking:

```python
def get_field_sources(config_list: List[BaseModel]) -> Dict[str, Dict[str, List[str]]]:
    """
    Extract field sources from config list.
    
    Returns a dictionary with three categories:
    - 'all': All fields and their source configs
    - 'processing': Fields from processing configs
    - 'specific': Fields from non-processing configs
    
    This is used for backward compatibility with the legacy field categorization.
    
    Args:
        config_list: List of configuration objects to analyze
        
    Returns:
        Dictionary of field sources by category
    """
```

This function maintains backward compatibility with the legacy field tracking system, providing valuable information about which configs contribute to each field. When configurations are merged and saved, the 'all' category of field sources is included in the metadata section:

```json
{
  "metadata": {
    "created_at": "timestamp",
    "config_types": {
      "StepName1": "ConfigClass1",
      "StepName2": "ConfigClass2"
    },
    "field_sources": {
      "field1": ["StepName1", "StepName2"],
      "field2": ["StepName1"],
      "field3": ["StepName2"]
    }
  },
  "configuration": {
    "shared": { "shared fields across all configs" },
    "specific": {
      "StepName1": { "step-specific fields" },
      "StepName2": { "step-specific fields" }
    }
  }
}
```

This allows for:
1. **Traceability**: Identify which configs contribute to each field
2. **Conflict Resolution**: Understand when multiple configs provide the same field
3. **Dependency Analysis**: Better understand relationships between configurations

## Public API Functions

Enhanced utility functions for the public API follow our Hybrid Design Approach:

```python
# Simple public API function - implementing Hybrid Design Approach
def merge_and_save_configs(config_list, output_file):
    """
    Merge and save multiple configs to JSON.
    
    Args:
        config_list: List of config objects
        output_file: Path to output file
        
    Returns:
        dict: Merged configuration
    """
    merger = ConfigMerger(config_list)
    return merger.save(output_file)
    
def load_configs(input_file, config_classes=None):
    """
    Load multiple configs from JSON.
    
    Args:
        input_file: Path to input file
        config_classes: Optional dictionary of config classes
        
    Returns:
        dict: Mapping of step names to config instances
    """
    config_classes = config_classes or ConfigRegistry.get_all_classes()
    if not config_classes:
        config_classes = build_complete_config_classes()
        
    # Create serializer with config classes - implementing Type-Safe Specifications
    serializer = TypeAwareSerializer(config_classes)
    
    with open(input_file) as f:
        data = json.load(f)
        
    meta = data['metadata']
    cfgs = data['configuration']
    types = meta['config_types']  # step_name -> class_name
    rebuilt = {}
    
    # First, identify processing and non-processing configs
    processing_steps = set()
    for step, cls_name in types.items():
        if cls_name not in config_classes:
            raise ValueError(f"Unknown config class: {cls_name}")
        cls = config_classes[cls_name]
        if issubclass(cls, ProcessingStepConfigBase):
            processing_steps.add(step)
    
    # Build each config with priority-based field loading - implementing Declarative Over Imperative
    for step, cls_name in types.items():
        cls = config_classes[cls_name]
        is_processing = step in processing_steps
        
        # Build field dictionary
        fields = {}
        valid_fields = set(cls.model_fields.keys())
        
        # Add shared values (lowest priority)
        for k, v in cfgs['shared'].items():
            if k in valid_fields:
                fields[k] = v
        
        # Add processing_shared values if applicable
        if is_processing:
            for k, v in cfgs['processing'].get('processing_shared', {}).items():
                if k in valid_fields:
                    fields[k] = v
                    
        # Add specific values (highest priority)
        if is_processing:
            for k, v in cfgs['processing'].get('processing_specific', {}).get(step, {}).items():
                if k in valid_fields:
                    fields[k] = v
        else:
            for k, v in cfgs['specific'].get(step, {}).items():
                if k in valid_fields:
                    fields[k] = v
        
        # Deserialize complex fields - implementing Type-Safe Specifications
        for field_name in list(fields.keys()):
            if field_name in valid_fields:
                field_value = fields[field_name]
                field_type = cls.model_fields[field_name].annotation
                
                # Handle complex field
                fields[field_name] = serializer.deserialize(field_value, field_name, field_type)
        
        # Create the instance with proper typing
        try:
            instance = cls(**fields)
            rebuilt[step] = instance
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create instance for {step}: {str(e)}")
            raise
    
    return rebuilt
```

## Benefits of the Refactored Design

The refactored design provides several significant benefits:

### 1. Enhanced Maintainability

- **Clean Separation of Concerns**: Each class has a single, well-defined responsibility
- **Explicit Rules**: Categorization rules are clearly defined and easy to understand
- **Improved Testability**: Components can be tested in isolation
- **Better Debugging**: Proper logging and explicit error handling make issues easier to diagnose

### 2. Improved Robustness

- **Type-Safe Serialization**: Preserves type information for correct reconstruction
- **Special Field Handling**: Consistent handling of special fields with verification
- **Mutual Exclusivity Enforcement**: Ensures categories don't overlap incorrectly
- **Comprehensive Error Checking**: Detailed error messages for troubleshooting

### 3. Greater Flexibility

- **Dependency Injection**: Components can be replaced or extended independently
- **Registry Pattern**: Easy registration of new config classes
- **Consistent Public API**: Maintains backward compatibility while improving internals
- **Customizable Rules**: Categorization rules can be modified without changing core logic

### 4. Better Performance

- **Efficient Field Information Collection**: Gathers all necessary information in a single pass
- **Optimized Category Placement**: Places fields correctly the first time
- **Reduced Redundancy**: Eliminates duplicate processing of fields
- **Streamlined Deserialization**: Type-aware deserialization for efficient object creation

## Job Type Variant Handling

The refactored system provides improved support for job type variants as outlined in the Job Type Variant Solution (July 4, 2025). This capability is essential for creating pipeline variants like training-only, calibration-only, and end-to-end pipelines.

### Key Features

1. **Attribute-Based Step Name Generation**
   - Step names include distinguishing attributes like `job_type`, `data_type`, and `mode`
   - For example: `CradleDataLoading_training` vs `CradleDataLoading_calibration`
   - This ensures proper identification of job type variants in step specifications

2. **Config Field Preservation**
   - The job_type field and other variant identifiers are preserved during serialization/deserialization
   - The categorization system respects job type fields when determining field placement

3. **Step Specification Integration**
   - Works seamlessly with the step specification system that relies on job type variants
   - Ensures correct dependency resolution between variants

### Example Usage

```python
# Create configs with job type variants
train_config = CradleDataLoadConfig(
    job_type="training",
    # other fields...
)

calib_config = CradleDataLoadConfig(
    job_type="calibration", 
    # other fields...
)

# When merged and saved, step names will include job type
merged = merge_and_save_configs([train_config, calib_config], "config.json")

# When loaded, job type information is preserved
loaded_configs = load_configs("config.json")
assert loaded_configs["CradleDataLoading_training"].job_type == "training"
assert loaded_configs["CradleDataLoading_calibration"].job_type == "calibration"
```

## Conclusion

The refactored Config Field Categorization system transforms a complex, error-prone process into a robust, maintainable architecture through the application of core design principles. By implementing Single Source of Truth, Declarative Over Imperative, Type-Safe Specifications, and Explicit Over Implicit principles, we've created a system that is not only more reliable but also easier to understand and extend.
The clear separation of responsibilities across dedicated classes makes the system easier to test, debug, and maintain, while preserving backward compatibility with existing code. This refactoring serves as an example of how applying our core architectural principles can significantly improve the quality and maintainability of our codebase.

Through this refactored design, we've demonstrated that following well-defined architectural principles doesn't just produce cleaner code—it creates more robust systems that are better prepared to handle evolving requirements and edge cases. The careful application of type safety, clear rule definition, and explicit interfaces ensures that configuration field categorization is no longer an error-prone process, but rather a reliable foundation for pipeline configuration management.
