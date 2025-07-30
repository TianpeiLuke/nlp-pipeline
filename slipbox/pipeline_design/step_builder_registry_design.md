# Step Builder Registry Design

**Version**: 1.0  
**Date**: July 30, 2025  
**Author**: MODS Development Team

## Overview

The Step Builder Registry serves as a centralized lookup system that maps pipeline step names to their corresponding builder classes. It plays a crucial role in the pipeline system by enabling automatic resolution of step builders during pipeline construction, establishing a single source of truth for step naming, and providing a standardized way to expand the system with new step types.

## Design Goals

1. **Single Source of Truth**: Maintain consistency with the step names registry
2. **Extensibility**: Make it easy to add new step builders as the system evolves
3. **Auto-Discovery**: Reduce manual maintenance by automatically discovering step builders
4. **Backward Compatibility**: Support legacy step type names for existing code
5. **Validation**: Provide tools to validate the registry and detect inconsistencies

## Architecture

### Core Components

1. **BUILDER_REGISTRY**: Central dictionary mapping canonical step types to builder classes
2. **LEGACY_ALIASES**: Dictionary mapping legacy step type names to canonical names
3. **StepBuilderRegistry**: Class that manages the builder registry and provides lookup methods
4. **Discovery Mechanism**: System for automatically finding and registering step builders
5. **Registration Decorator**: Decorator to auto-register step builder classes

### Integration with Step Names Registry

The Step Builder Registry is closely integrated with the Step Names Registry (`step_names.py`), which serves as the single source of truth for step naming across the system. This integration ensures that:

1. Step builders use canonical step names from the central registry
2. Legacy step names are properly mapped to canonical names
3. Step configuration classes are correctly associated with step builders

### Registration Flow

```mermaid
graph TD
    A[Add New Step Builder] --> B{Use Auto-Registration?}
    B -->|Yes| C[Apply @register_builder Decorator]
    B -->|No| D[Auto-Discovery at Runtime]
    C --> E[Builder Registered in Registry]
    D --> E
    E --> F[Lookup During Pipeline Construction]
```

## Implementation Details

### StepBuilderRegistry Class

The `StepBuilderRegistry` class maintains the registry and provides methods for lookup and validation:

```python
class StepBuilderRegistry:
    """
    Centralized registry mapping step types to builder classes.
    
    This registry maintains the mapping between step types and their
    corresponding step builder classes, enabling automatic resolution
    during pipeline construction.
    """
    
    # Core registry mapping step types to builders
    BUILDER_REGISTRY = {}  # Auto-populated during initialization
    
    # Legacy aliases for backward compatibility
    LEGACY_ALIASES = {
        "MIMSPackaging": "Package", 
        "MIMSPayload": "Payload",
        "ModelRegistration": "Registration",
        "PyTorchTraining": "PytorchTraining",
        "PyTorchModel": "PytorchModel",
    }
    
    def __init__(self):
        """Initialize the registry."""
        self._custom_builders = {}
        self.logger = logging.getLogger(__name__)
        
        # Populate the registry if empty (first initialization)
        if not self.__class__.BUILDER_REGISTRY:
            # Get core builders through discovery
            discovered = self.__class__.discover_builders()
            self.__class__.BUILDER_REGISTRY = discovered
            
            # Log discovery results
            self.logger.info(f"Discovered {len(discovered)} step builders")
```

### Builder Discovery

The discovery mechanism automatically finds and registers step builders:

```python
@classmethod
def discover_builders(cls):
    """Automatically discover and register step builders."""
    from ..pipeline_steps import builder_step_base
    import importlib
    import inspect
    import pkgutil
    
    # Get the package containing all step builders
    import src.pipeline_steps as steps_package
    
    discovered_builders = {}
    
    # Walk through all modules in the package
    for _, module_name, _ in pkgutil.iter_modules(steps_package.__path__):
        if module_name.startswith('builder_'):
            # Import the module
            module = importlib.import_module(f"src.pipeline_steps.{module_name}")
            
            # Find builder classes in the module
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, builder_step_base.StepBuilderBase) and 
                    obj != builder_step_base.StepBuilderBase):
                    
                    # Get step type from class name (remove 'StepBuilder' suffix)
                    if name.endswith('StepBuilder'):
                        step_type = name[:-11]  # Remove 'StepBuilder'
                    else:
                        step_type = name
                    
                    # Find canonical step name from step registry if possible
                    from ..pipeline_registry.step_names import BUILDER_STEP_NAMES
                    canonical_name = None
                    for step_name, builder_name in BUILDER_STEP_NAMES.items():
                        if builder_name == f"{step_type}Step":
                            canonical_name = step_name
                            break
                    
                    # Register with canonical name if found, otherwise use derived step_type
                    key = canonical_name or step_type
                    discovered_builders[key] = obj
    
    return discovered_builders
```

### Registration Decorator

The registration decorator provides an easy way to auto-register step builders:

```python
def register_builder(step_type: str = None):
    """
    Decorator to automatically register a step builder class.
    
    Args:
        step_type: Optional step type name. If not provided,
                  will be derived from the class name.
    """
    def decorator(cls):
        if not issubclass(cls, StepBuilderBase):
            raise TypeError(f"@register_builder can only be used on StepBuilderBase subclasses: {cls.__name__}")
        
        # Determine step type if not provided
        nonlocal step_type
        if step_type is None:
            if cls.__name__.endswith('StepBuilder'):
                derived_type = cls.__name__[:-11]  # Remove 'StepBuilder'
            else:
                derived_type = cls.__name__
            step_type = derived_type
        
        # Register the class
        StepBuilderRegistry.register_builder_class(step_type, cls)
        return cls
    
    return decorator
```

## Usage Examples

### Using the Registry in Pipeline Construction

```python
from src.pipeline_registry.builder_registry import get_global_registry

def create_pipeline(configs, dag_structure):
    """Create a pipeline from configs and DAG structure."""
    registry = get_global_registry()
    steps = {}
    
    # Create steps for each config
    for config_id, config in configs.items():
        # Get the builder for this config
        builder_class = registry.get_builder_for_config(config)
        builder = builder_class(config)
        
        # Create the step and store it
        step = builder.create_step()
        steps[config_id] = step
    
    # Connect steps based on DAG structure
    # ...
    
    return pipeline
```

### Adding a New Step Builder

With the auto-registration decorator:

```python
# In src/pipeline_steps/builder_new_step.py
from src.pipeline_registry.builder_registry import register_builder
from src.pipeline_steps.builder_step_base import StepBuilderBase

@register_builder()  # Auto-registers with derived name
class NewStepBuilder(StepBuilderBase):
    """Builder for NewStep processing step."""
    # Implementation here
```

Without the decorator (relies on auto-discovery):

```python
# In src/pipeline_steps/builder_new_step.py
from src.pipeline_steps.builder_step_base import StepBuilderBase

class NewStepBuilder(StepBuilderBase):
    """Builder for NewStep processing step."""
    # Implementation here
```

## Best Practices

1. **Use the Registration Decorator**: For explicit and clear registration of new step builders
2. **Update Step Names Registry**: Always add new step types to the step names registry first
3. **Validate the Registry**: Run validation checks to detect inconsistencies
4. **Follow Naming Conventions**: Use consistent naming (ClassNameStepBuilder for classes)
5. **Maintain Legacy Aliases**: Add aliases when renaming step types for backward compatibility

## Validation and Troubleshooting

The `validate_registry()` method helps identify issues with the registry:

```python
validation = registry.validate_registry()

if validation['invalid']:
    print("Invalid registry entries:")
    for entry in validation['invalid']:
        print(f"  - {entry}")

if validation['missing']:
    print("Missing registry entries:")
    for entry in validation['missing']:
        print(f"  - {entry}")
```

## Benefits and Impact

1. **Reduced Maintenance**: Auto-discovery and registration reduce manual maintenance
2. **Improved Consistency**: Better integration with step names registry ensures consistent naming
3. **Enhanced Developer Experience**: Clear patterns for adding new steps
4. **Future-Proof Design**: Flexible approach that accommodates project growth
5. **Better Error Messages**: Validation helps catch and diagnose issues early

## Related Documents

- [Step Names Registry](./step_names_registry_design.md)
- [DAG to Template API](./dag_to_template_api.md)
- [Step Builder Implementation](../developer_guide/step_builder.md)
- [Adding a New Pipeline Step](../developer_guide/adding_new_pipeline_step.md)
