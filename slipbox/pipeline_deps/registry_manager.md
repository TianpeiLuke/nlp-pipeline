# Registry Manager

## Overview
The Registry Manager provides centralized management of multiple isolated specification registries, ensuring complete isolation between different contexts (pipelines, environments, etc.). It enables context-scoped registries for multi-tenant pipeline systems.

## Core Functionality

### Context Isolation
- **Multiple Registries** - Manages separate registry instances for different contexts
- **Complete Isolation** - No cross-contamination between contexts
- **Dynamic Creation** - Creates registries on-demand for new contexts
- **Context Statistics** - Tracks usage and statistics per context

### Registry Lifecycle
- **Creation** - Automatic registry creation for new contexts
- **Retrieval** - Fast access to context-specific registries
- **Cleanup** - Individual or bulk registry clearing
- **Statistics** - Context usage monitoring and reporting

## Key Classes

### RegistryManager
Main manager class that coordinates multiple registry instances.

```python
class RegistryManager:
    def __init__(self):
        """Initialize the registry manager."""
        
    def get_registry(self, context_name: str = "default", 
                    create_if_missing: bool = True) -> Optional[SpecificationRegistry]:
        """Get the registry for a specific context."""
        
    def list_contexts(self) -> List[str]:
        """Get list of all registered context names."""
        
    def clear_context(self, context_name: str) -> bool:
        """Clear the registry for a specific context."""
        
    def clear_all_contexts(self):
        """Clear all registries."""
        
    def get_context_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics for all contexts."""
```

## Usage Examples

### Basic Registry Management
```python
from src.pipeline_deps.registry_manager import RegistryManager

# Create registry manager
manager = RegistryManager()

# Get registry for specific context
training_registry = manager.get_registry("training_pipeline")
validation_registry = manager.get_registry("validation_pipeline")

# Register specifications in different contexts
training_registry.register_specification("data_loading", training_data_spec)
validation_registry.register_specification("data_loading", validation_data_spec)

# List all contexts
contexts = manager.list_contexts()
print(f"Active contexts: {contexts}")
# Output: ['training_pipeline', 'validation_pipeline']
```

### Using Global Registry Manager
```python
from src.pipeline_deps.registry_manager import get_registry, list_contexts

# Get registry using convenience function
pipeline_registry = get_registry("my_pipeline")

# Register specifications
pipeline_registry.register_specification("preprocessing", preprocess_spec)
pipeline_registry.register_specification("training", training_spec)

# List all contexts
all_contexts = list_contexts()
print(f"All contexts: {all_contexts}")
```

### Context Statistics
```python
from src.pipeline_deps.registry_manager import get_context_stats

# Get statistics for all contexts
stats = get_context_stats()
for context_name, context_stats in stats.items():
    print(f"Context '{context_name}':")
    print(f"  Steps: {context_stats['step_count']}")
    print(f"  Step Types: {context_stats['step_type_count']}")

# Example output:
# Context 'training_pipeline':
#   Steps: 5
#   Step Types: 3
# Context 'validation_pipeline':
#   Steps: 3
#   Step Types: 2
```

### Context Cleanup
```python
from src.pipeline_deps.registry_manager import clear_context, registry_manager

# Clear specific context
success = clear_context("old_pipeline")
if success:
    print("Context cleared successfully")

# Clear all contexts
registry_manager.clear_all_contexts()
print("All contexts cleared")
```

## Integration Features

### Pipeline Builder Integration
The registry manager provides a decorator for automatic integration with pipeline builders:

```python
from src.pipeline_deps.registry_manager import integrate_with_pipeline_builder

@integrate_with_pipeline_builder
class MyPipelineBuilder:
    def __init__(self, base_config):
        self.base_config = base_config
        # self.registry is automatically created based on pipeline_name
        
    def build_pipeline(self):
        # Use self.registry for context-specific specifications
        data_spec = self.registry.get_specification("data_loading")
        return self._build_with_spec(data_spec)
```

### Automatic Context Detection
```python
# Registry manager automatically detects context from pipeline configuration
class PipelineConfig:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name

config = PipelineConfig("production_training")
builder = MyPipelineBuilder(config)
# builder.registry is automatically scoped to "production_training"
```

## Context Patterns

### Environment-Based Contexts
```python
# Different registries for different environments
dev_registry = get_registry("development")
staging_registry = get_registry("staging")
prod_registry = get_registry("production")

# Each environment can have different specifications
dev_registry.register_specification("data_loading", dev_data_spec)
prod_registry.register_specification("data_loading", prod_data_spec)
```

### Pipeline-Based Contexts
```python
# Different registries for different pipeline types
training_registry = get_registry("training_pipeline")
inference_registry = get_registry("inference_pipeline")
batch_registry = get_registry("batch_processing")

# Each pipeline type has its own specifications
training_registry.register_specification("model_training", training_spec)
inference_registry.register_specification("model_inference", inference_spec)
```

### Multi-Tenant Contexts
```python
# Different registries for different tenants/customers
customer_a_registry = get_registry("customer_a")
customer_b_registry = get_registry("customer_b")

# Each customer can have customized specifications
customer_a_registry.register_specification("preprocessing", custom_preprocess_a)
customer_b_registry.register_specification("preprocessing", custom_preprocess_b)
```

## Backward Compatibility

### Legacy Functions
The module provides backward compatibility functions for existing code:

```python
from src.pipeline_deps.registry_manager import (
    get_pipeline_registry,
    get_default_registry
)

# Legacy pipeline registry access
pipeline_registry = get_pipeline_registry("my_pipeline")

# Legacy default registry access
default_registry = get_default_registry()
```

### Migration Path
```python
# Old code
from src.pipeline_deps import SpecificationRegistry
registry = SpecificationRegistry()

# New code
from src.pipeline_deps.registry_manager import get_registry
registry = get_registry("my_context")
```

## Global Registry Manager

### Singleton Pattern
The module provides a global registry manager instance:

```python
from src.pipeline_deps.registry_manager import registry_manager

# Direct access to global manager
registry = registry_manager.get_registry("my_context")
contexts = registry_manager.list_contexts()
stats = registry_manager.get_context_stats()
```

### Thread Safety
The registry manager is designed to be thread-safe for concurrent access:

```python
import threading
from src.pipeline_deps.registry_manager import get_registry

def worker_function(context_name: str):
    # Each thread can safely access its own context
    registry = get_registry(f"worker_{context_name}")
    registry.register_specification("task", task_spec)

# Create multiple worker threads
threads = []
for i in range(5):
    thread = threading.Thread(target=worker_function, args=(str(i),))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()
```

## Best Practices

### 1. Context Naming
- Use descriptive context names: `training_pipeline`, `production_env`
- Follow consistent naming conventions across your organization
- Avoid special characters that might cause issues in logging or file systems

### 2. Context Lifecycle
- Create contexts when needed, don't pre-create all possible contexts
- Clean up contexts when pipelines are decommissioned
- Monitor context statistics to identify unused contexts

### 3. Registry Isolation
- Keep contexts completely isolated - don't share specifications across contexts
- Use context-specific specifications even if they're similar
- Document the purpose and scope of each context

### 4. Integration Patterns
- Use the decorator pattern for automatic pipeline builder integration
- Leverage context detection from configuration objects
- Implement consistent context naming across your pipeline ecosystem

## Error Handling

### Context Not Found
```python
from src.pipeline_deps.registry_manager import get_registry

# Handle missing context
registry = get_registry("nonexistent_context", create_if_missing=False)
if registry is None:
    print("Context not found")
else:
    # Use registry
    pass
```

### Registry Cleanup Errors
```python
from src.pipeline_deps.registry_manager import clear_context

# Handle cleanup failures
success = clear_context("context_to_clear")
if not success:
    print("Context was already cleared or didn't exist")
```

## Integration Points

### With Specification Registry
```python
# Registry manager creates and manages SpecificationRegistry instances
from src.pipeline_deps.specification_registry import SpecificationRegistry

# Each context gets its own SpecificationRegistry instance
registry = get_registry("my_context")
assert isinstance(registry, SpecificationRegistry)
```

### With Dependency Resolver
```python
from src.pipeline_deps.dependency_resolver import DependencyResolver
from src.pipeline_deps.registry_manager import get_registry

# Use context-specific registry with dependency resolver
registry = get_registry("training_context")
resolver = DependencyResolver(registry)
dependencies = resolver.resolve_dependencies(["data_loading", "training"])
```

### With Pipeline Builder
```python
# Automatic integration with pipeline builders
@integrate_with_pipeline_builder
class TrainingPipelineBuilder:
    def build(self):
        # self.registry is automatically context-scoped
        specs = self.registry.list_specifications()
        return self._build_pipeline_from_specs(specs)
```

## Related Design Documentation

For architectural context and design decisions, see:
- **[Registry Manager Design](../pipeline_design/registry_manager.md)** - Registry management architecture
- **[Specification Registry Design](../pipeline_design/specification_registry.md)** - Registry implementation patterns
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming and structure conventions

## Performance Considerations

### Memory Management
- Registries are kept in memory for fast access
- Clear unused contexts to free memory
- Monitor context statistics to identify memory usage patterns

### Concurrent Access
- Registry manager supports concurrent access from multiple threads
- Each context is isolated, preventing race conditions
- Use context-specific registries to avoid contention

### Scalability
- Registry manager scales to hundreds of contexts
- Context creation is lightweight and fast
- Statistics collection is optimized for frequent access
