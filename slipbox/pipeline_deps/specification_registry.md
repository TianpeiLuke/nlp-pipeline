# Specification Registry

## Overview
The Specification Registry provides context-aware storage and retrieval of step specifications with complete isolation between contexts. It serves as the central repository for pipeline step specifications, enabling efficient lookup, compatibility checking, and dependency resolution.

## Core Functionality

### Context Isolation
- **Context-Scoped Storage** - Each registry instance is tied to a specific context
- **Complete Isolation** - No cross-contamination between different contexts
- **Context Identification** - Clear context naming for debugging and management
- **Independent Lifecycle** - Each context can be managed independently

### Specification Management
- **Registration** - Store step specifications with validation
- **Retrieval** - Fast lookup by step name or type
- **Compatibility Checking** - Find compatible outputs for dependencies
- **Type Mapping** - Efficient lookup by step type

## Key Classes

### SpecificationRegistry
Main registry class that manages step specifications within a context.

```python
class SpecificationRegistry:
    def __init__(self, context_name: str = "default"):
        """Initialize a context-scoped registry."""
        
    def register(self, step_name: str, specification: StepSpecification):
        """Register a step specification."""
        
    def get_specification(self, step_name: str) -> Optional[StepSpecification]:
        """Get specification by step name."""
        
    def get_specifications_by_type(self, step_type: str) -> List[StepSpecification]:
        """Get all specifications of a given step type."""
        
    def find_compatible_outputs(self, dependency_spec: DependencySpec) -> List[tuple]:
        """Find outputs compatible with a dependency specification."""
```

## Usage Examples

### Basic Registry Operations
```python
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import StepSpecification, NodeType

# Create context-specific registry
registry = SpecificationRegistry("training_pipeline")

# Create and register a specification
data_loading_spec = StepSpecification(
    step_type="DataLoadingStep",
    node_type=NodeType.SOURCE,
    outputs=[output_spec]
)

registry.register("data_loading", data_loading_spec)

# Retrieve specification
spec = registry.get_specification("data_loading")
print(f"Retrieved spec: {spec.step_type}")
```

### Type-Based Retrieval
```python
# Register multiple specifications of the same type
registry.register("training_data_loading", training_data_spec)
registry.register("validation_data_loading", validation_data_spec)
registry.register("test_data_loading", test_data_spec)

# Get all data loading specifications
data_loading_specs = registry.get_specifications_by_type("DataLoadingStep")
print(f"Found {len(data_loading_specs)} data loading specifications")

# List all step types
step_types = registry.list_step_types()
print(f"Available step types: {step_types}")
```

### Compatibility Checking
```python
from src.pipeline_deps.base_specifications import DependencySpec, DependencyType

# Define a dependency requirement
training_data_dep = DependencySpec(
    logical_name="training_data",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    required=True,
    compatible_sources=["DataLoadingStep", "PreprocessingStep"],
    semantic_keywords=["data", "training"],
    data_type="S3Uri"
)

# Find compatible outputs
compatible_outputs = registry.find_compatible_outputs(training_data_dep)
for step_name, output_name, output_spec, score in compatible_outputs:
    print(f"{step_name}.{output_name}: {score:.3f}")

# Output:
# data_loading.processed_data: 0.8
# preprocessing.training_output: 0.9
```

### Registry Inspection
```python
# Get registry statistics
step_names = registry.list_step_names()
step_types = registry.list_step_types()

print(f"Registry '{registry.context_name}':")
print(f"  Total steps: {len(step_names)}")
print(f"  Step types: {len(step_types)}")
print(f"  Steps: {step_names}")
print(f"  Types: {step_types}")
```

## Compatibility Scoring

### Compatibility Algorithm
The registry uses a multi-factor scoring system to rank compatibility:

```python
def _calculate_compatibility_score(self, dep_spec: DependencySpec, 
                                 out_spec: OutputSpec, step_type: str) -> float:
    """Calculate compatibility score between dependency and output."""
    score = 0.5  # Base compatibility score
    
    # Compatible source bonus (+0.3)
    if dep_spec.compatible_sources and step_type in dep_spec.compatible_sources:
        score += 0.3
    
    # Semantic keyword matching (+0.2 max)
    if dep_spec.semantic_keywords:
        keyword_matches = sum(
            1 for keyword in dep_spec.semantic_keywords
            if keyword.lower() in out_spec.logical_name.lower()
        )
        score += (keyword_matches / len(dep_spec.semantic_keywords)) * 0.2
    
    return min(score, 1.0)  # Cap at 1.0
```

### Compatibility Factors
1. **Base Compatibility (0.5)** - Type and data type match
2. **Source Compatibility (+0.3)** - Step type is in compatible sources list
3. **Semantic Matching (+0.2)** - Keyword overlap in names

### Compatibility Examples
```python
# High compatibility (score: 1.0)
# - Exact type match
# - Step type in compatible sources
# - All semantic keywords match

# Medium compatibility (score: 0.7)
# - Type match
# - Step type in compatible sources
# - Partial semantic keyword match

# Low compatibility (score: 0.5)
# - Type match only
# - No source or semantic bonuses
```

## Advanced Features

### Batch Registration
```python
# Register multiple specifications at once
specifications = {
    "data_loading": data_loading_spec,
    "preprocessing": preprocessing_spec,
    "training": training_spec,
    "evaluation": evaluation_spec
}

for step_name, spec in specifications.items():
    registry.register(step_name, spec)
```

### Specification Validation
```python
# Registry validates specifications during registration
try:
    registry.register("invalid_step", invalid_spec)
except ValueError as e:
    print(f"Registration failed: {e}")
    
# Specifications are validated using Pydantic
# - Required fields must be present
# - Types must be correct
# - Node type constraints must be satisfied
```

### Context Management
```python
# Different registries for different contexts
training_registry = SpecificationRegistry("training_pipeline")
inference_registry = SpecificationRegistry("inference_pipeline")
batch_registry = SpecificationRegistry("batch_processing")

# Each registry maintains independent state
training_registry.register("model_training", training_spec)
inference_registry.register("model_inference", inference_spec)

# No cross-contamination between contexts
assert training_registry.get_specification("model_inference") is None
assert inference_registry.get_specification("model_training") is None
```

## Integration Points

### With Registry Manager
```python
from src.pipeline_deps.registry_manager import get_registry

# Registry manager creates and manages SpecificationRegistry instances
registry = get_registry("my_context")
assert isinstance(registry, SpecificationRegistry)
assert registry.context_name == "my_context"
```

### With Dependency Resolver
```python
from src.pipeline_deps.dependency_resolver import DependencyResolver

# Dependency resolver uses registry for specification lookup
resolver = DependencyResolver(registry)

# Resolver queries registry for compatible specifications
dependencies = resolver.resolve_dependencies(["data_loading", "training"])
```

### With Pipeline Builder
```python
# Pipeline builders use registry for step specification lookup
class PipelineBuilder:
    def __init__(self, registry: SpecificationRegistry):
        self.registry = registry
        
    def build_step(self, step_name: str):
        spec = self.registry.get_specification(step_name)
        if spec is None:
            raise ValueError(f"No specification found for step '{step_name}'")
        return self._create_step_from_spec(spec)
```

## Error Handling

### Registration Errors
```python
# Handle invalid specifications
try:
    registry.register("bad_step", invalid_spec)
except ValueError as e:
    print(f"Registration failed: {e}")
    # Handle error appropriately

# Handle duplicate registrations
if registry.get_specification("existing_step") is not None:
    print("Step already registered, updating...")
registry.register("existing_step", updated_spec)
```

### Retrieval Errors
```python
# Handle missing specifications
spec = registry.get_specification("nonexistent_step")
if spec is None:
    print("Specification not found")
    # Handle missing specification
else:
    # Use specification
    pass
```

### Compatibility Errors
```python
# Handle no compatible outputs
compatible = registry.find_compatible_outputs(dependency_spec)
if not compatible:
    print("No compatible outputs found")
    # Handle incompatibility
else:
    best_match = compatible[0]  # Highest scored match
    print(f"Best match: {best_match[0]}.{best_match[1]} (score: {best_match[3]:.3f})")
```

## Performance Considerations

### Memory Usage
- Registries store specifications in memory for fast access
- Each context maintains its own memory space
- Consider clearing unused contexts to free memory

### Lookup Performance
- Step name lookup: O(1) - Direct dictionary access
- Step type lookup: O(1) - Pre-computed type mappings
- Compatibility checking: O(n) - Linear scan of all outputs

### Optimization Strategies
```python
# Cache compatibility results for repeated queries
from functools import lru_cache

class OptimizedSpecificationRegistry(SpecificationRegistry):
    @lru_cache(maxsize=100)
    def find_compatible_outputs_cached(self, dep_spec_key: str) -> List[tuple]:
        # Convert dependency spec to hashable key and cache results
        return self.find_compatible_outputs(dep_spec)
```

## Best Practices

### 1. Context Naming
- Use descriptive context names: `training_pipeline`, `production_env`
- Follow consistent naming conventions
- Include environment or purpose in context names

### 2. Specification Organization
- Group related specifications in the same context
- Use consistent step naming within contexts
- Document the purpose and scope of each context

### 3. Compatibility Design
- Design specifications with compatibility in mind
- Use semantic keywords to improve matching
- Specify compatible sources explicitly

### 4. Error Handling
- Always check for None when retrieving specifications
- Validate specifications before registration
- Handle compatibility failures gracefully

## Debugging and Monitoring

### Registry Inspection
```python
# Debug registry contents
def debug_registry(registry: SpecificationRegistry):
    print(f"Registry: {registry}")
    print(f"Context: {registry.context_name}")
    print(f"Steps: {registry.list_step_names()}")
    print(f"Types: {registry.list_step_types()}")
    
    for step_name in registry.list_step_names():
        spec = registry.get_specification(step_name)
        print(f"  {step_name}: {spec.step_type} ({spec.node_type.value})")
        print(f"    Dependencies: {list(spec.dependencies.keys())}")
        print(f"    Outputs: {list(spec.outputs.keys())}")
```

### Compatibility Analysis
```python
# Analyze compatibility patterns
def analyze_compatibility(registry: SpecificationRegistry):
    all_deps = []
    all_outputs = []
    
    for spec in registry._specifications.values():
        all_deps.extend(spec.dependencies.values())
        all_outputs.extend(spec.outputs.values())
    
    print(f"Total dependencies: {len(all_deps)}")
    print(f"Total outputs: {len(all_outputs)}")
    
    # Check compatibility matrix
    for dep in all_deps:
        compatible = registry.find_compatible_outputs(dep)
        print(f"Dependency '{dep.logical_name}': {len(compatible)} compatible outputs")
```

## Related Design Documentation

For architectural context and design decisions, see:
- **[Specification Registry Design](../pipeline_design/specification_registry.md)** - Registry architecture and patterns
- **[Registry Manager Design](../pipeline_design/registry_manager.md)** - Multi-context registry management
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Step Specification Design](../pipeline_design/step_specification.md)** - Step specification patterns
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles
- **[Standardization Rules](../pipeline_design/standardization_rules.md)** - Naming and structure conventions

## Migration and Compatibility

### Legacy Support
The registry maintains backward compatibility with older specification formats:

```python
# Legacy specification creation (still supported)
legacy_spec = StepSpecification(
    step_type="ProcessingStep",
    dependencies=[dep1, dep2],  # List format
    outputs=[out1, out2]        # List format
)

# Modern specification creation (recommended)
modern_spec = StepSpecification(
    step_type="ProcessingStep",
    node_type=NodeType.INTERNAL,
    dependencies={"dep1": dep1, "dep2": dep2},  # Dict format
    outputs={"out1": out1, "out2": out2}        # Dict format
)
```

### Migration Path
```python
# Migrate from old registry to new context-aware registry
old_specs = old_registry.get_all_specifications()
new_registry = SpecificationRegistry("migrated_context")

for step_name, spec in old_specs.items():
    new_registry.register(step_name, spec)
