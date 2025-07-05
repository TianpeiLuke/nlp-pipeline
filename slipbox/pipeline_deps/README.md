# Pipeline Dependencies

This module provides dependency resolution and specification management for pipeline components. It enables automatic dependency resolution between pipeline steps based on their input/output specifications.

## Core Components

### Base Specifications
- **base_specifications.py** - Foundation classes for dependency specifications
- **specification_registry.py** - Registry for managing step specifications
- **registry_manager.py** - High-level registry management interface

### Dependency Resolution
- **dependency_resolver.py** - Core dependency resolution algorithms
- **semantic_matcher.py** - Semantic matching for compatible specifications

## Key Features

1. **Automatic Dependency Resolution** - Resolves step dependencies based on I/O specifications
2. **Semantic Matching** - Intelligent matching of compatible data types and formats
3. **Specification Registry** - Centralized management of step specifications
4. **Validation Framework** - Ensures dependency consistency and completeness

## Usage Pattern

```python
from src.pipeline_deps import SpecificationRegistry, DependencyResolver

# Register step specifications
registry = SpecificationRegistry()
registry.register_specification("data_loading", data_loading_spec)
registry.register_specification("preprocessing", preprocessing_spec)

# Resolve dependencies
resolver = DependencyResolver(registry)
dependencies = resolver.resolve_dependencies(["data_loading", "preprocessing"])
```

## Integration

This module integrates with:
- **Pipeline Step Specs** - Provides specifications for dependency resolution
- **Pipeline Builder** - Uses resolved dependencies for pipeline construction
- **Step Builders** - Validates step compatibility during configuration

## Related Design Documentation

For architectural context and design decisions, see:
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Dependency Resolver Design](../pipeline_design/dependency_resolver.md)** - Dependency resolution architecture
- **[Step Specification Design](../pipeline_design/step_specification.md)** - Step specification patterns
- **[Registry Manager Design](../pipeline_design/registry_manager.md)** - Registry management approach
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles
