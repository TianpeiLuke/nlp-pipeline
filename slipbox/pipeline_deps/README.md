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
