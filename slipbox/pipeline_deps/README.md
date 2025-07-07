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
5. **Script Contract Integration** - Automated script validation against step specifications
6. **Output Aliases System** - Multiple names for outputs with backward compatibility
7. **Contract Alignment Validation** - Built-in verification of specification-contract alignment
8. **Job Type Variant Handling** - Support for training vs calibration workflow differentiation

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

## New Features Examples

### Output Aliases System
```python
from src.pipeline_deps.base_specifications import OutputSpec, DependencyType

# Create output with aliases for backward compatibility
model_output = OutputSpec(
    logical_name="model_output",
    output_type=DependencyType.MODEL_ARTIFACTS,
    property_path="properties.ModelArtifacts.S3ModelArtifacts",
    aliases=["ModelArtifacts", "model_data", "output_path"]
)

# Access by primary name or any alias
primary = step_spec.get_output("model_output")
legacy = step_spec.get_output_by_name_or_alias("ModelArtifacts")
assert primary == legacy  # Same output object
```

### Script Contract Validation
```python
from src.pipeline_script_contracts.base_script_contract import ScriptContract

# Define contract matching actual script behavior
contract = ScriptContract(
    script_name="train.py",
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "config": "/opt/ml/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "data_output": "/opt/ml/output/data"
    }
)

# Create specification with contract reference
step_spec = StepSpecification(
    step_type="TrainingStep",
    script_contract=contract,
    dependencies=[...],
    outputs=[...]
)

# Validate alignment
result = step_spec.validate_contract_alignment()
if result.is_valid:
    print("✅ Perfect alignment!")
else:
    print("❌ Alignment issues:", result.errors)
```

## Integration

This module integrates with:
- **Pipeline Step Specs** - Provides specifications for dependency resolution
- **Pipeline Builder** - Uses resolved dependencies for pipeline construction
- **Step Builders** - Validates step compatibility during configuration

## File Structure

```
slipbox/pipeline_deps/
├── README.md                    # This overview document
├── base_specifications.md       # Core specification data structures (Pydantic V2)
├── dependency_resolver.md       # Dependency resolution algorithms and patterns
├── registry_manager.md          # Multi-context registry management
├── semantic_matcher.md          # Intelligent semantic similarity matching
└── specification_registry.md    # Context-aware specification storage
```

## Documentation Coverage

### Core Data Structures
- **[Base Specifications](base_specifications.md)** - Pydantic V2 models for step specifications, dependencies, and outputs
- **[Specification Registry](specification_registry.md)** - Context-aware storage and retrieval with compatibility checking
- **[Registry Manager](registry_manager.md)** - Multi-context registry management with complete isolation

### Advanced Features
- **[Dependency Resolver](dependency_resolver.md)** - Intelligent dependency resolution algorithms and patterns
- **[Semantic Matcher](semantic_matcher.md)** - Multi-metric similarity scoring for automatic component matching

## Related Design Documentation

For architectural context and design decisions, see:
- **[Specification Driven Design](../pipeline_design/specification_driven_design.md)** - Overall design philosophy
- **[Dependency Resolver Design](../pipeline_design/dependency_resolver.md)** - Dependency resolution architecture
- **[Step Specification Design](../pipeline_design/step_specification.md)** - Step specification patterns
- **[Registry Manager Design](../pipeline_design/registry_manager.md)** - Registry management approach
- **[Design Principles](../pipeline_design/design_principles.md)** - Core design principles

## Related Script Contract Documentation

For script contract implementation and validation details, see:
- **[Base Script Contract](../pipeline_script_contracts/base_script_contract.md)** - Foundation classes for script contracts
- **[Contract Validator](../pipeline_script_contracts/contract_validator.md)** - Contract validation mechanisms
- **[PyTorch Train Contract](../pipeline_script_contracts/pytorch_train_contract.md)** - PyTorch training contract example
- **[Script Contracts Overview](../pipeline_script_contracts/README.md)** - Complete script contracts documentation
