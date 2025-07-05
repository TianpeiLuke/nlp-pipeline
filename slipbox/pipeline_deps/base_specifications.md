# Base Specifications

## Overview
The Base Specifications module provides foundation classes for defining pipeline step specifications that enable automatic dependency resolution and semantic matching between pipeline components.

## Core Classes

### BaseSpecification
Foundation class for all pipeline step specifications.

```python
@dataclass
class BaseSpecification:
    step_name: str
    step_type: str
    inputs: Dict[str, DataSpecification]
    outputs: Dict[str, DataSpecification]
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
```

### DataSpecification
Defines the structure and requirements for data inputs/outputs.

```python
@dataclass
class DataSpecification:
    data_type: str
    format: str
    schema: Optional[Dict[str, Any]]
    constraints: List[str]
    semantic_tags: List[str]
```

## Key Features

### 1. Type Safety
- Strong typing for all specification components
- Validation of specification structure
- Runtime type checking

### 2. Semantic Tagging
- Semantic tags for intelligent matching
- Domain-specific vocabularies
- Extensible tagging system

### 3. Schema Validation
- Optional schema definitions for structured data
- JSON Schema support
- Custom validation rules

## Usage Example

```python
from src.pipeline_deps.base_specifications import BaseSpecification, DataSpecification

# Define input specification
input_spec = DataSpecification(
    data_type="tabular",
    format="parquet",
    schema={
        "type": "object",
        "properties": {
            "features": {"type": "array"},
            "labels": {"type": "array"}
        }
    },
    constraints=["non_empty", "valid_schema"],
    semantic_tags=["training_data", "supervised_learning"]
)

# Define output specification
output_spec = DataSpecification(
    data_type="model",
    format="pickle",
    schema=None,
    constraints=["serializable"],
    semantic_tags=["trained_model", "xgboost"]
)

# Create step specification
step_spec = BaseSpecification(
    step_name="xgboost_training",
    step_type="training",
    inputs={"training_data": input_spec},
    outputs={"trained_model": output_spec},
    parameters={"max_depth": 6, "learning_rate": 0.1},
    metadata={"framework": "xgboost", "version": "1.7.6"}
)
```

## Specification Types

### Data Types
- **tabular** - Structured tabular data (CSV, Parquet, etc.)
- **text** - Unstructured text data
- **model** - Trained machine learning models
- **metrics** - Performance metrics and evaluation results
- **artifacts** - General pipeline artifacts

### Formats
- **parquet** - Apache Parquet format
- **csv** - Comma-separated values
- **json** - JSON format
- **pickle** - Python pickle format
- **joblib** - Joblib serialization

### Semantic Tags
- **training_data** - Data used for model training
- **validation_data** - Data used for model validation
- **test_data** - Data used for model testing
- **preprocessed** - Data that has been preprocessed
- **raw** - Raw, unprocessed data

## Validation Rules

### Required Fields
- step_name must be non-empty string
- step_type must be valid type
- inputs and outputs must be dictionaries
- DataSpecification must have valid data_type and format

### Constraints
- Constraint validation based on data type
- Custom constraint functions
- Schema validation for structured data

## Integration Points

### With Dependency Resolver
Base specifications provide the foundation for dependency resolution:

```python
from src.pipeline_deps import DependencyResolver

resolver = DependencyResolver()
dependencies = resolver.resolve(step_spec)
```

### With Semantic Matcher
Semantic tags enable intelligent matching:

```python
from src.pipeline_deps import SemanticMatcher

matcher = SemanticMatcher()
compatibility = matcher.check_compatibility(output_spec, input_spec)
```

### With Registry
Specifications are registered for reuse:

```python
from src.pipeline_deps import SpecificationRegistry

registry = SpecificationRegistry()
registry.register_specification("xgboost_training", step_spec)
```

## Extension Points

### Custom Data Types
```python
# Register custom data type
registry.register_data_type("custom_format", CustomDataTypeHandler())
```

### Custom Constraints
```python
# Define custom constraint
def custom_constraint(data_spec: DataSpecification) -> bool:
    return validate_custom_logic(data_spec)

# Register constraint
registry.register_constraint("custom_constraint", custom_constraint)
```

### Custom Semantic Tags
```python
# Define domain-specific tags
DOMAIN_TAGS = [
    "financial_data",
    "time_series",
    "risk_model"
]

# Use in specifications
spec = DataSpecification(
    data_type="tabular",
    format="parquet",
    semantic_tags=["financial_data", "time_series"]
)
```

## Best Practices

1. **Use Semantic Tags** - Always include relevant semantic tags for better matching
2. **Define Schemas** - Provide schemas for structured data when possible
3. **Specify Constraints** - Include validation constraints to ensure data quality
4. **Consistent Naming** - Use consistent naming conventions for step and data names
5. **Version Metadata** - Include version information in metadata for reproducibility
