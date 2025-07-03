# Step Specification

## What is the Purpose of Step Specification?

Step Specifications serve as the **foundational declarative metadata layer** that defines the structural and topological characteristics of pipeline steps. They represent the evolution from imperative "how" to declarative "what" in pipeline architecture.

## Core Purpose

Step Specifications provide a **declarative way to define what a step is** rather than how it works, enabling:

1. **Pipeline Topology Classification** - Classify steps by their role using NodeType
2. **Automatic Constraint Validation** - Enforce architectural rules at design time
3. **Intelligent Dependency Resolution** - Enable semantic matching between steps
4. **Registry-Based Discovery** - Centralized management and compatibility checking
5. **Design-Time Validation** - Catch errors before implementation

## Key Features

### 1. Node Type Classification

Steps are classified by their role in the pipeline topology:

```python
# SOURCE: Entry points with no dependencies
DATA_LOADING_SPEC = StepSpecification(
    step_type="DataLoading",
    node_type=NodeType.SOURCE,  # No dependencies, produces outputs
    dependencies=[],  # SOURCE nodes cannot have dependencies
    outputs=[...]  # Must have outputs
)

# INTERNAL: Processing nodes with both inputs and outputs
PREPROCESSING_SPEC = StepSpecification(
    step_type="Preprocessing", 
    node_type=NodeType.INTERNAL,  # Has both dependencies and outputs
    dependencies=[...],  # Must have dependencies
    outputs=[...]  # Must have outputs
)

# SINK: Terminal nodes that consume but don't produce
REGISTRATION_SPEC = StepSpecification(
    step_type="ModelRegistration",
    node_type=NodeType.SINK,  # Has dependencies, no outputs
    dependencies=[...],  # Must have dependencies
    outputs=[]  # SINK nodes cannot have outputs
)
```

### 2. Semantic Dependency Matching

Specifications provide semantic metadata for intelligent matching:

```python
DependencySpec(
    logical_name="model_input",
    dependency_type=DependencyType.MODEL_ARTIFACTS,
    required=True,
    compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
    semantic_keywords=["model", "artifacts", "trained", "output"],
    data_type="S3Uri",
    description="Trained model artifacts for packaging"
)
```

### 3. Multi-Alias Output Support

Support multiple aliases for compatibility across naming conventions:

```python
outputs=[
    OutputSpec(
        logical_name="processed_data",  # Primary name
        output_type=DependencyType.PROCESSING_OUTPUT,
        property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
    ),
    OutputSpec(
        logical_name="ProcessedTabularData",  # Alias for compatibility
        output_type=DependencyType.PROCESSING_OUTPUT,
        property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri"
    )
]
```

### 4. Automatic Constraint Validation

Validation happens at specification creation time:

```python
# This will raise ValueError
invalid_spec = StepSpecification(
    step_type="InvalidSource",
    node_type=NodeType.SOURCE,
    dependencies=[some_dependency],  # ERROR: SOURCE cannot have dependencies
    outputs=[]
)
# ValueError: "SOURCE node 'InvalidSource' cannot have dependencies"
```

## Integration with Other Components

### With Registry System
```python
# Register specifications for discovery
global_registry.register("data_loading", DATA_LOADING_SPEC)
global_registry.register("preprocessing", PREPROCESSING_SPEC)

# Find compatible outputs for a dependency
compatible_outputs = global_registry.find_compatible_outputs(dependency_spec)
```

### With Smart Proxies
[Smart Proxies](smart_proxy.md) use specifications for intelligent dependency resolution and validation.

### With Step Builders
[Step Builders](step_builder.md) implement the behavior defined by specifications:

```python
class XGBoostTrainingStepBuilder(BuilderStepBase):
    @classmethod
    def get_specification(cls) -> StepSpecification:
        return XGBOOST_TRAINING_SPEC
```

## Strategic Value

Step Specifications enable:

1. **Separation of Concerns**: Structure (specification) vs. Behavior (implementation)
2. **Early Validation**: Catch errors at design time, not runtime
3. **Intelligent Automation**: Enable smart dependency resolution and validation
4. **Maintainability**: Changes to specifications automatically propagate
5. **Interoperability**: Common interface for different step implementations
6. **Scalability**: Registry-based discovery supports large step libraries

## Example Usage

```python
# Define a complete step specification
XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type="XGBoostTraining",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["PreprocessingStep", "DataLoadingStep"],
            semantic_keywords=["processed", "data", "training"],
            data_type="S3Uri"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri"
        )
    ]
)

# Automatic validation
errors = XGBOOST_TRAINING_SPEC.validate()  # Returns empty list if valid
```

Step Specifications are the **architectural foundation** that enables all higher-level abstractions ([Smart Proxies](smart_proxy.md), [Fluent APIs](fluent_api.md), [Step Contracts](step_contract.md)) to work intelligently and safely together.
