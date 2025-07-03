# Pipeline Dependency Management System

This system provides intelligent, automatic dependency resolution for ML pipeline steps using declarative specifications and semantic matching.

## Overview

The pipeline dependency management system consists of several key components:

1. **Base Specifications** (`base_specifications.py`): Core data structures and registry
2. **Step Specifications** (`../pipeline_step_specs/`): Declarative specifications for each step type
3. **Semantic Matcher** (`semantic_matcher.py`): Intelligent name matching using semantic similarity
4. **Dependency Resolver** (`dependency_resolver.py`): Main resolution engine
5. **Enhanced DAG** (`../pipeline_dag/enhanced_dag.py`): DAG structure with dependency metadata

## Key Features

### Automatic Dependency Resolution
- Automatically matches step inputs with compatible outputs from other steps
- Uses semantic similarity, type compatibility, and keyword matching
- Provides confidence scoring for dependency matches

### Declarative Specifications
- Each step type has a declarative specification defining its inputs and outputs
- Specifications include semantic keywords for intelligent matching
- Property paths define how to access outputs at runtime

### Intelligent Matching
- Semantic name similarity (e.g., "training_data" matches "processed_data")
- Type compatibility checking (e.g., MODEL_ARTIFACTS matches MODEL_ARTIFACTS)
- Keyword-based matching for better accuracy
- Compatible source validation

### Flexible Property Access
- Multiple property path formats supported
- Name-based and index-based output access
- Direct attribute access for special cases

## Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Step Builder      │    │  Step Specification  │    │  Dependency         │
│   (Implementation)  │───▶│  (Declarative)       │───▶│  Resolver           │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                      │                           │
                                      ▼                           ▼
                           ┌──────────────────────┐    ┌─────────────────────┐
                           │  Semantic Matcher    │    │  Enhanced DAG       │
                           │  (Name Similarity)   │    │  (Pipeline Graph)   │
                           └──────────────────────┘    └─────────────────────┘
```

## Supported Step Types

### 1. CradleDataLoading
- **Purpose**: Load data from external Cradle sources
- **Inputs**: None (source step)
- **Outputs**: DATA, METADATA, SIGNATURE

### 2. TabularPreprocessing  
- **Purpose**: Preprocess tabular data for training
- **Inputs**: input_data (from data loading)
- **Outputs**: processed_data, ProcessedTabularData

### 3. XGBoostTraining
- **Purpose**: Train XGBoost models
- **Inputs**: training_data (required), hyperparameters (optional)
- **Outputs**: model_artifacts, ModelArtifacts

### 4. Package
- **Purpose**: Package models for deployment
- **Inputs**: model_input (required), inference_scripts_input (required)
- **Outputs**: packaged_model_output, PackagedModel

### 5. Payload
- **Purpose**: Generate test payloads for models
- **Inputs**: model_input (required)
- **Outputs**: payload_sample, GeneratedPayloadSamples, payload_metadata, PayloadMetadata

### 6. Registration
- **Purpose**: Register models in MIMS
- **Inputs**: model_input (required), payload_input (optional)
- **Outputs**: registered_model, RegisteredModel

## Usage Example

```python
from src.pipeline_deps import UnifiedDependencyResolver, SpecificationRegistry
from src.pipeline_step_specs import *

# Create resolver with specifications
resolver = UnifiedDependencyResolver()
registry = resolver.registry

# Register step specifications
registry.register("data_loading", DATA_LOADING_SPEC)
registry.register("preprocessing", PREPROCESSING_SPEC)
registry.register("training", XGBOOST_TRAINING_SPEC)
registry.register("packaging", PACKAGING_SPEC)
registry.register("payload", PAYLOAD_SPEC)
registry.register("registration", REGISTRATION_SPEC)

# Resolve dependencies for a pipeline
available_steps = ["data_loading", "preprocessing", "training", "packaging", "payload", "registration"]
resolved_dependencies = resolver.resolve_all_dependencies(available_steps)

# Get resolution report
report = resolver.get_resolution_report(available_steps)
print(f"Resolution rate: {report['resolution_summary']['resolution_rate']:.2%}")
```

## Dependency Types

The system supports the following dependency types:

- **MODEL_ARTIFACTS**: Trained model files and artifacts
- **TRAINING_DATA**: Data used for model training  
- **PROCESSING_OUTPUT**: General processing step outputs
- **HYPERPARAMETERS**: Model hyperparameter configurations
- **PAYLOAD_SAMPLES**: Sample payloads for model testing
- **CUSTOM_PROPERTY**: Custom or specialized properties

## Property Path Formats

The system supports multiple property path formats for accessing step outputs:

1. **Name-based**: `properties.ProcessingOutputConfig.Outputs['OutputName'].S3Output.S3Uri`
2. **Index-based**: `properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`
3. **Direct attribute**: `properties.ModelArtifacts.S3ModelArtifacts`

## Semantic Matching

The semantic matcher uses various techniques to match dependency names:

- **Exact matching**: Direct string matches
- **Substring matching**: Partial string matches
- **Keyword matching**: Semantic keyword overlap
- **Similarity scoring**: Fuzzy string matching

## Configuration

The system is highly configurable:

- **Compatibility thresholds**: Minimum confidence scores for matches
- **Type compatibility matrix**: Define which types are compatible
- **Semantic keywords**: Customize keywords for better matching
- **Property path registration**: Register custom property paths

## Benefits

1. **Reduced Manual Configuration**: Automatic dependency resolution eliminates manual wiring
2. **Improved Maintainability**: Declarative specifications are easier to maintain
3. **Better Error Detection**: Early detection of incompatible dependencies
4. **Enhanced Flexibility**: Easy to add new step types and modify existing ones
5. **Intelligent Matching**: Semantic matching reduces configuration errors

## Future Enhancements

- **Machine Learning-based Matching**: Use ML models for better semantic matching
- **Dynamic Property Discovery**: Automatically discover available properties
- **Dependency Validation**: Runtime validation of resolved dependencies
- **Visual Dependency Graph**: Interactive visualization of dependency relationships
