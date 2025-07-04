# Pipeline Registry

## What is the Purpose of Pipeline Registry?

Pipeline Registries serve as **isolated dependency management containers** that enable multiple pipelines to coexist without interference. They represent the evolution from global, singleton registries to pipeline-scoped registries that respect pipeline boundaries.

## Core Purpose

Pipeline Registries provide a **pipeline-scoped approach to dependency management** rather than a global one, enabling:

1. **Pipeline Isolation** - Prevent cross-pipeline interference in multi-pipeline environments
2. **Shared Intra-Pipeline Context** - Maintain shared context within a single pipeline
3. **Centralized Management** - Manage multiple registries through a unified interface
4. **Backward Compatibility** - Support legacy code that expects a registry
5. **Pipeline-Aware Resolution** - Enable dependency resolution within pipeline boundaries

## Key Features

### 1. Pipeline-Scoped Registries

Each pipeline gets its own isolated registry instance:

```python
# Create pipeline-specific registries
training_registry = PipelineRegistry("training_pipeline")
inference_registry = PipelineRegistry("inference_pipeline")

# Register specifications in their respective registries
training_registry.register("training_step", TRAINING_SPEC)
inference_registry.register("inference_step", INFERENCE_SPEC)

# No cross-contamination between registries
assert inference_registry.get_specification("training_step") is None
```

### 2. Registry Manager

Centralized management of multiple pipeline registries:

```python
# Get registry for a specific pipeline
pipeline_registry = registry_manager.get_pipeline_registry("my_pipeline")

# List all registered pipelines
pipeline_names = registry_manager.list_pipeline_registries()

# Clear a specific pipeline registry
registry_manager.clear_pipeline_registry("old_pipeline")

# Access the default registry for shared specifications
default_registry = registry_manager.get_default_registry()
```

### 3. Automatic Pipeline Builder Integration

Decorator-based integration with pipeline builders:

```python
@integrate_with_pipeline_builder
class MyPipelineBuilder:
    def __init__(self, base_config):
        self.base_config = base_config
        # self.registry is automatically set to the pipeline-specific registry
        # based on base_config.pipeline_name
    
    def build(self):
        # Use self.registry for dependency resolution
        step = self.create_step("my_step")
        spec = self.registry.get_specification("my_step")
```

### 4. Convenience Functions

Simplified access to registries:

```python
# Get registry for a specific pipeline
pipeline_registry = get_pipeline_registry("my_pipeline")

# Get the default registry
default_registry = get_default_registry()
```

## Integration with Other Components

### With Dependency Resolver
```python
# Create resolver with pipeline-specific registry
resolver = UnifiedDependencyResolver(registry=get_pipeline_registry("my_pipeline"))

# Resolve dependencies within pipeline scope
resolved = resolver.resolve_all_dependencies(available_steps)
```

### With Step Builders
```python
@integrate_with_pipeline_builder
class XGBoostTrainingStepBuilder:
    def build(self):
        # Access pipeline-specific registry
        spec = self.registry.get_specification("xgboost_training")
        # Build step using specification
```

### With Pipeline Builder
```python
@integrate_with_pipeline_builder
class PipelineBuilder:
    def __init__(self, base_config):
        self.base_config = base_config
        # self.registry is automatically set
    
    def build_pipeline(self):
        # Use self.registry for all dependency resolution
        for step_name in self.steps:
            spec = self.registry.get_specification(step_name)
            # Build step using specification
```

## Strategic Value

Pipeline Registries enable:

1. **Multi-Pipeline Support**: Run multiple pipelines in the same environment without interference
2. **Isolation**: Changes to one pipeline don't affect others
3. **Contextual Awareness**: Steps know which pipeline they belong to
4. **Simplified Management**: Centralized management of multiple registries
5. **Backward Compatibility**: Legacy code continues to work with minimal changes
6. **Cleaner Architecture**: Respects pipeline boundaries and separation of concerns

## Example Usage

```python
# Create and configure pipeline registries
training_registry = get_pipeline_registry("training_pipeline")
inference_registry = get_pipeline_registry("inference_pipeline")

# Register step specifications in their respective registries
training_registry.register("data_loading", DATA_LOADING_SPEC)
training_registry.register("preprocessing", PREPROCESSING_SPEC)
training_registry.register("training", TRAINING_SPEC)

inference_registry.register("model_loading", MODEL_LOADING_SPEC)
inference_registry.register("inference", INFERENCE_SPEC)

# Create pipeline builders with automatic registry integration
@integrate_with_pipeline_builder
class TrainingPipelineBuilder:
    def __init__(self, config):
        self.base_config = config
        self.base_config.pipeline_name = "training_pipeline"
        # self.registry is automatically set to training_registry
    
    def build(self):
        # Use self.registry for dependency resolution
        data_step = self.create_step("data_loading")
        preprocessing_step = self.create_step("preprocessing")
        training_step = self.create_step("training")
        
        # All steps share the same registry
        return [data_step, preprocessing_step, training_step]

# Create and run the pipeline
builder = TrainingPipelineBuilder(config)
pipeline = builder.build()
```

Pipeline Registries are the **architectural foundation** for multi-pipeline environments, enabling isolation between pipelines while maintaining shared context within each pipeline. They work seamlessly with other components like [Step Specifications](step_specification.md), [Dependency Resolvers](dependency_resolver.md), and [Pipeline Builders](pipeline_template_builder_v2.md) to create a robust, scalable pipeline architecture.
