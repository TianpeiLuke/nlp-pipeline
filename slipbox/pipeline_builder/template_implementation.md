# Template-Based Pipeline Implementation

This document explains the template-based implementation of pipelines in the NLP Pipeline framework and how they handle placeholder variables and step connections.

## Overview

The template-based implementations in `src/pipeline_builder/` (such as `template_pipeline_pytorch_end_to_end.py`, `template_pipeline_pytorch_model_registration.py`, and `template_pipeline_xgboost_end_to_end.py`) use the `PipelineBuilderTemplate` class to automatically handle connections between pipeline steps. This is in contrast to traditional implementations that directly create and connect pipeline steps, explicitly passing outputs from one step as inputs to subsequent steps.

## Handling Placeholder Variables

In the original implementation, placeholder variables like these are used to pass outputs between steps:

```python
# Example 1: Accessing processing output
dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri

# Example 2: Accessing model artifacts
dependency_step.properties.ModelArtifacts.S3ModelArtifacts
```

The template-based implementation handles these placeholders through several mechanisms:

### 1. DAG-Based Message Propagation

The template uses a Directed Acyclic Graph (DAG) to represent the pipeline structure. Steps are connected through edges in the DAG, and the template automatically propagates messages (outputs) from one step to the next based on this structure.

```python
def _create_pipeline_dag(self) -> PipelineDAG:
    """Create the DAG structure for the pipeline."""
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("train_data_load")
    dag.add_node("train_preprocess")
    # ...
    
    # Add edges
    dag.add_edge("train_data_load", "train_preprocess")
    dag.add_edge("train_preprocess", "xgboost_train")
    # ...
    
    return dag
```

### 2. Automatic Property Extraction

The template includes methods that automatically extract common properties from steps:

```python
# In PipelineBuilderTemplate._extract_common_outputs
if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
    try:
        # Try to get the first output
        s3_uri = prev_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
        # ...
    except (AttributeError, IndexError) as e:
        # ...
```

### 3. Step-Specific Handlers

The template includes specialized handlers for different step types:

```python
# In PipelineBuilderTemplate._extract_inputs_from_dependencies
if step_type == "TabularPreprocessingStep":
    self._handle_tabular_preprocessing_step(kwargs, step_name, dependency_steps)
elif step_type == "PytorchTrainingStep" or step_type == "XGBoostTrainingStep":
    self._handle_training_step(kwargs, step_name, dependency_steps, step_type)
# ...
```

### 4. Pattern Matching

The template uses pattern matching to connect inputs to outputs when direct name matches aren't available:

```python
# In PipelineBuilderTemplate._match_inputs_to_outputs
common_patterns = {
    "model": ["model", "model_data", "model_artifacts", "model_path"],
    "data": ["data", "dataset", "input_data", "training_data"],
    "output": ["output", "result", "artifacts", "s3_uri"]
}
```

## Advantages of the Template-Based Approach

1. **Reduced Boilerplate**: The template eliminates the need to write repetitive code for connecting steps.

2. **Automatic Placeholder Handling**: The template automatically handles placeholder variables, reducing the risk of errors.

3. **Declarative Pipeline Definition**: The pipeline structure is defined declaratively through the DAG, making it easier to understand and modify.

4. **Separation of Concerns**: The template separates the pipeline structure (DAG) from the step implementations, making the code more modular and maintainable.

5. **Reusable Components**: The template can be reused for different pipelines, promoting code reuse.

## Example Usage

The template-based implementations follow two main patterns:

### 1. Function-Based Implementation

Used in pipelines like `template_pipeline_pytorch_end_to_end.py` and `template_pipeline_xgboost_end_to_end.py`:

```python
# Create the pipeline using the template function
pipeline = create_pipeline_from_template(
    config_path="path/to/config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    notebook_root=Path.cwd()
)
```

### 2. Class-Based Implementation

Used in pipelines like `template_pipeline_pytorch_model_registration.py`:

```python
# Create the pipeline builder
builder = TemplatePytorchPipelineBuilder(
    config_path="path/to/config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    notebook_root=Path.cwd()
)

# Generate the pipeline with a model path
pipeline = builder.generate_pipeline("s3://bucket/path/to/model.tar.gz")
```

See the individual template implementations in `src/pipeline_builder/` for complete examples.

## Related

- [Pipeline DAG](pipeline_dag.md)
- [Pipeline Builder Template](pipeline_builder_template.md)
- [Pipeline Examples](pipeline_examples.md)
- [Pipeline Steps](../pipeline_steps/README.md)
