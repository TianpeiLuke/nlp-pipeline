# Pipeline Examples

## Overview

This document provides an overview of the example pipelines built using the [Pipeline Builder Template](pipeline_builder_template.md). These examples demonstrate how to use the template system to create various types of SageMaker pipelines.

## Available Examples

### 1. PyTorch End-to-End Pipeline

**Source**: `src/pipeline_builder/template_pipeline_pytorch_end_to_end.py`

**Description**: This pipeline performs:
1. Data Loading (for training set)
2. Tabular Preprocessing (for training set)
3. PyTorch Model Training
4. Model Creation
5. Packaging
6. Payload Generation
7. Registration
8. Data Loading (for calibration set)
9. Tabular Preprocessing (for calibration set)

**Key Features**:
- Uses a function-based approach with the template
- Demonstrates a complete end-to-end ML workflow for PyTorch models
- Shows how to use the template's message passing algorithm to automatically connect outputs from one step to inputs of subsequent steps
- Includes helper functions for finding configurations by type and attributes

### 2. PyTorch Model Registration Pipeline

**Source**: `src/pipeline_builder/template_pipeline_pytorch_model_registration.py`

**Description**: This pipeline focuses on the model registration steps:
1. PyTorch Model Creation (using an existing model artifact)
2. Packaging
3. Payload Generation
4. Registration

**Key Features**:
- Uses a class-based approach with the template
- Demonstrates a pipeline focused on model registration without training
- Shows how to validate and prepare model configurations
- Includes error handling and validation for model paths

### 3. XGBoost End-to-End Pipeline

**Source**: `src/pipeline_builder/template_pipeline_xgboost_end_to_end.py`

**Description**: This pipeline performs:
1. Data Loading (for training set)
2. Tabular Preprocessing (for training set)
3. XGBoost Model Training
4. Model Creation
5. Packaging
6. Payload Generation
7. Registration
8. Data Loading (for calibration set)
9. Tabular Preprocessing (for calibration set)

**Key Features**:
- Uses a function-based approach with the template
- Demonstrates a complete end-to-end ML workflow for XGBoost models
- Shows how to create and register a model

### 4. XGBoost Data Load and Preprocess Pipeline

**Source**: `src/pipeline_builder/template_pipeline_xgboost_dataload_preprocess.py`

**Description**: This pipeline focuses on the data preparation steps:
1. Data Loading
2. Tabular Preprocessing

**Key Features**:
- Demonstrates a simpler pipeline focused on data preparation
- Shows how to use the template for specific parts of the ML workflow

## Common Patterns

Across these examples, several common patterns emerge:

1. **DAG Definition**: Each example defines a DAG that represents the structure of the pipeline.
2. **Config Map**: Each example creates a config map that maps step names to configuration instances.
3. **Step Builder Map**: Each example creates a step builder map that maps step types to step builder classes.
4. **Template Instantiation**: Each example creates a PipelineBuilderTemplate instance with the DAG, config map, and step builder map.
5. **Pipeline Generation**: Each example generates a SageMaker pipeline using the template.

## Implementation Details

### DAG Definition

The DAG can be defined in two ways:

1. With a list of nodes and edges:
   ```python
   nodes = ["step1", "step2", "step3"]
   edges = [("step1", "step2"), ("step2", "step3")]
   dag = PipelineDAG(nodes=nodes, edges=edges)
   ```

2. With incremental addition:
   ```python
   dag = PipelineDAG()
   dag.add_node("step1")
   dag.add_node("step2")
   dag.add_edge("step1", "step2")
   ```

### Config Map

The config map is a dictionary that maps step names to configuration instances:

```python
config_map = {
    "step1": step1_config,
    "step2": step2_config,
}
```

### Step Builder Map

The step builder map is a dictionary that maps step types to step builder classes:

```python
step_builder_map = {
    "Step1Type": Step1Builder,
    "Step2Type": Step2Builder,
}
```

### Template Instantiation

The template is instantiated with the DAG, config map, and step builder map:

```python
template = PipelineBuilderTemplate(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=sagemaker_session,
    role=role,
    pipeline_parameters=pipeline_parameters,
    notebook_root=notebook_root
)
```

Additional parameters include:
- `pipeline_parameters`: List of SageMaker pipeline parameters
- `notebook_root`: Root directory of the notebook (for resolving relative paths)

### Pipeline Generation

The pipeline is generated using the template:

```python
pipeline = template.generate_pipeline("my-pipeline")
```

## Benefits of Using Templates

These examples demonstrate the benefits of using the template system:

1. **Reduced Boilerplate**: The template eliminates the need to write repetitive code for connecting steps.
2. **Automatic Placeholder Handling**: The template automatically handles placeholder variables, reducing the risk of errors.
3. **Declarative Pipeline Definition**: The pipeline structure is defined declaratively through the DAG, making it easier to understand and modify.
4. **Separation of Concerns**: The template separates the pipeline structure (DAG) from the step implementations, making the code more modular and maintainable.
5. **Reusable Components**: The template can be reused for different pipelines, promoting code reuse.

## Related

- [Pipeline DAG](pipeline_dag.md)
- [Pipeline Builder Template](pipeline_builder_template.md)
- [Template Implementation](template_implementation.md)
- [Pipeline Steps](../pipeline_steps/README.md)
