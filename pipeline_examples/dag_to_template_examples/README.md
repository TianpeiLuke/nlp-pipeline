# DAG to Pipeline Examples

This directory contains examples of using the DAG-to-pipeline approach instead of the template-based approach for creating SageMaker pipelines.

## Available Examples

1. **Simplified DAG Example**: The [`simplified_dag_example.py`](./simplified_dag_example.py) demonstrates the core concept of DAG-based pipeline structure without any dependency issues.

2. **XGBoost Train-Calibrate-Evaluate DAG**: The [`xgboost_train_calibrate_evaluate_dag.py`](./xgboost_train_calibrate_evaluate_dag.py) is a full implementation that converts a template-based pipeline to a DAG-to-pipeline approach.

3. **Test DAG Conversion**: The [`test_dag_conversion.py`](./test_dag_conversion.py) shows how to test and validate DAG structure before conversion.

4. **Visualize DAG**: The [`visualize_dag.py`](./visualize_dag.py) creates visual representations of pipeline DAGs using Graphviz.

## How to Run the Examples

### Simplified Example (Recommended Starting Point)

```bash
python pipeline_examples/dag_to_template_examples/simplified_dag_example.py
```

This example shows the DAG structure and explains the conversion concept without requiring complex dependencies. It's the best place to start understanding the approach.

### Full Implementation (Requires Configuration File)

```bash
python pipeline_examples/dag_to_template_examples/xgboost_train_calibrate_evaluate_dag.py \
  --config-path /path/to/config.json \
  --role arn:aws:iam::123456789012:role/SageMakerRole
```

## Overview

The traditional approach in this codebase uses template classes that inherit from `PipelineTemplateBase` and implement required methods like `_create_pipeline_dag()`, `_create_config_map()`, etc. While this approach works well, it requires creating new template classes for each pipeline pattern.

The DAG-to-pipeline approach provides a more direct API that allows you to:
1. Create a DAG structure
2. Convert it directly to a pipeline using the `dag_to_pipeline_template` function or `PipelineDAGConverter` class

This approach is more flexible and easier to understand, especially for users who are new to the codebase.

## Examples

### XGBoost Train-Calibrate-Evaluate Pipeline

The [`xgboost_train_calibrate_evaluate_dag.py`](./xgboost_train_calibrate_evaluate_dag.py) example shows how to convert a template-based pipeline to a DAG-to-pipeline approach. It:

1. Creates a DAG with the same structure as the template-based pipeline
2. Uses the `PipelineDAGConverter` to convert it to a SageMaker pipeline
3. Includes validation and execution document handling

### Usage

To use the example:

```bash
python pipeline_examples/dag_to_template_examples/xgboost_train_calibrate_evaluate_dag.py \
  --config-path /path/to/config.json \
  --role arn:aws:iam::123456789012:role/SageMakerRole
```

For a preview of the node resolution without creating the pipeline:

```bash
python pipeline_examples/dag_to_template_examples/xgboost_train_calibrate_evaluate_dag.py \
  --config-path /path/to/config.json \
  --role arn:aws:iam::123456789012:role/SageMakerRole \
  --preview-only
```

## Benefits of the DAG Approach

1. **Simplified API**: No need to create template classes for each pipeline pattern
2. **Explicit Flow**: The DAG structure and its conversion are explicitly defined
3. **Better Validation**: Built-in validation capabilities from the converter
4. **Enhanced Debugging**: More visibility into node resolution and configuration mapping
5. **Easier to Understand**: More intuitive for users who are new to the codebase

## Key Components

The DAG-to-pipeline approach uses these main components:

- **PipelineDAG**: Represents the directed acyclic graph of pipeline steps
- **PipelineDAGConverter**: Converts a DAG to a SageMaker pipeline
- **ValidationResult**: Provides validation information about the DAG
- **ResolutionPreview**: Provides insight into how DAG nodes will be resolved to configurations
- **ConversionReport**: Provides a detailed report of the conversion process

## Template vs. DAG Comparison

| Template Approach | DAG Approach |
|------------------|--------------|
| Requires creating new template classes | Allows direct API usage |
| Implementation spread across methods | Linear flow from DAG to pipeline |
| Implicit DAG creation | Explicit DAG creation |
| Less visibility into node resolution | Better visibility with preview and validation |
| Good for standard pipelines | Flexible for custom pipelines |

## Implementation Notes

### Handling Import Dependencies

If you encounter circular import errors when running the full examples, you can use the simplified example to understand the core concepts without the dependency issues. The simplified example focuses on the DAG structure and explains how it would integrate with the configuration and conversion process.

The full implementation in `xgboost_train_calibrate_evaluate_dag.py` might require modifications to work with different codebase states due to complex import dependencies. In particular:

1. The `pipeline_api` module might have circular imports with other modules.
2. Some methods referenced in the examples might not be available in all versions of the codebase.

### Visualization

To generate a visual representation of the DAG structure, you need to have Graphviz installed:

```bash
pip install graphviz
# Also install the system Graphviz package
# On Ubuntu: sudo apt-get install graphviz
# On macOS: brew install graphviz
```

Then run:

```bash
python pipeline_examples/dag_to_template_examples/visualize_dag.py --output dag_visualization.png
```
