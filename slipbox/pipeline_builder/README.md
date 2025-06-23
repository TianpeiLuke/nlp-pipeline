# Pipeline Builder

## Overview

The Pipeline Builder is a template-based system for creating SageMaker pipelines. It provides a declarative approach to defining pipeline structure and automatically handles the connections between steps, eliminating the need for manual wiring of inputs and outputs.

## Key Components

### 1. [Pipeline DAG](pipeline_dag.md)

The Pipeline DAG (Directed Acyclic Graph) represents the structure of a pipeline as a directed acyclic graph, where nodes are pipeline steps and edges represent dependencies between steps. It provides methods for:

- Adding nodes and edges
- Querying dependencies
- Topological sorting

### 2. [Pipeline Builder Template](pipeline_builder_template.md)

The Pipeline Builder Template is the core component that uses the DAG to generate a SageMaker pipeline. It implements a message passing algorithm that automatically connects outputs from one step to inputs of subsequent steps. Key features include:

- Automatic handling of placeholder variables
- Step-specific handlers for different types of steps
- Pattern matching for connecting inputs to outputs

### 3. [Pipeline Examples](pipeline_examples.md)

The Pipeline Examples document provides an overview of the example pipelines built using the template system, including:

- XGBoost Train-Evaluate E2E Pipeline
- XGBoost End-to-End Pipeline
- XGBoost Data Load and Preprocess Pipeline
- PyTorch End-to-End Pipeline
- PyTorch Model Registration Pipeline

### 4. [Template Implementation](template_implementation.md)

The Template Implementation document explains how the template-based implementation handles placeholder variables like `dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri` through:

- DAG-Based Message Propagation
- Automatic Property Extraction
- Step-Specific Handlers
- Pattern Matching

## Importance of Topological Ordering and Message Passing

### Topological Ordering

Topological ordering is crucial for pipeline execution because:

1. **Dependency Resolution**: It ensures that all dependencies of a step are executed before the step itself.
2. **Parallel Execution**: It identifies steps that can be executed in parallel (steps that don't depend on each other).
3. **Cycle Detection**: It helps detect cycles in the graph, which would make the pipeline impossible to execute.

In the context of the pipeline builder template, topological ordering is used to determine the order in which steps should be instantiated and connected.

### Message Passing Algorithm

The message passing algorithm is a core feature of the Pipeline Builder Template. It automatically connects outputs from one step to inputs of subsequent steps, eliminating the need for manual wiring.

How it works:

1. **Collection Phase**: The template collects input requirements and output properties from all steps.
2. **Propagation Phase**: The template propagates messages between steps based on the DAG topology.
3. **Matching Phase**: The template matches inputs to outputs based on name similarity and common patterns.
4. **Extraction Phase**: The template extracts the actual output values from previous steps and passes them as inputs to subsequent steps.

This algorithm is particularly valuable for handling placeholder variables like:

```python
# Example 1: Accessing processing output
dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri

# Example 2: Accessing model artifacts
dependency_step.properties.ModelArtifacts.S3ModelArtifacts
```

## Benefits of Using the Pipeline Builder

1. **Reduced Boilerplate**: The template eliminates the need to write repetitive code for connecting steps.
2. **Automatic Placeholder Handling**: The template automatically handles placeholder variables, reducing the risk of errors.
3. **Declarative Pipeline Definition**: The pipeline structure is defined declaratively through the DAG, making it easier to understand and modify.
4. **Separation of Concerns**: The template separates the pipeline structure (DAG) from the step implementations, making the code more modular and maintainable.
5. **Reusable Components**: The template can be reused for different pipelines, promoting code reuse.

## Related

- [Pipeline Steps](../pipeline_steps/README.md)
