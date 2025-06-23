# Pipeline Builder Template

## Overview

The Pipeline Builder Template is a powerful abstraction that simplifies the creation of complex SageMaker pipelines. It uses a declarative approach to define pipeline structure and automatically handles the connections between steps, eliminating the need for manual wiring of inputs and outputs.

## Class Definition

```python
class PipelineBuilderTemplate:
    """
    Generic pipeline builder using a DAG and step builders.
    """
    def __init__(
        self,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_builder_map: Dict[str, Type[StepBuilderBase]],
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List[ParameterString]] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        dag: PipelineDAG instance
        config_map: Mapping from step name to config instance
        step_builder_map: Mapping from step type to StepBuilderBase subclass
        """
```

## Key Components

### 1. Pipeline DAG

The [Pipeline DAG](pipeline_dag.md) defines the structure of the pipeline, including the steps and their dependencies. It is used to determine the order in which steps should be instantiated and connected.

### 2. Config Map

The config map is a dictionary that maps step names to configuration instances. Each configuration instance contains the parameters needed to create a specific step.

### 3. Step Builder Map

The step builder map is a dictionary that maps step types to step builder classes. Each step builder class knows how to create a specific type of step.

## Key Methods

### Collecting Step I/O Requirements

```python
def _collect_step_io_requirements(self) -> None:
    """
    Collect input and output requirements from all steps.
    
    This method initializes the step builders for all steps in the DAG
    and collects their input requirements and output properties.
    """
```

This method initializes the step builders for all steps in the DAG and collects their input requirements and output properties. This information is used by the message passing algorithm to connect steps.

### Message Propagation

```python
def _propagate_messages(self) -> None:
    """
    Propagate messages between steps based on the DAG topology.
    
    This method implements a message passing algorithm where each step:
    1. Collects messages from its dependencies (previous steps)
    2. Verifies that its input requirements can be satisfied
    3. Prepares its own output messages for downstream steps
    """
```

This method implements a message passing algorithm that propagates information about outputs from one step to inputs of subsequent steps. It is a key part of the automatic connection mechanism.

### Step Instantiation

```python
def _instantiate_step(self, step_name: str) -> Step:
    """
    Instantiate a pipeline step with appropriate inputs from dependencies.
    
    This method dynamically determines the inputs required by each step based on:
    1. The step's configuration
    2. The outputs available from dependency steps
    3. The step type and its expected input parameters
    """
```

This method creates a SageMaker pipeline step with the appropriate inputs from its dependencies. It uses the information collected during the message propagation phase to determine what outputs from previous steps should be passed as inputs to the current step.

### Pipeline Generation

```python
def generate_pipeline(self, pipeline_name: str) -> Pipeline:
    """
    Build and return a SageMaker Pipeline object.
    """
```

This method generates a SageMaker pipeline by:
1. Collecting input/output requirements from all steps
2. Propagating messages between steps
3. Instantiating steps in topological order
4. Creating a SageMaker Pipeline object with the instantiated steps

## Message Passing Algorithm

The message passing algorithm is a core feature of the Pipeline Builder Template. It automatically connects outputs from one step to inputs of subsequent steps, eliminating the need for manual wiring.

### How It Works

1. **Collection Phase**: The template collects input requirements and output properties from all steps.
2. **Propagation Phase**: The template propagates messages between steps based on the DAG topology.
3. **Matching Phase**: The template matches inputs to outputs based on name similarity and common patterns.
4. **Extraction Phase**: The template extracts the actual output values from previous steps and passes them as inputs to subsequent steps.

### Handling Placeholder Variables

The message passing algorithm automatically handles placeholder variables like:

```python
# Example 1: Accessing processing output
dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri

# Example 2: Accessing model artifacts
dependency_step.properties.ModelArtifacts.S3ModelArtifacts
```

It does this through several mechanisms:

1. **Automatic Property Extraction**: The template includes methods that automatically extract common properties from steps.
2. **Step-Specific Handlers**: The template includes specialized handlers for different step types.
3. **Pattern Matching**: The template uses pattern matching to connect inputs to outputs when direct name matches aren't available.

## Importance of Topological Ordering

Topological ordering is crucial for the Pipeline Builder Template because:

1. **Dependency Resolution**: It ensures that all dependencies of a step are instantiated before the step itself.
2. **Message Propagation**: It determines the order in which messages are propagated between steps.
3. **Step Instantiation**: It determines the order in which steps are instantiated.

The template uses the topological sort provided by the [Pipeline DAG](pipeline_dag.md) to determine the order in which steps should be processed.

## Advantages of the Template-Based Approach

1. **Reduced Boilerplate**: The template eliminates the need to write repetitive code for connecting steps.
2. **Automatic Placeholder Handling**: The template automatically handles placeholder variables, reducing the risk of errors.
3. **Declarative Pipeline Definition**: The pipeline structure is defined declaratively through the DAG, making it easier to understand and modify.
4. **Separation of Concerns**: The template separates the pipeline structure (DAG) from the step implementations, making the code more modular and maintainable.
5. **Reusable Components**: The template can be reused for different pipelines, promoting code reuse.

## Usage Example

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("step1")
dag.add_node("step2")
dag.add_edge("step1", "step2")

# Create the config map
config_map = {
    "step1": step1_config,
    "step2": step2_config,
}

# Create the step builder map
step_builder_map = {
    "Step1Type": Step1Builder,
    "Step2Type": Step2Builder,
}

# Create the template
template = PipelineBuilderTemplate(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=sagemaker_session,
    role=role,
)

# Generate the pipeline
pipeline = template.generate_pipeline("my-pipeline")
```

## Related

- [Pipeline DAG](pipeline_dag.md)
- [Template Implementation](template_implementation.md)
- [Pipeline Steps](../pipeline_steps/README.md)
- [Pipeline Examples](pipeline_examples.md)
