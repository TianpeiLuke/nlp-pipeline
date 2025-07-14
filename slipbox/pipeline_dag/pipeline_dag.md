# Pipeline DAG

## Overview

The Pipeline DAG (Directed Acyclic Graph) is a fundamental component of the pipeline builder template system. It represents the structure of a pipeline as a directed acyclic graph, where nodes are pipeline steps and edges represent dependencies between steps. This graph structure enables automatic dependency resolution and proper step ordering.

## Class Definition

```python
class PipelineDAG:
    """
    Represents a pipeline topology as a directed acyclic graph (DAG).
    Each node is a step name; edges define dependencies.
    """
    def __init__(self, nodes: Optional[List[str]] = None, edges: Optional[List[tuple]] = None):
        """
        nodes: List of step names (str)
        edges: List of (from_step, to_step) tuples
        """
```

## Key Methods

### Initialization

The PipelineDAG can be initialized in two ways:

1. With a list of nodes and edges:
   ```python
   nodes = ["step1", "step2", "step3"]
   edges = [("step1", "step2"), ("step2", "step3")]
   dag = PipelineDAG(nodes=nodes, edges=edges)
   ```

2. Empty initialization with incremental addition:
   ```python
   dag = PipelineDAG()
   dag.add_node("step1")
   dag.add_node("step2")
   dag.add_edge("step1", "step2")
   ```

### Adding Nodes and Edges

```python
def add_node(self, node: str) -> None:
    """Add a single node to the DAG."""
    
def add_edge(self, src: str, dst: str) -> None:
    """Add a directed edge from src to dst."""
```

These methods allow for the incremental construction of the DAG. The `add_edge` method automatically adds any missing nodes, making it convenient to build pipelines with a single method.

### Querying Dependencies

```python
def get_dependencies(self, node: str) -> List[str]:
    """Return immediate dependencies (parents) of a node."""
```

This method returns the immediate dependencies (parent nodes) of a given node. In the context of pipeline building, this is used to determine which steps provide inputs to a given step.

### Topological Sorting

```python
def topological_sort(self) -> List[str]:
    """Return nodes in topological order."""
```

This method returns the nodes in topological order, which is critical for determining the execution order of pipeline steps. A topological ordering ensures that for every directed edge (u, v), node u comes before node v in the ordering.

If the graph contains cycles, this method raises a `ValueError` with the message "DAG has cycles or disconnected nodes". This ensures that pipeline definitions are valid directed acyclic graphs.

## Internal Data Structures

The PipelineDAG maintains several internal data structures:

1. **nodes**: A list of all nodes in the graph.
2. **edges**: A list of all edges in the graph as (source, destination) tuples.
3. **adj_list**: An adjacency list representing outgoing edges from each node (forward graph).
4. **reverse_adj**: An adjacency list representing incoming edges to each node (backward graph).

These data structures enable efficient traversal and querying of the graph in both forward and backward directions.

## Importance of Topological Ordering

Topological ordering is crucial for pipeline execution because:

1. **Dependency Resolution**: It ensures that all dependencies of a step are executed before the step itself.
2. **Parallel Execution**: It identifies steps that can be executed in parallel (steps that don't depend on each other).
3. **Cycle Detection**: It helps detect cycles in the graph, which would make the pipeline impossible to execute.

The implementation uses Kahn's algorithm for topological sorting, which has the following steps:
1. Calculate in-degree for each node (number of incoming edges)
2. Start with nodes that have in-degree zero (source nodes)
3. Process nodes in a queue, decrementing in-degree of neighboring nodes
4. Add nodes to the result list in the order they are processed

## Usage with Specification-Driven Dependency Resolution

The PipelineDAG works in conjunction with the specification-driven dependency resolution system:

1. **Graph Structure**: The DAG defines which steps can potentially provide inputs to other steps.
2. **Specification Matching**: For each edge in the DAG, the dependency resolver analyzes step specifications to match outputs to inputs.
3. **Compatibility Scoring**: The resolver calculates compatibility scores between dependencies and outputs based on types, names, and semantic similarity.
4. **Message Propagation**: The results of dependency resolution are propagated from source nodes to sink nodes following the DAG structure.

This combination of graph structure and intelligent matching provides a powerful system for automatically connecting pipeline steps.

## Usage in PipelineAssembler

The PipelineAssembler uses the PipelineDAG in the following way:

```python
# Inside PipelineAssembler.generate_pipeline:

# Topological sort to determine build order
try:
    build_order = self.dag.topological_sort()
    logger.info(f"Build order: {build_order}")
except ValueError as e:
    logger.error(f"Error in topological sort: {e}")
    raise ValueError(f"Failed to determine build order: {e}") from e

# Instantiate steps in topological order
for step_name in build_order:
    try:
        step = self._instantiate_step(step_name)
        self.step_instances[step_name] = step
    except Exception as e:
        logger.error(f"Error instantiating step {step_name}: {e}")
        raise ValueError(f"Failed to instantiate step {step_name}: {e}") from e
```

## Example: Creating a Complex Pipeline DAG

```python
# Create an empty DAG
dag = PipelineDAG()

# Data loading and preprocessing branch
dag.add_node("data_loading")
dag.add_node("preprocessing")
dag.add_edge("data_loading", "preprocessing")

# Model training branch
dag.add_node("training")
dag.add_edge("preprocessing", "training")

# Model evaluation branch
dag.add_node("evaluation")
dag.add_edge("training", "evaluation")
dag.add_edge("preprocessing", "evaluation")  # Evaluation also needs raw data

# Model registration branch
dag.add_node("registration")
dag.add_edge("evaluation", "registration")

# Determine build order
build_order = dag.topological_sort()
print(f"Build order: {build_order}")
# Output: ['data_loading', 'preprocessing', 'training', 'evaluation', 'registration']
```

## Best Practices

1. **Step Naming**: Use descriptive names for nodes that reflect their function
2. **Direct Dependencies Only**: Create edges only between steps with direct dependencies
3. **Avoid Redundant Dependencies**: If A→B→C, don't add A→C unless C directly depends on A
4. **Validation**: Always validate the DAG by calling topological_sort() to detect cycles
5. **Modular Construction**: Build complex DAGs by combining smaller sub-graphs
6. **DAG Visualization**: Consider visualizing the DAG for complex pipelines using tools like Graphviz

## Related Documentation

- [Pipeline Builder Template](../pipeline_builder/pipeline_builder_template.md)
- [Template Implementation](../pipeline_builder/template_implementation.md)
- [Pipeline Deps: Dependency Resolver](../pipeline_deps/dependency_resolver.md)
- [Pipeline Steps](../pipeline_steps/README.md)
- [Pipeline Examples](../pipeline_builder/pipeline_examples.md)
