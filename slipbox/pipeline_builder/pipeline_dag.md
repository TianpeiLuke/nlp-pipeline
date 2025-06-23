# Pipeline DAG

## Overview

The Pipeline DAG (Directed Acyclic Graph) is a fundamental component of the pipeline builder template system. It represents the structure of a pipeline as a directed acyclic graph, where nodes are pipeline steps and edges represent dependencies between steps.

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

These methods allow for the incremental construction of the DAG. The `add_edge` method automatically adds any missing nodes.

### Querying Dependencies

```python
def get_dependencies(self, node: str) -> List[str]:
    """Return immediate dependencies (parents) of a node."""
```

This method returns the immediate dependencies (parent nodes) of a given node, which is useful for determining what steps need to be completed before a particular step can be executed.

### Topological Sorting

```python
def topological_sort(self) -> List[str]:
    """Return nodes in topological order."""
```

This method returns the nodes in topological order, which is critical for determining the execution order of pipeline steps. A topological ordering ensures that for every directed edge (u, v), node u comes before node v in the ordering.

## Importance of Topological Ordering

Topological ordering is crucial for pipeline execution because:

1. **Dependency Resolution**: It ensures that all dependencies of a step are executed before the step itself.
2. **Parallel Execution**: It identifies steps that can be executed in parallel (steps that don't depend on each other).
3. **Cycle Detection**: It helps detect cycles in the graph, which would make the pipeline impossible to execute.

In the context of the pipeline builder template, topological ordering is used to determine the order in which steps should be instantiated and connected.

## Internal Data Structures

The PipelineDAG maintains several internal data structures:

1. **nodes**: A list of all nodes in the graph.
2. **edges**: A list of all edges in the graph as (source, destination) tuples.
3. **adj_list**: An adjacency list representing outgoing edges from each node.
4. **reverse_adj**: An adjacency list representing incoming edges to each node.

These data structures enable efficient traversal and querying of the graph.

## Usage in Pipeline Builder Template

The PipelineDAG is a core component of the [Pipeline Builder Template](pipeline_builder_template.md), which uses it to:

1. Define the structure of the pipeline
2. Determine the order in which steps should be instantiated
3. Establish connections between steps based on their dependencies

See the [Pipeline Builder Template](pipeline_builder_template.md) documentation for more details on how the DAG is used in the template system.

## Related

- [Pipeline Builder Template](pipeline_builder_template.md)
- [Template Implementation](template_implementation.md)
- [Pipeline Steps](../pipeline_steps/README.md)
- [Pipeline Examples](pipeline_examples.md)
