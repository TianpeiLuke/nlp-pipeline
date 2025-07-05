# Dependency Resolver

## Overview
The Dependency Resolver automatically analyzes pipeline step specifications to determine execution order and data flow dependencies. It uses semantic matching and constraint validation to build valid pipeline execution graphs.

## Core Functionality

### Dependency Analysis
- **Input/Output Matching** - Matches step outputs to compatible inputs
- **Semantic Compatibility** - Uses semantic tags for intelligent matching
- **Constraint Validation** - Ensures data format and type compatibility
- **Circular Dependency Detection** - Prevents invalid dependency cycles

### Resolution Algorithms
- **Topological Sorting** - Determines valid execution order
- **Graph Construction** - Builds dependency graphs from specifications
- **Conflict Resolution** - Handles multiple compatible matches
- **Optimization** - Minimizes data transfer and processing overhead

## Key Classes

### DependencyResolver
Main resolver class that orchestrates dependency analysis.

```python
class DependencyResolver:
    def __init__(self, registry: SpecificationRegistry):
        self.registry = registry
        self.semantic_matcher = SemanticMatcher()
        
    def resolve_dependencies(self, step_names: List[str]) -> DependencyGraph:
        """Resolve dependencies for given steps"""
        
    def build_execution_order(self, dependencies: DependencyGraph) -> List[str]:
        """Determine optimal execution order"""
        
    def validate_dependencies(self, dependencies: DependencyGraph) -> ValidationResult:
        """Validate dependency graph for consistency"""
```

### DependencyGraph
Represents the dependency relationships between pipeline steps.

```python
@dataclass
class DependencyGraph:
    nodes: Dict[str, StepNode]
    edges: List[DependencyEdge]
    execution_order: List[str]
    
class StepNode:
    step_name: str
    specification: BaseSpecification
    dependencies: List[str]
    dependents: List[str]
    
class DependencyEdge:
    source_step: str
    target_step: str
    output_name: str
    input_name: str
    compatibility_score: float
```

## Usage Examples

### Basic Dependency Resolution
```python
from src.pipeline_deps import DependencyResolver, SpecificationRegistry

# Create registry with step specifications
registry = SpecificationRegistry()
registry.register_specification("data_loading", DATA_LOADING_SPEC)
registry.register_specification("preprocessing", PREPROCESSING_SPEC)
registry.register_specification("training", TRAINING_SPEC)

# Resolve dependencies
resolver = DependencyResolver(registry)
dependency_graph = resolver.resolve_dependencies([
    "data_loading", "preprocessing", "training"
])

# Get execution order
execution_order = resolver.build_execution_order(dependency_graph)
print(f"Execution order: {execution_order}")
# Output: ['data_loading', 'preprocessing', 'training']
```

### Advanced Resolution with Options
```python
# Resolve with multiple compatible steps
dependency_graph = resolver.resolve_dependencies([
    "data_loading_training",
    "preprocessing_training", 
    "preprocessing_validation",
    "xgboost_training"
])

# Analyze dependency relationships
for edge in dependency_graph.edges:
    print(f"{edge.source_step}.{edge.output_name} -> {edge.target_step}.{edge.input_name}")
    print(f"Compatibility: {edge.compatibility_score}")
```

### Validation and Error Handling
```python
# Validate dependency graph
validation_result = resolver.validate_dependencies(dependency_graph)

if not validation_result.is_valid:
    print("Dependency validation failed:")
    for error in validation_result.errors:
        print(f"  - {error}")
        
    print("Warnings:")
    for warning in validation_result.warnings:
        print(f"  - {warning}")
```

## Resolution Process

### 1. Specification Loading
```python
# Load step specifications from registry
specifications = []
for step_name in step_names:
    spec = registry.get_specification(step_name)
    specifications.append(spec)
```

### 2. Compatibility Analysis
```python
# Find compatible input/output pairs
compatible_pairs = []
for source_spec in specifications:
    for target_spec in specifications:
        if source_spec != target_spec:
            matches = semantic_matcher.find_matches(
                source_spec.outputs, 
                target_spec.inputs
            )
            compatible_pairs.extend(matches)
```

### 3. Graph Construction
```python
# Build dependency graph
graph = DependencyGraph()
for spec in specifications:
    graph.add_node(StepNode(spec.step_name, spec))
    
for match in compatible_pairs:
    graph.add_edge(DependencyEdge(
        source_step=match.source_step,
        target_step=match.target_step,
        output_name=match.output_name,
        input_name=match.input_name,
        compatibility_score=match.score
    ))
```

### 4. Execution Order Determination
```python
# Topological sort for execution order
execution_order = topological_sort(graph)
if not execution_order:
    raise CircularDependencyError("Circular dependency detected")
```

## Semantic Matching

### Compatibility Scoring
The resolver uses semantic matching to score compatibility between outputs and inputs:

```python
def calculate_compatibility_score(output_spec: DataSpecification, 
                                input_spec: DataSpecification) -> float:
    score = 0.0
    
    # Data type compatibility
    if output_spec.data_type == input_spec.data_type:
        score += 0.4
    elif are_compatible_types(output_spec.data_type, input_spec.data_type):
        score += 0.2
        
    # Format compatibility  
    if output_spec.format == input_spec.format:
        score += 0.3
    elif are_compatible_formats(output_spec.format, input_spec.format):
        score += 0.1
        
    # Semantic tag overlap
    tag_overlap = len(set(output_spec.semantic_tags) & set(input_spec.semantic_tags))
    score += min(0.3, tag_overlap * 0.1)
    
    return score
```

### Matching Thresholds
- **Perfect Match**: Score >= 0.9 (exact type, format, and semantic alignment)
- **Good Match**: Score >= 0.7 (compatible with minor conversions)
- **Acceptable Match**: Score >= 0.5 (requires data transformation)
- **Poor Match**: Score < 0.5 (not recommended, may require manual intervention)

## Error Handling

### Common Errors
```python
class DependencyResolutionError(Exception):
    """Base class for dependency resolution errors"""
    
class CircularDependencyError(DependencyResolutionError):
    """Raised when circular dependencies are detected"""
    
class IncompatibleSpecificationError(DependencyResolutionError):
    """Raised when specifications are incompatible"""
    
class MissingDependencyError(DependencyResolutionError):
    """Raised when required dependencies are missing"""
```

### Error Recovery
```python
try:
    dependency_graph = resolver.resolve_dependencies(step_names)
except CircularDependencyError as e:
    # Attempt to break cycles by removing lowest-scored edges
    dependency_graph = resolver.resolve_with_cycle_breaking(step_names)
except IncompatibleSpecificationError as e:
    # Suggest compatible alternatives
    suggestions = resolver.suggest_compatible_steps(e.incompatible_step)
    print(f"Consider using: {suggestions}")
```

## Integration Points

### With Pipeline Builder
```python
from src.pipeline_builder import PipelineBuilderTemplate

class PipelineBuilderTemplate:
    def __init__(self, dependency_resolver: DependencyResolver):
        self.resolver = dependency_resolver
        
    def build_pipeline(self, step_names: List[str]) -> Pipeline:
        # Resolve dependencies automatically
        dependencies = self.resolver.resolve_dependencies(step_names)
        execution_order = self.resolver.build_execution_order(dependencies)
        
        # Build pipeline in dependency order
        return self._build_ordered_pipeline(execution_order, dependencies)
```

### With Specification Registry
```python
# Resolver uses registry for specification lookup
resolver = DependencyResolver(registry)

# Registry can trigger re-resolution when specifications change
registry.on_specification_updated = lambda: resolver.invalidate_cache()
```

## Performance Optimization

### Caching
```python
class DependencyResolver:
    def __init__(self):
        self._resolution_cache = {}
        self._compatibility_cache = {}
        
    def resolve_dependencies(self, step_names: List[str]) -> DependencyGraph:
        cache_key = tuple(sorted(step_names))
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
            
        # Perform resolution
        result = self._resolve_dependencies_impl(step_names)
        self._resolution_cache[cache_key] = result
        return result
```

### Parallel Processing
```python
# Compatibility analysis can be parallelized
from concurrent.futures import ThreadPoolExecutor

def analyze_compatibility_parallel(self, specifications: List[BaseSpecification]):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for source_spec in specifications:
            for target_spec in specifications:
                if source_spec != target_spec:
                    future = executor.submit(
                        self.semantic_matcher.find_matches,
                        source_spec.outputs,
                        target_spec.inputs
                    )
                    futures.append(future)
        
        # Collect results
        matches = []
        for future in futures:
            matches.extend(future.result())
        return matches
```

## Best Practices

1. **Specification Quality** - Ensure specifications have rich semantic tags
2. **Compatibility Thresholds** - Set appropriate thresholds for your use case
3. **Error Handling** - Implement robust error handling and recovery
4. **Performance** - Use caching for repeated resolutions
5. **Validation** - Always validate resolved dependencies before execution
6. **Monitoring** - Log resolution decisions for debugging and optimization
