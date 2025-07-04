# Dependency Resolver

## What is the Purpose of Dependency Resolver?

The Dependency Resolver serves as the **intelligent matching engine** that automatically connects pipeline steps by analyzing their [step specifications](step_specification.md) and finding compatible input-output relationships. It transforms manual, error-prone dependency wiring into intelligent, automated resolution based on semantic compatibility scoring.

## Core Purpose

The Dependency Resolver provides **intelligent automation for pipeline dependency management** that:

1. **Automatic Dependency Resolution** - Eliminate manual property path wiring between steps
2. **Semantic Compatibility Matching** - Use intelligent scoring to find best matches
3. **Type Safety Validation** - Ensure data type compatibility between connected steps
4. **Multi-Criteria Scoring** - Consider type, semantics, keywords, and source compatibility
5. **Pipeline-Scoped Resolution** - Maintain isolation between different pipeline contexts
6. **Performance Optimization** - Cache resolution results and optimize matching algorithms

## Key Features

### 1. Intelligent Compatibility Scoring

The resolver uses a multi-factor scoring algorithm to determine the best matches:

```python
def _calculate_compatibility(self, dep_spec: DependencySpec, output_spec: OutputSpec,
                           provider_spec: StepSpecification) -> float:
    """Calculate compatibility score between dependency and output."""
    score = 0.0
    
    # 1. Dependency type compatibility (40% weight)
    if dep_spec.dependency_type == output_spec.output_type:
        score += 0.4
    elif self._are_types_compatible(dep_spec.dependency_type, output_spec.output_type):
        score += 0.2
    
    # 2. Data type compatibility (20% weight)
    if dep_spec.data_type == output_spec.data_type:
        score += 0.2
    
    # 3. Semantic name similarity (25% weight)
    semantic_score = self.semantic_matcher.calculate_similarity(
        dep_spec.logical_name, output_spec.logical_name
    )
    score += semantic_score * 0.25
    
    # 4. Compatible source check (10% weight)
    if provider_spec.step_type in dep_spec.compatible_sources:
        score += 0.1
    
    # 5. Keyword matching bonus (5% weight)
    keyword_score = self._calculate_keyword_match(dep_spec.semantic_keywords, output_spec.logical_name)
    score += keyword_score * 0.05
    
    return min(score, 1.0)
```

### 2. Type Compatibility Matrix

The resolver maintains a compatibility matrix for different dependency types:

```python
def _are_types_compatible(self, dep_type: DependencyType, output_type: DependencyType) -> bool:
    """Check if dependency and output types are compatible."""
    compatibility_matrix = {
        DependencyType.MODEL_ARTIFACTS: [DependencyType.MODEL_ARTIFACTS],
        DependencyType.TRAINING_DATA: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
        DependencyType.PROCESSING_OUTPUT: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
        DependencyType.HYPERPARAMETERS: [DependencyType.HYPERPARAMETERS, DependencyType.CUSTOM_PROPERTY],
        DependencyType.PAYLOAD_SAMPLES: [DependencyType.PAYLOAD_SAMPLES, DependencyType.PROCESSING_OUTPUT],
        DependencyType.CUSTOM_PROPERTY: [DependencyType.CUSTOM_PROPERTY]
    }
    
    compatible_types = compatibility_matrix.get(dep_type, [])
    return output_type in compatible_types
```

### 3. Semantic Matching Integration

The resolver integrates with semantic matching for intelligent name similarity:

```python
# Example of semantic matching in action
preprocessing_output = OutputSpec(
    logical_name="processed_data",
    output_type=DependencyType.PROCESSING_OUTPUT
)

training_dependency = DependencySpec(
    logical_name="training_input",
    dependency_type=DependencyType.PROCESSING_OUTPUT,
    semantic_keywords=["processed", "data", "input"]
)

# Semantic matcher calculates similarity between "processed_data" and "training_input"
# Plus keyword matching for "processed" and "data"
# Results in high compatibility score
```

### 4. Pipeline-Scoped Resolution

The resolver works within pipeline-specific registries to maintain isolation:

```python
# Each pipeline has its own resolver instance
class PipelineRegistry(SpecificationRegistry):
    def __init__(self, pipeline_name: str):
        super().__init__()
        self.pipeline_name = pipeline_name
        self._dependency_resolver = None
    
    @property
    def dependency_resolver(self) -> 'UnifiedDependencyResolver':
        """Lazy-loaded dependency resolver for this pipeline."""
        if self._dependency_resolver is None:
            self._dependency_resolver = UnifiedDependencyResolver(self)
        return self._dependency_resolver
    
    def resolve_pipeline_dependencies(self, step_names: List[str]) -> Dict[str, Dict[str, PropertyReference]]:
        """Resolve dependencies for all steps in this pipeline."""
        return self.dependency_resolver.resolve_all_dependencies(step_names)
```

### 5. Performance Optimization

The resolver includes caching and optimization features:

```python
class UnifiedDependencyResolver:
    def __init__(self, registry: Optional[SpecificationRegistry] = None):
        self.registry = registry or SpecificationRegistry()
        self.semantic_matcher = SemanticMatcher()
        self._resolution_cache: Dict[str, Dict[str, PropertyReference]] = {}
    
    def resolve_step_dependencies(self, consumer_step: str, available_steps: List[str]) -> Dict[str, PropertyReference]:
        """Resolve dependencies with caching."""
        # Check cache first
        cache_key = f"{consumer_step}:{':'.join(sorted(available_steps))}"
        if cache_key in self._resolution_cache:
            return self._resolution_cache[cache_key]
        
        # Perform resolution and cache result
        resolved = self._perform_resolution(consumer_step, available_steps)
        self._resolution_cache[cache_key] = resolved
        return resolved
```

### 6. Comprehensive Error Handling

The resolver provides detailed error reporting and debugging capabilities:

```python
def get_resolution_report(self, available_steps: List[str]) -> Dict[str, any]:
    """Generate detailed resolution report for debugging."""
    report = {
        'total_steps': len(available_steps),
        'registered_steps': len([s for s in available_steps if self.registry.get_specification(s)]),
        'step_details': {},
        'unresolved_dependencies': [],
        'resolution_summary': {}
    }
    
    for step_name in available_steps:
        # Detailed analysis for each step
        step_report = self._analyze_step_resolution(step_name, available_steps)
        report['step_details'][step_name] = step_report
    
    return report
```

## Integration with Other Components

### With Step Specifications

The resolver consumes [step specifications](step_specification.md) to understand step capabilities:

```python
# Step specification defines what a step needs and provides
XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type="XGBoostTraining",
    node_type=NodeType.INTERNAL,
    dependencies={
        "training_data": DependencySpec(
            logical_name="training_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing"],
            semantic_keywords=["data", "processed", "training"]
        )
    },
    outputs={
        "model_artifacts": OutputSpec(
            logical_name="model_artifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts"
        )
    }
)

# Resolver uses this specification to find compatible connections
resolver.register_specification("training", XGBOOST_TRAINING_SPEC)
resolved = resolver.resolve_step_dependencies("training", ["preprocessing", "training"])
```

### With Pipeline Registry

The resolver is embedded within pipeline registries for scoped resolution:

```python
# Pipeline registry with embedded resolver
pipeline_registry = get_pipeline_registry("fraud_detection")

# Register steps
pipeline_registry.register("preprocessing", PREPROCESSING_SPEC)
pipeline_registry.register("training", XGBOOST_TRAINING_SPEC)

# Resolve dependencies through registry
resolved = pipeline_registry.resolve_pipeline_dependencies(["preprocessing", "training"])
```

### With Enhanced DAG

The resolver integrates with [enhanced DAG](pipeline_dag.md) for topology-aware resolution:

```python
class EnhancedPipelineDAG(PipelineDAG):
    def __init__(self, pipeline_registry: PipelineRegistry = None):
        super().__init__()
        self.registry = pipeline_registry or global_pipeline_registry
    
    def auto_resolve_dependencies(self) -> List[DependencyEdge]:
        """Use resolver to automatically create dependency edges."""
        step_names = list(self.registry._specifications.keys())
        resolved = self.registry.resolve_pipeline_dependencies(step_names)
        
        edges = []
        for target_step, dependencies in resolved.items():
            for dep_name, prop_ref in dependencies.items():
                edge = DependencyEdge(
                    source_step=prop_ref.step_name,
                    target_step=target_step,
                    source_output=prop_ref.output_spec,
                    target_dependency=self.registry.get_specification(target_step).dependencies[dep_name]
                )
                edges.append(edge)
                self.add_edge(prop_ref.step_name, target_step)
        
        return edges
```

### With Smart Proxies

[Smart proxies](smart_proxy.md) use the resolver for intelligent connection suggestions:

```python
class SmartProxy:
    def suggest_compatible_sources(self):
        """Use resolver to suggest compatible source steps."""
        suggestions = {}
        
        for dep_name, dep_spec in self.specification.dependencies.items():
            # Use resolver to find compatible outputs
            compatible = self.registry.dependency_resolver.find_compatible_outputs(dep_spec)
            suggestions[dep_name] = compatible
        
        return suggestions
    
    def auto_connect(self):
        """Automatically connect to best compatible sources."""
        available_steps = list(self.registry._specifications.keys())
        resolved = self.registry.dependency_resolver.resolve_step_dependencies(
            self.step_name, available_steps
        )
        
        for dep_name, prop_ref in resolved.items():
            self._create_connection(prop_ref.step_name, prop_ref.output_spec)
```

## Architecture Patterns

### 1. Unified Resolver Pattern (Current)

The current implementation uses a unified resolver that works with any registry:

```python
# Unified resolver can work with any registry
resolver = UnifiedDependencyResolver(registry)
resolved = resolver.resolve_all_dependencies(step_names)
```

**Pros:**
- Flexible and reusable
- Can work with different registry types
- Clear separation of concerns

**Cons:**
- Requires explicit registry management
- More complex integration

### 2. Embedded Resolver Pattern (Proposed)

The proposed enhancement embeds the resolver within the registry:

```python
# Registry with embedded resolver
class PipelineRegistry(SpecificationRegistry):
    @property
    def dependency_resolver(self) -> 'UnifiedDependencyResolver':
        if self._dependency_resolver is None:
            self._dependency_resolver = UnifiedDependencyResolver(self)
        return self._dependency_resolver
    
    def resolve_pipeline_dependencies(self, step_names: List[str]) -> Dict[str, Dict[str, PropertyReference]]:
        """Direct resolution through embedded resolver."""
        return self.dependency_resolver.resolve_all_dependencies(step_names)
```

**Pros:**
- Simpler API - one object to manage
- Better encapsulation
- Automatic lifecycle management

**Cons:**
- Tighter coupling between registry and resolver
- Less flexibility for custom resolver implementations

## Resolution Algorithm

### 1. Step-by-Step Resolution Process

```python
def resolve_all_dependencies(self, available_steps: List[str]) -> Dict[str, Dict[str, PropertyReference]]:
    """
    1. For each step in available_steps:
       a. Get step specification from registry
       b. For each dependency in the specification:
          i. Find all compatible outputs from other steps
          ii. Score each compatibility using multi-factor algorithm
          iii. Select best match above threshold
          iv. Create PropertyReference for the match
       c. Validate all required dependencies are resolved
    2. Return complete resolution mapping
    """
```

### 2. Compatibility Scoring Factors

| Factor | Weight | Description |
|--------|--------|-------------|
| **Dependency Type Match** | 40% | Exact or compatible dependency type matching |
| **Data Type Compatibility** | 20% | S3Uri, String, Integer, Float, Boolean compatibility |
| **Semantic Name Similarity** | 25% | Fuzzy matching between logical names |
| **Compatible Source Check** | 10% | Step type listed in compatible_sources |
| **Keyword Matching** | 5% | Semantic keywords found in output name |

### 3. Resolution Confidence Levels

```python
# Confidence thresholds for resolution decisions
CONFIDENCE_THRESHOLDS = {
    'EXCELLENT': 0.9,   # Perfect or near-perfect match
    'GOOD': 0.7,        # Strong match, safe to use
    'ACCEPTABLE': 0.5,  # Minimum threshold for automatic resolution
    'POOR': 0.3,        # Requires manual review
    'INCOMPATIBLE': 0.0 # Cannot be resolved automatically
}
```

## Strategic Value

### For Pipeline Development

1. **Elimination of Manual Wiring** - No more fragile property path management
2. **Intelligent Automation** - System understands step relationships
3. **Error Prevention** - Type safety and compatibility checking
4. **Rapid Prototyping** - Quick pipeline construction without detailed wiring

### For System Architecture

1. **Separation of Concerns** - Resolution logic isolated from step implementation
2. **Extensibility** - Easy to add new compatibility rules and scoring factors
3. **Maintainability** - Centralized dependency logic
4. **Testability** - Resolution logic can be thoroughly tested in isolation

### For Developer Experience

1. **Reduced Cognitive Load** - Focus on business logic, not wiring details
2. **Better Error Messages** - Detailed resolution reports for debugging
3. **IDE Support** - Type-safe property references
4. **Documentation** - Self-documenting dependency relationships

## Usage Examples

### 1. Basic Resolution

```python
# Create resolver with registry
resolver = UnifiedDependencyResolver(pipeline_registry)

# Register step specifications
resolver.register_specification("preprocessing", PREPROCESSING_SPEC)
resolver.register_specification("training", XGBOOST_TRAINING_SPEC)
resolver.register_specification("registration", REGISTRATION_SPEC)

# Resolve all dependencies
available_steps = ["preprocessing", "training", "registration"]
resolved = resolver.resolve_all_dependencies(available_steps)

# Result:
# {
#   "training": {
#     "training_data": PropertyReference(step="preprocessing", output="processed_data")
#   },
#   "registration": {
#     "model_artifacts": PropertyReference(step="training", output="model_artifacts")
#   }
# }
```

### 2. Pipeline-Scoped Resolution

```python
# Get pipeline-specific registry
fraud_registry = get_pipeline_registry("fraud_detection")

# Register steps in pipeline context
fraud_registry.register("data_loading", DATA_LOADING_SPEC)
fraud_registry.register("preprocessing", PREPROCESSING_SPEC)
fraud_registry.register("training", XGBOOST_TRAINING_SPEC)

# Resolve within pipeline scope
resolved = fraud_registry.resolve_pipeline_dependencies([
    "data_loading", "preprocessing", "training"
])
```

### 3. Enhanced DAG Integration

```python
# Create enhanced DAG with automatic resolution
dag = EnhancedPipelineDAG(fraud_registry)

# Register steps with DAG
dag.register_step_spec("data_loading", DATA_LOADING_SPEC)
dag.register_step_spec("preprocessing", PREPROCESSING_SPEC)
dag.register_step_spec("training", XGBOOST_TRAINING_SPEC)

# Auto-resolve dependencies and create edges
dependency_edges = dag.auto_resolve_dependencies()

# Validate complete pipeline
errors = dag.validate_dependencies()
if not errors:
    print("Pipeline dependencies successfully resolved!")
```

### 4. Resolution Debugging

```python
# Generate detailed resolution report
report = resolver.get_resolution_report(["preprocessing", "training", "registration"])

print(f"Resolution Summary:")
print(f"  Total Dependencies: {report['resolution_summary']['total_dependencies']}")
print(f"  Resolved: {report['resolution_summary']['resolved_dependencies']}")
print(f"  Resolution Rate: {report['resolution_summary']['resolution_rate']:.1%}")

# Detailed step analysis
for step_name, details in report['step_details'].items():
    print(f"\nStep: {step_name}")
    print(f"  Type: {details['step_type']}")
    print(f"  Dependencies: {details['total_dependencies']} ({details['required_dependencies']} required)")
    print(f"  Resolved: {len(details['resolved_dependencies'])}")
    
    if details['unresolved_dependencies']:
        print(f"  Unresolved: {details['unresolved_dependencies']}")
```

### 5. Custom Compatibility Rules

```python
class CustomDependencyResolver(UnifiedDependencyResolver):
    """Custom resolver with domain-specific compatibility rules."""
    
    def _calculate_compatibility(self, dep_spec: DependencySpec, output_spec: OutputSpec,
                               provider_spec: StepSpecification) -> float:
        """Override with custom scoring logic."""
        base_score = super()._calculate_compatibility(dep_spec, output_spec, provider_spec)
        
        # Add domain-specific bonuses
        if "fraud" in dep_spec.semantic_keywords and "fraud" in output_spec.logical_name.lower():
            base_score += 0.1  # Fraud domain bonus
        
        if provider_spec.step_type.startswith("Custom") and dep_spec.logical_name.startswith("custom"):
            base_score += 0.05  # Custom step bonus
        
        return min(base_score, 1.0)
```

## Future Enhancements

### 1. Machine Learning-Based Matching

```python
# Future: ML-based compatibility scoring
class MLDependencyResolver(UnifiedDependencyResolver):
    def __init__(self, registry, ml_model=None):
        super().__init__(registry)
        self.ml_model = ml_model or self._load_pretrained_model()
    
    def _calculate_compatibility(self, dep_spec, output_spec, provider_spec):
        """Use ML model for compatibility scoring."""
        features = self._extract_features(dep_spec, output_spec, provider_spec)
        ml_score = self.ml_model.predict_compatibility(features)
        
        # Combine with rule-based score
        rule_score = super()._calculate_compatibility(dep_spec, output_spec, provider_spec)
        return (ml_score * 0.6) + (rule_score * 0.4)
```

### 2. Cross-Pipeline Dependency Resolution

```python
# Future: Resolve dependencies across multiple pipelines
class CrossPipelineDependencyResolver:
    def resolve_cross_pipeline_dependencies(self, pipeline_names: List[str]):
        """Resolve dependencies that span multiple pipelines."""
        # Implementation for cross-pipeline resolution
        pass
```

### 3. Real-Time Resolution Monitoring

```python
# Future: Monitor resolution performance in production
class MonitoredDependencyResolver(UnifiedDependencyResolver):
    def resolve_step_dependencies(self, consumer_step, available_steps):
        start_time = time.time()
        result = super().resolve_step_dependencies(consumer_step, available_steps)
        resolution_time = time.time() - start_time
        
        # Log performance metrics
        self.metrics_collector.record_resolution_time(consumer_step, resolution_time)
        return result
```

## Related Components

- **[Step Specification](step_specification.md)** - Provides the declarative metadata that drives resolution
- **[Pipeline Registry](pipeline_registry.md)** - Manages step specifications and embeds resolver
- **[Smart Proxy](smart_proxy.md)** - Uses resolver for intelligent connection suggestions
- **[Enhanced DAG](pipeline_dag.md)** - Integrates resolver for automatic topology construction
- **[Fluent API](fluent_api.md)** - Leverages resolver for seamless pipeline construction

---

The Dependency Resolver represents the **intelligent automation layer** that transforms pipeline construction from manual, error-prone wiring into intelligent, specification-driven automation. It serves as the foundation for all higher-level abstractions while maintaining the flexibility and extensibility needed for complex ML pipeline scenarios.
