# Unified Dependency Resolver

## Overview
The Unified Dependency Resolver automatically determines input/output connections between pipeline steps using specification-driven analysis. It evaluates compatibility between step outputs and inputs using semantic matching and type compatibility rules, enabling automatic pipeline assembly with minimal manual configuration.

## Core Functionality

### Key Features
- **Automatic Dependency Resolution**: Resolves dependencies between steps based on their specifications
- **Semantic Name Matching**: Uses semantic similarity to match input/output names
- **Compatibility Scoring**: Evaluates multiple factors to determine best matches
- **Caching**: Caches resolution results for performance optimization
- **Error Reporting**: Detailed error reporting for unresolved dependencies

## Key Components

### UnifiedDependencyResolver
Main resolver class that analyzes and resolves dependencies between pipeline steps.

```python
class UnifiedDependencyResolver:
    def __init__(self, registry: SpecificationRegistry, semantic_matcher: SemanticMatcher):
        """
        Initialize the dependency resolver.
        
        Args:
            registry: Specification registry
            semantic_matcher: Semantic matcher for name similarity calculations
        """
        
    def resolve_all_dependencies(self, available_steps: List[str]) -> Dict[str, Dict[str, PropertyReference]]:
        """
        Resolve dependencies for all registered steps.
        
        Args:
            available_steps: List of step names available in the pipeline
            
        Returns:
            Dictionary mapping step names to their resolved dependencies
        """
        
    def resolve_step_dependencies(self, consumer_step: str, available_steps: List[str]) -> Dict[str, PropertyReference]:
        """
        Resolve dependencies for a single step.
        
        Args:
            consumer_step: Name of the step whose dependencies to resolve
            available_steps: List of available step names
            
        Returns:
            Dictionary mapping dependency names to property references
        """
        
    def get_resolution_report(self, available_steps: List[str]) -> Dict[str, Any]:
        """
        Generate a detailed resolution report for debugging.
        
        Args:
            available_steps: List of available step names
            
        Returns:
            Detailed report of resolution process
        """
        
    def clear_cache(self):
        """Clear the resolution cache."""
```

### DependencyResolutionError
Exception raised when dependencies cannot be resolved.

```python
class DependencyResolutionError(Exception):
    """Raised when dependencies cannot be resolved."""
    pass
```

### PropertyReference
Reference to a property of another step, used to represent resolved dependencies.

```python
@dataclass
class PropertyReference:
    step_name: str
    output_spec: OutputSpec
```

## Dependency Resolution Process

### 1. Step Specification Analysis
The resolver examines the specifications of all steps in the pipeline to understand their inputs and outputs.

### 2. Dependency Matching
For each step, the resolver:
1. Identifies its required and optional dependencies from its specification
2. Searches available steps for compatible outputs
3. Calculates compatibility scores for each potential match
4. Selects the best match for each dependency

### 3. Compatibility Scoring
Compatibility between a dependency and an output is determined by evaluating:

- **Type Compatibility (40%)**: Whether the dependency type (MODEL_ARTIFACTS, TRAINING_DATA, etc.) matches the output type
- **Data Type Compatibility (20%)**: Whether the data types (String, S3Uri, Integer, etc.) are compatible
- **Semantic Name Matching (25%)**: Similarity between dependency and output names, including alias support
- **Source Compatibility (10%)**: Whether the provider step type is in the list of compatible sources
- **Keyword Matching (5%)**: Matching of semantic keywords

Each factor contributes to an overall compatibility score between 0.0 and 1.0, with higher scores indicating better matches.

### 4. Resolution Result
The resolution process produces a mapping from each step's dependency names to PropertyReferences pointing to other steps' outputs.

## Usage Examples

### Basic Dependency Resolution

```python
from src.pipeline_deps.dependency_resolver import create_dependency_resolver
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.semantic_matcher import SemanticMatcher

# Create the resolver components
registry = SpecificationRegistry()
semantic_matcher = SemanticMatcher()
resolver = create_dependency_resolver(registry, semantic_matcher)

# Register step specifications
registry.register("data_load", DATA_LOADING_SPEC)
registry.register("preprocess", PREPROCESSING_SPEC)
registry.register("train", TRAINING_SPEC)

# Resolve dependencies for all steps
available_steps = ["data_load", "preprocess", "train"]
dependencies = resolver.resolve_all_dependencies(available_steps)

# Print the resolved dependencies
for step_name, step_deps in dependencies.items():
    print(f"Step: {step_name}")
    for dep_name, prop_ref in step_deps.items():
        print(f"  {dep_name} -> {prop_ref.step_name}.{prop_ref.output_spec.logical_name}")
```

### Resolving Dependencies for a Single Step

```python
# Resolve dependencies for just the training step
train_deps = resolver.resolve_step_dependencies("train", available_steps)

# Print the resolved dependencies
print(f"Training step dependencies:")
for dep_name, prop_ref in train_deps.items():
    print(f"  {dep_name} -> {prop_ref.step_name}.{prop_ref.output_spec.logical_name}")
```

### Generating a Resolution Report

```python
# Get detailed resolution information for debugging
report = resolver.get_resolution_report(available_steps)

# Print summary statistics
print(f"Total steps: {report['total_steps']}")
print(f"Registered steps: {report['registered_steps']}")
print(f"Resolution rate: {report['resolution_summary']['resolution_rate']:.2f}")
print(f"Steps with errors: {report['resolution_summary']['steps_with_errors']}")

# Print detailed information for each step
for step_name, details in report['step_details'].items():
    print(f"\nStep: {step_name} ({details['step_type']})")
    print(f"  Total dependencies: {details['total_dependencies']}")
    print(f"  Required dependencies: {details['required_dependencies']}")
    print(f"  Resolved dependencies: {len(details['resolved_dependencies'])}")
    
    if details.get('unresolved_dependencies'):
        print(f"  Unresolved dependencies: {details['unresolved_dependencies']}")
    
    if 'error' in details:
        print(f"  Error: {details['error']}")
```

## Compatibility Calculation Logic

```python
def _calculate_compatibility(self, dep_spec: DependencySpec, output_spec: OutputSpec,
                           provider_spec: StepSpecification) -> float:
    score = 0.0
    
    # 1. Dependency type compatibility (40% weight)
    if dep_spec.dependency_type == output_spec.output_type:
        score += 0.4
    elif self._are_types_compatible(dep_spec.dependency_type, output_spec.output_type):
        score += 0.2
    else:
        # If types are not compatible at all, return 0
        return 0.0
    
    # 2. Data type compatibility (20% weight)
    if dep_spec.data_type == output_spec.data_type:
        score += 0.2
    elif self._are_data_types_compatible(dep_spec.data_type, output_spec.data_type):
        score += 0.1
    
    # 3. Enhanced semantic name matching with alias support (25% weight)
    semantic_score = self.semantic_matcher.calculate_similarity_with_aliases(
        dep_spec.logical_name, output_spec
    )
    score += semantic_score * 0.25
    
    # Optional: Add direct match bonus for exact matches
    if dep_spec.logical_name == output_spec.logical_name:
        score += 0.05  # Exact logical name match bonus
    elif dep_spec.logical_name in output_spec.aliases:
        score += 0.05  # Exact alias match bonus
    
    # 4. Compatible source check (10% weight)
    if dep_spec.compatible_sources:
        if provider_spec.step_type in dep_spec.compatible_sources:
            score += 0.1
    else:
        # If no compatible sources specified, give small bonus for any match
        score += 0.05
    
    # 5. Keyword matching bonus (5% weight)
    if dep_spec.semantic_keywords:
        keyword_score = self._calculate_keyword_match(dep_spec.semantic_keywords, output_spec.logical_name)
        score += keyword_score * 0.05
    
    return min(score, 1.0)  # Cap at 1.0
```

## Type Compatibility Rules

The resolver defines compatibility matrices for dependency types and data types:

### Dependency Type Compatibility

```python
# Define compatibility matrix
compatibility_matrix = {
    DependencyType.MODEL_ARTIFACTS: [DependencyType.MODEL_ARTIFACTS],
    DependencyType.TRAINING_DATA: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
    DependencyType.PROCESSING_OUTPUT: [DependencyType.PROCESSING_OUTPUT, DependencyType.TRAINING_DATA],
    DependencyType.HYPERPARAMETERS: [DependencyType.HYPERPARAMETERS, DependencyType.CUSTOM_PROPERTY],
    DependencyType.PAYLOAD_SAMPLES: [DependencyType.PAYLOAD_SAMPLES, DependencyType.PROCESSING_OUTPUT],
    DependencyType.CUSTOM_PROPERTY: [DependencyType.CUSTOM_PROPERTY]
}
```

### Data Type Compatibility

```python
# Define data type compatibility
compatibility_map = {
    'S3Uri': ['S3Uri', 'String'],  # S3Uri can sometimes be used as String
    'String': ['String', 'S3Uri'],  # String can sometimes accept S3Uri
    'Integer': ['Integer', 'Float'],  # Integer can be used as Float
    'Float': ['Float', 'Integer'],   # Float can accept Integer
    'Boolean': ['Boolean'],
}
```

## Best Practices

1. **Rich Specifications**: Create detailed step specifications with accurate dependency and output information
2. **Logical Naming**: Use consistent, descriptive names for dependencies and outputs
3. **Type Accuracy**: Correctly specify dependency types and data types
4. **Aliases**: Add aliases to outputs for common alternative names
5. **Semantic Keywords**: Add relevant semantic keywords to improve matching accuracy
6. **Error Handling**: Check for unresolved dependencies before executing pipelines
7. **Reporting**: Use resolution reports to identify and fix dependency issues

## Integration with Pipeline Builder

The UnifiedDependencyResolver is a key component of the Pipeline Builder Template:

```python
from src.pipeline_builder.template import PipelineBuilderTemplate
from src.pipeline_deps.dependency_resolver import create_dependency_resolver

# Create registry and resolver
registry = SpecificationRegistry()
resolver = create_dependency_resolver(registry)

# Register specifications
for name, spec in specifications.items():
    registry.register(name, spec)

# Create pipeline template
template = PipelineBuilderTemplate(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    registry=registry,
    dependency_resolver=resolver
)

# Generate pipeline with automatic dependency resolution
pipeline = template.generate_pipeline("my-pipeline")
```

## Error Handling

```python
try:
    # Attempt to resolve dependencies
    deps = resolver.resolve_step_dependencies(step_name, available_steps)
    
    # Use resolved dependencies
    for dep_name, prop_ref in deps.items():
        print(f"Resolved {dep_name} to {prop_ref.step_name}.{prop_ref.output_spec.logical_name}")
        
except DependencyResolutionError as e:
    # Handle resolution errors
    print(f"Failed to resolve dependencies: {e}")
    
    # Get resolution report for debugging
    report = resolver.get_resolution_report(available_steps)
    
    # Look for specific unresolved dependencies
    unresolved_deps = [
        dep for step in report['step_details'].values()
        for dep in step.get('unresolved_dependencies', [])
    ]
    
    print(f"Unresolved dependencies: {unresolved_deps}")
