# Dependency Resolution Enhancement: Alias Support Implementation Plan (COMPLETED)

**Date:** July 8, 2025  
**Author:** Cline  
**Topic:** Adding Alias Support to Semantic Matching

## Background

Currently, our dependency resolution system only considers the logical name of an output when calculating semantic similarity, ignoring any aliases defined in the OutputSpec. This limits the flexibility of our matching system and makes evolving naming conventions difficult.

## Implementation Plan

### Phase 1: Core Implementation

#### 1.1 Add New Method to SemanticMatcher

**File:** `src/v2/pipeline_deps/semantic_matcher.py`

```python
def calculate_similarity_with_aliases(self, name: str, output_spec: OutputSpec) -> float:
    """
    Calculate semantic similarity between a name and an output specification,
    considering both logical_name and all aliases.
    
    Args:
        name: The name to compare (typically the dependency's logical_name)
        output_spec: OutputSpec with logical_name and potential aliases
        
    Returns:
        The highest similarity score (0.0 to 1.0) between name and any name in output_spec
    """
    # Start with similarity to logical_name
    best_score = self.calculate_similarity(name, output_spec.logical_name)
    best_match = output_spec.logical_name
    
    # Check each alias
    for alias in output_spec.aliases:
        alias_score = self.calculate_similarity(name, alias)
        if alias_score > best_score:
            best_score = alias_score
            best_match = alias
    
    # Log which name gave the best match (only for meaningful matches)
    if best_score > 0.5:
        logger.debug(f"Best match for '{name}': '{best_match}' (score: {best_score:.3f})")
        
    return best_score
```

#### 1.2 Update Compatibility Calculation

**File:** `src/v2/pipeline_deps/dependency_resolver.py`

```python
def _calculate_compatibility(self, dep_spec: DependencySpec, output_spec: OutputSpec, 
                           provider_spec: StepSpecification) -> float:
    # ... existing type compatibility check (40% weight) ...
    
    # ... existing data type compatibility check (20% weight) ...
    
    # Enhanced semantic name matching with alias support (25% weight)
    semantic_score = self.semantic_matcher.calculate_similarity_with_aliases(
        dep_spec.logical_name, output_spec
    )
    score += semantic_score * 0.25
    
    # Optional: Add direct match bonus for exact matches
    if dep_spec.logical_name == output_spec.logical_name:
        score += 0.05  # Exact logical name match bonus
    elif dep_spec.logical_name in output_spec.aliases:
        score += 0.05  # Exact alias match bonus
        
    # ... rest of method remains unchanged ...
    
    return min(score, 1.0)  # Cap at 1.0
```

### Phase 2: Testing

#### 2.1 Unit Tests for Semantic Matcher

**File:** `test/v2/pipeline_deps/test_semantic_matcher.py`

```python
def test_calculate_similarity_with_aliases():
    """Test that calculate_similarity_with_aliases returns the highest similarity."""
    matcher = SemanticMatcher()
    
    # Create test output spec with aliases
    output_spec = OutputSpec(
        logical_name="processed_data",
        aliases=["training_data", "input_features", "ModelData"],
        output_type=DependencyType.PROCESSING_OUTPUT,
        property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri"
    )
    
    # Test with various dependency names
    test_cases = [
        # (dep_name, expected_min_score, expected_best_match)
        ("training_data", 1.0, "training_data"),        # Exact match with alias
        ("train_data", 0.8, "training_data"),           # Close match with alias
        ("processed_data", 1.0, "processed_data"),      # Exact match with logical_name
        ("model_data", 0.7, "ModelData"),               # Close match with alias
        ("feature_data", 0.6, "input_features"),        # Moderate match with alias
        ("unrelated_name", 0.3, "processed_data")       # No good match (defaults to logical)
    ]
    
    for dep_name, expected_min_score, expected_best_match in test_cases:
        score = matcher.calculate_similarity_with_aliases(dep_name, output_spec)
        assert score >= expected_min_score, f"Score for '{dep_name}' below expectation"
```

#### 2.2 Integration Tests for Dependency Resolution

**File:** `test/v2/pipeline_deps/test_dependency_resolver.py`

```python
def test_dependency_resolution_with_aliases():
    """Test that dependency resolution uses aliases for matching."""
    registry = SpecificationRegistry()
    resolver = UnifiedDependencyResolver(registry)
    
    # Create producer step specification with aliases
    producer_spec = StepSpecification(
        step_type="PreprocessingStep",
        node_type=NodeType.INTERNAL,
        dependencies=[],  # No dependencies for this test
        outputs=[
            OutputSpec(
                logical_name="processed_data",
                aliases=["training_data", "model_input"],
                output_type=DependencyType.PROCESSING_OUTPUT,
                property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
                data_type="S3Uri"
            )
        ]
    )
    
    # Create consumer step specification
    consumer_spec = StepSpecification(
        step_type="TrainingStep",
        node_type=NodeType.INTERNAL,
        dependencies=[
            DependencySpec(
                logical_name="training_data",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                required=True,
                compatible_sources=["PreprocessingStep"],
                data_type="S3Uri"
            )
        ],
        outputs=[]  # No outputs needed for this test
    )
    
    # Register step specifications
    registry.register("producer", producer_spec)
    registry.register("consumer", consumer_spec)
    
    # Resolve dependencies
    resolved = resolver.resolve_step_dependencies("consumer", ["producer"])
    
    # Assert that resolution was successful
    assert "training_data" in resolved
    assert resolved["training_data"].step_name == "producer"
    assert resolved["training_data"].output_spec.logical_name == "processed_data"
```

### Phase 3: Documentation and Rollout

#### 3.1 Update Documentation

1. Update docstrings in code
2. Update developer guides to explain alias support
3. Create usage examples showing how to leverage aliases

#### 3.2 Performance Testing

1. Create benchmarks comparing resolution speed before and after changes
2. Verify that any overhead from checking aliases is acceptable

#### 3.3 Phased Rollout

1. Deploy to development environment
2. Test with real-world pipelines
3. Gather feedback from pipeline authors
4. Deploy to production

## Acceptance Criteria

1. The implementation correctly finds the highest similarity score between a dependency name and any of an output's names (logical name or aliases)
2. Performance impact is negligible (< 5% increase in resolution time)
3. Unit and integration tests pass
4. Documentation is updated to reflect new capabilities
5. Real-world pipelines validate successfully with the new implementation

## Timeline

- Day 1: Implement core changes (Phases 1.1 & 1.2)
- Day 2: Implement unit and integration tests (Phases 2.1 & 2.2)
- Day 3: Update documentation and perform performance testing (Phase 3)
- Day 4: Test with real-world pipelines and gather feedback
- Day 5: Address feedback and deploy to production

## Expected Benefits

1. **Better Matching**: The system will find more appropriate matches, especially when naming conventions evolve
2. **Backward Compatibility**: Support for both older and newer naming styles simultaneously
3. **Improved Developer Experience**: Step authors can provide multiple names for the same output
4. **More Resilient Pipelines**: Less likelihood of missing connections due to name mismatches
