# Removing Global Singleton Objects

**Date:** July 8, 2025  
**Status:** 🔄 IMPLEMENTATION IN PROGRESS  
**Priority:** 🔥 HIGH - Foundation for Testing Reliability

## Executive Summary

This document outlines the comprehensive plan to remove global singleton objects from the pipeline dependencies system. This change will significantly improve testability, debugging, and concurrent execution by eliminating unwanted correlations between different parts of the codebase. The plan involves a phased approach to minimize disruption while ensuring comprehensive refactoring of all affected components.

## Related Documents

- [Phase 1: Registry Manager Implementation](./2025-07-08_phase1_registry_manager_implementation.md)
- [Phase 1: Dependency Resolver Implementation](./2025-07-08_phase1_dependency_resolver_implementation.md)
- [Phase 1: Semantic Matcher Implementation](./2025-07-08_phase1_semantic_matcher_implementation.md)

## Current Issues with Global Singletons

As documented in [global_vs_local_objects.md](../pipeline_design/global_vs_local_objects.md), the current architecture has several pain points:

1. **Testing Challenges**
   - Tests that pass individually may fail when run together due to shared state
   - Requires complex setUp/tearDown cleanup methods
   - Test isolation issues lead to inconsistent results

2. **Developer Coordination Issues**
   - Unwanted correlation between different developers' work
   - Changes in one area can unexpectedly impact another
   - Difficult to debug issues with shared global state

3. **Concurrency Limitations**
   - Potential race conditions in multi-threaded environments
   - Need for synchronization creates bottlenecks
   - Difficulty in parallel execution

4. **Hidden Dependencies**
   - Function dependencies not explicit in signatures
   - "Spooky action at a distance" where changes in one part affect another
   - Complex dependency graph that's difficult to understand

## Identified Global Singletons

The codebase currently uses three primary global singleton objects that need to be refactored:

1. **Registry Manager**:
   - Global instance: `registry_manager` in `registry_manager.py`
   - Used for managing multiple isolated specification registries
   - Used via convenience functions like `get_registry()`

2. **Dependency Resolver**:
   - Global instance: `global_resolver` in `dependency_resolver.py`
   - Used for resolving dependencies between steps
   - Maintains resolution cache for performance

3. **Semantic Matcher**:
   - Global instance: `semantic_matcher` in `semantic_matcher.py`
   - Used for calculating semantic similarity between names
   - Contains predefined synonym dictionaries and matching algorithms

## Component Relationships

These components have the following dependencies:

```
SemanticMatcher <-- UnifiedDependencyResolver <-- StepBuilderBase
                                              ^
                                              |
SpecificationRegistry <---------------------->+
         ^
         |
RegistryManager
```

The migration needs to respect these relationships by ensuring that components higher in the dependency chain are updated first.

## Implementation Strategy

### Phase 1: Core Component Refactoring (Week 1) - ✅ COMPLETED

#### 1.1 Semantic Matcher Refactoring - ✅ COMPLETED

Detailed implementation plan: [Phase 1: Semantic Matcher Implementation](./2025-07-08_phase1_semantic_matcher_implementation.md)

- ✅ Removed global `semantic_matcher` instance from `semantic_matcher.py`
- ✅ Updated class implementation to be instance-based
- ✅ Updated `__all__` list to reflect changes
- ✅ Updated class docstrings to reflect the change

#### 1.2 Registry Manager Refactoring - ✅ COMPLETED

Detailed implementation plan: [Phase 1: Registry Manager Implementation](./2025-07-08_phase1_registry_manager_implementation.md)

- ✅ Removed global `registry_manager` instance from `registry_manager.py`
- ✅ Updated convenience functions to accept a manager instance
- ✅ Updated `integrate_with_pipeline_builder` to accept a registry manager
- ✅ Updated `__all__` list to reflect changes

#### 1.3 Dependency Resolver Refactoring - ✅ COMPLETED

Detailed implementation plan: [Phase 1: Dependency Resolver Implementation](./2025-07-08_phase1_dependency_resolver_implementation.md)

- ✅ Removed global `global_resolver` instance from `dependency_resolver.py`
- ✅ Updated `UnifiedDependencyResolver` constructor to accept both registry and semantic matcher
- ✅ Added factory function `create_dependency_resolver()` for simplified object creation
- ✅ Updated `__all__` list to reflect changes

### Phase 2: Dependency Injection Implementation (Week 2) - ✅ COMPLETED

#### 2.1 Create Factory Module - ✅ COMPLETED

- ✅ Created new module `factory.py` in `pipeline_deps` package
- ✅ Implemented factory functions for creating properly configured components
- ✅ Added convenience methods for common use cases
- ✅ Included comprehensive documentation and examples

The implementation provides a centralized factory function that creates all dependency components with proper wiring:

```python
def create_pipeline_components(context_name=None):
    """Create all necessary pipeline components with proper dependencies."""
    semantic_matcher = SemanticMatcher()
    registry_manager = RegistryManager()
    registry = registry_manager.get_registry(context_name or "default")
    resolver = UnifiedDependencyResolver(registry, semantic_matcher)
    
    return {
        "semantic_matcher": semantic_matcher,
        "registry_manager": registry_manager,
        "registry": registry,
        "resolver": resolver
    }
```

#### 2.2 Update Step Builder Base Class - ✅ COMPLETED

- ✅ Modified `StepBuilderBase` constructor to accept necessary dependencies:

```python
def __init__(self, config, spec=None, sagemaker_session=None, role=None,
             notebook_root=None, registry_manager=None, dependency_resolver=None):
    # ...
    self._registry_manager = registry_manager
    self._dependency_resolver = dependency_resolver
    # ...
```

- ✅ Added helper methods for creating or getting dependencies:

```python
def _get_context_name(self) -> str:
    """Get the context name to use for registry operations."""
    if hasattr(self.config, 'pipeline_name') and self.config.pipeline_name:
        return self.config.pipeline_name
    return "default"

def _get_registry_manager(self) -> RegistryManager:
    """Get or create a registry manager."""
    if not hasattr(self, '_registry_manager') or self._registry_manager is None:
        self._registry_manager = RegistryManager()
        self.log_debug("Created new registry manager")
    return self._registry_manager

def _get_registry(self):
    """Get the appropriate registry for this step."""
    registry_manager = self._get_registry_manager()
    context_name = self._get_context_name()
    return registry_manager.get_registry(context_name)

def _get_dependency_resolver(self) -> UnifiedDependencyResolver:
    """Get or create a dependency resolver."""
    if not hasattr(self, '_dependency_resolver') or self._dependency_resolver is None:
        registry = self._get_registry()
        semantic_matcher = SemanticMatcher()
        self._dependency_resolver = create_dependency_resolver(registry, semantic_matcher)
        self.log_debug(f"Created new dependency resolver for context '{self._get_context_name()}'")
    return self._dependency_resolver
```

- ✅ Updated `extract_inputs_from_dependencies` method to use the injected or lazily created resolver:

```python
def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
    # ...
    # Use the injected resolver or create one
    resolver = self._get_dependency_resolver()
    resolver.register_specification(step_name, self.spec)
    # ...
```

#### 2.3 Add Context Management - ✅ COMPLETED

- ✅ Implemented context managers for scoped component usage:

```python
@contextmanager
def dependency_resolution_context(clear_on_exit=True):
    """Create a scoped dependency resolution context."""
    components = create_pipeline_components()
    try:
        yield components
    finally:
        if clear_on_exit:
            components["resolver"].clear_cache()
            components["registry_manager"].clear_all_contexts()
```

#### 2.4 Add Thread-Local Storage - ✅ COMPLETED

- ✅ Implemented thread-local storage for per-thread component instances:

```python
# Thread-local storage for per-thread instances
_thread_local = threading.local()

def get_thread_components():
    """Get thread-specific component instances."""
    if not hasattr(_thread_local, 'components'):
        _thread_local.components = create_pipeline_components()
    return _thread_local.components
```

### Phase 3: Application Updates (Week 3)

#### 3.1 Update Pipeline Builder Classes - ✅ COMPLETED

This phase focuses on updating all pipeline builder classes to use the new dependency injection pattern instead of global singletons.

##### Required Changes

1. **Constructor Modifications**:
   - Update all pipeline builder constructors to accept registry_manager and dependency_resolver parameters
   - Implement proper default creation with factory methods when parameters are not provided
   
   ```python
   class PipelineBuilderBase:
       def __init__(self, config, registry_manager=None, dependency_resolver=None):
           self.config = config
           
           # Use injected or create new registry manager
           self.registry_manager = registry_manager or RegistryManager()
           
           # Get registry for this pipeline
           context_name = getattr(config, 'pipeline_name', 'default_pipeline')
           self.registry = self.registry_manager.get_registry(context_name)
           
           # Create resolver if not provided
           if dependency_resolver is None:
               from ..pipeline_deps.factory import create_dependency_resolver
               semantic_matcher = SemanticMatcher()
               dependency_resolver = create_dependency_resolver(self.registry, semantic_matcher)
           self.dependency_resolver = dependency_resolver
   ```

2. **Builder Factory Methods**:
   - Add factory methods that use the component factory for pipeline builders:

   ```python
   @classmethod
   def create_with_components(cls, config, context_name=None):
       """Create pipeline builder with managed components."""
       components = create_pipeline_components(context_name)
       return cls(
           config=config,
           registry_manager=components["registry_manager"],
           dependency_resolver=components["resolver"]
       )
   ```

3. **Component Passing**:
   - Update all methods that create step builders to pass the components:

   ```python
   def create_training_step(self, **kwargs):
       """Create training step with properly configured components."""
       training_builder = TrainingStepBuilder(
           config=self.config.training,
           registry_manager=self.registry_manager,
           dependency_resolver=self.dependency_resolver,
           **kwargs
       )
       return training_builder.create_step()
   ```

##### Files to Update

- `src/v2/pipeline_builder/builder_base.py`
- `src/v2/pipeline_builder/training_pipeline_builder.py`
- `src/v2/pipeline_builder/processing_pipeline_builder.py`
- Any custom pipeline builder implementations

#### 3.2 Update Step Builder Implementations - ✅ COMPLETED

This phase focuses on updating all step builder classes to properly support dependency injection and use the new helper methods in StepBuilderBase.

##### Required Changes

1. **Constructor Updates**:
   - Update all step builder constructors to accept and forward registry_manager and dependency_resolver parameters:

   ```python
   class ProcessingStepBuilder(StepBuilderBase):
       def __init__(
           self,
           config,
           spec=None, 
           sagemaker_session=None,
           role=None,
           notebook_root=None,
           registry_manager=None,      # New parameter
           dependency_resolver=None    # New parameter
       ):
           super().__init__(
               config=config,
               spec=spec,
               sagemaker_session=sagemaker_session,
               role=role,
               notebook_root=notebook_root,
               registry_manager=registry_manager,        # Pass to parent
               dependency_resolver=dependency_resolver   # Pass to parent
           )
   ```

2. **Replace Direct Global References**:
   - Remove any code that directly imports or uses global singletons
   - Replace with calls to the new helper methods:
     - `self._get_dependency_resolver()`
     - `self._get_registry_manager()`
     - `self._get_registry()`

   Before:
   ```python
   from ..pipeline_deps.dependency_resolver import global_resolver
   
   # ...
   
   def resolve_dependencies(self, steps):
       resolver = global_resolver
       resolver.register_specification(...)
   ```

   After:
   ```python
   def resolve_dependencies(self, steps):
       resolver = self._get_dependency_resolver()
       resolver.register_specification(...)
   ```

3. **Update Registry Access**:
   - Replace direct registry access with context-aware methods

   Before:
   ```python
   from ..pipeline_deps.registry_manager import get_registry
   
   # ...
   
   registry = get_registry("some_context")
   ```

   After:
   ```python
   registry = self._get_registry()  # Gets registry for appropriate context
   ```

##### Updated Step Builder Files
- ✅ `builder_training_step_pytorch.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_training_step_xgboost.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_model_step_pytorch.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_model_step_xgboost.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_model_eval_step_xgboost.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_tabular_preprocessing_step.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_batch_transform_step.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_data_load_step_cradle.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_currency_conversion_step.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_mims_packaging_step.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_mims_registration_step.py`: Added RegistryManager and UnifiedDependencyResolver parameters
- ✅ `builder_mims_payload_step.py`: Added RegistryManager and UnifiedDependencyResolver parameters

#### 3.3 Update Pipeline Examples - 🔄 IN PROGRESS

This phase focuses on updating example pipeline scripts to demonstrate proper component creation and dependency injection.

##### Required Changes

1. **Component Creation**:
   - Update pipeline examples to use the factory module for component creation:

   ```python
   from src.v2.pipeline_deps.factory import create_pipeline_components
   
   def build_pipeline():
       # Create components for this pipeline context
       components = create_pipeline_components("example_pipeline")
       
       # Create pipeline builder with components
       pipeline_builder = ExamplePipelineBuilder(
           config=config,
           registry_manager=components["registry_manager"],
           dependency_resolver=components["resolver"]
       )
       
       # Use the pipeline builder to create the pipeline
       pipeline = pipeline_builder.build()
       return pipeline
   ```

2. **Context Manager Usage**:
   - Add examples of using context managers for scoped component usage:

   ```python
   from src.v2.pipeline_deps.factory import dependency_resolution_context
   
   def run_pipeline_example():
       with dependency_resolution_context(clear_on_exit=True) as components:
           # Create pipeline builder with scoped components
           pipeline_builder = ExamplePipelineBuilder(
               config=config,
               registry_manager=components["registry_manager"],
               dependency_resolver=components["resolver"]
           )
           
           # Use the pipeline builder to create and run the pipeline
           pipeline = pipeline_builder.build()
           pipeline.run()
           # Components automatically cleaned up when context exits
   ```

3. **Thread Safety Examples**:
   - Add examples showing how to use thread-local storage for thread-safe pipelines:

   ```python
   from src.v2.pipeline_deps.factory import get_thread_components
   import threading
   
   def run_in_thread():
       # Each thread gets its own isolated components
       components = get_thread_components()
       
       # Create pipeline builder with thread-local components
       pipeline_builder = ExamplePipelineBuilder(
           config=config,
           registry_manager=components["registry_manager"],
           dependency_resolver=components["resolver"]
       )
       
       # Use the pipeline builder to create and run the pipeline
       pipeline = pipeline_builder.build()
       pipeline.run()
   
   # Create and start threads for concurrent execution
   threads = []
   for i in range(3):
       thread = threading.Thread(target=run_in_thread)
       threads.append(thread)
       thread.start()
   
   # Wait for all threads to complete
   for thread in threads:
       thread.join()
   ```

4. **Best Practices Documentation**:
   - Add examples demonstrating best practices for dependency management:
     - Consistent component creation and passing
     - Explicit dependency resolution
     - Context-scoped registries for isolation

##### Files to Update

- `pipeline_examples/pytorch_bsm/mods_pipeline_bsm_pytorch_end_to_end.py`
- `pipeline_examples/pytorch_bsm/mods_pipeline_bsm_pytorch.py`
- `pipeline_examples/xgboost_atoz/mods_pipeline_xgboost_end_to_end.py`
- Any other example pipeline scripts

##### Implementation Approach

1. Create a reference implementation for one pipeline example first
2. Document the patterns and approaches used
3. Apply the same pattern to other examples
4. Add comprehensive comments explaining the dependency injection approach

### Phase 4: Testing Framework (Week 4)

#### 4.1 Create Test Helpers

- Create helper classes for test isolation:

```python
class IsolatedTestCase(unittest.TestCase):
    """Base class for tests that need isolation from global state."""
    
    def setUp(self):
        self.components = create_pipeline_components()
        self.registry_manager = self.components["registry_manager"]
        self.resolver = self.components["resolver"]
        self.semantic_matcher = self.components["semantic_matcher"]
        
    def tearDown(self):
        self.resolver.clear_cache()
        self.registry_manager.clear_all_contexts()
```

#### 4.2 Update Existing Tests

- Refactor tests to use the new approach
- Replace direct usage of global instances
- Add validation to ensure no state leakage between tests

#### 4.3 Add New Tests

- Create tests specifically for the new factory module
- Add tests for thread-local storage and context managers
- Verify concurrency safety with multi-threaded tests

### Phase 5: Documentation and Cleanup (Week 5)

#### 5.1 Update API Documentation

- Update all docstrings to reflect the new approach
- Create comprehensive API documentation
- Document best practices and patterns

#### 5.2 Create Migration Guide

- Create a detailed migration guide for developers
- Provide examples of before/after code changes
- Document common patterns and pitfalls

#### 5.3 Final Cleanup

- Remove any remaining references to global singletons
- Ensure consistent coding style throughout
- Add deprecation warnings for any transitional code

## Key Design Patterns

### Factory Method Pattern

```python
def create_dependency_resolver(registry=None, semantic_matcher=None):
    """Factory method for creating a properly configured dependency resolver."""
    registry = registry or SpecificationRegistry()
    semantic_matcher = semantic_matcher or SemanticMatcher()
    return UnifiedDependencyResolver(registry, semantic_matcher)
```

### Dependency Injection Pattern

```python
class StepBuilder:
    def __init__(self, config, dependency_resolver=None):
        self.config = config
        self.dependency_resolver = dependency_resolver or create_dependency_resolver()
        
    def resolve_dependencies(self, dependencies):
        # Use the injected dependency resolver
        return self.dependency_resolver.resolve_step_dependencies(...)
```

### Context Manager Pattern

```python
with dependency_resolution_context() as components:
    resolver = components["resolver"]
    result = resolver.resolve_step_dependencies(...)
    # Automatically cleans up when context exits
```

### Thread-Local Storage Pattern

```python
def get_current_components():
    """Get thread-specific component instances."""
    if not hasattr(_thread_local, 'components'):
        _thread_local.components = create_pipeline_components()
    return _thread_local.components
```

## Implementation Timeline

| Phase | Task | Status | Weeks | Dependencies |
|-------|------|--------|-------|-------------|
| 1.1 | Semantic Matcher Refactoring | ✅ COMPLETED | 0.5 | None |
| 1.2 | Registry Manager Refactoring | ✅ COMPLETED | 0.5 | None |
| 1.3 | Dependency Resolver Refactoring | ✅ COMPLETED | 0.5 | 1.1 |
| 2.1 | Create Factory Module | ✅ COMPLETED | 0.5 | 1.1, 1.2, 1.3 |
| 2.2 | Update Step Builder Base Class | ✅ COMPLETED | 0.5 | 1.3, 2.1 |
| 2.3 | Add Context Management | ✅ COMPLETED | 0.5 | 2.1 |
| 2.4 | Add Thread-Local Storage | ✅ COMPLETED | 0.5 | 2.1 |
| 3.1 | Update Pipeline Builder Classes | ✅ COMPLETED | 0.5 | 2.2 |
| 3.2 | Update Step Builder Implementations | ✅ COMPLETED | 1.0 | 2.2, 3.1 |
| 3.3 | Update Pipeline Examples | 🔄 IN PROGRESS | 0.5 | 3.1, 3.2 |
| 4.1 | Create Test Helpers | 📝 PLANNED | 0.5 | 2.1 |
| 4.2 | Update Existing Tests | 📝 PLANNED | 1.0 | 4.1 |
| 4.3 | Add New Tests | 📝 PLANNED | 0.5 | 4.1 |
| 5.1 | Update API Documentation | 📝 PLANNED | 0.5 | All |
| 5.2 | Create Migration Guide | 📝 PLANNED | 0.5 | All |
| 5.3 | Final Cleanup | 📝 PLANNED | 0.5 | All |

Total estimated time: 9 weeks of developer effort, likely spanning 4-5 calendar weeks with parallel work.

## Expected Benefits

### 1. Improved Testability

- Perfect isolation between tests
- No more test failures due to shared state
- Simpler test setup/teardown
- More reliable test results

### 2. Better Development Experience

- Clear dependencies between components
- No more "spooky action at a distance"
- Easier debugging of component interactions
- Better separation of concerns

### 3. Enhanced Concurrency Support

- Thread-safe operation without synchronization bottlenecks
- Support for parallel pipeline execution
- Per-thread component instances for isolation

### 4. More Maintainable Code

- Explicit dependencies in function signatures
- Easier to understand and modify component relationships
- Clear ownership of component lifecycles
- Consistent patterns throughout the codebase

## Risk Mitigation

### 1. Backward Compatibility

- Provide factory methods that mimic global access patterns
- Add comprehensive warning messages
- Ensure clear error messages for migration issues
- Create detailed migration guide

### 2. Performance Impact

- Use caching in factory methods to avoid overhead
- Optimize component creation
- Add performance tests to compare before/after
- Monitor key performance metrics during migration

### 3. Migration Complexity

- Break changes into small, manageable PRs
- Focus on one component at a time
- Provide detailed examples and guidance
- Create comprehensive tests for each changed component

## Special Path Handling for Model Evaluation

The recent improvements to script path resolution revealed inconsistencies in how different step types handle path resolution. Specifically, we identified and fixed a special case with the XGBoost Model Evaluation step:

1. **Path Resolution Issue**:
   - The `XGBoostModelEvalStepBuilder` expects to receive the entry point name and source directory separately
   - However, the `get_script_path()` method was combining these into a single path
   - This inconsistency caused errors when the pipeline was executed

2. **Implementation Solution**:
   - Modified `get_script_path()` in `XGBoostModelEvalConfig` to return only the entry point name without combining with source directory
   - Updated the builder code to explicitly document this special case
   - Created comprehensive documentation in `model_evaluation_path_handling.md` to explain the design decision

3. **Key Benefits**:
   - Fixed pipeline execution errors without requiring changes to multiple components
   - Preserved backward compatibility with existing code
   - Added clear documentation to prevent future confusion
   - Implemented the least invasive solution to minimize risk

This change demonstrates our commitment to maintaining backward compatibility while addressing technical debt. The solution follows the principle of least surprise and provides a clear path forward for future path handling standardization.

## Progress Summary as of July 10, 2025

### Pipeline Dependencies Modernization

Today we completed significant updates to the pipeline dependency system:

1. **Relative Imports Implementation**:
   - Verified all files in `src/v2/pipeline_deps` are using proper relative imports
   - Ensured consistent import patterns across all dependency-related modules
   - Confirmed proper imports in `dependency_resolver.py`, `registry_manager.py`, `semantic_matcher.py`, `specification_registry.py`, `base_specifications.py`, and `factory.py`

2. **Template Pipeline Modernization**:
   - Updated `src/v2/pipeline_builder/template_pipeline_xgboost_train_evaluate_e2e.py` to use relative imports
   - Removed `HyperparameterPrepConfig` from the pipeline template
   - Restructured pipeline DAG to match the new flow requirements:
     - data load training → tabular preprocessing training → xgboost training
     - xgboost training → package step + payload step → mims registration
     - data load calibration → tabular preprocessing calibration → model evaluation
     - xgboost training → model evaluation
   - Updated template to properly handle payload config's inheritance change
   - Modified _create_execution_doc_config and _store_pipeline_metadata to handle new PayloadConfig structure

3. **Step Naming Standardization**:
   - Fixed naming inconsistencies between configuration classes and step builders
   - Removed redundant "ModelRegistration" entry from step_names.py registry 
   - Removed redundant "XGBoostModelEvaluation" entry from step_names.py registry
   - Updated model_eval_spec.py to use the canonical "XGBoostModelEval" name
   - Updated pipeline_builder_template.py to use the centralized step names registry
   - Implemented proper fallback mechanism when configs aren't in the registry
   - Updated XGBoostTrainEvaluateE2ETemplate to use spec_type from the step names registry
   - Verified there are no other duplicate entries in the step name registry
   - This change ensures consistent step naming across the entire pipeline system

4. **Config Hierarchy Refactoring**:
   - Changed PayloadConfig to inherit from ProcessingStepConfigBase instead of ModelRegistrationConfig
   - Added necessary registration-related fields that were previously inherited
   - Updated builder validation to align with the new inheritance structure
   - Simplified step name generation and argument handling in the payload builder
   - This change makes the config inheritance hierarchy more logical and consistent

5. **Data Loading Step Specification Improvements**:
   - Created job type-specific specifications for all Cradle data loading steps: 
     - Created `data_loading_calibration_spec.py` for calibration data
     - Created `data_loading_validation_spec.py` for validation data
     - Created `data_loading_testing_spec.py` for testing data
   - Ensured each specification has proper OutputSpec objects with correct properties
   - Updated CradleDataLoadingStepBuilder to use the appropriate spec for each job type
   - Fixed the issue where the calibration data loading step was failing with: "'str' object has no attribute 'logical_name'"
   - This improves spec-based pipeline dependency management for data loading steps

6. **Enum Hashability Fix**:
   - Added `__hash__` method to `DependencyType` enum in `base_specifications.py`
   - Added `__hash__` method to `NodeType` enum for consistency
   - Fixed the error: `TypeError: unhashable type: 'DependencyType'` that was occurring in the dependency resolver
   - The error happened because Python makes classes that override `__eq__` without `__hash__` unhashable
   - This ensures that DependencyType can be properly used as dictionary keys in the compatibility matrix
   - The fix maintains consistent hash values based on the enum value, which preserves dictionary lookup behavior

7. **MIMS Packaging Step Fixes**:
   - Fixed error in `builder_mims_packaging_step.py` that was referencing non-existent `job_type` attribute
   - Removed references to `job_type` in `_create_processor` method, replacing with static "ModelPackaging" name
   - Updated `_get_job_arguments` method to use "--mode standard" instead of job_type-based arguments
   - Removed references to `job_type` in `create_step` method for step name generation
   - This fixes the error: `AttributeError: 'PackageStepConfig' object has no attribute 'job_type'`
   - Improved consistency by using static naming which is more appropriate for packaging steps that don't have job types

8. **ProcessingStep Parameter Fix**:
   - Fixed error in `builder_mims_packaging_step.py` and `builder_mims_payload_step.py` where ProcessingStep was being created with invalid parameter
   - Removed `source_dir` parameter from ProcessingStep constructor call in create_step() for both step builders
   - Updated script path handling to use `config.get_script_path()` which returns the full path
   - Added fallback to use contract's entry_point when config.get_script_path() returns None
   - Completely removed the unnecessary `source_dir` variable reference in both files
   - This fixes the error: `ValueError: Failed to build step model_packaging: ProcessingStep.__init__() got an unexpected keyword argument 'source_dir'`
   - ProcessingStep only accepts the `code` parameter for script path, not separate source_dir and entry_point parameters
   - Improved code reuse by implementing the same script path resolution logic in both step builders

9. **MIMS Registration Step Integration Fix**:
   - Fixed error in `builder_mims_registration_step.py` where the MIMS SDK validation was failing on property objects
   - Removed explicit StringLikeWrapper usage from the MIMS registration step builder
   - Modified the _get_inputs method to work directly with source inputs without wrapping them
   - Added comprehensive documentation in `slipbox/v2/pipeline_design/mims_registration_integration.md` explaining the proxy pattern design
   - Improved logging to document how inputs are being handled
   - This fixes the errors: `AttributeError: 'dict' object has no attribute 'startswith'` and `AttributeError: 'dict' object has no attribute 'expr'`
   - Eliminated error-prone code paths related to property reference manipulation
   - The updated approach now relies on automatic property reference handling in the base class
   - Property references are preserved intact for SageMaker's runtime resolution while satisfying the MIMS SDK validation requirements
   - This change simplifies step builder code and makes it more maintainable by removing special-case handling

10. **Cradle Data Loading Property Reference Fix**:
   - Solved error in Cradle Data Loading step that caused: `AttributeError: 'dict' object has no attribute 'decode'`
   - Created a specialized `CradleOnlyTemplate` pipeline template that applies the StringLikeWrapper to PIPELINE_EXECUTION_TEMP_DIR
   - Used monkey patching approach to temporarily replace constants during pipeline assembly without modifying third-party code
   - Implemented proper cleanup in finally block to restore original values even if exceptions occur
   - Created detailed documentation in `slipbox/v2/pipeline_design/cradle_data_loading_property_reference_fix.md`
   - This approach complements the StringLikeWrapper used in MIMS registration and the handle_property_reference method in other step builders
   - The solution demonstrates a non-invasive approach to fixing property reference handling issues in third-party components

11. **Property Reference Wrapper Enhancement**:
   - Enhanced the `StringLikeWrapper` class to fully handle all URL parsing operations
   - Added implementation of the `decode()` method to address `AttributeError: 'dict' object has no attribute 'decode'`
   - Extended wrapper application to cover more constants and directly wrap property references in the pipeline assembler
   - Added comprehensive logging to assist with debugging property reference handling
   - Created detailed design document `slipbox/v2/pipeline_design/property_reference_wrapper_enhancement.md`
   - Solved recurring validation issues with SageMaker property references that appear in multiple template types
   - Implemented standardized wrapping approach that automatically handles property references during pipeline assembly
   - This enhancement ensures that property references consistently pass validation while preserving their runtime behavior

12. **MIMS Registration Step Modernization**:
   - Updated the approach to property reference handling in the MIMS Registration step
   - Removed explicit StringLikeWrapper usage from the MIMS registration step builder
   - Modified the _get_inputs method to work directly with source inputs without wrapping them
   - Added comprehensive documentation in `slipbox/v2/pipeline_design/mims_registration_integration.md` explaining the proxy pattern design
   - The step now relies on the automatic property reference handling in the base class
   - This design centralizes property reference handling and makes the step builder more maintainable
   - Property references are preserved intact for SageMaker's runtime resolution
   - Simplified step builder code by eliminating special-case handling for property references
   - This change completes the modernization of all MIMS-related steps to use the unified property reference approach

13. **Property Reference Handling Cleanup**:
   - Removed the redundant `property_reference_wrapper.py` module from `src/v2/pipeline_builder`
   - The module contained a `StringLikeWrapper` class and methods for monkey-patching constants, which have been superseded by our enhanced `PropertyReference` class
   - Deleted redundant documentation files:
     - `slipbox/v2/pipeline_design/property_reference_handling.md`
     - `slipbox/v2/pipeline_design/property_reference_wrapper_enhancement.md` 
     - `slipbox/v2/pipeline_design/property_reference_handling_cleanup.md`
     - `slipbox/v2/pipeline_design/cradle_data_loading_property_reference_fix.md`
   - Consolidated all property reference documentation in `slipbox/v2/pipeline_design/enhanced_property_reference.md`
   - Verified no remaining references to the removed module or its classes
   - Verified that `pipeline_assembler.py` correctly uses the enhanced `PropertyReference` class
   - This cleanup simplifies our codebase by eliminating multiple competing solutions to the same problem
   - Standardized on a single approach to property reference handling through the robust `PropertyReference` class
