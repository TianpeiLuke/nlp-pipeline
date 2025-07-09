# Removing Global Singleton Objects

**Date:** July 8, 2025  
**Status:** ✅ COMPLETED
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

### Phase 1: Core Component Refactoring (Week 1)

#### 1.1 Semantic Matcher Refactoring

Detailed implementation plan: [Phase 1: Semantic Matcher Implementation](./2025-07-08_phase1_semantic_matcher_implementation.md)

- Remove global `semantic_matcher` instance from `semantic_matcher.py`
- Update `__all__` list to reflect changes
- Document the changes in the class docstrings

#### 1.2 Registry Manager Refactoring

Detailed implementation plan: [Phase 1: Registry Manager Implementation](./2025-07-08_phase1_registry_manager_implementation.md)

- Remove global `registry_manager` instance from `registry_manager.py`
- Update convenience functions to accept a manager instance
- Update `integrate_with_pipeline_builder` to accept a registry manager
- Update `__all__` list to reflect changes

#### 1.3 Dependency Resolver Refactoring

Detailed implementation plan: [Phase 1: Dependency Resolver Implementation](./2025-07-08_phase1_dependency_resolver_implementation.md)

- Remove global `global_resolver` instance from `dependency_resolver.py`
- Update `UnifiedDependencyResolver` constructor to accept both registry and semantic matcher
- Add factory function `create_dependency_resolver()` for simplified object creation
- Update `__all__` list to reflect changes

### Phase 2: Dependency Injection Implementation (COMPLETED)

#### 2.1 Create Factory Module

- Create a new module `factory.py` in `pipeline_deps` package
- Implement factory functions for creating properly configured components
- Add convenience methods for common use cases
- Include comprehensive documentation and examples

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

#### 2.2 Update Step Builder Base Class

- Modify `StepBuilderBase` constructor to accept necessary dependencies:

```python
def __init__(self, config, spec=None, sagemaker_session=None, role=None,
             notebook_root=None, registry_manager=None, dependency_resolver=None):
    # ...
    self.registry_manager = registry_manager
    self.dependency_resolver = dependency_resolver
    # ...
```

- Add helper methods for creating or getting dependencies:

```python
def _get_registry_manager(self):
    """Get or create a registry manager."""
    if not hasattr(self, '_registry_manager') or self._registry_manager is None:
        from ..pipeline_deps.registry_manager import RegistryManager
        self._registry_manager = RegistryManager()
    return self._registry_manager

def _get_dependency_resolver(self):
    """Get or create a dependency resolver."""
    if not hasattr(self, '_dependency_resolver') or self._dependency_resolver is None:
        from ..pipeline_deps.factory import create_dependency_resolver
        registry = self._get_registry().get_registry(self._get_context_name())
        self._dependency_resolver = create_dependency_resolver(registry)
    return self._dependency_resolver
```

#### 2.3 Add Context Management

- Implement context managers for scoped component usage:

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
            components["registry_manager"].clear_context("default")
```

#### 2.4 Add Thread-Local Storage

- Implement thread-local storage for per-thread component instances:

```python
# Thread-local storage for per-thread instances
_thread_local = threading.local()

def get_thread_components():
    """Get thread-specific component instances."""
    if not hasattr(_thread_local, 'components'):
        _thread_local.components = create_pipeline_components()
    return _thread_local.components
```

### Phase 3: Application Updates (In Progress)

#### 3.1 Update Pipeline Builder Classes

- Modify pipeline builder classes to use dependency injection:

```python
class PipelineBuilderBase:
    def __init__(self, config, registry_manager=None, dependency_resolver=None):
        self.config = config
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

#### 3.2 Update Step Builder Implementations

- Modify step builder classes to use the injected dependencies
- Update methods that currently use global instances
- Ensure proper dependency passing between components

#### 3.3 Update Pipeline Examples

- Update example pipeline scripts to demonstrate proper component creation
- Add examples of using the factory module
- Document best practices for dependency management

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

| Phase | Task | Weeks | Dependencies |
|-------|------|-------|-------------|
| 1.1 | Semantic Matcher Refactoring | 0.5 | None |
| 1.2 | Registry Manager Refactoring | 0.5 | None |
| 1.3 | Dependency Resolver Refactoring | 0.5 | 1.1 |
| 2.1 | Create Factory Module | 0.5 | 1.1, 1.2, 1.3 |
| 2.2 | Update Step Builder Base Class | 0.5 | 1.3, 2.1 |
| 2.3 | Add Context Management | 0.5 | 2.1 |
| 2.4 | Add Thread-Local Storage | 0.5 | 2.1 |
| 3.1 | Update Pipeline Builder Classes | 0.5 | 2.2 |
| 3.2 | Update Step Builder Implementations | 1.0 | 2.2, 3.1 |
| 3.3 | Update Pipeline Examples | 0.5 | 3.1, 3.2 |
| 4.1 | Create Test Helpers | 0.5 | 2.1 |
| 4.2 | Update Existing Tests | 1.0 | 4.1 |
| 4.3 | Add New Tests | 0.5 | 4.1 |
| 5.1 | Update API Documentation | 0.5 | All |
| 5.2 | Create Migration Guide | 0.5 | All |
| 5.3 | Final Cleanup | 0.5 | All |

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

## Conclusion

Removing global singleton objects from the pipeline dependencies system will significantly improve testability, debugging, and concurrent execution. The phased approach outlined in this plan will minimize disruption while ensuring comprehensive refactoring of all affected components. The end result will be a more maintainable, testable, and reliable codebase.

By implementing proper dependency injection, factory methods, and context management, we can maintain the convenience of the current approach while eliminating the issues associated with global state. This change aligns with software engineering best practices and will set a strong foundation for future development.
