# Removing Global Singleton Objects

## Overview

This document outlines the plan to remove all references to global singleton objects in the `src/pipeline_deps` directory and related components. The goal is to improve testability, reduce global state issues, and make the codebase more maintainable.

## Identified Global Singletons

1. **Registry Manager**:
   - Global instance: `registry_manager` in `registry_manager.py`
   - Used for managing multiple isolated specification registries

2. **Semantic Matcher**:
   - Global instance: `semantic_matcher` in `semantic_matcher.py`
   - Used for calculating semantic similarity between names

3. **Dependency Resolver**:
   - Global instance: `global_resolver` in `dependency_resolver.py`
   - Used for resolving dependencies between steps

## Implementation Plan

### Phase 1: Core Refactoring

- [ ] **Remove Global Instances**
  - [ ] Remove `registry_manager` from `registry_manager.py`
  - [ ] Remove `semantic_matcher` from `semantic_matcher.py`
  - [ ] Remove `global_resolver` from `dependency_resolver.py`

- [ ] **Update Convenience Functions**
  - [ ] Modify functions in `registry_manager.py` to accept a manager instance
  - [ ] Update any other convenience functions in related modules

### Phase 2: Dependency Injection

- [ ] **Update Class Constructors**
  - [ ] Modify `UnifiedDependencyResolver` to accept a `SemanticMatcher` instance
  - [ ] Update other classes to accept dependencies as constructor parameters

- [ ] **Create Factory Functions**
  - [ ] Add factory functions to simplify object creation
  - [ ] Ensure proper dependency injection throughout the codebase

### Phase 3: Step Builder Updates

- [ ] **Update Step Builder Base Class**
  - [ ] Modify `StepBuilderBase` to accept necessary instances
  - [ ] Update methods that interact with dependency resolver

- [ ] **Update Pipeline Builder Integration**
  - [ ] Modify `integrate_with_pipeline_builder` decorator
  - [ ] Update pipeline builder classes to use the new non-singleton approach

### Phase 4: Testing and Validation

- [ ] **Update Tests**
  - [ ] Fix tests in `test_registry_manager.py`
  - [ ] Fix tests in `test_global_state_isolation.py`
  - [ ] Update other affected tests

- [ ] **Validate Changes**
  - [ ] Run all tests to ensure they pass
  - [ ] Verify that global state issues are resolved

### Phase 5: Documentation and Cleanup

- [ ] **Update Documentation**
  - [ ] Update docstrings to reflect the new approach
  - [ ] Add examples of proper usage

- [ ] **Final Cleanup**
  - [ ] Remove any remaining references to global singletons
  - [ ] Ensure consistent coding style throughout

## Detailed Implementation Notes

### Removing Global Instances

```python
# BEFORE (at the end of each module)
registry_manager = RegistryManager()
global_resolver = UnifiedDependencyResolver()
semantic_matcher = SemanticMatcher()

# AFTER
# Remove these global instances completely
```

### Updating Convenience Functions

```python
# BEFORE
def get_registry(context_name: str = "default") -> SpecificationRegistry:
    return registry_manager.get_registry(context_name)

# AFTER
def get_registry(manager: RegistryManager, context_name: str = "default") -> SpecificationRegistry:
    return manager.get_registry(context_name)
```

### Updating Class Constructors

```python
# BEFORE
class UnifiedDependencyResolver:
    def __init__(self, registry: Optional[SpecificationRegistry] = None):
        self.registry = registry or SpecificationRegistry()
        self.semantic_matcher = SemanticMatcher()

# AFTER
class UnifiedDependencyResolver:
    def __init__(self, registry: SpecificationRegistry, semantic_matcher: SemanticMatcher):
        self.registry = registry
        self.semantic_matcher = semantic_matcher
```

### Updating Step Builders

```python
# BEFORE
class StepBuilderBase(ABC):
    def __init__(self, config, spec=None, sagemaker_session=None, role=None, notebook_root=None):
        # ...

# AFTER
class StepBuilderBase(ABC):
    def __init__(self, config, spec=None, sagemaker_session=None, role=None, notebook_root=None, 
                 dependency_resolver=None):
        self.dependency_resolver = dependency_resolver
        # ...
```

### Creating Factory Functions

```python
def create_pipeline_components():
    """Create all necessary pipeline components with proper dependencies."""
    semantic_matcher = SemanticMatcher()
    registry_manager = RegistryManager()
    default_registry = registry_manager.get_registry("default")
    dependency_resolver = UnifiedDependencyResolver(default_registry, semantic_matcher)
    
    return {
        "semantic_matcher": semantic_matcher,
        "registry_manager": registry_manager,
        "dependency_resolver": dependency_resolver
    }
```

## Benefits

1. **Improved Testability**: Tests can create isolated instances without affecting global state
2. **Reduced Global State Issues**: No more issues with tests failing when run together
3. **Better Dependency Management**: Clear dependencies between components
4. **Enhanced Maintainability**: Easier to understand and modify the codebase

## Timeline

- Phase 1: 1-2 days
- Phase 2: 1-2 days
- Phase 3: 2-3 days
- Phase 4: 1-2 days
- Phase 5: 1 day

Total estimated time: 6-10 days
