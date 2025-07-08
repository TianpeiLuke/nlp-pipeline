# Specification-Driven Step Builder Implementation Plan

**Date:** July 8, 2025  
**Status:** ✅ IMPLEMENTATION PHASE  
**Priority:** 🔥 HIGH - Foundation for Pipeline Simplification  
**Related Documents:** 
- [2025-07-07_step_name_consistency_implementation_status.md](./2025-07-07_step_name_consistency_implementation_status.md)
- [2025-07-05_alignment_validation_implementation_plan.md](./2025-07-05_alignment_validation_implementation_plan.md)
- [2025-07-07_phase5_training_step_modernization_summary.md](./2025-07-07_phase5_training_step_modernization_summary.md)
- [2025-07-07_phase6_model_steps_implementation_summary.md](./2025-07-07_phase6_model_steps_implementation_summary.md)
- [2025-07-07_phase6_2_registration_step_implementation_summary.md](./2025-07-07_phase6_2_registration_step_implementation_summary.md)

## Executive Summary

This document outlines a comprehensive plan to simplify step builders by leveraging step specifications and script contracts. The goal is to eliminate redundant code in step builders by using the declarative specifications as the source of truth for input/output mappings. This approach will make step builders more maintainable, consistent, and less error-prone.

## Current Architecture Analysis

Our current pipeline architecture has well-designed components:

1. **Step Specifications** (`src/v2/pipeline_step_specs`)
   - Define logical names for inputs/outputs
   - Define property paths for accessing outputs at runtime
   - Provide semantic keywords for dependency matching
   - Example: `processed_data` as the logical name

2. **Script Contracts** (`src/v2/pipeline_script_contracts`)
   - Define mapping between logical names and container paths
   - Example: Maps `processed_data` → `/opt/ml/processing/output`
   - Defines required and optional environment variables

3. **Step Builders** (`src/v2/pipeline_steps`)
   - Currently reimplement input/output mappings manually
   - Do not fully leverage specifications and contracts
   - Contain redundant code that could be simplified

4. **Dependency Resolution** (`src/v2/pipeline_deps`)
   - `UnifiedDependencyResolver`: Resolves dependencies between steps
   - `SpecificationRegistry`: Manages step specifications
   - `SemanticMatcher`: Calculates semantic similarity between names
   - Provides sophisticated scoring and matching algorithms

## Current Pain Points

1. **Redundant Input/Output Mapping**: Step builders duplicate logic already defined in contracts
   - `_get_processor_inputs()` reimplements mapping that's defined in the contract
   - `_get_processor_outputs()` reimplements mapping that's defined in the contract

2. **Inconsistent Method Naming**: Different step types use different method names
   - ProcessingStep uses `_get_processor_inputs()` and `_get_processor_outputs()`
   - TrainingStep uses different patterns for channel mapping
   - Creates confusion and makes maintenance harder

3. **Complex Configuration**: Config classes require `input_names` and `output_names`
   - Duplicates information already in contracts
   - Error-prone when manually maintained

4. **Error-prone String Handling**: Many hardcoded paths and strings
   - Container paths duplicated between contracts and step builders
   - Easy to make typos or get paths wrong

5. **Redundant Dependency Resolution**: StepBuilderBase has complex matching logic
   - `extract_inputs_from_dependencies` duplicates functionality in `UnifiedDependencyResolver`
   - Multiple matching methods that could be simplified with the specification system

6. **Duplicate Property Path Storage**: 
   - `StepBuilderBase._PROPERTY_PATH_REGISTRY` duplicates information in `OutputSpec.property_path`
   - `register_property_path` method creates a parallel system to specifications

## Proposed Solution

### 1. Remove input_names and output_names from Config Classes

```python
class BasePipelineConfig:
    """
    Base class for pipeline step configurations.
    
    This class no longer requires input_names and output_names mappings,
    as these are now sourced directly from script contracts.
    """
    
    def __init__(self, region: str, pipeline_s3_loc: str, *args, **kwargs):
        """Initialize base pipeline configuration."""
        self.region = region
        self.pipeline_s3_loc = pipeline_s3_loc
        
        # No longer initializing input_names and output_names
    
    def get_script_contract(self):
        """Get script contract for this configuration."""
        # Base implementation - derived classes should override
        return None
    
    @property
    def script_contract(self):
        """Property accessor for script contract."""
        return self.get_script_contract()
```

### 2. Standardize StepBuilderBase Methods

```python
class StepBuilderBase(ABC):
    """Base class for all step builders."""
    
    def __init__(self, config, spec=None, sagemaker_session=None, role=None, notebook_root=None):
        """Initialize with specification if available."""
        self.config = config
        self.spec = spec  # Store the specification
        self.contract = getattr(spec, 'script_contract', None) if spec else None
        # ...
    
    @abstractmethod
    def _get_inputs(self, inputs: Dict[str, Any]) -> Any:
        """
        Get inputs for the step.
        
        This is a unified method that all derived classes must implement.
        """
        pass
        
    @abstractmethod
    def _get_outputs(self, outputs: Dict[str, Any]) -> Any:
        """
        Get outputs for the step.
        
        This is a unified method that all derived classes must implement.
        """
        pass
```

### 3. Add Specification-Driven Helper Methods

```python
def _map_logical_to_container_path(self, logical_name: str) -> Optional[str]:
    """Map a logical input/output name to its container path using the contract."""
    if not self.contract:
        return None
        
    # Try input paths
    if hasattr(self.contract, 'expected_input_paths'):
        if logical_name in self.contract.expected_input_paths:
            return self.contract.expected_input_paths[logical_name]
            
    # Try output paths
    if hasattr(self.contract, 'expected_output_paths'):
        if logical_name in self.contract.expected_output_paths:
            return self.contract.expected_output_paths[logical_name]
            
    return None
    
def _create_standard_processing_input(self, logical_name: str, inputs: Dict[str, Any]) -> Any:
    """
    Create a standard ProcessingInput for the given logical name.
    
    This simplified version automatically gets the container path from the contract.
    """
    # Import ProcessingInput here to avoid circular imports
    from sagemaker.processing import ProcessingInput
    
    if logical_name not in inputs:
        raise ValueError(f"Input '{logical_name}' not found in inputs dictionary")
    
    # Get container path from contract
    container_path = self._map_logical_to_container_path(logical_name)
    if not container_path:
        raise ValueError(f"No container path found for logical name '{logical_name}' in contract")
        
    # Create ProcessingInput
    return ProcessingInput(
        input_name=logical_name,      # Use logical name directly
        source=inputs[logical_name],  # S3 URI from pipeline
        destination=container_path     # From contract
    )
```

## Revised Implementation Plan

### Phase 1: StepBuilderBase Updates (Week 1) - COMPLETED

#### 1.1 Add Specification and Contract Support
- [x] Add `spec` parameter to constructor
- [x] Get contract from specification
- [x] Validate specification-contract alignment

#### 1.2 Add Standardized Abstract Methods
- [x] Add abstract `_get_inputs` method
- [x] Add abstract `_get_outputs` method
- [x] Add helper methods for mapping logical names to container paths

#### 1.3 Add Helper Methods for ProcessingStep
- [x] Add `_create_standard_processing_input` method
- [x] Add `_create_standard_processing_output` method
- [x] Add helper methods for finding specifications

### Phase 2: Initial Implementation (Week 2) - COMPLETED

#### 2.1 Update BasePipelineConfig
- [x] Update `BasePipelineConfig` to no longer require input_names/output_names
- [x] Add `get_script_contract` method
- [x] Add `script_contract` property accessor
- [x] Update `get_script_path` to use contract

#### 2.2 Update TabularPreprocessingConfig
- [x] Update `TabularPreprocessingConfig` to remove input_names/output_names
- [x] Add `get_script_contract` implementation
- [x] Update constructors to use script contracts

#### 2.3 Redesign TabularPreprocessingStepBuilder
- [x] Replace `_get_processor_inputs` with specification-driven `_get_inputs`
- [x] Replace `_get_processor_outputs` with specification-driven `_get_outputs`
- [x] Remove redundant methods like `_get_processor_inputs_traditional`
- [x] Simplify code to use contract and spec exclusively
- [x] Integrate with `UnifiedDependencyResolver` for dependency extraction (in StepBuilderBase)
- [x] Update `build` method to use resolver (in StepBuilderBase)

#### 2.4 Enhance StepBuilderBase Dependency Resolution
- [x] Add `extract_inputs_using_resolver` to StepBuilderBase for centralized resolver integration
- [x] Modify `extract_inputs_from_dependencies` to prioritize resolver over traditional methods
- [x] Deprecate redundant matching methods: `_match_list_outputs`, `_match_dict_outputs`
- [x] Add type hints and comprehensive logging for dependency resolution
- [x] Create comprehensive tests for resolver-based dependency resolution

#### 2.5 Remove Redundant Helper Methods
- [x] Add deprecation warnings to `get_input_requirements` and `get_output_properties`
- [x] Modify `extract_inputs_from_dependencies` to use spec.dependencies directly
- [x] Modify `_check_missing_inputs` to use spec.dependencies directly
- [x] Create replacement methods for accessing dependencies and outputs directly from specs
- [x] Create tests for the direct specification access methods

### Phase 3: Complete StepBuilderBase Integration (HIGH PRIORITY) (Week 3) - COMPLETED

#### 3.1 Implement Direct Specification Access
- [x] Create `get_required_dependencies` method that directly uses spec.dependencies
- [x] Create `get_optional_dependencies` method that directly uses spec.dependencies
- [x] Create `get_outputs` method that directly uses spec.outputs
- [x] Add comprehensive tests for the new accessor methods

#### 3.2 Replace Property Path Registry (From Previous Phase 6.1)
- [x] Add specification-based `get_property_path` method
- [x] Update `get_all_property_paths` to use specifications
- [x] Add template support to `OutputSpec` via format_args parameter
- [x] Mark registry methods as deprecated
- [x] Create migration guidance for transitioning from registry to specifications

#### 3.3 Complete Dependency Resolution Integration (From Previous Phase 6.2)
- [x] Create specification-driven alternatives for `_match_model_artifacts`
- [x] Create specification-driven alternatives for `_match_processing_outputs`
- [x] Implement improved fallback mechanism for steps without specifications
- [x] Create adapter for backward compatibility with custom builders
- [x] Implement comprehensive tests for the unified dependency resolution system

#### 3.4 Create Compatibility Module for Redundant Methods
- [x] Identify all deprecated and redundant methods
- [x] Create compatibility module for property path methods
- [x] Create compatibility module for dependency matching methods
- [x] Create compatibility module for input/output information methods
- [x] Update docstrings with migration guidance

#### 3.5 Fully Specification-Driven StepBuilderBase (NEW)
- [x] Remove LegacyCompatibility inheritance from StepBuilderBase
- [x] Replace extract_inputs_from_dependencies with resolver-only version
- [x] Make extract_inputs_from_dependencies require a specification
- [x] Raise ValueError if specification is not provided
- [x] Add module initialization to ensure proper imports
- [x] Remove redundant class variables (INPUT_PATTERNS, _PROPERTY_PATH_REGISTRY)
- [x] Remove redundant instance variables (_instance_property_paths)
- [x] Remove deprecated property path registration methods
- [x] Remove legacy input/output information methods
- [x] Remove legacy matching methods (_match_inputs_to_outputs, etc.)
- [x] Update remaining methods to require specifications

### Phase 4: Apply to Processing Steps (Week 4) - COMPLETED

#### 4.1 Update CradleDataLoadingStepBuilder
- [x] Select specification based on job type during initialization
- [x] Implement `_get_inputs` and `_get_outputs` abstract methods
- [x] Update `create_step` to work with custom `CradleDataLoadingStep`
- [x] Enhance `get_output_location` to use specification-based property paths
- [x] Store specification in the step using `_spec` attribute
- [x] Remove redundant property path registrations

#### 4.2 Update TabularPreprocessingStepBuilder
- [x] Refactor to remove any dependency on legacy methods
- [x] Update to use specification-only implementation
- [x] Update constructor to use specification directly
- [x] Ensure compatibility with the new StepBuilderBase
- [x] Remove any redundant methods and variables
- [x] Add specification storage in the step using _spec attribute
- [x] Improve error handling for missing specifications

#### 4.3 Update ModelEvaluationStepBuilder
- [x] Import and use MODEL_EVAL_SPEC from specification
- [x] Implement specification-driven `_get_inputs` method
- [x] Implement specification-driven `_get_outputs` method
- [x] Use script contract for mapping logical names to container paths
- [x] Simplify code using UnifiedDependencyResolver
- [x] Store specification in the step using `_spec` attribute
- [x] Improve step naming with job type capitalization

#### 4.4 Update Other ProcessingStep Builders
- [x] Create MIMSPayloadStepBuilder with specification-driven approach
- [x] Create MIMSPackagingStepBuilder with specification-driven approach
- [x] Update CurrencyConversionStepBuilder with specification-driven approach
- [x] Update HyperparameterPrepStepBuilder with specification-driven approach
- [x] Update remaining ProcessingStep builders

#### 4.5 Update Processing Step Configurations
- [x] Update TabularPreprocessingConfig to use get_script_contract method
- [x] Update MIMSPayloadConfig to remove input_names/output_names
- [x] Update MIMSPackagingConfig to remove input_names/output_names
- [x] Update CurrencyConversionConfig to remove input_names/output_names
- [x] Update HyperparameterPrepConfig to remove redundant fields
- [x] Update XGBoostModelEvalConfig to remove input_names/output_names and use script contract
- [x] Update CradleDataLoadConfig to remove input_names/output_names
- [x] Add validation methods to configs to ensure script contracts are properly loaded

#### 4.6 Test ProcessingStep Implementation
- [x] Add unit tests for processing step configurations
- [x] Add unit tests for processing step builders
- [x] Verify dependency resolution works correctly
- [x] Ensure output paths are correctly resolved
- [x] Validate that job capitalization is consistent

### Phase 5: Extend to TrainingStep Builders (Week 5) - COMPLETED

#### 5.1 Update XGBoostTrainingStepBuilder
- [x] Implement specification-driven `_get_inputs` for training channels
- [x] Implement specification-driven `_get_outputs` for training outputs
- [x] Update `create_step` to use the new methods
- [x] Integrate with UnifiedDependencyResolver

#### 5.2 Update PytorchTrainingStepBuilder
- [x] Apply patterns established in XGBoostTrainingStepBuilder
- [x] Leverage script contracts and specs
- [x] Simplify code using UnifiedDependencyResolver

### Phase 6: Complete Model and Registration Steps (Week 6) - COMPLETED

#### 6.1 Update Model Creation Steps
- [x] Update `XGBoostModelStepBuilder` to use specification-driven approach
- [x] Update `PyTorchModelStepBuilder` to use specification-driven approach
- [x] Implement specification-driven `_get_inputs` and `_get_outputs` methods
- [x] Integrate with UnifiedDependencyResolver
- [x] Remove redundant code and simplify implementations

#### 6.2 Update Registration and Packaging Steps
- [x] Update `ModelRegistrationStepBuilder` to use specification-driven approach
- [x] Implement specification-driven `_get_inputs` and `_get_outputs` methods
- [x] Apply specification-driven patterns throughout
- [x] Simplify code using UnifiedDependencyResolver
- [x] Add backward compatibility for legacy parameter formats

### Phase 7: Final Testing and Documentation (Week 7) - IN PROGRESS

#### 7.1 Comprehensive Testing
- [ ] Test end-to-end pipelines with fully specification-driven steps
- [ ] Test mixed pipelines with both old and new style steps
- [ ] Performance testing of dependency resolution
- [ ] Verify compatibility with existing pipelines
- [ ] Validate error handling for edge cases

#### 7.2 Documentation Updates
- [ ] Update docstrings for all modified classes
- [ ] Create examples for different step types
- [ ] Update developer guide with new approach
- [ ] Create migration guide for updating existing builders
- [ ] Document common patterns and best practices

#### 7.3 Performance Optimization
- [ ] Identify and optimize any performance bottlenecks
- [ ] Benchmark dependency resolution performance
- [ ] Create performance comparison metrics (before vs. after)
- [ ] Implement caching improvements if needed

#### 7.4 Final Code Cleanup
- [ ] Remove any remaining deprecated methods
- [ ] Eliminate all redundant code paths
- [ ] Standardize naming and interfaces across all builders
- [ ] Apply consistent logging patterns

## Progress Tracking

| Phase | Task | Status | Date | Priority |
|-------|------|--------|------|----------|
| 1.1 | Add spec parameter to constructor | ✅ Complete | July 7, 2025 | High |
| 1.2 | Add standardized abstract methods | ✅ Complete | July 7, 2025 | High |
| 1.3 | Add helper methods for ProcessingStep | ✅ Complete | July 7, 2025 | High |
| 2.1 | Update BasePipelineConfig | ✅ Complete | July 7, 2025 | High |
| 2.2 | Update TabularPreprocessingConfig | ✅ Complete | July 7, 2025 | High |
| 2.3 | Redesign TabularPreprocessingStepBuilder | ✅ Complete | July 7, 2025 | High |
| 2.4 | Enhance StepBuilderBase dependency resolution | ✅ Complete | July 7, 2025 | High |
| 2.5 | Add deprecation warnings to redundant methods | ✅ Complete | July 7, 2025 | High |
| 3.1 | Implement direct specification access methods | ✅ Complete | July 7, 2025 | **HIGH** |
| 3.2 | Replace property path registry | ✅ Complete | July 7, 2025 | **HIGH** |
| 3.3 | Complete dependency resolution integration | ✅ Complete | July 7, 2025 | **HIGH** |
| 3.4 | Create compatibility module for redundant methods | ✅ Complete | July 7, 2025 | **HIGH** |
| 3.5 | Fully Specification-Driven StepBuilderBase | ✅ Complete | July 7, 2025 | **HIGH** |
| 4.1 | Update CradleDataLoadingStepBuilder | ✅ Complete | July 7, 2025 | Medium |
| 4.2 | Update TabularPreprocessingStepBuilder | ✅ Complete | July 7, 2025 | Medium |
| 4.3 | Update ModelEvaluationStepBuilder | ✅ Complete | July 7, 2025 | Medium |
| 4.4 | Update Other ProcessingStep Builders | ✅ Complete | July 7, 2025 | Medium |
| 4.5 | Update Processing Step Configurations | ✅ Complete | July 7, 2025 | Medium |
| 4.6 | Test ProcessingStep Implementation | ✅ Complete | July 7, 2025 | Medium |
| 5.1 | Update XGBoostTrainingStepBuilder | ✅ Complete | July 7, 2025 | Medium |
| 5.2 | Update PytorchTrainingStepBuilder | ✅ Complete | July 7, 2025 | Medium |
| 6.1 | Update Model Creation Steps | ✅ Complete | July 7, 2025 | Medium |
| 6.2 | Update Registration and Packaging Steps | ✅ Complete | July 7, 2025 | Medium |
| 7.1 | Comprehensive Testing | 🔄 In Progress | July 8, 2025 | High |
| 7.2 | Documentation Updates | 🔄 In Progress | July 8, 2025 | High |
| 7.3 | Performance Optimization | 📝 Planned | July 9, 2025 | Medium |
| 7.4 | Final Code Cleanup | 📝 Planned | July 9, 2025 | Medium |

## Summary of Achievements

### ✅ Completed: Processing Steps (Phase 4)
- All processing step builders now use specification-driven architecture
- CurrencyConversionStepBuilder demonstrates dynamic specification loading based on job type
- Comprehensive test coverage added for dependency resolution
- Consistent interfaces across all processing step builders

### ✅ Completed: Training Steps (Phase 5)
- Both XGBoost and PyTorch training step builders now follow specification-driven pattern
- ~300 lines of complex custom matching code eliminated
- No more manual property path registrations
- Single source of truth for input/output specifications

### ✅ Completed: Model Steps (Phase 6.1)
- XGBoost and PyTorch model step builders modernized
- ~380 lines of legacy code removed
- 47% code reduction in model builders
- Perfect alignment between specifications and implementations

### ✅ Completed: Registration Step (Phase 6.2)
- ModelRegistrationStepBuilder fully modernized
- 66% code reduction (330 lines removed)
- Complex parameter handling eliminated
- Backward compatibility maintained for existing pipelines

### 🔄 In Progress: Final Testing & Documentation (Phase 7)
- End-to-end pipeline testing with fully specification-driven steps
- Documentation updates for new architecture
- Performance benchmarking and optimization
- Final code cleanup and standardization

## Next Steps

1. Complete the final comprehensive testing to verify the end-to-end pipeline with fully specification-driven steps
2. Finalize documentation updates for the new architecture
3. Implement any performance optimizations identified during testing
4. Complete the final code cleanup to ensure consistency across all builders

The project has successfully transformed the pipeline framework into a **unified, consistent, and maintainable architecture** with automatic dependency resolution and specification-driven step creation.
