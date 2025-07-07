# Step Name Consistency Implementation Plan

**Date:** July 7, 2025  
**Objective:** Create a single source of truth for step names across all pipeline components  
**Status:** Planning Phase

## Problem Statement

Currently, step names are defined in multiple places with inconsistencies:

1. **Base Configuration** (`src/pipeline_steps/config_base.py`) - STEP_REGISTRY mapping
2. **Base Step Builder** (`src/pipeline_steps/builder_step_base.py`) - STEP_NAMES mapping  
3. **Step Specifications** (`src/pipeline_step_specs/*.py`) - step_type fields
4. **Pipeline Templates** (`src/pipeline_builder/*.py`) - step references

### Key Issues Identified:
- **179+ references** found across codebase with potential inconsistencies
- **Casing variations**: `PyTorchTraining` vs `PytorchTraining`
- **Multiple definitions** of the same logical step names
- **Maintenance burden** when adding new steps or changing names

## Solution Architecture

### Single Source of Truth Registry

Create a central registry that defines all step names and their relationships:

```python
# src/pipeline_registry/step_names.py
STEP_NAMES = {
    "PytorchTraining": {
        "config_class": "PytorchTrainingConfig",      # For config registry
        "builder_step_name": "PytorchTrainingStep",   # For builder registry
        "spec_type": "PytorchTraining",               # For StepSpecification.step_type
        "description": "PyTorch model training step"
    },
    # ... all other steps
}
```

### Component Integration

1. **Config Registry**: Import `CONFIG_STEP_REGISTRY` from central registry
2. **Builder Registry**: Import `BUILDER_STEP_NAMES` from central registry
3. **Step Specifications**: Use `get_spec_step_type()` helper function
4. **Pipeline Templates**: Reference consistent step names

## Implementation Plan

### Phase 1: Create Central Registry (Priority: High)

#### Files to Create:
- [ ] `src/pipeline_registry/__init__.py`
- [ ] `src/pipeline_registry/step_names.py`

#### Registry Contents:
```python
STEP_NAMES = {
    # Base Steps
    "Base": {
        "config_class": "BasePipelineConfig",
        "builder_step_name": "BaseStep",
        "spec_type": "Base",
        "description": "Base pipeline configuration"
    },
    
    # Processing Steps
    "Processing": {
        "config_class": "ProcessingStepConfigBase", 
        "builder_step_name": "ProcessingStep",
        "spec_type": "Processing",
        "description": "Base processing step"
    },
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStep",
        "spec_type": "TabularPreprocessing",
        "description": "Tabular data preprocessing step"
    },
    "CurrencyConversion": {
        "config_class": "CurrencyConversionConfig",
        "builder_step_name": "CurrencyConversionStep",
        "spec_type": "CurrencyConversion",
        "description": "Currency conversion processing step"
    },
    
    # Data Loading Steps
    "CradleDataLoading": {
        "config_class": "CradleDataLoadConfig",
        "builder_step_name": "CradleDataLoadingStep",
        "spec_type": "CradleDataLoading",
        "description": "Cradle data loading step"
    },
    
    # Training Steps
    "PytorchTraining": {  # Canonical: PytorchTraining (not PyTorchTraining)
        "config_class": "PytorchTrainingConfig",
        "builder_step_name": "PytorchTrainingStep",
        "spec_type": "PytorchTraining",
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig", 
        "builder_step_name": "XGBoostTrainingStep",
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    
    # Model Creation Steps
    "PytorchModel": {  # Canonical: PytorchModel (not PyTorchModel)
        "config_class": "PytorchModelCreationConfig",
        "builder_step_name": "CreatePytorchModelStep",
        "spec_type": "PytorchModel",
        "description": "PyTorch model creation step"
    },
    "XGBoostModel": {
        "config_class": "XGBoostModelCreationConfig",
        "builder_step_name": "CreateXGBoostModelStep",
        "spec_type": "XGBoostModel",
        "description": "XGBoost model creation step"
    },
    
    # Evaluation Steps
    "XGBoostModelEval": {
        "config_class": "XGBoostModelEvalConfig",
        "builder_step_name": "XGBoostModelEvaluationStep",
        "spec_type": "XGBoostModelEval",
        "description": "XGBoost model evaluation step"
    },
    "PytorchModelEval": {
        "config_class": "PytorchModelEvalConfig", 
        "builder_step_name": "PytorchModelEvaluationStep",
        "spec_type": "PytorchModelEval",
        "description": "PyTorch model evaluation step"
    },
    
    # Deployment Steps
    "Package": {
        "config_class": "PackageStepConfig",
        "builder_step_name": "PackagingStep",
        "spec_type": "Package",
        "description": "Model packaging step"
    },
    "Registration": {
        "config_class": "ModelRegistrationConfig",
        "builder_step_name": "RegistrationStep",
        "spec_type": "Registration",
        "description": "Model registration step"
    },
    "Payload": {
        "config_class": "PayloadConfig",
        "builder_step_name": "PayloadTestStep",
        "spec_type": "Payload",
        "description": "Payload testing step"
    },
    
    # Transform Steps
    "BatchTransform": {
        "config_class": "BatchTransformStepConfig",
        "builder_step_name": "BatchTransformStep",
        "spec_type": "BatchTransform",
        "description": "Batch transform step"
    },
    
    # Utility Steps
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStep",
        "spec_type": "HyperparameterPrep",
        "description": "Hyperparameter preparation step"
    }
}
```

### Phase 2: Update Core Components (Priority: High)

#### Files to Update:
- [ ] `src/pipeline_steps/config_base.py`
  - Replace `STEP_REGISTRY` with import from central registry
  - Change: `from ..pipeline_registry.step_names import CONFIG_STEP_REGISTRY as STEP_REGISTRY`

- [ ] `src/pipeline_steps/builder_step_base.py`
  - Replace `STEP_NAMES` with import from central registry
  - Change: `from ..pipeline_registry.step_names import BUILDER_STEP_NAMES as STEP_NAMES`

### Phase 3: Update Step Specifications (Priority: High)

#### Files to Update (Fix Casing & Use Central Constants):

- [ ] `src/pipeline_step_specs/pytorch_training_spec.py`
  - Change: `step_type="PyTorchTraining"` → `step_type=get_spec_step_type("PytorchTraining")`

- [ ] `src/pipeline_step_specs/pytorch_model_spec.py`
  - Change: `step_type="PyTorchModel"` → `step_type=get_spec_step_type("PytorchModel")`

- [ ] `src/pipeline_step_specs/xgboost_training_spec.py`
  - Ensure: `step_type=get_spec_step_type("XGBoostTraining")`

- [ ] `src/pipeline_step_specs/xgboost_model_spec.py`
  - Ensure: `step_type=get_spec_step_type("XGBoostModel")`

- [ ] `src/pipeline_step_specs/preprocessing_spec.py`
  - Ensure: `step_type=get_spec_step_type("TabularPreprocessing")`

- [ ] `src/pipeline_step_specs/preprocessing_training_spec.py`
  - Change: `step_type="TabularPreprocessing_Training"` → Handle job type variants

- [ ] `src/pipeline_step_specs/preprocessing_calibration_spec.py`
  - Change: `step_type="TabularPreprocessing_Calibration"` → Handle job type variants

- [ ] `src/pipeline_step_specs/preprocessing_validation_spec.py`
  - Change: `step_type="TabularPreprocessing_Validation"` → Handle job type variants

- [ ] `src/pipeline_step_specs/preprocessing_testing_spec.py`
  - Change: `step_type="TabularPreprocessing_Testing"` → Handle job type variants

- [ ] `src/pipeline_step_specs/data_loading_spec.py`
  - Ensure: `step_type=get_spec_step_type("CradleDataLoading")`

- [ ] `src/pipeline_step_specs/data_loading_training_spec.py`
  - Handle job type variants for data loading

- [ ] `src/pipeline_step_specs/data_loading_calibration_spec.py`
  - Handle job type variants for data loading

- [ ] `src/pipeline_step_specs/data_loading_validation_spec.py`
  - Handle job type variants for data loading

- [ ] `src/pipeline_step_specs/data_loading_testing_spec.py`
  - Handle job type variants for data loading

- [ ] `src/pipeline_step_specs/packaging_spec.py`
  - Ensure: `step_type=get_spec_step_type("Package")`

- [ ] `src/pipeline_step_specs/registration_spec.py`
  - Ensure: `step_type=get_spec_step_type("Registration")`

- [ ] `src/pipeline_step_specs/payload_spec.py`
  - Ensure: `step_type=get_spec_step_type("Payload")`

- [ ] `src/pipeline_step_specs/model_eval_spec.py`
  - Ensure: `step_type=get_spec_step_type("XGBoostModelEval")`

### Phase 4: Update Pipeline Templates (Priority: Medium)

#### Files to Update (Ensure Consistent Step References):

- [ ] `src/pipeline_builder/template_pipeline_xgboost_end_to_end.py`
  - Verify all step name references use canonical names
  - Update step builder mappings and config mappings

- [ ] `src/pipeline_builder/template_pipeline_xgboost_dataload_preprocess.py`
  - Verify step name consistency

- [ ] `src/pipeline_builder/template_pipeline_xgboost_train_evaluate_e2e.py`
  - Verify step name consistency

- [ ] `src/pipeline_builder/template_pipeline_pytorch_end_to_end.py`
  - Verify step name consistency

- [ ] `src/pipeline_builder/pipeline_builder_template.py`
  - Ensure step type resolution uses consistent naming

### Phase 5: Handle Job Type Variants (Priority: Medium)

#### Special Consideration for Job Type Variants:

Some steps have job type variants (e.g., `TabularPreprocessing_Training`, `TabularPreprocessing_Calibration`).

**Options:**
1. **Extend Registry**: Add job type variants to central registry
2. **Dynamic Generation**: Generate job type variants programmatically
3. **Separate Handling**: Keep base names in registry, handle variants in specs

**Recommended Approach**: Dynamic generation with helper functions:

```python
def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type
```

### Phase 6: Create Validation Tools (Priority: Medium)

#### Files to Create:
- [ ] `tools/validate_step_names.py`
  - Validate consistency across all four components
  - Check for orphaned references
  - Verify all step names exist in central registry

#### Validation Checks:
1. **Config Registry Consistency**: All config classes map to valid step names
2. **Builder Registry Consistency**: All builder classes map to valid step names
3. **Spec Type Consistency**: All step_type values exist in central registry
4. **Template Consistency**: All step references in templates are valid
5. **No Orphaned References**: No hardcoded step names outside registry

### Phase 7: Testing & Documentation (Priority: Low)

#### Testing:
- [ ] Run existing unit tests to ensure no regressions
- [ ] Test pipeline building with updated step names
- [ ] Validate step specification loading
- [ ] Test step builder instantiation

#### Documentation:
- [ ] Update developer documentation on step naming conventions
- [ ] Document the central registry usage patterns
- [ ] Create migration guide for future step additions

## Implementation Timeline

### Week 1: Core Infrastructure
- [ ] Create central registry (`src/pipeline_registry/step_names.py`)
- [ ] Update base config and builder imports
- [ ] Create validation tools

### Week 2: Step Specifications
- [ ] Update all step specification files
- [ ] Handle job type variants
- [ ] Test step specification loading

### Week 3: Pipeline Templates
- [ ] Update all pipeline template files
- [ ] Test pipeline building
- [ ] Validate end-to-end functionality

### Week 4: Testing & Documentation
- [ ] Comprehensive testing
- [ ] Documentation updates
- [ ] Code review and refinement

## Risk Assessment

### High Risk:
- **Breaking Changes**: Incorrect step name references could break pipeline building
- **Import Cycles**: New registry imports could create circular dependencies

### Medium Risk:
- **Job Type Variants**: Complex handling of step variants with job types
- **Template Complexity**: Pipeline templates have complex step interdependencies

### Low Risk:
- **Documentation**: Updates to documentation are low risk
- **Validation Tools**: New validation tools don't affect existing functionality

## Mitigation Strategies

1. **Incremental Implementation**: Implement in phases with testing at each step
2. **Backward Compatibility**: Maintain existing interfaces during transition
3. **Comprehensive Testing**: Test all pipeline building scenarios
4. **Code Review**: Thorough review of all changes before deployment

## Success Criteria

1. **Single Source of Truth**: All step names defined in one central location
2. **Zero Inconsistencies**: No mismatched step names across components
3. **Easy Maintenance**: Adding new steps requires only one registry entry
4. **No Regressions**: All existing functionality continues to work
5. **Improved Developer Experience**: Clear, consistent step naming conventions

## Dependencies

- No external dependencies
- Requires coordination with pipeline development team
- May need to coordinate with any ongoing pipeline development work

## Rollback Plan

If issues arise during implementation:

1. **Phase-by-Phase Rollback**: Can rollback individual phases
2. **Git Revert**: All changes are version controlled
3. **Backup Strategy**: Keep original files as backup during transition
4. **Testing Checkpoints**: Validate functionality at each phase

---

**Next Steps:**
1. Review and approve this implementation plan
2. Begin Phase 1: Create central registry
3. Set up validation tools early for continuous checking
4. Coordinate with team on implementation timeline
