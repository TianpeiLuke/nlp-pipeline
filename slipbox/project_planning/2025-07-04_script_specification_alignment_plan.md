# Script-Specification Alignment Solution

**Created**: July 4, 2025 8:29 PM PST  
**Status**: Ready for Implementation  
**Priority**: High - Critical Infrastructure Risk  
**Timeline**: 2-3 weeks  
**Related**: [Job Type Variant Solution](./2025-07-04_job_type_variant_solution.md), [Script Contract Design](../pipeline_design/script_contract.md)

## Context

This document addresses the critical risk of **script-specification misalignment** identified during the analysis of pipeline scripts and step specifications. Current scripts hardcode input/output paths and environment variables, creating fragile implicit dependencies that can cause runtime failures.

## Problem Analysis

### Current Misalignments Identified

1. **Tabular Preprocessing Script**:
   - ❌ **Spec expects**: `DATA`, `METADATA`, `SIGNATURE` inputs
   - ❌ **Script uses**: Only `/opt/ml/processing/input/data` (hardcoded)
   - ❌ **Gap**: Script doesn't handle metadata/signature inputs

2. **Model Evaluation Script**:
   - ❌ **Spec expects**: `model_input`, `eval_data_input`, `hyperparameters_input`
   - ❌ **Script uses**: Fixed paths for model, eval_data, code
   - ❌ **Gap**: No explicit hyperparameters input handling

3. **MIMS Package Script**:
   - ❌ **Spec expects**: `model_input`, `inference_scripts_input`
   - ❌ **Script uses**: Fixed `/model` and `/script` paths
   - ❌ **Gap**: Input channel names don't match spec logical names

4. **MIMS Payload Script**:
   - ❌ **Spec expects**: `model_input`
   - ❌ **Script uses**: Fixed `/model` path
   - ❌ **Gap**: Input channel name doesn't match spec logical name

### Risk Assessment

**High Risk Areas:**
- ❌ **Runtime Failures**: Scripts fail if step builders change path mappings
- ❌ **Silent Failures**: Scripts may ignore optional inputs without validation
- ❌ **Maintenance Burden**: Changes require manual coordination between specs and scripts
- ❌ **Development Friction**: No validation that scripts match their specifications
- ❌ **Documentation Drift**: Script requirements not explicitly documented

## Solution: Script Contract Architecture

Implement [Script Contracts](../pipeline_design/script_contract.md) to bridge the gap between specifications and script implementations.

### Architecture Overview

```
Step Specification ←→ Script Contract ←→ Script Implementation
     (What)              (How)              (Implementation)
```

**Key Components:**
1. **Script Contract**: Explicit I/O and environment contracts
2. **Specification Extension**: Add contracts to existing specifications
3. **Runtime Framework**: Specification-aware script runtime
4. **Validation System**: Build-time and runtime validation

## Implementation Plan

### Phase 1: Foundation (Week 1)

#### 1.1 Extend Base Specifications (2 days)

**File**: `src/pipeline_deps/base_specifications.py`

```python
@dataclass
class ScriptContract:
    """Script execution contract"""
    entry_point: str
    expected_input_paths: Dict[str, str]
    expected_output_paths: Dict[str, str]
    required_env_vars: List[str]
    optional_env_vars: Dict[str, str] = field(default_factory=dict)
    framework_requirements: Dict[str, str] = field(default_factory=dict)

@dataclass
class StepSpecification:
    # ... existing fields ...
    script_contract: Optional[ScriptContract] = None  # NEW
```

#### 1.2 Create Script Runtime Framework (3 days)

**File**: `src/pipeline_runtime/script_runtime.py`

```python
class SpecificationAwareScriptRuntime:
    """Runtime that validates against specifications"""
    
    def __init__(self, step_type: str):
        self.spec = SpecificationRegistry.get_specification(step_type)
        self.channel_mapping = self._build_channel_mapping()
    
    def get_input_path(self, logical_name: str) -> Optional[str]:
        """Get validated input path"""
        
    def validate_inputs(self) -> ValidationResult:
        """Validate all required inputs exist"""
```

### Phase 2: Script Contract Implementation (Week 2)

#### 2.1 Define Script Contracts (2 days)

**Tabular Preprocessing Contract**:
```python
PREPROCESSING_SCRIPT_CONTRACT = ScriptContract(
    entry_point="tabular_preprocess.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "METADATA": "/opt/ml/processing/input/metadata",
        "SIGNATURE": "/opt/ml/processing/input/signature"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output"
    },
    required_env_vars=["LABEL_FIELD", "TRAIN_RATIO", "TEST_VAL_RATIO"]
)
```

**Model Evaluation Contract**:
```python
MODEL_EVAL_SCRIPT_CONTRACT = ScriptContract(
    entry_point="model_eval_xgb.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "eval_data_input": "/opt/ml/processing/input/eval_data",
        "code_input": "/opt/ml/processing/input/code"
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
        "metrics_output": "/opt/ml/processing/output/metrics"
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"]
)
```

#### 2.2 Enhance Step Specifications (1 day)

**Files to Update**:
- `src/pipeline_step_specs/preprocessing_spec.py`
- `src/pipeline_step_specs/model_eval_spec.py`
- `src/pipeline_step_specs/packaging_spec.py`
- `src/pipeline_step_specs/payload_spec.py`

#### 2.3 Update Step Builders (2 days)

**Enhanced Validation**:
```python
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        super().validate_configuration()
        
        # NEW: Validate script compliance
        script_path = self.config.get_script_path()
        validation = SpecificationRegistry.validate_script_compliance(
            "TabularPreprocessing", script_path
        )
        
        if not validation.is_valid:
            raise ValueError(f"Script validation failed: {validation.errors}")
```

### Phase 3: Script Migration (Week 3)

#### 3.1 Migrate Tabular Preprocessing Script (2 days)

**Before (Hardcoded)**:
```python
def main(job_type, label_field, train_ratio, test_val_ratio, input_base_dir, output_dir):
    input_data_dir = os.path.join(input_base_dir, "data")  # Hardcoded
    df = combine_shards(input_data_dir)
```

**After (Specification-Aware)**:
```python
def main(job_type, label_field, train_ratio, test_val_ratio, input_base_dir, output_dir):
    runtime = SpecificationAwareScriptRuntime("TabularPreprocessing")
    runtime.validate_inputs()
    
    data_path = runtime.get_input_path("DATA")
    metadata_path = runtime.get_input_path("METADATA")  # Optional
    
    df = combine_shards(data_path)
    if metadata_path:
        process_metadata(metadata_path)
```

#### 3.2 Migrate Model Evaluation Script (2 days)

**Key Changes**:
- Use `runtime.get_input_path("model_input")` instead of hardcoded paths
- Validate all inputs exist before processing
- Handle optional hyperparameters input

#### 3.3 Migrate MIMS Scripts (1 day)

**Package Script**:
- Use `runtime.get_input_path("model_input")` and `runtime.get_input_path("inference_scripts_input")`

**Payload Script**:
- Use `runtime.get_input_path("model_input")`

### Phase 4: Validation & Testing (Ongoing)

#### 4.1 Build-Time Validation

**CI/CD Integration**:
```python
# Add to build pipeline
def validate_all_scripts():
    for step_type in SpecificationRegistry.get_all_step_types():
        script_path = get_script_path(step_type)
        validation = SpecificationRegistry.validate_script_compliance(step_type, script_path)
        if not validation.is_valid:
            raise BuildError(f"Script validation failed for {step_type}: {validation.errors}")
```

#### 4.2 Runtime Validation Tests

**Test Cases**:
- ✅ Scripts validate inputs before processing
- ✅ Scripts handle optional inputs gracefully
- ✅ Scripts fail fast with clear error messages
- ✅ Environment variables are validated
- ✅ Path mappings work correctly

#### 4.3 Integration Tests

**Pipeline Tests**:
- ✅ End-to-end pipeline execution with script contracts
- ✅ Error handling when inputs are missing
- ✅ Backward compatibility with existing pipelines

## Expected Outcomes

### Immediate Benefits (Phase 1-2)
- ✅ **Explicit Documentation**: Script requirements clearly documented
- ✅ **Build-Time Validation**: Catch misalignments before deployment
- ✅ **Development Safety**: Developers know scripts will work if validated

### Long-Term Benefits (Phase 3-4)
- ✅ **Runtime Safety**: Scripts validate inputs before processing
- ✅ **Maintainability**: Changes to specifications validate all scripts
- ✅ **Flexibility**: Scripts adapt to specification changes automatically
- ✅ **Reliability**: Eliminate runtime failures due to path mismatches

## Success Criteria

### Phase 1 (Foundation)
- [ ] `ScriptContract` class implemented and tested
- [ ] `SpecificationAwareScriptRuntime` framework created
- [ ] Base specification classes extended

### Phase 2 (Contracts)
- [ ] All 4 step types have script contracts defined
- [ ] Step specifications enhanced with contracts
- [ ] Step builders validate script compliance

### Phase 3 (Migration)
- [ ] All 4 scripts migrated to use specification runtime
- [ ] Scripts handle optional inputs correctly
- [ ] Backward compatibility maintained

### Phase 4 (Validation)
- [ ] Build-time validation integrated into CI/CD
- [ ] >95% test coverage for script contracts
- [ ] End-to-end pipeline tests pass

## Risk Mitigation

### Backward Compatibility
- Script contracts are optional additions to specifications
- Existing scripts continue to work during migration
- Gradual migration reduces deployment risk

### Performance Impact
- Runtime validation is minimal overhead
- Input validation happens once at script startup
- No impact on processing performance

### Development Workflow
- Clear migration path for each script
- Comprehensive testing at each phase
- Documentation and examples provided

## Timeline

**Week 1 (July 7-11, 2025)**: Foundation
- Days 1-2: Extend base specifications
- Days 3-5: Create script runtime framework

**Week 2 (July 14-18, 2025)**: Contracts
- Days 1-2: Define script contracts
- Day 3: Enhance step specifications
- Days 4-5: Update step builders

**Week 3 (July 21-25, 2025)**: Migration
- Days 1-2: Migrate tabular preprocessing
- Days 3-4: Migrate model evaluation
- Day 5: Migrate MIMS scripts

**Ongoing**: Validation & Testing
- Continuous integration of validation
- Performance monitoring
- Documentation updates

---

**Next Steps**: Begin Phase 1 foundation work, targeting script contract framework completion by July 11, 2025.

**Dependencies**: This plan builds on the [Job Type Variant Solution](./2025-07-04_job_type_variant_solution.md) and implements the [Script Contract Design](../pipeline_design/script_contract.md).
