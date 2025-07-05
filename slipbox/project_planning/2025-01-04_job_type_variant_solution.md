# Job Type Variant Handling Solution

**Created**: January 4, 2025 6:20 PM PST  
**Status**: Ready for Implementation  
**Priority**: High - Completes Phase 1 (89% → 100%)  
**Timeline**: 4-7 days  

## Problem

The existing step specifications don't distinguish between job type variants:
1. **CradleDataLoading_Training** vs **CradleDataLoading_Calibration**
2. **TabularPreprocessing_Training** vs **TabularPreprocessing_Calibration**

This prevents proper dependency resolution and pipeline variant creation.

## Solution: Job Type Variant Specifications

Create separate specifications for each job type variant with distinct `step_type` identifiers and semantic keywords.

### Implementation Plan

#### Phase 1: Create Variant Specifications (1-2 days)

**Files to Create:**
- `src/pipeline_step_specs/data_loading_training_spec.py`
- `src/pipeline_step_specs/data_loading_calibration_spec.py`
- `src/pipeline_step_specs/preprocessing_training_spec.py`
- `src/pipeline_step_specs/preprocessing_calibration_spec.py`

**Example Structure:**
```python
# data_loading_training_spec.py
DATA_LOADING_TRAINING_SPEC = StepSpecification(
    step_type="CradleDataLoading_Training",
    node_type=NodeType.SOURCE,
    dependencies=[],
    outputs=[
        OutputSpec(
            logical_name="DATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Training data output from Cradle data loading",
            semantic_keywords=["training", "train", "data", "input", "raw", "dataset"]
        ),
        # ... METADATA and SIGNATURE outputs
    ]
)
```

**Key Differentiation:**
- **Training**: `semantic_keywords=["training", "train", "data", "input", "raw"]`
- **Calibration**: `semantic_keywords=["calibration", "calib", "eval", "data", "input"]`

#### Phase 2: Enhanced Pipeline Builder (2-3 days)

**Step Name to Config Mapping:**
```python
def _create_step_with_builder(self, step_name: str, resolved_inputs: Dict[str, Any], created_steps: Dict[str, Step]) -> Step:
    # Handle CradleDataLoading variants
    if step_name.startswith("CradleDataLoading_"):
        job_type = step_name.split("_")[1].lower()  # "training" or "calibration"
        config = self.cradle_train_cfg if job_type == "training" else self.cradle_calib_cfg
        return self._create_data_load_step(config)
        
    # Handle TabularPreprocessing variants
    elif step_name.startswith("TabularPreprocessing_"):
        job_type = step_name.split("_")[1].lower()
        config = self.tp_train_cfg if job_type == "training" else self.tp_calib_cfg
        dependency_step = created_steps[f"CradleDataLoading_{job_type.capitalize()}"]
        return self._create_tabular_preprocess_step(config, dependency_step)
```

#### Phase 3: Validation & Testing (1-2 days)

**Test Cases:**
- Training flow resolution: `CradleDataLoading_Training → TabularPreprocessing_Training`
- Calibration flow resolution: `CradleDataLoading_Calibration → TabularPreprocessing_Calibration`
- No cross-contamination between training and calibration flows
- Pipeline equivalence with manual implementation

## Expected Outcomes

- ✅ **100% Phase 1 Completion**: From 89% to 100% specification coverage
- ✅ **Precise Dependency Resolution**: Training and calibration flows properly separated
- ✅ **Pipeline Variants**: Enable training-only, evaluation-only pipelines
- ✅ **Future-Proof**: Easy to add validation, testing job types

## Timeline

**Week 1 (Jan 6-10, 2025):**
- Days 1-2: Create variant specifications
- Days 3-4: Pipeline builder integration
- Day 5: Validation and testing

**Week 2 (Jan 13-17, 2025):** Buffer/Polish
- Advanced testing and optimization
- Documentation and deployment prep

## Success Criteria

- [ ] All 4 job type variant specifications created
- [ ] Dependency resolver correctly matches variants
- [ ] No cross-contamination between flows
- [ ] Pipeline equivalence with manual approach
- [ ] >95% test coverage

---

**Next Steps**: Begin Phase 1 specification creation, targeting completion by January 10, 2025.
