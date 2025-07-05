# Script-Specification Alignment Solution Plan
*Date: 2025-07-04*

## Executive Summary

This document provides a comprehensive solution for aligning pipeline scripts with step specifications, addressing the identified gaps in the specification-driven XGBoost pipeline implementation.

## Problem Statement

The original analysis identified a critical gap between step specifications and script implementations:

> The only remaining gap is **job type variant handling** for:
> 1. **CradleDataLoading_Training** vs **CradleDataLoading_Calibration**
> 2. **TabularPreprocessing_Training** vs **TabularPreprocessing_Calibration**

## Solution Architecture

### 1. Script Contract System

I've implemented a comprehensive **Script Contract System** that bridges the gap between step specifications and script implementations:

#### Core Components:
- **`ScriptContract`**: Pydantic V2 model defining explicit I/O and environment requirements
- **`ScriptAnalyzer`**: AST-based analyzer that extracts actual script behavior
- **`ValidationResult`**: Structured validation reporting
- **`ScriptContractValidator`**: Comprehensive validation and compliance reporting

#### Key Features:
- **Explicit I/O Declaration**: Each script contract declares expected input/output paths
- **Environment Variable Requirements**: Required and optional environment variables
- **Framework Dependencies**: Version requirements for external libraries
- **Automated Validation**: AST analysis to verify script compliance
- **Gap Detection**: Identifies missing inputs, outputs, and environment variables

### 2. Contract Implementation Status

Current validation results show **4/6 scripts compliant**:

#### ✅ Compliant Scripts:
- `mims_package.py` - Model packaging for deployment
- `model_evaluation_xgb.py` - Model evaluation and metrics
- `currency_conversion.py` - Multi-currency data processing
- `risk_table_mapping.py` - Categorical feature risk mapping

#### ❌ Non-Compliant Scripts:
- `tabular_preprocess.py` - **Gap**: Missing metadata/signature inputs
- `mims_payload.py` - **Gap**: Output path structure mismatch

### 3. Job Type Variant Handling Solution

#### Current State Analysis:
The job type variants (Training vs Calibration) are handled through:
1. **Command-line arguments**: `--job_type` parameter
2. **Conditional logic**: Scripts adapt behavior based on job type
3. **Environment variables**: Configuration through env vars

#### Recommended Approach:
Instead of creating separate step specifications for each job type variant, use **parameterized specifications** with job type as a parameter:

```python
# Single specification with job type parameter
TabularPreprocessingSpec(
    job_type=JobType.TRAINING,  # or JobType.CALIBRATION
    input_requirements=...,
    output_requirements=...,
    environment_variables=...
)
```

### 4. Specific Gap Resolutions

#### Gap 1: Tabular Preprocessing Input Mismatch
**Issue**: Script only uses `/opt/ml/processing/input/data` but specification expects DATA, METADATA, and SIGNATURE inputs.

**Solution Options**:
1. **Update Script** (Recommended): Modify script to handle metadata and signature inputs
2. **Update Contract**: Align contract with current script behavior
3. **Hybrid Approach**: Make metadata/signature inputs optional

#### Gap 2: MIMS Payload Output Structure
**Issue**: Script creates nested output structure but contract expects flat structure.

**Solution**: Update contract to match actual script output structure:
```python
expected_output_paths={
    "payload_output": "/opt/ml/processing/output",  # Parent directory
    # Script creates subdirectories: payload_sample/, payload_metadata/
}
```

### 5. Implementation Roadmap

#### Phase 1: Contract Alignment (Immediate)
1. ✅ **Complete**: Implement script contract system
2. ✅ **Complete**: Create contracts for all 6 pipeline scripts
3. ✅ **Complete**: Build validation and compliance reporting
4. 🔄 **In Progress**: Fix identified contract misalignments

#### Phase 2: Specification Integration (Next)
1. **Integrate contracts with step specifications**
2. **Add job type parameterization to specifications**
3. **Update step builders to use contract validation**
4. **Create specification-contract alignment tests**

#### Phase 3: Pipeline Enhancement (Future)
1. **Add contract validation to pipeline execution**
2. **Implement automatic contract generation from specifications**
3. **Create contract-driven step builder templates**
4. **Add runtime contract compliance checking**

## Technical Implementation Details

### Script Contract Example
```python
TABULAR_PREPROCESS_CONTRACT = ScriptContract(
    entry_point="tabular_preprocess.py",
    expected_input_paths={
        "DATA": "/opt/ml/processing/input/data",
        "METADATA": "/opt/ml/processing/input/metadata",  # Gap identified
        "SIGNATURE": "/opt/ml/processing/input/signature"  # Gap identified
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output"
    },
    required_env_vars=["LABEL_FIELD", "TRAIN_RATIO", "TEST_VAL_RATIO"],
    framework_requirements={
        "pandas": ">=1.3.0",
        "numpy": ">=1.21.0",
        "scikit-learn": ">=1.0.0"
    }
)
```

### Validation Usage
```python
from src.pipeline_script_contracts import ScriptContractValidator

validator = ScriptContractValidator()
reports = validator.validate_all_scripts()
summary = validator.generate_compliance_summary(reports)
print(summary)
```

## Benefits of This Solution

### 1. **Explicit Contract Definition**
- Clear I/O requirements for each script
- Environment variable documentation
- Framework dependency tracking

### 2. **Automated Compliance Checking**
- AST-based script analysis
- Gap detection and reporting
- Continuous validation capability

### 3. **Bridge Between Specifications and Implementation**
- Contracts serve as intermediate layer
- Specifications can reference contracts
- Implementation validated against contracts

### 4. **Developer Experience Improvement**
- Clear requirements for script development
- Automated validation feedback
- Standardized script interfaces

### 5. **Pipeline Reliability**
- Early detection of I/O mismatches
- Environment configuration validation
- Dependency requirement checking

## Next Steps

1. **Fix Identified Gaps**: Address the 2 non-compliant scripts
2. **Integrate with Step Specifications**: Connect contracts to specification system
3. **Add Job Type Parameterization**: Implement parameterized specifications
4. **Create Integration Tests**: Validate end-to-end specification-contract-script alignment
5. **Documentation**: Create developer guide for contract-driven development

## Conclusion

The Script Contract System provides a robust solution for aligning pipeline scripts with step specifications. By creating explicit contracts that define I/O requirements and validating script implementations against these contracts, we've established a reliable bridge between high-level specifications and low-level script implementations.

The system successfully identifies gaps (4/6 scripts compliant) and provides actionable feedback for resolution. This approach scales well and can be extended to handle job type variants through parameterization rather than specification duplication.

---

*This solution addresses the core gap identified in the specification-driven XGBoost pipeline plan while providing a foundation for future pipeline development and maintenance.*
