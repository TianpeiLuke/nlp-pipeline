# Script-Specification Alignment Prevention Plan

## Problem Statement

With the implementation of script contracts integrated into step specifications, there's a critical need to prevent misalignment between:

1. **Step Specifications** - Declarative metadata defining step dependencies and outputs
2. **Script Contracts** - Execution requirements for SageMaker container scripts
3. **Actual Script Implementation** - The runtime behavior of pipeline scripts

Misalignments can cause runtime failures, incorrect dependency resolution, and maintenance issues.

## Current Misalignment Issues Identified

### 1. **Input Path Mismatch**
- **Script Contract** expects: `"eval_data_input": "/opt/ml/processing/input/eval_data"`
- **Step Specification** defines: `logical_name="eval_data_input"` but doesn't enforce exact path
- **Actual Script** uses: `eval_data_dir = "/opt/ml/processing/input/eval_data"` (hardcoded)

### 2. **Missing Input in Contract**
- **Script Contract** only defines 2 inputs: `model_input` and `eval_data_input`
- **Step Specification** defines 3 dependencies: `model_input`, `eval_data_input`, and `hyperparameters_input`
- **Actual Script** loads hyperparameters from model artifacts, not as separate input

### 3. **Output Path Inconsistency**
- **Script Contract** expects: `"eval_output": "/opt/ml/processing/output/eval"`, `"metrics_output": "/opt/ml/processing/output/metrics"`
- **Step Specification** defines: `"EvaluationResults"` and `"EvaluationMetrics"` with specific S3Output paths
- **Actual Script** creates subdirectories and files within these paths

## Solution Architecture

### Phase 1: Contract-Specification Alignment Framework

#### 1.1 Enhanced Contract Validation
```python
# Add to StepSpecification class in src/pipeline_deps/base_specifications.py
def validate_contract_alignment(self) -> ValidationResult:
    """Validate that script contract aligns with step specification"""
    if not self.script_contract:
        return ValidationResult.success("No contract to validate")
    
    errors = []
    
    # Validate input alignment
    contract_inputs = set(self.script_contract.expected_input_paths.keys())
    spec_inputs = set(dep.logical_name for dep in self.dependencies if dep.required)
    
    missing_in_contract = spec_inputs - contract_inputs
    if missing_in_contract:
        errors.append(f"Contract missing required inputs: {missing_in_contract}")
    
    # Validate output alignment
    contract_outputs = set(self.script_contract.expected_output_paths.keys())
    spec_outputs = set(output.logical_name for output in self.outputs)
    
    missing_in_contract = spec_outputs - contract_outputs
    if missing_in_contract:
        errors.append(f"Contract missing required outputs: {missing_in_contract}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

#### 1.2 Specification Validation Against Contracts
```python
# Add to src/pipeline_deps/specification_registry.py
def validate_spec_against_contract(spec: StepSpecification) -> ValidationResult:
    """Ensure specification is compatible with its contract"""
    if not spec.script_contract:
        return ValidationResult.success("No contract defined")
    
    errors = []
    
    # Check that all required dependencies have corresponding contract inputs
    required_deps = [dep for dep in spec.dependencies if dep.required]
    contract_inputs = set(spec.script_contract.expected_input_paths.keys())
    
    for dep in required_deps:
        if dep.logical_name not in contract_inputs:
            errors.append(f"Required dependency '{dep.logical_name}' not in contract inputs")
    
    # Check that all outputs have corresponding contract outputs
    spec_outputs = set(output.logical_name for output in spec.outputs)
    contract_outputs = set(spec.script_contract.expected_output_paths.keys())
    
    missing_outputs = spec_outputs - contract_outputs
    if missing_outputs:
        errors.append(f"Specification outputs not in contract: {missing_outputs}")
    
    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
```

### Phase 2: SageMaker-Compatible Script Standardization

#### 2.1 Contract-Aware Utility Functions
```python
# Create src/pipeline_scripts/contract_utils.py
import os
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def validate_contract_environment(contract):
    """Validate SageMaker environment matches contract expectations"""
    errors = []
    
    # Check required environment variables
    for var in contract.required_env_vars:
        if var not in os.environ:
            errors.append(f"Missing required environment variable: {var}")
    
    # Check input paths exist (SageMaker mounts these)
    for logical_name, path in contract.expected_input_paths.items():
        if not os.path.exists(path):
            errors.append(f"Input path not found: {path} ({logical_name})")
    
    # Ensure output directories exist
    for logical_name, path in contract.expected_output_paths.items():
        os.makedirs(path, exist_ok=True)
        logger.info(f"Ensured output directory exists: {path}")
    
    if errors:
        raise RuntimeError(f"Contract validation failed: {errors}")
    
    logger.info("Contract environment validation passed")

def get_contract_paths(contract):
    """Get input/output paths from contract for easy access"""
    return {
        'inputs': contract.expected_input_paths,
        'outputs': contract.expected_output_paths
    }
```

#### 2.2 Refactored Script Pattern
```python
# Standard pattern for all SageMaker scripts
def get_script_contract():
    """Get the contract for this script"""
    # Import at runtime to avoid circular imports
    from ..pipeline_script_contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT
    return MODEL_EVALUATION_CONTRACT

def main():
    """Main entry point with contract validation"""
    # 1. Get and validate contract
    contract = get_script_contract()
    validate_contract_environment(contract)
    paths = get_contract_paths(contract)
    
    # 2. Use contract paths instead of hardcoded paths
    model_dir = paths['inputs']['model_input']
    eval_data_dir = paths['inputs']['eval_data_input']
    output_eval_dir = paths['outputs']['eval_output']
    output_metrics_dir = paths['outputs']['metrics_output']
    
    # 3. Access validated environment variables
    ID_FIELD = os.environ["ID_FIELD"]  # Contract ensures this exists
    LABEL_FIELD = os.environ["LABEL_FIELD"]
    
    # 4. Business logic remains unchanged
    # ... existing script logic
```

### Phase 3: Automated Alignment Enforcement

#### 3.1 Pre-commit Validation Hook
```python
# Create tools/validate_contracts.py
def validate_all_contracts():
    """Validate all script contracts against their specifications"""
    from src.pipeline_step_specs import (
        MODEL_EVAL_SPEC,
        PREPROCESSING_TRAINING_SPEC,
        XGBOOST_TRAINING_SPEC
    )
    
    specs_with_contracts = [
        MODEL_EVAL_SPEC,
        PREPROCESSING_TRAINING_SPEC,
        XGBOOST_TRAINING_SPEC
    ]
    
    all_valid = True
    for spec in specs_with_contracts:
        result = spec.validate_contract_alignment()
        if not result.is_valid:
            print(f"❌ {spec.step_type}: {result.errors}")
            all_valid = False
        else:
            print(f"✅ {spec.step_type}: Contract aligned")
    
    return all_valid

if __name__ == "__main__":
    import sys
    if not validate_all_contracts():
        sys.exit(1)
```

#### 3.2 Runtime Contract Enforcement
```python
# Enhanced ScriptContract with runtime validation
class ScriptContract(BaseModel):
    # ... existing fields
    
    def enforce_at_runtime(self):
        """Enforce contract compliance at script runtime"""
        # Validate all required inputs exist
        for logical_name, path in self.expected_input_paths.items():
            if not os.path.exists(path):
                raise RuntimeError(f"Contract violation: Missing input {logical_name} at {path}")
        
        # Ensure output directories exist
        for logical_name, path in self.expected_output_paths.items():
            os.makedirs(path, exist_ok=True)
        
        # Validate environment variables
        for var in self.required_env_vars:
            if var not in os.environ:
                raise RuntimeError(f"Contract violation: Missing environment variable {var}")
```

### Phase 4: Specification-Contract Co-evolution

#### 4.1 Contract Generation from Specifications
```python
# Create src/pipeline_deps/contract_generator.py
def generate_contract_from_spec(spec: StepSpecification) -> ScriptContract:
    """Generate a script contract template from step specification"""
    
    # Map dependencies to input paths
    input_paths = {}
    for dep in spec.dependencies:
        if dep.required:
            # Generate standard SageMaker path
            input_paths[dep.logical_name] = f"/opt/ml/processing/input/{dep.logical_name}"
    
    # Map outputs to output paths
    output_paths = {}
    for output in spec.outputs:
        output_paths[output.logical_name] = f"/opt/ml/processing/output/{output.logical_name}"
    
    return ScriptContract(
        entry_point=f"{spec.step_type.lower()}.py",
        expected_input_paths=input_paths,
        expected_output_paths=output_paths,
        required_env_vars=[],  # To be filled manually
        framework_requirements={}  # To be filled manually
    )
```

#### 4.2 Bidirectional Validation
```python
# Create src/pipeline_deps/alignment_validator.py
class AlignmentValidator:
    """Validates alignment between specifications, contracts, and scripts"""
    
    def validate_complete_alignment(self, spec: StepSpecification, script_path: str) -> ValidationResult:
        """Validate complete alignment between spec, contract, and script"""
        results = []
        
        # 1. Validate spec-contract alignment
        results.append(spec.validate_contract_alignment())
        
        # 2. Validate contract-script alignment
        if spec.script_contract:
            results.append(spec.script_contract.validate_implementation(script_path))
        
        # 3. Validate spec-script semantic alignment
        results.append(self._validate_semantic_alignment(spec, script_path))
        
        return ValidationResult.combine(results)
    
    def _validate_semantic_alignment(self, spec: StepSpecification, script_path: str) -> ValidationResult:
        """Validate that script semantically matches specification intent"""
        # This could include checks like:
        # - Script uses expected processing patterns
        # - Output files match expected formats
        # - Environment variables are used appropriately
        return ValidationResult.success("Semantic validation not implemented yet")
```

## Implementation Strategy

### ✅ Immediate Actions (Week 1) - COMPLETED
1. **✅ Fix Current Misalignments**
   - ✅ Updated model evaluation specification to remove unnecessary hyperparameters dependency
   - ✅ Added alias outputs to contract for proper alignment
   - ✅ Implemented contract-specification validation methods

2. **✅ Create Contract Utils**
   - ✅ Implemented `src/pipeline_scripts/contract_utils.py` with comprehensive validation
   - ✅ Created `dockers/xgboost_atoz/pipeline_scripts/contract_utils.py` for XGBoost containers
   - ✅ Added validation and path helper functions with context managers

### ✅ Short-term (Week 2-3) - PARTIALLY COMPLETED
3. **✅ Refactor Existing Scripts**
   - ✅ Updated `model_evaluation_xgb.py` to use contract-aware pattern
   - ✅ Implemented ContractEnforcer context manager usage
   - ✅ Added runtime contract enforcement with comprehensive validation

4. **✅ Add Validation Framework**
   - ✅ Implemented `validate_contract_alignment()` method in StepSpecification
   - ✅ Created `tools/validate_contracts.py` pre-commit validation tool
   - 🔄 CI/CD integration (pending)

### Medium-term (Week 4-6)
5. **Automated Validation**
   - Integrate validation into development workflow
   - Add specification-contract alignment checks
   - Create contract generation utilities

6. **Documentation and Training**
   - Update development guidelines
   - Create alignment best practices
   - Train team on new patterns

### Long-term (Month 2+)
7. **Advanced Features**
   - Semantic alignment validation
   - Automated contract generation
   - Integration with pipeline builder

## Success Metrics

### Technical Metrics
- **Zero Runtime Failures** due to contract misalignment
- **100% Contract Coverage** for all pipeline scripts
- **Automated Validation** in CI/CD pipeline
- **Sub-second Validation** time for all contracts

### Process Metrics
- **Pre-commit Hook Adoption** by all developers
- **Contract-First Development** for new scripts
- **Reduced Debug Time** for pipeline issues
- **Improved Developer Confidence** in deployments

## Risk Mitigation

### Development Risks
- **Learning Curve**: Provide clear examples and documentation
- **Performance Impact**: Keep validation lightweight and fast
- **Maintenance Overhead**: Automate as much validation as possible

### Operational Risks
- **SageMaker Compatibility**: Test thoroughly in SageMaker environments
- **Backward Compatibility**: Maintain support for existing scripts during transition
- **Deployment Issues**: Gradual rollout with rollback capabilities

## Files to Create/Modify

### New Files
- `src/pipeline_scripts/contract_utils.py` - Contract validation utilities
- `src/pipeline_deps/contract_generator.py` - Contract generation from specs
- `src/pipeline_deps/alignment_validator.py` - Comprehensive alignment validation
- `tools/validate_contracts.py` - Pre-commit validation hook

### Modified Files
- `src/pipeline_deps/base_specifications.py` - Add alignment validation methods
- `src/pipeline_script_contracts/base_script_contract.py` - Add runtime enforcement
- `src/pipeline_scripts/model_evaluation_xgb.py` - Refactor to use contract pattern
- All other pipeline scripts - Apply contract-aware pattern

### Configuration Files
- `.pre-commit-config.yaml` - Add contract validation hook
- CI/CD pipeline configuration - Add validation steps

## Conclusion

This plan provides a comprehensive approach to preventing script-specification misalignment through:

1. **Automated Validation** - Catch misalignments before they reach production
2. **Standardized Patterns** - Consistent approach across all SageMaker scripts
3. **Runtime Enforcement** - Contract validation within SageMaker containers
4. **Developer Tools** - Pre-commit hooks and validation utilities
5. **Continuous Improvement** - Automated contract generation and alignment checking

The solution maintains SageMaker compatibility while providing robust alignment guarantees, reducing runtime failures and improving development confidence.
