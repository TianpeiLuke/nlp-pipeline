# Script Contract

## What is the Purpose of Script Contract?

Script Contracts serve as the **execution bridge** between declarative Step Specifications and imperative script implementations. They provide explicit validation that scripts conform to their architectural specifications, eliminating the risk of runtime failures due to script-specification misalignment.

## ✅ Implementation Status (January 2025)

**FULLY IMPLEMENTED** - Complete script contract system with 8 contracts covering all major pipeline scripts:

- **Processing Scripts (6)**: tabular_preprocess.py, mims_package.py, mims_payload.py, model_evaluation_xgb.py, currency_conversion.py, risk_table_mapping.py
- **Training Scripts (2)**: train.py (PyTorch), train_xgb.py (XGBoost)
- **Validation Framework**: Automated compliance checking with AST analysis
- **Contract Types**: Base contracts for processing scripts, specialized contracts for training scripts

## Core Purpose

Script Contracts provide a **declarative way to define script execution requirements** and validate implementation compliance, enabling:

1. **Explicit I/O Validation** - Ensure scripts use expected input/output channels
2. **Runtime Environment Contracts** - Define and validate environment variable requirements
3. **Implementation Compliance** - Verify scripts match their specifications
4. **Development Safety** - Catch misalignments at build time, not runtime
5. **Documentation as Code** - Self-documenting script requirements

## Implemented Contract Types

### 1. Processing Script Contracts (SageMaker Processing Jobs)

For scripts running in SageMaker Processing containers:

```python
from src.pipeline_script_contracts import ScriptContract

TABULAR_PREPROCESS_CONTRACT = ScriptContract(
    entry_point="tabular_preprocess.py",
    expected_input_paths={
        "input_data": "/opt/ml/processing/input/data",
        "metadata": "/opt/ml/processing/input/metadata",
        "signature": "/opt/ml/processing/input/signature"
    },
    expected_output_paths={
        "processed_data": "/opt/ml/processing/output/data"
    },
    required_env_vars=["LABEL_FIELD", "TRAIN_RATIO", "TEST_VAL_RATIO"],
    framework_requirements={
        "pandas": ">=1.3.0",
        "scikit-learn": ">=1.0.0",
        "numpy": ">=1.19.0"
    }
)
```

### 2. Training Script Contracts (SageMaker Training Jobs)

For scripts running in SageMaker Training containers:

```python
from src.pipeline_script_contracts import TrainingScriptContract

PYTORCH_TRAIN_CONTRACT = TrainingScriptContract(
    entry_point="train.py",
    expected_input_paths={
        "train_data": "/opt/ml/input/data/train",
        "val_data": "/opt/ml/input/data/val", 
        "test_data": "/opt/ml/input/data/test",
        "config": "/opt/ml/input/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "data_output": "/opt/ml/output/data",
        "checkpoints": "/opt/ml/checkpoints"
    },
    framework_requirements={
        "torch": "==2.1.2",
        "lightning": "==2.1.3",
        "transformers": "==4.37.2",
        "pandas": "==2.1.4"
    }
)
```

## Implemented Validation System

### Automated Contract Validation

```python
from src.pipeline_script_contracts import ScriptContractValidator

# Validate single script
validator = ScriptContractValidator('src/pipeline_scripts')
report = validator.validate_script('tabular_preprocess.py')

print(report.summary)
# Output: "tabular_preprocess.py vs ScriptContract: ✅ COMPLIANT"

if not report.is_compliant:
    print("Errors:", report.errors)
    print("Missing inputs:", report.missing_inputs)
    print("Missing outputs:", report.missing_outputs)
```

### Comprehensive Validation Reports

```python
# Validate all scripts
reports = validator.validate_all_scripts()
summary = validator.generate_compliance_summary(reports)
print(summary)

# Example output:
# ============================================================
# SCRIPT CONTRACT COMPLIANCE REPORT
# ============================================================
# Overall Compliance: 6/8 scripts compliant
# 
# ✅ COMPLIANT SCRIPTS:
# --------------------
#   • tabular_preprocess.py
#   • mims_package.py
#   • mims_payload.py
#   • model_evaluation_xgb.py
#   • currency_conversion.py
#   • risk_table_mapping.py
# 
# ❌ NON-COMPLIANT SCRIPTS:
# -------------------------
#   • train.py
#     Errors: 2
#     Missing Inputs: ['/opt/ml/input/data/train', '/opt/ml/input/data/val']
```

## Real Contract Examples

### 1. XGBoost Training Contract

```python
XGBOOST_TRAIN_CONTRACT = TrainingScriptContract(
    entry_point="train_xgb.py",
    expected_input_paths={
        "train_data": "/opt/ml/input/data/train",
        "val_data": "/opt/ml/input/data/val",
        "test_data": "/opt/ml/input/data/test",
        "config": "/opt/ml/input/data/config/hyperparameters.json"
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "data_output": "/opt/ml/output/data"
    },
    framework_requirements={
        "xgboost": "==1.7.6",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "matplotlib": ">=3.0.0"
    },
    description="""
    XGBoost training script for tabular data classification that:
    1. Loads training, validation, and test datasets from split directories
    2. Applies numerical imputation using mean strategy for missing values
    3. Fits risk tables on categorical features using training data
    4. Trains XGBoost model with configurable hyperparameters
    5. Supports both binary and multiclass classification
    6. Evaluates model performance with comprehensive metrics
    7. Saves model artifacts and preprocessing components
    """
)
```

### 2. Model Evaluation Contract

```python
MODEL_EVALUATION_CONTRACT = ScriptContract(
    entry_point="model_evaluation_xgb.py",
    expected_input_paths={
        "model_input": "/opt/ml/processing/input/model",
        "eval_data_input": "/opt/ml/processing/input/eval_data",
        "code_input": "/opt/ml/processing/input/code"
    },
    expected_output_paths={
        "eval_output": "/opt/ml/processing/output/eval",
        "metrics_output": "/opt/ml/processing/output/metrics"
    },
    required_env_vars=["ID_FIELD", "LABEL_FIELD"],
    framework_requirements={
        "xgboost": ">=1.6.0",
        "scikit-learn": ">=1.0.0",
        "pandas": ">=1.3.0",
        "matplotlib": ">=3.5.0"
    }
)
```

## AST-Based Script Analysis

The validation system uses Abstract Syntax Tree (AST) analysis to detect:

### Input/Output Path Usage
```python
# Detects hardcoded paths in scripts
"/opt/ml/processing/input/data"
"/opt/ml/model"

# Detects path construction
os.path.join("/opt/ml", "processing", "input", "data")
```

### Environment Variable Access
```python
# Detects various env var patterns
os.environ["LABEL_FIELD"]
os.environ.get("TRAIN_RATIO", "0.8")
os.getenv("DEBUG_MODE")
```

### Framework Import Analysis
```python
# Detects framework usage
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
```

## CLI Usage

### Command Line Validation
```bash
# Validate specific script
python -m src.pipeline_script_contracts.contract_validator --script train.py

# Validate all scripts with detailed output
python -m src.pipeline_script_contracts.contract_validator --verbose

# Validate scripts in specific directory
python -m src.pipeline_script_contracts.contract_validator --scripts-dir dockers/pytorch_bsm
```

### Programmatic Usage
```python
from src.pipeline_script_contracts import (
    ScriptContractValidator, 
    PYTORCH_TRAIN_CONTRACT,
    XGBOOST_TRAIN_CONTRACT
)

# Direct contract access
print(PYTORCH_TRAIN_CONTRACT.description)
print(PYTORCH_TRAIN_CONTRACT.framework_requirements)

# Validation for training scripts in different directories
pytorch_validator = ScriptContractValidator('dockers/pytorch_bsm')
xgboost_validator = ScriptContractValidator('dockers/xgboost_atoz')

pytorch_report = pytorch_validator.validate_script('train.py')
xgboost_report = xgboost_validator.validate_script('train_xgb.py')
```

## Integration Points

### With Step Specifications

**✅ IMPLEMENTED** - Script Contracts are now fully integrated with Step Specifications:

```python
# Step specifications now include script contracts
MODEL_EVAL_SPEC = StepSpecification(
    step_type="XGBoostModelEvaluation",
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_evaluation_contract(),  # Integrated contract
    dependencies=[...],
    outputs=[...]
)

# Automatic script validation
result = MODEL_EVAL_SPEC.validate_script_compliance("src/pipeline_scripts/model_evaluation_xgb.py")
if not result.is_valid:
    print(f"Script validation errors: {result.errors}")
```

**See Also**: [Step Specifications](step_specification.md) for complete integration details and job type variant handling.

### With Step Builders
Step builders can validate script compliance during configuration:

```python
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        super().validate_configuration()
        
        # Validate script compliance
        script_path = self.config.get_script_path()
        validation = TABULAR_PREPROCESS_CONTRACT.validate_implementation(script_path)
        
        if not validation.is_valid:
            raise ValueError(f"Script validation failed: {validation.errors}")
```

## Problem Solved: Script-Specification Misalignment

### Before Script Contracts (Implicit, Fragile)

```python
# Script hardcodes paths - no validation
def main():
    # Implicit assumption about input location
    data_path = "/opt/ml/processing/input/data"  # Hardcoded
    
    # Implicit assumption about environment
    label_field = os.environ["LABEL_FIELD"]  # May not exist
    
    # No validation that inputs exist
    df = pd.read_csv(data_path)  # May fail at runtime
```

**Problems:**
- ❌ Runtime failures if paths don't match step builder configuration
- ❌ No validation that required inputs exist
- ❌ Implicit dependencies on environment variables
- ❌ No documentation of script requirements

### After Script Contracts (Explicit, Validated)

```python
# Script with explicit contract validation
from src.pipeline_script_contracts import TABULAR_PREPROCESS_CONTRACT

def main():
    # Validate script compliance at startup
    validation = TABULAR_PREPROCESS_CONTRACT.validate_implementation(__file__)
    if not validation.is_valid:
        raise RuntimeError(f"Contract validation failed: {validation.errors}")
    
    # Use contract-defined paths
    data_path = "/opt/ml/processing/input/data"  # From contract
    
    # Validated environment access
    label_field = os.environ["LABEL_FIELD"]  # Contract ensures this exists
    
    # Safe processing with validated inputs
    df = pd.read_csv(data_path)
```

**Benefits:**
- ✅ Build-time validation prevents runtime failures
- ✅ Explicit documentation of script requirements
- ✅ Automatic validation that scripts match specifications
- ✅ Safe environment variable access
- ✅ Framework dependency documentation

## Current Implementation Status

### ✅ Completed Features
- **8 Complete Contracts**: All major pipeline scripts covered
- **Dual Contract Types**: Processing and Training script patterns
- **AST Validation**: Static code analysis for compliance checking
- **Framework Requirements**: Exact version specifications from requirements.txt
- **CLI Interface**: Command-line validation tools
- **Comprehensive Reports**: Detailed validation results with error categorization

### 📊 Contract Coverage
```
Total Contracts: 8
├── Processing Scripts: 6
│   ├── tabular_preprocess.py ✅
│   ├── mims_package.py ✅
│   ├── mims_payload.py ✅
│   ├── model_evaluation_xgb.py ✅
│   ├── currency_conversion.py ✅
│   └── risk_table_mapping.py ✅
└── Training Scripts: 2
    ├── train.py (PyTorch) ✅
    └── train_xgb.py (XGBoost) ✅
```

### 🔍 Validation Results
Current validation shows expected non-compliance for training scripts due to dynamic path construction patterns, which is normal and expected behavior.

## Strategic Value

Script Contracts enable:

1. **Fail-Fast Validation**: Catch errors at build time, not runtime
2. **Explicit Documentation**: Script requirements are self-documenting
3. **Safe Refactoring**: Changes to specifications validate all implementations
4. **Development Confidence**: Developers know scripts will work if they pass validation
5. **Maintainability**: Clear contracts make scripts easier to understand and modify
6. **Framework Standardization**: Exact dependency versions prevent conflicts
7. **CI/CD Integration**: Automated validation prevents non-compliant deployments

## File Structure

```
src/pipeline_script_contracts/
├── __init__.py                     # Module exports
├── base_script_contract.py         # Base contract for processing scripts
├── training_script_contract.py     # Specialized contract for training scripts
├── contract_validator.py           # Validation framework
├── tabular_preprocess_contract.py  # Tabular preprocessing contract
├── mims_package_contract.py        # MIMS packaging contract
├── mims_payload_contract.py        # MIMS payload contract
├── model_evaluation_contract.py    # Model evaluation contract
├── currency_conversion_contract.py # Currency conversion contract
├── risk_table_mapping_contract.py  # Risk table mapping contract
├── pytorch_train_contract.py       # PyTorch training contract
└── xgboost_train_contract.py       # XGBoost training contract
```

Script Contracts bridge the gap between **architectural intent** ([Step Specifications](step_specification.md)) and **implementation reality** (actual scripts), ensuring they remain aligned throughout the development lifecycle with automated validation and explicit documentation.
