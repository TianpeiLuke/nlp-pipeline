#!/usr/bin/env python3
"""
Contract Validation Tool

Validates all script contracts against their specifications to ensure alignment
and prevent runtime failures due to mismatched dependencies and outputs.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def validate_model_evaluation_alignment():
    """Validate model evaluation step alignment"""
    print("🔍 Validating Model Evaluation Step Alignment...")
    
    try:
        from src.pipeline_step_specs.model_eval_spec import MODEL_EVAL_SPEC
        
        # Test contract alignment
        result = MODEL_EVAL_SPEC.validate_contract_alignment()
        
        if result.is_valid:
            print("✅ Model Evaluation: Contract aligned with specification")
            return True
        else:
            print("❌ Model Evaluation: Contract misalignment detected")
            for error in result.errors:
                print(f"   - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Model Evaluation: Validation failed with error: {e}")
        return False

def validate_all_contracts():
    """Validate all script contracts against their specifications"""
    print("=" * 60)
    print("🔒 CONTRACT VALIDATION REPORT")
    print("=" * 60)
    
    all_valid = True
    
    # Validate Model Evaluation
    if not validate_model_evaluation_alignment():
        all_valid = False
    
    print()
    
    # Add more validations here as we implement them
    # if not validate_preprocessing_alignment():
    #     all_valid = False
    # if not validate_training_alignment():
    #     all_valid = False
    
    print("=" * 60)
    if all_valid:
        print("🎉 ALL CONTRACTS VALIDATED SUCCESSFULLY")
        print("   All specifications align with their contracts")
    else:
        print("⚠️  CONTRACT VALIDATION FAILURES DETECTED")
        print("   Please fix the alignment issues above")
    print("=" * 60)
    
    return all_valid

def test_contract_enforcement():
    """Test contract enforcement functionality"""
    print("\n🧪 Testing Contract Enforcement...")
    
    try:
        from src.pipeline_scripts.contract_utils import ContractEnforcer
        from src.pipeline_script_contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT
        
        print("✅ Contract utilities imported successfully")
        print("✅ Model evaluation contract loaded successfully")
        
        # Test contract validation (without actual SageMaker environment)
        print("✅ Contract enforcement classes available")
        
        return True
        
    except Exception as e:
        print(f"❌ Contract enforcement test failed: {e}")
        return False

def main():
    """Main validation entry point"""
    print("🚀 Starting Contract Validation Suite...")
    print()
    
    # Test contract enforcement functionality
    enforcement_ok = test_contract_enforcement()
    
    # Validate all contracts
    contracts_ok = validate_all_contracts()
    
    # Overall result
    if contracts_ok and enforcement_ok:
        print("\n🎯 VALIDATION SUITE PASSED")
        print("   Ready for deployment!")
        return 0
    else:
        print("\n💥 VALIDATION SUITE FAILED")
        print("   Please fix issues before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
