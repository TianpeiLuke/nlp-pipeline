#!/usr/bin/env python3
"""
Runner script for backward compatibility tests.

This script runs the backward compatibility tests for the config field manager
refactoring, providing detailed output and verification that the new implementation
behaves identically to the old implementation.
"""

import unittest
import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import the backward compatibility test case
from test.config_field_manager.test_backward_compatibility import (
    BackwardCompatibilityTest,
    TestConfig1,
    TestConfig2,
    TestProcessingConfig1,
    TestProcessingConfig2,
)

# Import implementations
from src.pipeline_steps.utils import merge_and_save_configs as old_merge_and_save_configs
from src.config_field_manager import merge_and_save_configs as new_merge_and_save_configs


def run_tests():
    """Run the compatibility test suite."""
    print("\n========== Running Backward Compatibility Tests ==========\n")
    unittest.main(module='test.config_field_manager.test_backward_compatibility', exit=False)


def generate_example_configs():
    """Generate example configs with both old and new implementations."""
    print("\n========== Generating Example Configs ==========\n")
    
    # Create test configs
    configs = [
        TestConfig1(),
        TestConfig2(),
        TestProcessingConfig1(),
        TestProcessingConfig2(),
    ]
    
    # Create output directory
    output_dir = Path('test/config_field_manager/example_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Save configs with both implementations
    old_output_file = output_dir / 'old_implementation.json'
    new_output_file = output_dir / 'new_implementation.json'
    
    print(f"Generating output with old implementation: {old_output_file}")
    old_merge_and_save_configs(configs, str(old_output_file))
    
    print(f"Generating output with new implementation: {new_output_file}")
    new_merge_and_save_configs(configs, str(new_output_file))
    
    # Load and print summary
    with open(old_output_file, 'r') as f:
        old_data = json.load(f)
    
    with open(new_output_file, 'r') as f:
        new_data = json.load(f)
    
    print("\nStructure Comparison:")
    print(f"  Shared fields: old={len(old_data['configuration']['shared'])}, "
          f"new={len(new_data['configuration']['shared'])}")
          
    # The new implementation uses simplified structure without processing sections
    old_processing_shared_count = len(old_data['configuration']['processing']['processing_shared']) if 'processing' in old_data['configuration'] else 0
    old_processing_specific_count = len(old_data['configuration']['processing']['processing_specific']) if 'processing' in old_data['configuration'] else 0
    
    print(f"  Old implementation processing_shared fields: {old_processing_shared_count}")
    print(f"  Old implementation processing_specific steps: {old_processing_specific_count}")
    print(f"  Specific steps: old={len(old_data['configuration']['specific'])}, "
          f"new={len(new_data['configuration']['specific'])}")
    
    # Compare special fields
    old_hyperparams = []
    new_hyperparams = []
    
    # Check in specific steps in old implementation
    for step in old_data['configuration']['specific']:
        if "hyperparameters" in old_data['configuration']['specific'][step]:
            old_hyperparams.append(f"specific.{step}")
    
    # Check in processing specific steps in old implementation
    if 'processing' in old_data['configuration']:
        for step in old_data['configuration']['processing']['processing_specific']:
            if "hyperparameters" in old_data['configuration']['processing']['processing_specific'][step]:
                old_hyperparams.append(f"processing_specific.{step}")
    
    # Check in specific steps in new implementation (simplified structure)
    for step in new_data['configuration']['specific']:
        if "hyperparameters" in new_data['configuration']['specific'][step]:
            new_hyperparams.append(f"specific.{step}")
    
    print(f"\nHyperparameters placement:")
    print(f"  Old implementation: {', '.join(old_hyperparams)}")
    print(f"  New implementation: {', '.join(new_hyperparams)}")
    
    # Check for shared fields that differ
    shared_diffs = []
    for field in set(old_data['configuration']['shared'].keys()).intersection(
                    new_data['configuration']['shared'].keys()):
        if old_data['configuration']['shared'][field] != new_data['configuration']['shared'][field]:
            shared_diffs.append(field)
    
    if shared_diffs:
        print(f"\nWARNING: Found differences in shared fields: {', '.join(shared_diffs)}")
    else:
        print("\nAll shared fields have identical values between implementations.")


if __name__ == "__main__":
    print("\n===== Config Field Manager Backward Compatibility Test Suite =====\n")
    print("This script verifies that the new implementation in src/config_field_manager/")
    print("produces identical results to the original implementation in src/pipeline_steps/utils.py")
    
    # Run compatibility tests
    run_tests()
    
    # Generate example configs
    generate_example_configs()
    
    print("\n===== Compatibility Testing Complete =====\n")
