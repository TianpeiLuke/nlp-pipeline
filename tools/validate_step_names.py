#!/usr/bin/env python3
"""
Step Name Consistency Validation Tool

This tool validates that step names are consistent across all pipeline components:
1. Config Registry (src/pipeline_steps/config_base.py)
2. Builder Registry (src/pipeline_steps/builder_step_base.py)
3. Step Specifications (src/pipeline_step_specs/*.py)
4. Pipeline Templates (src/pipeline_builder/*.py)

Usage:
    python tools/validate_step_names.py
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import importlib.util
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_module_from_file(file_path: Path, module_name: str):
    """Load a Python module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_central_registry() -> Dict[str, Dict[str, str]]:
    """Get the central step names registry."""
    try:
        # Import with package path since we added project root to sys.path
        from src.pipeline_registry.step_names import STEP_NAMES
        return STEP_NAMES
    except ImportError as e:
        print(f"Error importing central registry: {e}")
        return {}

def validate_config_registry() -> Tuple[bool, List[str]]:
    """Validate that config registry uses central registry."""
    issues = []
    
    try:
        # Import with package paths since we added project root to sys.path
        from src.pipeline_steps.config_base import STEP_REGISTRY
        from src.pipeline_registry.step_names import CONFIG_STEP_REGISTRY
        
        # Check if they match
        if STEP_REGISTRY != CONFIG_STEP_REGISTRY:
            issues.append("Config STEP_REGISTRY does not match central CONFIG_STEP_REGISTRY")
            
            # Find differences
            config_keys = set(STEP_REGISTRY.keys())
            central_keys = set(CONFIG_STEP_REGISTRY.keys())
            
            missing_in_config = central_keys - config_keys
            extra_in_config = config_keys - central_keys
            
            if missing_in_config:
                issues.append(f"Missing in config registry: {missing_in_config}")
            if extra_in_config:
                issues.append(f"Extra in config registry: {extra_in_config}")
                
            # Check value differences
            common_keys = config_keys & central_keys
            for key in common_keys:
                if STEP_REGISTRY[key] != CONFIG_STEP_REGISTRY[key]:
                    issues.append(f"Value mismatch for {key}: config='{STEP_REGISTRY[key]}' vs central='{CONFIG_STEP_REGISTRY[key]}'")
        
    except ImportError as e:
        issues.append(f"Error importing config registry: {e}")
    
    return len(issues) == 0, issues

def validate_builder_registry() -> Tuple[bool, List[str]]:
    """Validate that builder registry uses central registry."""
    issues = []
    
    try:
        # Import with package paths since we added project root to sys.path
        from src.pipeline_steps.builder_step_base import STEP_NAMES as BUILDER_STEP_NAMES_IMPORTED
        from src.pipeline_registry.step_names import BUILDER_STEP_NAMES
        
        # Check if they match
        if BUILDER_STEP_NAMES_IMPORTED != BUILDER_STEP_NAMES:
            issues.append("Builder STEP_NAMES does not match central BUILDER_STEP_NAMES")
            
            # Find differences
            builder_keys = set(BUILDER_STEP_NAMES_IMPORTED.keys())
            central_keys = set(BUILDER_STEP_NAMES.keys())
            
            missing_in_builder = central_keys - builder_keys
            extra_in_builder = builder_keys - central_keys
            
            if missing_in_builder:
                issues.append(f"Missing in builder registry: {missing_in_builder}")
            if extra_in_builder:
                issues.append(f"Extra in builder registry: {extra_in_builder}")
                
            # Check value differences
            common_keys = builder_keys & central_keys
            for key in common_keys:
                if BUILDER_STEP_NAMES_IMPORTED[key] != BUILDER_STEP_NAMES[key]:
                    issues.append(f"Value mismatch for {key}: builder='{BUILDER_STEP_NAMES_IMPORTED[key]}' vs central='{BUILDER_STEP_NAMES[key]}'")
        
    except ImportError as e:
        issues.append(f"Error importing builder registry: {e}")
    
    return len(issues) == 0, issues

def validate_step_specifications() -> Tuple[bool, List[str]]:
    """Validate step specifications use consistent step_type values."""
    issues = []
    spec_dir = Path(__file__).parent.parent / "src" / "pipeline_step_specs"
    
    if not spec_dir.exists():
        issues.append(f"Step specifications directory not found: {spec_dir}")
        return False, issues
    
    # Get central registry for validation
    central_registry = get_central_registry()
    if not central_registry:
        issues.append("Could not load central registry for validation")
        return False, issues
    
    # Valid step types from central registry
    valid_step_types = set(info["spec_type"] for info in central_registry.values())
    
    # Check each specification file
    for spec_file in spec_dir.glob("*.py"):
        if spec_file.name.startswith("__"):
            continue
            
        # Read file content to look for step_type definitions
        try:
            content = spec_file.read_text()
            
            # Look for step_type assignments
            step_type_matches = re.findall(r'step_type\s*=\s*["\']([^"\']+)["\']', content)
            get_spec_matches = re.findall(r'step_type\s*=\s*get_spec_step_type\s*\(\s*["\']([^"\']+)["\']\s*\)', content)
            
            # Check hardcoded step_type values
            for step_type in step_type_matches:
                # Handle job type variants (e.g., "TabularPreprocessing_Training")
                base_step_type = step_type.split('_')[0] if '_' in step_type else step_type
                
                if base_step_type not in valid_step_types:
                    issues.append(f"{spec_file.name}: Unknown step_type '{step_type}' (base: '{base_step_type}')")
                else:
                    # Check if it should use get_spec_step_type instead
                    if step_type in valid_step_types:
                        issues.append(f"{spec_file.name}: Should use get_spec_step_type('{step_type}') instead of hardcoded '{step_type}'")
            
            # Check get_spec_step_type calls
            for step_name in get_spec_matches:
                if step_name not in central_registry:
                    issues.append(f"{spec_file.name}: Unknown step name '{step_name}' in get_spec_step_type()")
                    
        except Exception as e:
            issues.append(f"Error reading {spec_file.name}: {e}")
    
    return len(issues) == 0, issues

def validate_pipeline_templates() -> Tuple[bool, List[str]]:
    """Validate pipeline templates use consistent step names."""
    issues = []
    template_dir = Path(__file__).parent.parent / "src" / "pipeline_builder"
    
    if not template_dir.exists():
        issues.append(f"Pipeline builder directory not found: {template_dir}")
        return False, issues
    
    # Get central registry for validation
    central_registry = get_central_registry()
    if not central_registry:
        issues.append("Could not load central registry for validation")
        return False, issues
    
    # Valid step names and types
    valid_step_names = set(central_registry.keys())
    valid_step_types = set(info["spec_type"] for info in central_registry.values())
    valid_config_classes = set(info["config_class"] for info in central_registry.values())
    valid_builder_classes = set(info["builder_step_name"] for info in central_registry.values())
    
    # Check each template file
    for template_file in template_dir.glob("*.py"):
        if template_file.name.startswith("__"):
            continue
            
        try:
            content = template_file.read_text()
            
            # Look for potential step name references
            # This is a heuristic check - may need refinement
            
            # Check for hardcoded step type strings
            for step_type in valid_step_types:
                if f'"{step_type}"' in content or f"'{step_type}'" in content:
                    # This might be okay, but flag for review
                    issues.append(f"{template_file.name}: Contains hardcoded step type '{step_type}' - verify consistency")
            
            # Check for config class references
            for config_class in valid_config_classes:
                if config_class in content:
                    # This is expected in templates
                    pass
                    
        except Exception as e:
            issues.append(f"Error reading {template_file.name}: {e}")
    
    return len(issues) == 0, issues

def find_orphaned_references() -> Tuple[bool, List[str]]:
    """Find hardcoded step name references outside the registry."""
    issues = []
    src_dir = Path(__file__).parent.parent / "src"
    
    # Get central registry
    central_registry = get_central_registry()
    if not central_registry:
        issues.append("Could not load central registry for validation")
        return False, issues
    
    # Step types that should only come from registry
    step_types_to_check = set(info["spec_type"] for info in central_registry.values())
    
    # Files to check (excluding registry files)
    files_to_check = []
    for pattern in ["**/*.py"]:
        files_to_check.extend(src_dir.glob(pattern))
    
    # Exclude registry files
    registry_files = {
        src_dir / "pipeline_registry" / "__init__.py",
        src_dir / "pipeline_registry" / "step_names.py"
    }
    
    for file_path in files_to_check:
        if file_path in registry_files or file_path.name.startswith("__"):
            continue
            
        try:
            content = file_path.read_text()
            
            # Look for hardcoded step type strings
            for step_type in step_types_to_check:
                # Skip very common words that might appear in comments
                if step_type.lower() in ["base", "processing"]:
                    continue
                    
                # Look for quoted strings
                if f'"{step_type}"' in content or f"'{step_type}'" in content:
                    # Check if it's in a get_spec_step_type call (which is okay)
                    if f'get_spec_step_type("{step_type}")' not in content and f"get_spec_step_type('{step_type}')" not in content:
                        rel_path = file_path.relative_to(src_dir)
                        issues.append(f"{rel_path}: Contains hardcoded step type '{step_type}' - should use registry")
                        
        except Exception as e:
            rel_path = file_path.relative_to(src_dir)
            issues.append(f"Error reading {rel_path}: {e}")
    
    return len(issues) == 0, issues

def main():
    """Main validation function."""
    print("Step Name Consistency Validation")
    print("=" * 40)
    
    all_passed = True
    
    # Test 1: Central Registry
    print("\n1. Checking Central Registry...")
    central_registry = get_central_registry()
    if central_registry:
        print(f"   ✓ Central registry loaded with {len(central_registry)} step definitions")
        for step_name, info in central_registry.items():
            print(f"     - {step_name}: {info['spec_type']}")
    else:
        print("   ✗ Failed to load central registry")
        all_passed = False
    
    # Test 2: Config Registry
    print("\n2. Validating Config Registry...")
    config_passed, config_issues = validate_config_registry()
    if config_passed:
        print("   ✓ Config registry is consistent with central registry")
    else:
        print("   ✗ Config registry issues found:")
        for issue in config_issues:
            print(f"     - {issue}")
        all_passed = False
    
    # Test 3: Builder Registry
    print("\n3. Validating Builder Registry...")
    builder_passed, builder_issues = validate_builder_registry()
    if builder_passed:
        print("   ✓ Builder registry is consistent with central registry")
    else:
        print("   ✗ Builder registry issues found:")
        for issue in builder_issues:
            print(f"     - {issue}")
        all_passed = False
    
    # Test 4: Step Specifications
    print("\n4. Validating Step Specifications...")
    spec_passed, spec_issues = validate_step_specifications()
    if spec_passed:
        print("   ✓ Step specifications are consistent")
    else:
        print("   ✗ Step specification issues found:")
        for issue in spec_issues:
            print(f"     - {issue}")
        all_passed = False
    
    # Test 5: Pipeline Templates
    print("\n5. Validating Pipeline Templates...")
    template_passed, template_issues = validate_pipeline_templates()
    if template_passed:
        print("   ✓ Pipeline templates are consistent")
    else:
        print("   ✗ Pipeline template issues found:")
        for issue in template_issues:
            print(f"     - {issue}")
        all_passed = False
    
    # Test 6: Orphaned References
    print("\n6. Checking for Orphaned References...")
    orphan_passed, orphan_issues = find_orphaned_references()
    if orphan_passed:
        print("   ✓ No orphaned step name references found")
    else:
        print("   ✗ Orphaned references found:")
        for issue in orphan_issues:
            print(f"     - {issue}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 40)
    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("Step name consistency is maintained across all components.")
        return 0
    else:
        print("✗ VALIDATION FAILURES DETECTED")
        print("Please address the issues above to ensure step name consistency.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
