#!/usr/bin/env python3
"""
Step Name Consistency Validation Tool

This tool validates that all step names across the pipeline system are consistent
with the central registry. It checks:
1. Step specifications use central registry
2. Pipeline templates use consistent step names
3. Config classes align with registry
4. No hardcoded step names exist

Usage:
    python tools/validate_step_names.py
    ./tools/validate_step_names.py
"""

import os
import sys
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from src.pipeline_registry.step_names import (
        STEP_NAMES,
        get_spec_step_type,
        get_builder_step_name
    )
except ImportError as e:
    print(f"❌ ERROR: Could not import central registry: {e}")
    print("Make sure src/pipeline_registry/step_names.py exists and is properly configured")
    sys.exit(1)


@dataclass
class ValidationResult:
    """Result of a validation check"""
    component: str
    check_type: str
    status: bool
    message: str
    details: str = ""


class StepNameValidator:
    """Validates step name consistency across the pipeline system"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.results: List[ValidationResult] = []
        self.errors: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print("🔍 Step Name Consistency Validation")
        print("=" * 50)
        
        # Phase 1: Validate central registry
        self._validate_central_registry()
        
        # Phase 2: Validate step specifications
        self._validate_step_specifications()
        
        # Phase 3: Validate pipeline templates
        self._validate_pipeline_templates()
        
        # Phase 4: Validate config classes
        self._validate_config_classes()
        
        # Phase 5: Check for hardcoded step names
        self._check_hardcoded_step_names()
        
        # Print results
        self._print_results()
        
        # Return overall success
        return len(self.errors) == 0
    
    def _validate_central_registry(self):
        """Validate the central registry is properly configured"""
        print("\n📋 Validating Central Registry...")
        
        try:
            # Check registry has required step types
            required_steps = [
                "CradleDataLoading", "TabularPreprocessing", "PytorchTraining",
                "PytorchModel", "XGBoostTraining", "XGBoostModel", 
                "Registration", "Package", "Payload"
            ]
            
            missing_steps = []
            for step in required_steps:
                try:
                    result = get_spec_step_type(step)
                    if not result:
                        missing_steps.append(step)
                except Exception:
                    missing_steps.append(step)
            
            if missing_steps:
                self._add_error("Central Registry", "Missing step types", 
                              f"Missing: {missing_steps}")
            else:
                self._add_success("Central Registry", "All required step types present")
                
        except Exception as e:
            self._add_error("Central Registry", "Import failed", str(e))
    
    def _validate_step_specifications(self):
        """Validate step specifications use central registry"""
        print("\n📝 Validating Step Specifications...")
        
        spec_dir = self.project_root / "src" / "pipeline_step_specs"
        if not spec_dir.exists():
            self._add_error("Step Specifications", "Directory not found", str(spec_dir))
            return
        
        spec_files = list(spec_dir.glob("*.py"))
        if not spec_files:
            self._add_error("Step Specifications", "No specification files found", str(spec_dir))
            return
        
        registry_import_pattern = re.compile(r'from.*pipeline_registry.*import.*get_spec_step_type')
        registry_usage_pattern = re.compile(r'get_spec_step_type\s*\(\s*["\']([^"\']+)["\']\s*\)')
        
        for spec_file in spec_files:
            if spec_file.name.startswith("__"):
                continue
                
            try:
                content = spec_file.read_text()
                
                # Check for registry import
                has_import = bool(registry_import_pattern.search(content))
                
                # Check for registry usage
                usage_matches = registry_usage_pattern.findall(content)
                
                if has_import and usage_matches:
                    self._add_success("Step Specifications", f"{spec_file.name} uses central registry")
                elif "step_type=" in content:
                    # File defines step_type but doesn't use registry
                    self._add_error("Step Specifications", f"{spec_file.name} hardcodes step_type", 
                                  "Should use get_spec_step_type()")
                else:
                    # File might not define step specifications
                    self._add_success("Step Specifications", f"{spec_file.name} no step_type defined")
                    
            except Exception as e:
                self._add_error("Step Specifications", f"Error reading {spec_file.name}", str(e))
    
    def _validate_pipeline_templates(self):
        """Validate pipeline templates use consistent step names"""
        print("\n🏗️ Validating Pipeline Templates...")
        
        template_dir = self.project_root / "src" / "pipeline_builder"
        if not template_dir.exists():
            self._add_error("Pipeline Templates", "Directory not found", str(template_dir))
            return
        
        template_files = [f for f in template_dir.glob("template_*.py")]
        template_files.append(template_dir / "pipeline_builder_template.py")
        
        builder_map_pattern = re.compile(r'BUILDER_MAP\s*=\s*\{([^}]+)\}', re.DOTALL)
        step_name_pattern = re.compile(r'["\']([^"\']+)["\']:\s*\w+StepBuilder')
        
        for template_file in template_files:
            if not template_file.exists():
                continue
                
            try:
                content = template_file.read_text()
                
                # Find BUILDER_MAP definitions
                builder_maps = builder_map_pattern.findall(content)
                
                if not builder_maps:
                    # Check if file should have BUILDER_MAP
                    if "StepBuilder" in content:
                        self._add_error("Pipeline Templates", f"{template_file.name} missing BUILDER_MAP", 
                                      "Should define BUILDER_MAP with consistent step names")
                    else:
                        self._add_success("Pipeline Templates", f"{template_file.name} no step builders")
                    continue
                
                # Validate step names in BUILDER_MAP
                for builder_map in builder_maps:
                    step_names = step_name_pattern.findall(builder_map)
                    
                    for step_name in step_names:
                        # Check if step name exists in registry
                        try:
                            registry_name = get_spec_step_type(step_name)
                            if registry_name == step_name:
                                self._add_success("Pipeline Templates", 
                                                f"{template_file.name} uses consistent step name: {step_name}")
                            else:
                                self._add_error("Pipeline Templates", 
                                              f"{template_file.name} step name mismatch", 
                                              f"Uses '{step_name}', registry has '{registry_name}'")
                        except Exception:
                            # Step name not in registry - might be valid for specialized steps
                            self._add_success("Pipeline Templates", 
                                            f"{template_file.name} uses step name: {step_name} (not in registry)")
                            
            except Exception as e:
                self._add_error("Pipeline Templates", f"Error reading {template_file.name}", str(e))
    
    def _validate_config_classes(self):
        """Validate config classes use central registry"""
        print("\n⚙️ Validating Config Classes...")
        
        config_dir = self.project_root / "src" / "pipeline_steps"
        if not config_dir.exists():
            self._add_error("Config Classes", "Directory not found", str(config_dir))
            return
        
        config_files = [f for f in config_dir.glob("config_*.py")]
        config_files.append(config_dir / "config_base.py")
        
        registry_import_pattern = re.compile(r'from.*pipeline_registry.*import')
        
        for config_file in config_files:
            if not config_file.exists():
                continue
                
            try:
                content = config_file.read_text()
                
                # Check for registry import
                has_import = bool(registry_import_pattern.search(content))
                
                if has_import:
                    self._add_success("Config Classes", f"{config_file.name} imports from registry")
                elif "get_step_name" in content or "step_type" in content:
                    self._add_error("Config Classes", f"{config_file.name} missing registry import", 
                                  "Should import from pipeline_registry")
                else:
                    self._add_success("Config Classes", f"{config_file.name} no step name usage")
                    
            except Exception as e:
                self._add_error("Config Classes", f"Error reading {config_file.name}", str(e))
    
    def _check_hardcoded_step_names(self):
        """Check for hardcoded step names that should use registry"""
        print("\n🔍 Checking for Hardcoded Step Names...")
        
        # Common hardcoded patterns to look for
        hardcoded_patterns = [
            re.compile(r'step_type\s*=\s*["\']([A-Z][a-zA-Z]+Step)["\']'),
            re.compile(r'["\']([A-Z][a-zA-Z]*Training)["\']'),
            re.compile(r'["\']([A-Z][a-zA-Z]*Model)["\']'),
            re.compile(r'["\']([A-Z][a-zA-Z]*Processing)["\']'),
        ]
        
        # Directories to check
        check_dirs = [
            self.project_root / "src" / "pipeline_step_specs",
            self.project_root / "src" / "pipeline_builder",
            self.project_root / "src" / "pipeline_steps",
        ]
        
        for check_dir in check_dirs:
            if not check_dir.exists():
                continue
                
            for py_file in check_dir.glob("*.py"):
                if py_file.name.startswith("__"):
                    continue
                    
                try:
                    content = py_file.read_text()
                    
                    # Skip files that properly import registry
                    if "from.*pipeline_registry.*import" in content:
                        continue
                    
                    # Check for hardcoded patterns
                    for pattern in hardcoded_patterns:
                        matches = pattern.findall(content)
                        for match in matches:
                            self._add_error("Hardcoded Names", 
                                          f"{py_file.relative_to(self.project_root)} contains hardcoded step name", 
                                          f"Found: '{match}' - should use registry")
                            
                except Exception as e:
                    self._add_error("Hardcoded Names", f"Error reading {py_file.name}", str(e))
    
    def _add_success(self, component: str, message: str, details: str = ""):
        """Add a successful validation result"""
        self.results.append(ValidationResult(component, "SUCCESS", True, message, details))
    
    def _add_error(self, component: str, message: str, details: str = ""):
        """Add a failed validation result"""
        self.results.append(ValidationResult(component, "ERROR", False, message, details))
        self.errors.append(f"{component}: {message}")
    
    def _print_results(self):
        """Print validation results"""
        print("\n" + "=" * 50)
        print("📊 Validation Results")
        print("=" * 50)
        
        # Group results by component
        by_component = {}
        for result in self.results:
            if result.component not in by_component:
                by_component[result.component] = []
            by_component[result.component].append(result)
        
        # Print results by component
        for component, results in by_component.items():
            print(f"\n{component}:")
            for result in results:
                status_icon = "✅" if result.status else "❌"
                print(f"  {status_icon} {result.message}")
                if result.details:
                    print(f"     {result.details}")
        
        # Print summary
        total_checks = len(self.results)
        successful_checks = len([r for r in self.results if r.status])
        failed_checks = total_checks - successful_checks
        
        print(f"\n" + "=" * 50)
        print(f"📈 Summary: {successful_checks}/{total_checks} checks passed")
        
        if failed_checks == 0:
            print("🎉 ALL VALIDATIONS PASSED!")
            print("✅ Step name consistency is maintained across the system")
        else:
            print(f"❌ {failed_checks} validation(s) failed")
            print("🔧 Please fix the issues above to maintain step name consistency")
        
        print("=" * 50)


def main():
    """Main entry point"""
    validator = StepNameValidator()
    success = validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
