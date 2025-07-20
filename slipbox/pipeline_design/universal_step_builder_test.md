# Universal Step Builder Test

This document outlines the design and implementation of a standardized, universal test suite for validating step builder classes. The universal test serves as a quality gate to ensure that all step builders align with architectural standards and can seamlessly integrate into the specification-driven pipeline system.

## Purpose

The Universal Step Builder Test provides an automated validation mechanism that:

1. **Enforces Interface Compliance** - Ensures step builders implement required methods and inheritance
2. **Validates Specification Integration** - Verifies proper use of step specifications and script contracts
3. **Confirms Dependency Handling** - Tests correct resolution of inputs from dependencies
4. **Evaluates Environment Variable Processing** - Validates contract-driven environment variable management
5. **Verifies Step Creation** - Tests that the builder produces valid and properly configured steps
6. **Assesses Error Handling** - Confirms builders respond appropriately to invalid inputs

## Design Principles

The universal test is designed following these key principles:

1. **Parameterized Testing** - A single test suite that can be applied to any step builder
2. **Comprehensive Coverage** - Tests all aspects of step builder functionality
3. **Minimized Boilerplate** - Test logic is centralized to avoid duplication
4. **Realistic Mocking** - Uses realistic mock objects to simulate the SageMaker environment
5. **Self-Contained** - Tests can run without external dependencies or SageMaker connectivity

## Test Structure

The universal test is structured as a parameterizable test class:

```python
class UniversalStepBuilderTest:
    """
    Universal test suite for validating step builder implementation compliance.
    
    This test can be applied to any step builder class to verify that it meets
    the architectural requirements for integration into the pipeline system.
    
    Usage:
        # Test a specific builder
        tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder)
        tester.run_all_tests()
        
        # Or register with pytest
        @pytest.mark.parametrize("builder_class", [
            XGBoostTrainingStepBuilder,
            TabularPreprocessingStepBuilder,
            ModelEvalStepBuilder
        ])
        def test_step_builder_compliance(builder_class):
            tester = UniversalStepBuilderTest(builder_class)
            tester.run_all_tests()
    """
    
    def __init__(self, builder_class):
        """Initialize with the step builder class to test."""
        self.builder_class = builder_class
        self._setup_test_environment()
    
    def run_all_tests(self):
        """Run all tests and return a consolidated result."""
        test_methods = [
            self.test_inheritance,
            self.test_required_methods,
            self.test_specification_usage,
            self.test_contract_alignment,
            self.test_environment_variable_handling,
            self.test_dependency_resolution,
            self.test_step_creation,
            self.test_error_handling
        ]
        
        results = {}
        for test_method in test_methods:
            results[test_method.__name__] = self._run_test(test_method)
            
        return results
    
    def test_inheritance(self):
        """Test that the builder inherits from StepBuilderBase."""
        # Implementation...
    
    def test_required_methods(self):
        """Test that the builder implements all required methods."""
        # Implementation...
    
    def test_specification_usage(self):
        """Test that the builder uses a valid specification."""
        # Implementation...
    
    def test_contract_alignment(self):
        """Test that the specification aligns with the script contract."""
        # Implementation...
    
    def test_environment_variable_handling(self):
        """Test that the builder handles environment variables correctly."""
        # Implementation...
    
    def test_dependency_resolution(self):
        """Test that the builder resolves dependencies correctly."""
        # Implementation...
    
    def test_step_creation(self):
        """Test that the builder creates a valid step."""
        # Implementation...
    
    def test_error_handling(self):
        """Test that the builder handles errors appropriately."""
        # Implementation...
    
    def _setup_test_environment(self):
        """Set up mock objects and test fixtures."""
        # Implementation...
    
    def _run_test(self, test_method):
        """Run a single test method and capture results."""
        # Implementation...
```

## Test Cases

### 1. Inheritance Test

Verifies that the builder inherits from `StepBuilderBase`:

```python
def test_inheritance(self):
    """Test that the builder inherits from StepBuilderBase."""
    from src.pipeline_steps.builder_step_base import StepBuilderBase
    
    self.assertTrue(
        issubclass(self.builder_class, StepBuilderBase),
        f"{self.builder_class.__name__} must inherit from StepBuilderBase"
    )
```

### 2. Required Methods Test

Verifies that all required methods are implemented:

```python
def test_required_methods(self):
    """Test that the builder implements all required methods."""
    required_methods = [
        'validate_configuration',
        '_get_inputs',
        '_get_outputs',
        'create_step'
    ]
    
    for method_name in required_methods:
        method = getattr(self.builder_class, method_name, None)
        self.assertIsNotNone(
            method,
            f"Builder must implement {method_name}()"
        )
        self.assertTrue(
            callable(method),
            f"{method_name} must be callable"
        )
```

### 3. Specification Usage Test

Verifies that the builder uses a valid specification:

```python
def test_specification_usage(self):
    """Test that the builder uses a valid specification."""
    # Create instance with mock config
    builder = self._create_builder_instance()
    
    # Check that spec is available
    self.assertIsNotNone(
        builder.spec,
        f"Builder must have a non-None spec attribute"
    )
    
    # Verify spec has required attributes
    required_spec_attrs = [
        'step_type',
        'node_type',
        'dependencies',
        'outputs'
    ]
    
    for attr in required_spec_attrs:
        self.assertTrue(
            hasattr(builder.spec, attr),
            f"Specification must have {attr} attribute"
        )
```

### 4. Contract Alignment Test

Verifies that the specification aligns with the script contract:

```python
def test_contract_alignment(self):
    """Test that the specification aligns with the script contract."""
    # Create instance with mock config
    builder = self._create_builder_instance()
    
    # Check contract is available
    self.assertIsNotNone(
        builder.contract,
        f"Builder must have a non-None contract attribute"
    )
    
    # Verify contract has required attributes
    required_contract_attrs = [
        'entry_point',
        'expected_input_paths',
        'expected_output_paths'
    ]
    
    for attr in required_contract_attrs:
        self.assertTrue(
            hasattr(builder.contract, attr),
            f"Contract must have {attr} attribute"
        )
    
    # Verify all dependency logical names have corresponding paths in contract
    if hasattr(builder.spec, 'dependencies'):
        for dep_name, dep_spec in builder.spec.dependencies.items():
            logical_name = dep_spec.logical_name
            if logical_name != "hyperparameters_s3_uri":  # Special case
                self.assertIn(
                    logical_name,
                    builder.contract.expected_input_paths,
                    f"Dependency {logical_name} must have corresponding path in contract"
                )
    
    # Verify all output logical names have corresponding paths in contract
    if hasattr(builder.spec, 'outputs'):
        for out_name, out_spec in builder.spec.outputs.items():
            logical_name = out_spec.logical_name
            self.assertIn(
                logical_name,
                builder.contract.expected_output_paths,
                f"Output {logical_name} must have corresponding path in contract"
            )
```

### 5. Environment Variable Handling Test

Verifies that the builder correctly handles environment variables:

```python
def test_environment_variable_handling(self):
    """Test that the builder handles environment variables correctly."""
    # Create instance with mock config
    builder = self._create_builder_instance()
    
    # Get environment variables
    env_vars = builder._get_environment_variables()
    
    # Verify environment variables include required variables from contract
    if hasattr(builder.contract, 'required_env_vars'):
        for env_var in builder.contract.required_env_vars:
            self.assertIn(
                env_var,
                env_vars,
                f"Environment variables must include required variable {env_var}"
            )
    
    # Verify environment variables include optional variables with defaults
    if hasattr(builder.contract, 'optional_env_vars'):
        for env_var, default in builder.contract.optional_env_vars.items():
            self.assertIn(
                env_var,
                env_vars,
                f"Environment variables must include optional variable {env_var}"
            )
```

### 6. Dependency Resolution Test

Verifies that the builder correctly resolves dependencies:

```python
def test_dependency_resolution(self):
    """Test that the builder resolves dependencies correctly."""
    # Create instance with mock config
    builder = self._create_builder_instance()
    
    # Create mock dependencies
    dependencies = self._create_mock_dependencies()
    
    # Test extraction of inputs from dependencies
    try:
        extracted_inputs = builder.extract_inputs_from_dependencies(dependencies)
        
        # Verify extracted inputs include required dependencies
        required_deps = builder.get_required_dependencies()
        for dep_name in required_deps:
            self.assertIn(
                dep_name,
                extracted_inputs,
                f"Extracted inputs must include required dependency {dep_name}"
            )
    except Exception as e:
        self.fail(f"Dependency resolution failed: {str(e)}")
```

### 7. Step Creation Test

Verifies that the builder correctly creates a step:

```python
def test_step_creation(self):
    """Test that the builder creates a valid step."""
    # Create instance with mock config
    builder = self._create_builder_instance()
    
    # Create mock dependencies
    dependencies = self._create_mock_dependencies()
    
    # Test step creation
    try:
        step = builder.create_step(
            dependencies=dependencies,
            enable_caching=True
        )
        
        # Verify step has required attributes
        self.assertIsNotNone(
            step,
            "Step must be created successfully"
        )
        
        # Verify step has spec attached
        self.assertTrue(
            hasattr(step, '_spec'),
            "Step must have _spec attribute"
        )
        
        # Verify step has name
        self.assertTrue(
            hasattr(step, 'name'),
            "Step must have name attribute"
        )
    except Exception as e:
        self.fail(f"Step creation failed: {str(e)}")
```

### 8. Error Handling Test

Verifies that the builder handles errors appropriately:

```python
def test_error_handling(self):
    """Test that the builder handles errors appropriately."""
    # Test with invalid configuration
    try:
        # Create config without required attributes
        invalid_config = self._create_invalid_config()
        
        # Create builder with invalid config
        builder = self.builder_class(
            config=invalid_config,
            sagemaker_session=self.mock_session,
            role=self.mock_role
        )
        
        # Should raise ValueError
        builder.validate_configuration()
        
        # If we get here, validation didn't fail
        self.fail("validate_configuration should raise ValueError for invalid config")
    except ValueError:
        # Expected behavior
        pass
    except Exception as e:
        self.fail(f"validate_configuration should raise ValueError, not {type(e).__name__}")
```

## Mock Implementation

The test suite uses comprehensive mocking to simulate the SageMaker environment:

```python
def _setup_test_environment(self):
    """Set up mock objects and test fixtures."""
    # Mock SageMaker session
    self.mock_session = MagicMock()
    self.mock_session.boto_session.client.return_value = MagicMock()
    
    # Mock IAM role
    self.mock_role = 'arn:aws:iam::123456789012:role/MockRole'
    
    # Create mock registry manager and dependency resolver
    self.mock_registry_manager = MagicMock()
    self.mock_dependency_resolver = MagicMock()
    
    # Configure dependency resolver for successful resolution
    self.mock_dependency_resolver.resolve_step_dependencies.return_value = {
        dep: MagicMock() for dep in self._get_expected_dependencies()
    }
    
def _create_builder_instance(self):
    """Create a builder instance with mock configuration."""
    # Create mock configuration
    mock_config = self._create_mock_config()
    
    # Create builder instance
    builder = self.builder_class(
        config=mock_config,
        sagemaker_session=self.mock_session,
        role=self.mock_role,
        registry_manager=self.mock_registry_manager,
        dependency_resolver=self.mock_dependency_resolver
    )
    
    return builder

def _create_mock_config(self):
    """Create a mock configuration for the builder."""
    # Basic config with required attributes
    mock_config = SimpleNamespace()
    mock_config.region = 'NA'
    
    # Add builder-specific attributes
    self._add_builder_specific_config(mock_config)
    
    return mock_config
    
def _add_builder_specific_config(self, mock_config):
    """Add builder-specific configuration attributes."""
    # Get builder class name
    builder_name = self.builder_class.__name__
    
    if "XGBoostTraining" in builder_name:
        self._add_xgboost_training_config(mock_config)
    elif "TabularPreprocessing" in builder_name:
        self._add_tabular_preprocessing_config(mock_config)
    elif "ModelEval" in builder_name:
        self._add_model_eval_config(mock_config)
    # Add more builder-specific configurations as needed
    else:
        self._add_generic_config(mock_config)
```

## Test Execution

The universal test can be executed in two ways:

### 1. Standalone Usage

```python
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from test.pipeline_steps.universal_step_builder_test import UniversalStepBuilderTest

# Test a specific builder
tester = UniversalStepBuilderTest(XGBoostTrainingStepBuilder)
results = tester.run_all_tests()

for test_name, result in results.items():
    if result['passed']:
        print(f"✅ {test_name} PASSED")
    else:
        print(f"❌ {test_name} FAILED: {result['error']}")
```

### 2. Pytest Integration

```python
import pytest
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from test.pipeline_steps.universal_step_builder_test import UniversalStepBuilderTest

@pytest.mark.parametrize("builder_class", [
    XGBoostTrainingStepBuilder,
    TabularPreprocessingStepBuilder,
    # Add more builders to test
])
def test_step_builder_compliance(builder_class):
    tester = UniversalStepBuilderTest(builder_class)
    results = tester.run_all_tests()
    
    # Assert all tests passed
    for test_name, result in results.items():
        assert result['passed'], f"{test_name} failed for {builder_class.__name__}: {result['error']}"
```

## Complete Implementation

The complete test implementation is available in `test/pipeline_steps/universal_step_builder_test.py`:

```python
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import inspect
import importlib
from pathlib import Path

# Import for type hints only
from typing import Dict, List, Any, Optional, Union, Type, Callable
from sagemaker.workflow.steps import Step

# Import StepBuilderBase for inheritance check
from src.pipeline_steps.builder_step_base import StepBuilderBase


class UniversalStepBuilderTest:
    """
    Universal test suite for validating step builder implementation compliance.
    
    This test can be applied to any step builder class to verify that it meets
    the architectural requirements for integration into the pipeline system.
    """
    
    def __init__(
        self, 
        builder_class: Type[StepBuilderBase],
        verbose: bool = False,
        test_reporter: Optional[Callable] = None
    ):
        """
        Initialize with the step builder class to test.
        
        Args:
            builder_class: The step builder class to test
            verbose: Whether to print verbose output
            test_reporter: Optional function to report test results
        """
        self.builder_class = builder_class
        self.verbose = verbose
        self.test_reporter = test_reporter or (lambda *args, **kwargs: None)
        self._setup_test_environment()
    
    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """
        Run all tests and return a consolidated result.
        
        Returns:
            Dictionary mapping test names to their results
        """
        test_methods = [
            self.test_inheritance,
            self.test_required_methods,
            self.test_specification_usage,
            self.test_contract_alignment,
            self.test_environment_variable_handling,
            self.test_dependency_resolution,
            self.test_step_creation,
            self.test_error_handling
        ]
        
        results = {}
        for test_method in test_methods:
            results[test_method.__name__] = self._run_test(test_method)
            
        # Report overall results
        self._report_overall_results(results)
        
        return results
    
    def test_inheritance(self) -> None:
        """Test that the builder inherits from StepBuilderBase."""
        self._assert(
            issubclass(self.builder_class, StepBuilderBase),
            f"{self.builder_class.__name__} must inherit from StepBuilderBase"
        )
    
    def test_required_methods(self) -> None:
        """Test that the builder implements all required methods."""
        required_methods = [
            'validate_configuration',
            '_get_inputs',
            '_get_outputs',
            'create_step'
        ]
        
        for method_name in required_methods:
            method = getattr(self.builder_class, method_name, None)
            self._assert(
                method is not None,
                f"Builder must implement {method_name}()"
            )
            self._assert(
                callable(method),
                f"{method_name} must be callable"
            )
            
            # Check if method is abstract or implemented
            if method_name not in ['create_step']:  # create_step is often overridden
                self._assert(
                    not getattr(method, '__isabstractmethod__', False),
                    f"{method_name}() must be implemented, not abstract"
                )
    
    def test_specification_usage(self) -> None:
        """Test that the builder uses a valid specification."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Check that spec is available
        self._assert(
            hasattr(builder, 'spec'),
            f"Builder must have a spec attribute"
        )
        
        self._assert(
            builder.spec is not None,
            f"Builder must have a non-None spec attribute"
        )
        
        # Verify spec has required attributes
        required_spec_attrs = [
            'step_type',
            'node_type',
            'dependencies',
            'outputs'
        ]
        
        for attr in required_spec_attrs:
            self._assert(
                hasattr(builder.spec, attr),
                f"Specification must have {attr} attribute"
            )
    
    def test_contract_alignment(self) -> None:
        """Test that the specification aligns with the script contract."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Check contract is available
        self._assert(
            hasattr(builder, 'contract'),
            f"Builder must have a contract attribute"
        )
        
        if builder.contract is None:
            self._log("Contract is None, skipping contract alignment tests")
            return
        
        # Verify contract has required attributes
        required_contract_attrs = [
            'entry_point',
            'expected_input_paths',
            'expected_output_paths'
        ]
        
        for attr in required_contract_attrs:
            self._assert(
                hasattr(builder.contract, attr),
                f"Contract must have {attr} attribute"
            )
        
        # Verify all dependency logical names have corresponding paths in contract
        if hasattr(builder.spec, 'dependencies'):
            for dep_name, dep_spec in builder.spec.dependencies.items():
                logical_name = dep_spec.logical_name
                if logical_name != "hyperparameters_s3_uri":  # Special case
                    self._assert(
                        logical_name in builder.contract.expected_input_paths,
                        f"Dependency {logical_name} must have corresponding path in contract"
                    )
        
        # Verify all output logical names have corresponding paths in contract
        if hasattr(builder.spec, 'outputs'):
            for out_name, out_spec in builder.spec.outputs.items():
                logical_name = out_spec.logical_name
                self._assert(
                    logical_name in builder.contract.expected_output_paths,
                    f"Output {logical_name} must have corresponding path in contract"
                )
    
    def test_environment_variable_handling(self) -> None:
        """Test that the builder handles environment variables correctly."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Get environment variables
        env_vars = builder._get_environment_variables()
        
        self._assert(
            isinstance(env_vars, dict),
            f"_get_environment_variables() must return a dictionary"
        )
        
        # Verify environment variables include required variables from contract
        if hasattr(builder, 'contract') and builder.contract is not None:
            if hasattr(builder.contract, 'required_env_vars'):
                for env_var in builder.contract.required_env_vars:
                    self._assert(
                        env_var in env_vars,
                        f"Environment variables must include required variable {env_var}"
                    )
            
            # Verify environment variables include optional variables with defaults
            if hasattr(builder.contract, 'optional_env_vars'):
                for env_var, default in builder.contract.optional_env_vars.items():
                    self._assert(
                        env_var in env_vars,
                        f"Environment variables must include optional variable {env_var}"
                    )
    
    def test_dependency_resolution(self) -> None:
        """Test that the builder resolves dependencies correctly."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Create mock dependencies
        dependencies = self._create_mock_dependencies()
        
        # Test extraction of inputs from dependencies
        try:
            extracted_inputs = builder.extract_inputs_from_dependencies(dependencies)
            
            self._assert(
                isinstance(extracted_inputs, dict),
                f"extract_inputs_from_dependencies() must return a dictionary"
            )
            
            # Verify extracted inputs include required dependencies
            try:
                required_deps = builder.get_required_dependencies()
                for dep_name in required_deps:
                    self._assert(
                        dep_name in extracted_inputs,
                        f"Extracted inputs must include required dependency {dep_name}"
                    )
            except Exception as e:
                self._log(f"Could not get required dependencies: {str(e)}")
        except Exception as e:
            self._assert(
                False,
                f"Dependency resolution failed: {str(e)}"
            )
    
    def test_step_creation(self) -> None:
        """Test that the builder creates a valid step."""
        # Create instance with mock config
        builder = self._create_builder_instance()
        
        # Create mock dependencies
        dependencies = self._create_mock_dependencies()
        
        # Test step creation
        try:
            step = builder.create_step(
                dependencies=dependencies,
                enable_caching=True
            )
            
            # Verify step has required attributes
            self._assert(
                step is not None,
                "Step must be created successfully"
            )
            
            # Verify step has spec attached
            self._assert(
                hasattr(step, '_spec'),
                "Step must have _spec attribute"
            )
            
            # Verify step has name
            self._assert(
                hasattr(step, 'name'),
                "Step must have name attribute"
            )
        except Exception as e:
            self._assert(
                False,
                f"Step creation failed: {str(e)}"
            )
    
    def test_error_handling(self) -> None:
        """Test that the builder handles errors appropriately."""
        # Test with invalid configuration
        try:
            # Create config without required attributes
            invalid_config = self._create_invalid_config()
            
            # Create builder with invalid config
            with self._assert_raises(ValueError):
                builder = self.builder_class(
                    config=invalid_config,
                    sagemaker_session=self.mock_session,
                    role=self.mock_role,
                    registry_manager=self.mock_registry_manager,
                    dependency_resolver=self.mock_dependency_resolver
                )
                
                # Should raise ValueError
                builder.validate_configuration()
        except Exception as e:
            self._assert(
                False,
                f"Error handling test failed: {str(e)}"
            )
    
    def _setup_test_environment(self) -> None:
        """Set up mock objects and test fixtures."""
        # Mock SageMaker session
        self.mock_session = MagicMock()
        self.mock_session.boto_session.client.return_value = MagicMock()
        
        # Mock IAM role
        self.mock_role = 'arn:aws:iam::123456789012:role/MockRole'
        
        # Create mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Configure dependency resolver for successful resolution
        self.mock_dependency_resolver.resolve_step_dependencies.return_value = {
            dep: MagicMock() for dep in self._get_expected_dependencies()
        }
        
        # Mock boto3 client
        self.mock_boto3_client = MagicMock()
        
        # Track assertions for reporting
        self.assertions = []
    
    def _create_builder_instance(self) -> StepBuilderBase:
        """Create a builder instance with mock configuration."""
        # Create mock configuration
        mock_config = self._create_mock_config()
        
        # Create builder instance
        builder = self.builder_class(
            config=mock_config,
            sagemaker_session=self.mock_session,
            role=self.mock_role,
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        return builder
    
    def _create_mock_config(self) -> SimpleNamespace:
        """Create a mock configuration for the builder."""
        # Basic config with required attributes
        mock_config = SimpleNamespace()
        mock_config.region = 'NA'
        mock_config.pipeline_name = 'test-pipeline'
        mock_config.pipeline_s3_loc = 's3://bucket/prefix'
        
        # Add hyperparameters if needed
        mock_hp = SimpleNamespace()
        mock_hp.model_dump = lambda: {'param': 'value'}
        mock_config.hyperparameters = mock_hp
        
        # Add common methods
        mock_config.get_image_uri = lambda: 'mock-image-uri'
        mock_config.get_script_path = lambda: 'mock_script.py'
        mock_config.get_script_contract = lambda: None
        
        # Add builder-specific attributes
        self._add_builder_specific_config(mock_config)
        
        return mock_config
    
    def _add_builder_specific_config(self, mock_config: SimpleNamespace) -> None:
        """Add builder-specific configuration attributes."""
        # Get builder class name
        builder_name = self.builder_class.__name__
        
        if "XGBoostTraining" in builder_name:
            self._add_xgboost_training_config(mock_config)
        elif "TabularPreprocessing" in builder_name:
            self._add_tabular_preprocessing_config(mock_config)
        elif "ModelEval" in builder_name:
            self._add_model_eval_config(mock_config)
        # Add more builder-specific configurations as needed
        else:
            self._add_generic_config(mock_config)
    
    def _add_xgboost_training_config(self, mock_config: SimpleNamespace) -> None:
        """Add XGBoost training-specific configuration attributes."""
        mock_config.training_instance_type = 'ml.m5.xlarge'
        mock_config.training_instance_count = 1
        mock_config.training_volume_size = 30
        mock_config.training_entry_point = 'train_xgb.py'
        mock_config.source_dir = 'src/pipeline_scripts'
        mock_config.framework_version = '1.7-1'
        mock_config.py_version = 'py3'
    
    def _add_tabular_preprocessing_config(self, mock_config: SimpleNamespace) -> None:
        """Add tabular preprocessing-specific configuration attributes."""
        mock_config.processing_instance_type = 'ml.m5.large'
        mock_config.processing_instance_count = 1
        mock_config.processing_volume_size = 30
        mock_config.processing_entry_point = 'tabular_preprocess.py'
        mock_config.source_dir = 'src/pipeline_scripts'
    
    def _add_model_eval_config(self, mock_config: SimpleNamespace) -> None:
        """Add model evaluation-specific configuration attributes."""
        mock_config.processing_instance_type = 'ml.m5.large'
        mock_config.processing_instance_count = 1
        mock_config.processing_volume_size = 30
        mock_config.processing_entry_point = 'model_evaluation_xgb.py'
        mock_config.source_dir = 'src/pipeline_scripts'
        mock_config.id_field = 'id'
        mock_config.label_field = 'label'
    
    def _add_generic_config(self, mock_config: SimpleNamespace) -> None:
        """Add generic configuration attributes for unknown builder types."""
        mock_config.instance_type = 'ml.m5.large'
        mock_config.instance_count =
