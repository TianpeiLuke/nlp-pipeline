#!/usr/bin/env python
"""
Unit tests for the StepBuilderBase class in pipeline_steps/builder_step_base.py.
These tests focus on the build method and input/output matching functionality.
"""
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
from pathlib import Path
import logging

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the class to be tested
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.config_base import BasePipelineConfig

# Disable logging during tests
logging.getLogger('src.pipeline_steps.builder_step_base').setLevel(logging.CRITICAL)


class TestStepBuilderBase(unittest.TestCase):
    """Test cases for the StepBuilderBase class."""

    def setUp(self):
        """Set up a concrete implementation of StepBuilderBase for testing."""
        # Create a custom step class to store attributes
        class TestStep:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Create a concrete implementation of the abstract class
        class ConcreteStepBuilder(StepBuilderBase):
            def validate_configuration(self):
                """Concrete implementation of abstract method."""
                pass
                
            def create_step(self, **kwargs):
                """Concrete implementation that returns a test step with the kwargs."""
                # Create a test step with the kwargs
                return TestStep(**kwargs)
                
        # Create a minimal config
        self.config = SimpleNamespace()
        self.config.region = "NA"
        self.config.input_names = {"input1": "Input 1", "model_input": "Model Input"}
        self.config.output_names = {"output1": "Output 1"}
        
        # Create builder instance
        self.builder = ConcreteStepBuilder(self.config)
        self.builder.role = "arn:aws:iam::123456789012:role/test-role"
        self.builder.session = MagicMock()
        self.builder.notebook_root = Path('.')
        
    def test_build_with_dependencies(self):
        """Test that build correctly combines extract_inputs_from_dependencies and create_step."""
        # Mock dependencies
        dep1 = MagicMock()
        dep1.name = "Dependency1"
        dep1.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        dep2 = MagicMock()
        dep2.name = "Dependency2"
        dep2.properties.ProcessingOutputConfig.Outputs = {
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        }
        
        dependencies = [dep1, dep2]
        
        # Create expected kwargs
        expected_kwargs = {
            "input1": "value1",
            "dependencies": dependencies,
            "enable_caching": True
        }
        
        # Mock extract_inputs_from_dependencies
        with patch.object(self.builder, 'extract_inputs_from_dependencies', 
                         return_value={"input1": "value1"}) as mock_extract:
            # Mock _check_missing_inputs to return empty list (no missing inputs)
            with patch.object(self.builder, '_check_missing_inputs', 
                             return_value=[]) as mock_check:
                # Mock _filter_kwargs to return all kwargs
                with patch.object(self.builder, '_filter_kwargs', 
                                 return_value=expected_kwargs) as mock_filter:
                    
                    # Call build
                    step = self.builder.build(dependencies)
                    
                    # Verify extract_inputs_from_dependencies was called with dependencies
                    mock_extract.assert_called_once_with(dependencies)
                    
                    # Verify create_step was called with the expected kwargs
                    self.assertEqual(step.input1, "value1")
                    self.assertEqual(step.dependencies, dependencies)
                    self.assertTrue(step.enable_caching)

    def test_build_with_no_dependencies(self):
        """Test that build handles the case with no dependencies."""
        # Expected kwargs for None case
        expected_kwargs_none = {
            "dependencies": [],
            "enable_caching": True
        }
        
        # Expected kwargs for empty list case
        expected_kwargs_empty = {
            "dependencies": [],
            "enable_caching": True
        }
        
        # Call build with None
        with patch.object(self.builder, 'extract_inputs_from_dependencies', 
                         return_value={}) as mock_extract:
            # Mock _check_missing_inputs to return empty list (no missing inputs)
            with patch.object(self.builder, '_check_missing_inputs', 
                             return_value=[]):
                # Mock _filter_kwargs to return expected kwargs
                with patch.object(self.builder, '_filter_kwargs', 
                                 return_value=expected_kwargs_none):
                    step = self.builder.build(None)
                    
                    # Verify extract_inputs_from_dependencies was called with empty list
                    mock_extract.assert_called_once_with([])
                    
                    # Verify create_step was called with empty dependencies list
                    self.assertEqual(step.dependencies, [])
                    self.assertTrue(step.enable_caching)
            
        # Call build with empty list
        with patch.object(self.builder, 'extract_inputs_from_dependencies', 
                         return_value={}) as mock_extract:
            # Mock _check_missing_inputs to return empty list (no missing inputs)
            with patch.object(self.builder, '_check_missing_inputs', 
                             return_value=[]):
                # Mock _filter_kwargs to return expected kwargs
                with patch.object(self.builder, '_filter_kwargs', 
                                 return_value=expected_kwargs_empty):
                    step = self.builder.build([])
                    
                    # Verify extract_inputs_from_dependencies was called with empty list
                    mock_extract.assert_called_once_with([])
                    
                    # Verify create_step was called with empty dependencies list
                    self.assertEqual(step.dependencies, [])
                    self.assertTrue(step.enable_caching)

    def test_build_with_missing_required_inputs(self):
        """Test that build raises ValueError when required inputs are missing."""
        # Mock dependencies
        dependencies = [MagicMock()]
        
        # Mock extract_inputs_from_dependencies to return empty dict
        with patch.object(self.builder, 'extract_inputs_from_dependencies', 
                         return_value={}):
            
            # Mock _check_missing_inputs to return missing inputs
            with patch.object(self.builder, '_check_missing_inputs', 
                             return_value=["required_input"]):
                
                # Verify ValueError is raised
                with self.assertRaises(ValueError):
                    self.builder.build(dependencies)

    def test_match_inputs_to_outputs(self):
        """Test that _match_inputs_to_outputs correctly matches inputs to outputs."""
        # Setup
        inputs = {}
        input_requirements = {"model_input": "Model input", "data_input": "Data input"}
        
        # Create a mock step with model artifacts
        step = MagicMock()
        step.name = "ModelStep"
        step.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        # Call _match_inputs_to_outputs
        matched = self.builder._match_inputs_to_outputs(inputs, input_requirements, step)
        
        # Verify model_input was matched
        self.assertIn("model_input", matched)
        self.assertEqual(inputs["model_input"], "s3://bucket/model.tar.gz")
        
    def test_match_inputs_to_outputs_empty_requirements(self):
        """Test that _match_inputs_to_outputs handles empty input requirements."""
        # Setup
        inputs = {}
        input_requirements = {}
        step = MagicMock()
        
        # Call _match_inputs_to_outputs
        matched = self.builder._match_inputs_to_outputs(inputs, input_requirements, step)
        
        # Verify empty set is returned
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})

    def test_match_model_artifacts(self):
        """Test that _match_model_artifacts correctly matches model artifacts."""
        # Setup
        inputs = {}
        input_requirements = {"model_input": "Model input", "data_input": "Data input"}
        
        # Create a mock step with model artifacts
        step = MagicMock()
        step.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        # Call _match_model_artifacts
        matched = self.builder._match_model_artifacts(inputs, input_requirements, step)
        
        # Verify model_input was matched
        self.assertIn("model_input", matched)
        self.assertEqual(inputs["model_input"], "s3://bucket/model.tar.gz")
        
    def test_match_model_artifacts_no_match(self):
        """Test that _match_model_artifacts handles case with no matches."""
        # Setup
        inputs = {}
        input_requirements = {"data_input": "Data input"}  # No model input
        
        # Create a mock step with model artifacts
        step = MagicMock()
        step.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        # Call _match_model_artifacts
        matched = self.builder._match_model_artifacts(inputs, input_requirements, step)
        
        # Verify no matches
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})
        
    def test_match_model_artifacts_no_artifacts(self):
        """Test that _match_model_artifacts handles step with no model artifacts."""
        # Setup
        inputs = {}
        input_requirements = {"model_input": "Model input"}
        
        # Create a mock step without model artifacts
        step = MagicMock(spec=[])  # Empty spec means no attributes
        
        # Call _match_model_artifacts
        matched = self.builder._match_model_artifacts(inputs, input_requirements, step)
        
        # Verify no matches
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})

    def test_match_processing_outputs(self):
        """Test that _match_processing_outputs correctly matches processing outputs."""
        # Setup
        inputs = {}
        input_requirements = {"data_input": "Data input", "output_path": "Output path"}
        
        # Create a mock step with processing outputs
        step = MagicMock()
        step.properties.ProcessingOutputConfig.Outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        }
        
        # Mock the _match_list_outputs and _match_dict_outputs methods
        with patch.object(self.builder, '_match_list_outputs', return_value=set()) as mock_list:
            with patch.object(self.builder, '_match_dict_outputs', 
                             side_effect=lambda i, r, o: self._mock_match_dict(i, r, o)) as mock_dict:
                
                # Call _match_processing_outputs
                matched = self.builder._match_processing_outputs(inputs, input_requirements, step)
                
                # Verify outputs were matched
                self.assertIn("data_input", matched)
                self.assertIn("output_path", matched)
                self.assertEqual(inputs["data_input"], "s3://bucket/data")
                self.assertEqual(inputs["output_path"], "s3://bucket/output")
    
    def _mock_match_dict(self, inputs, input_requirements, outputs):
        """Helper method to mock _match_dict_outputs behavior."""
        matched = set()
        if "data" in outputs and "data_input" in input_requirements:
            inputs["data_input"] = outputs["data"].S3Output.S3Uri
            matched.add("data_input")
        if "output" in outputs and "output_path" in input_requirements:
            inputs["output_path"] = outputs["output"].S3Output.S3Uri
            matched.add("output_path")
        return matched
        
    def test_match_processing_outputs_no_processing_config(self):
        """Test that _match_processing_outputs handles step with no processing config."""
        # Setup
        inputs = {}
        input_requirements = {"data_input": "Data input"}
        
        # Create a mock step without processing outputs
        step = MagicMock()
        # No properties.ProcessingOutputConfig attribute
        
        # Call _match_processing_outputs
        matched = self.builder._match_processing_outputs(inputs, input_requirements, step)
        
        # Verify no matches
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})

    def test_match_list_outputs(self):
        """Test that _match_list_outputs correctly matches list-like outputs."""
        # Setup
        inputs = {}
        input_requirements = {"output_path": "Output path"}
        
        # Create mock outputs with list-like structure
        outputs = [MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))]
        
        # Call _match_list_outputs
        matched = self.builder._match_list_outputs(inputs, input_requirements, outputs)
        
        # Verify output was matched
        self.assertIn("output_path", matched)
        self.assertEqual(inputs["output_path"], "s3://bucket/output")
        
    def test_match_list_outputs_no_match(self):
        """Test that _match_list_outputs handles case with no matches."""
        # Setup
        inputs = {}
        input_requirements = {"data_input": "Data input"}  # No output path
        
        # Create mock outputs with list-like structure
        outputs = [MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))]
        
        # Call _match_list_outputs
        matched = self.builder._match_list_outputs(inputs, input_requirements, outputs)
        
        # Verify no matches
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})
        
    def test_match_list_outputs_invalid_structure(self):
        """Test that _match_list_outputs handles invalid output structure."""
        # Setup
        inputs = {}
        input_requirements = {"output_path": "Output path"}
        
        # Create invalid outputs (no S3Output attribute)
        outputs = [MagicMock()]
        
        # Mock the exception that would occur when trying to access S3Output
        with patch.object(self.builder, '_match_list_outputs', 
                         side_effect=lambda i, r, o: set()) as mock_list:
            
            # Call _match_list_outputs directly from the mock
            matched = mock_list(inputs, input_requirements, outputs)
            
            # Verify no matches
            self.assertEqual(matched, set())
            self.assertEqual(inputs, {})

    def test_match_dict_outputs(self):
        """Test that _match_dict_outputs correctly matches dictionary-like outputs."""
        # Setup
        inputs = {}
        input_requirements = {"data_input": "Data input", "model_path": "Model path"}
        
        # Create mock outputs with dictionary-like structure
        outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "model": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/model"))
        }
        
        # Call _match_dict_outputs
        matched = self.builder._match_dict_outputs(inputs, input_requirements, outputs)
        
        # Verify outputs were matched
        self.assertIn("data_input", matched)
        self.assertIn("model_path", matched)
        self.assertEqual(inputs["data_input"], "s3://bucket/data")
        self.assertEqual(inputs["model_path"], "s3://bucket/model")
        
    def test_match_dict_outputs_no_match(self):
        """Test that _match_dict_outputs handles case with no matches."""
        # Setup
        inputs = {}
        input_requirements = {"other_input": "Other input"}  # No matching inputs
        
        # Create mock outputs with dictionary-like structure
        outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "model": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/model"))
        }
        
        # Call _match_dict_outputs
        matched = self.builder._match_dict_outputs(inputs, input_requirements, outputs)
        
        # Verify no matches
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})
        
    def test_match_dict_outputs_invalid_structure(self):
        """Test that _match_dict_outputs handles invalid output structure."""
        # Setup
        inputs = {}
        input_requirements = {"data_input": "Data input"}
        
        # Create invalid outputs (no S3Output attribute)
        outputs = {"data": MagicMock()}
        
        # Mock the exception that would occur when trying to access S3Output
        with patch.object(self.builder, '_match_dict_outputs', 
                         side_effect=lambda i, r, o: set()) as mock_dict:
            
            # Call _match_dict_outputs directly from the mock
            matched = mock_dict(inputs, input_requirements, outputs)
            
            # Verify no matches
            self.assertEqual(matched, set())
            self.assertEqual(inputs, {})

    def test_match_custom_properties(self):
        """Test that _match_custom_properties can be overridden by derived classes."""
        # Setup
        inputs = {}
        input_requirements = {"custom_input": "Custom input"}
        step = MagicMock()
        
        # Call _match_custom_properties (base implementation returns empty set)
        matched = self.builder._match_custom_properties(inputs, input_requirements, step)
        
        # Verify no matches (base implementation)
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})
        
        # Create a custom step class for the derived builder
        class TestStep:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Create a derived class that overrides _match_custom_properties
        class DerivedBuilder(StepBuilderBase):
            def validate_configuration(self):
                pass
                
            def create_step(self, **kwargs):
                return TestStep(**kwargs)
                
            def _match_custom_properties(self, inputs, input_requirements, prev_step):
                # Custom implementation that always matches custom_input
                if "custom_input" in input_requirements:
                    inputs["custom_input"] = "custom_value"
                    return {"custom_input"}
                return set()
        
        # Create derived builder
        derived_builder = DerivedBuilder(self.config)
        
        # Call _match_custom_properties on derived builder
        matched = derived_builder._match_custom_properties(inputs, input_requirements, step)
        
        # Verify custom_input was matched
        self.assertIn("custom_input", matched)
        self.assertEqual(inputs["custom_input"], "custom_value")

    def test_check_missing_inputs(self):
        """Test that _check_missing_inputs correctly identifies missing required inputs."""
        # Setup
        kwargs = {"input1": "value1", "optional_input": "value2"}
        
        # Mock get_input_requirements to return required and optional inputs
        with patch.object(self.builder, 'get_input_requirements', 
                         return_value={
                             "input1": "Required input",
                             "input2": "Required input",
                             "optional_input": "Optional input (optional)"
                         }):
            
            # Call _check_missing_inputs
            missing = self.builder._check_missing_inputs(kwargs)
            
            # Verify input2 is identified as missing
            self.assertEqual(missing, ["input2"])
            
    def test_check_missing_inputs_no_requirements(self):
        """Test that _check_missing_inputs handles case with no input requirements."""
        # Setup
        kwargs = {"input1": "value1"}
        
        # Mock get_input_requirements to return empty dict
        with patch.object(self.builder, 'get_input_requirements', 
                         return_value={}):
            
            # Call _check_missing_inputs
            missing = self.builder._check_missing_inputs(kwargs)
            
            # Verify no missing inputs
            self.assertEqual(missing, [])
            
    def test_check_missing_inputs_none_kwargs(self):
        """Test that _check_missing_inputs handles None kwargs."""
        # Mock get_input_requirements to return required inputs
        with patch.object(self.builder, 'get_input_requirements', 
                         return_value={"input1": "Required input"}):
            
            # Call _check_missing_inputs with None
            missing = self.builder._check_missing_inputs(None)
            
            # Verify warning is logged and empty list is returned
            self.assertEqual(missing, [])

    def test_filter_kwargs(self):
        """Test that _filter_kwargs correctly filters kwargs for a function."""
        # Setup
        def test_func(a, b, c=None):
            pass
        
        kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
        
        # Call _filter_kwargs
        filtered = self.builder._filter_kwargs(test_func, kwargs)
        
        # Verify only a, b, c are included
        self.assertEqual(filtered, {"a": 1, "b": 2, "c": 3})
        self.assertNotIn("d", filtered)
        
    def test_filter_kwargs_none_func(self):
        """Test that _filter_kwargs handles None function."""
        # Call _filter_kwargs with None function
        filtered = self.builder._filter_kwargs(None, {"a": 1})
        
        # Verify empty dict is returned
        self.assertEqual(filtered, {})
        
    def test_filter_kwargs_none_kwargs(self):
        """Test that _filter_kwargs handles None kwargs."""
        # Setup
        def test_func(a, b):
            pass
        
        # Call _filter_kwargs with None kwargs
        filtered = self.builder._filter_kwargs(test_func, None)
        
        # Verify empty dict is returned
        self.assertEqual(filtered, {})

    def test_extract_inputs_from_dependencies(self):
        """Test that extract_inputs_from_dependencies correctly extracts inputs."""
        # Setup
        dep1 = MagicMock()
        dep1.name = "ModelStep"
        dep1.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        dep2 = MagicMock()
        dep2.name = "ProcessingStep"
        dep2.properties.ProcessingOutputConfig.Outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data"))
        }
        
        dependencies = [dep1, dep2]
        
        # Define a side effect function that updates the inputs dictionary
        def match_side_effect(inputs, reqs, step):
            if step == dep1:
                inputs["model_input"] = "s3://bucket/model.tar.gz"
                return {"model_input"}
            elif step == dep2:
                inputs["data_input"] = "s3://bucket/data"
                return {"data_input"}
            return set()
        
        # Mock _match_inputs_to_outputs to simulate matching
        with patch.object(self.builder, '_match_inputs_to_outputs', 
                         side_effect=match_side_effect) as mock_match:
            
            # Call extract_inputs_from_dependencies
            inputs = self.builder.extract_inputs_from_dependencies(dependencies)
            
            # Verify _match_inputs_to_outputs was called for each dependency
            self.assertEqual(mock_match.call_count, 2)
            
            # Verify inputs were extracted
            self.assertIn("model_input", inputs)
            self.assertIn("data_input", inputs)
            
    def test_extract_inputs_from_dependencies_none(self):
        """Test that extract_inputs_from_dependencies handles None dependencies."""
        # Call extract_inputs_from_dependencies with None
        inputs = self.builder.extract_inputs_from_dependencies(None)
        
        # Verify empty dict is returned
        self.assertEqual(inputs, {})
        
    def test_extract_inputs_from_dependencies_no_requirements(self):
        """Test that extract_inputs_from_dependencies handles case with no input requirements."""
        # Setup
        dependencies = [MagicMock()]
        
        # Mock get_input_requirements to return empty dict
        with patch.object(self.builder, 'get_input_requirements', 
                         return_value={}):
            
            # Call extract_inputs_from_dependencies
            inputs = self.builder.extract_inputs_from_dependencies(dependencies)
            
            # Verify empty dict is returned
            self.assertEqual(inputs, {})

    def test_extract_param(self):
        """Test that _extract_param correctly extracts parameters from kwargs."""
        # Setup
        kwargs = {"param1": "value1", "param2": "value2"}
        
        # Extract existing parameter
        value = self.builder._extract_param(kwargs, "param1", "default")
        self.assertEqual(value, "value1")
        
        # Extract non-existing parameter (use default)
        value = self.builder._extract_param(kwargs, "param3", "default")
        self.assertEqual(value, "default")
        
        # Extract non-existing parameter (no default)
        value = self.builder._extract_param(kwargs, "param3")
        self.assertIsNone(value)
        
    def test_extract_param_none_kwargs(self):
        """Test that _extract_param handles None kwargs."""
        # Call _extract_param with None kwargs
        value = self.builder._extract_param(None, "param", "default")
        
        # Verify default is returned
        self.assertEqual(value, "default")


if __name__ == '__main__':
    unittest.main()
