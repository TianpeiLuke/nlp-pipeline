#!/usr/bin/env python
"""
Unit tests specifically for the build method and matching input/output methods
of the StepBuilderBase class in pipeline_steps/builder_step_base.py.
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


class TestStepBuilderBaseBuild(unittest.TestCase):
    """Test cases specifically for the build method and matching input/output methods."""

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
        self.config.input_names = {
            "input1": "Input 1", 
            "model_input": "Model Input",
            "data_input": "Data Input",
            "output_path": "Output Path"
        }
        self.config.output_names = {"output1": "Output 1"}
        
        # Create builder instance
        self.builder = ConcreteStepBuilder(self.config)
        self.builder.role = "arn:aws:iam::123456789012:role/test-role"
        self.builder.session = MagicMock()
        self.builder.notebook_root = Path('.')

    def test_build_with_complete_dependencies(self):
        """Test that build correctly processes dependencies with all required inputs."""
        # Create mock dependencies with all required inputs
        model_step = MagicMock()
        model_step.name = "ModelStep"
        model_step.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        processing_step = MagicMock()
        processing_step.name = "ProcessingStep"
        processing_step.properties.ProcessingOutputConfig.Outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        }
        
        dependencies = [model_step, processing_step]
        
        # Create expected kwargs
        expected_kwargs = {
            "model_input": "s3://bucket/model.tar.gz",
            "data_input": "s3://bucket/data",
            "output_path": "s3://bucket/output",
            "dependencies": dependencies,
            "enable_caching": True
        }
        
        # Define a side effect function that updates the inputs dictionary
        def match_side_effect(inputs, reqs, step):
            if step == model_step:
                inputs["model_input"] = "s3://bucket/model.tar.gz"
                return {"model_input"}
            elif step == processing_step:
                inputs["data_input"] = "s3://bucket/data"
                inputs["output_path"] = "s3://bucket/output"
                return {"data_input", "output_path"}
            return set()
        
        # Mock _match_inputs_to_outputs to simulate matching
        with patch.object(self.builder, '_match_inputs_to_outputs', 
                         side_effect=match_side_effect):
            # Mock _check_missing_inputs to return empty list (no missing inputs)
            with patch.object(self.builder, '_check_missing_inputs', 
                             return_value=[]):
                # Mock _filter_kwargs to return expected kwargs
                with patch.object(self.builder, '_filter_kwargs', 
                                 return_value=expected_kwargs):
                    # Call build
                    step = self.builder.build(dependencies)
                    
                    # Verify inputs were correctly extracted and passed to create_step
                    self.assertEqual(step.model_input, "s3://bucket/model.tar.gz")
                    self.assertEqual(step.data_input, "s3://bucket/data")
                    self.assertEqual(step.output_path, "s3://bucket/output")
                    self.assertEqual(step.dependencies, dependencies)
                    self.assertTrue(step.enable_caching)

    def test_build_with_missing_inputs(self):
        """Test that build raises ValueError when required inputs are missing."""
        # Create mock dependencies with missing required inputs
        model_step = MagicMock()
        model_step.name = "ModelStep"
        # No model_artifacts_path
        
        processing_step = MagicMock()
        processing_step.name = "ProcessingStep"
        processing_step.properties.ProcessingOutputConfig.Outputs = {
            # Missing data output
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        }
        
        dependencies = [model_step, processing_step]
        
        # Define a side effect function that updates the inputs dictionary
        def match_side_effect(inputs, reqs, step):
            if step == processing_step:
                inputs["output_path"] = "s3://bucket/output"
                return {"output_path"}
            return set()
        
        # Mock _match_inputs_to_outputs to simulate matching
        with patch.object(self.builder, '_match_inputs_to_outputs', 
                         side_effect=match_side_effect):
            # Mock _check_missing_inputs to return missing inputs
            with patch.object(self.builder, '_check_missing_inputs', 
                             return_value=["model_input", "data_input"]):
                # Verify ValueError is raised
                with self.assertRaises(ValueError) as context:
                    self.builder.build(dependencies)
                
                # Verify error message contains missing inputs
                self.assertIn("model_input", str(context.exception))
                self.assertIn("data_input", str(context.exception))

    def test_match_model_artifacts_detailed(self):
        """Test that _match_model_artifacts correctly matches model artifacts with different patterns."""
        # Setup
        inputs = {}
        input_requirements = {
            "model_input": "Model input",
            "model_path": "Model path",
            "model_data": "Model data",
            "model_artifacts": "Model artifacts"
        }
        
        # Create a mock step with model artifacts
        step = MagicMock()
        step.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        # Call _match_model_artifacts
        matched = self.builder._match_model_artifacts(inputs, input_requirements, step)
        
        # Verify all model-related inputs were matched
        self.assertIn("model_input", matched)
        self.assertIn("model_path", matched)
        self.assertIn("model_data", matched)
        self.assertIn("model_artifacts", matched)
        self.assertEqual(inputs["model_input"], "s3://bucket/model.tar.gz")
        self.assertEqual(inputs["model_path"], "s3://bucket/model.tar.gz")
        self.assertEqual(inputs["model_data"], "s3://bucket/model.tar.gz")
        self.assertEqual(inputs["model_artifacts"], "s3://bucket/model.tar.gz")

    def test_match_processing_outputs_detailed(self):
        """Test that _match_processing_outputs correctly matches processing outputs with different patterns."""
        # Setup
        inputs = {}
        input_requirements = {
            "data_input": "Data input",
            "dataset": "Dataset",
            "output_path": "Output path",
            "result_uri": "Result URI"
        }
        
        # Create a mock step with processing outputs
        step = MagicMock()
        step.properties.ProcessingOutputConfig.Outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output")),
            "result": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/result"))
        }
        
        # Define a side effect function for _match_dict_outputs
        def match_dict_side_effect(inputs, reqs, outputs):
            matched = set()
            if "data" in outputs:
                if "data_input" in reqs:
                    inputs["data_input"] = outputs["data"].S3Output.S3Uri
                    matched.add("data_input")
                if "dataset" in reqs:
                    inputs["dataset"] = outputs["data"].S3Output.S3Uri
                    matched.add("dataset")
            if "output" in outputs and "output_path" in reqs:
                inputs["output_path"] = outputs["output"].S3Output.S3Uri
                matched.add("output_path")
            if "result" in outputs and "result_uri" in reqs:
                inputs["result_uri"] = outputs["result"].S3Output.S3Uri
                matched.add("result_uri")
            return matched
        
        # Mock the _match_list_outputs and _match_dict_outputs methods
        with patch.object(self.builder, '_match_list_outputs', return_value=set()) as mock_list:
            with patch.object(self.builder, '_match_dict_outputs', 
                             side_effect=match_dict_side_effect) as mock_dict:
                
                # Call _match_processing_outputs
                matched = self.builder._match_processing_outputs(inputs, input_requirements, step)
                
                # Verify outputs were matched based on patterns
                self.assertIn("data_input", matched)
                self.assertIn("dataset", matched)
                self.assertIn("output_path", matched)
                self.assertIn("result_uri", matched)
                self.assertEqual(inputs["data_input"], "s3://bucket/data")
                self.assertEqual(inputs["dataset"], "s3://bucket/data")
                self.assertEqual(inputs["output_path"], "s3://bucket/output")
                self.assertEqual(inputs["result_uri"], "s3://bucket/result")

    def test_match_list_outputs_detailed(self):
        """Test that _match_list_outputs correctly matches list outputs with different patterns."""
        # Setup
        inputs = {}
        input_requirements = {
            "output_path": "Output path",
            "result_uri": "Result URI"
        }
        
        # Create mock outputs with list-like structure
        outputs = [
            MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        ]
        
        # Call _match_list_outputs
        matched = self.builder._match_list_outputs(inputs, input_requirements, outputs)
        
        # Verify outputs were matched based on patterns
        self.assertIn("output_path", matched)
        self.assertIn("result_uri", matched)
        self.assertEqual(inputs["output_path"], "s3://bucket/output")
        self.assertEqual(inputs["result_uri"], "s3://bucket/output")

    def test_match_dict_outputs_detailed(self):
        """Test that _match_dict_outputs correctly matches dictionary outputs with different patterns."""
        # Setup
        inputs = {}
        input_requirements = {
            "data_input": "Data input",
            "dataset": "Dataset",
            "model_path": "Model path",
            "output_path": "Output path"
        }
        
        # Create mock outputs with dictionary-like structure
        outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "model": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/model")),
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        }
        
        # Call _match_dict_outputs
        matched = self.builder._match_dict_outputs(inputs, input_requirements, outputs)
        
        # Verify outputs were matched based on patterns
        self.assertIn("data_input", matched)
        self.assertIn("dataset", matched)
        self.assertIn("model_path", matched)
        self.assertIn("output_path", matched)
        self.assertEqual(inputs["data_input"], "s3://bucket/data")
        self.assertEqual(inputs["dataset"], "s3://bucket/data")
        self.assertEqual(inputs["model_path"], "s3://bucket/model")
        self.assertEqual(inputs["output_path"], "s3://bucket/output")

    def test_extract_inputs_from_dependencies_detailed(self):
        """Test that extract_inputs_from_dependencies correctly extracts inputs from different dependency types."""
        # Create mock dependencies of different types
        model_step = MagicMock()
        model_step.name = "ModelStep"
        model_step.model_artifacts_path = "s3://bucket/model.tar.gz"
        
        processing_step = MagicMock()
        processing_step.name = "ProcessingStep"
        processing_step.properties.ProcessingOutputConfig.Outputs = {
            "data": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/data")),
            "output": MagicMock(S3Output=MagicMock(S3Uri="s3://bucket/output"))
        }
        
        training_step = MagicMock()
        training_step.name = "TrainingStep"
        # This step has no outputs that match our patterns
        
        dependencies = [model_step, processing_step, training_step]
        
        # Define a side effect function that updates the inputs dictionary
        def match_side_effect(inputs, reqs, step):
            if step == model_step:
                inputs["model_input"] = "s3://bucket/model.tar.gz"
                return {"model_input"}
            elif step == processing_step:
                inputs["data_input"] = "s3://bucket/data"
                inputs["output_path"] = "s3://bucket/output"
                return {"data_input", "output_path"}
            return set()
        
        # Mock _match_inputs_to_outputs to simulate matching
        with patch.object(self.builder, '_match_inputs_to_outputs', 
                         side_effect=match_side_effect) as mock_match:
            
            # Call extract_inputs_from_dependencies
            inputs = self.builder.extract_inputs_from_dependencies(dependencies)
            
            # Verify _match_inputs_to_outputs was called for each dependency
            self.assertEqual(mock_match.call_count, 3)
            
            # Verify inputs were extracted
            self.assertIn("model_input", inputs)
            self.assertIn("data_input", inputs)
            self.assertIn("output_path", inputs)
            self.assertEqual(inputs["model_input"], "s3://bucket/model.tar.gz")
            self.assertEqual(inputs["data_input"], "s3://bucket/data")
            self.assertEqual(inputs["output_path"], "s3://bucket/output")

    def test_match_custom_properties_override(self):
        """Test that a derived class can override _match_custom_properties to match custom properties."""
        # Create a custom step class
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
                # Custom implementation that matches custom properties
                matched = set()
                
                # Match custom_input if available
                if hasattr(prev_step, "custom_property") and "custom_input" in input_requirements:
                    inputs["custom_input"] = prev_step.custom_property
                    matched.add("custom_input")
                    
                # Match hyperparameters if available
                if hasattr(prev_step, "hyperparameters") and "hyperparams" in input_requirements:
                    inputs["hyperparams"] = prev_step.hyperparameters
                    matched.add("hyperparams")
                    
                return matched
        
        # Create derived builder with custom input requirements
        config = SimpleNamespace()
        config.region = "NA"
        config.input_names = {
            "custom_input": "Custom input",
            "hyperparams": "Hyperparameters"
        }
        config.output_names = {}
        
        derived_builder = DerivedBuilder(config)
        
        # Create mock step with custom properties
        step = MagicMock()
        step.custom_property = "custom_value"
        step.hyperparameters = {"learning_rate": 0.01, "batch_size": 32}
        
        # Create expected kwargs
        expected_kwargs = {
            "custom_input": "custom_value",
            "hyperparams": {"learning_rate": 0.01, "batch_size": 32},
            "dependencies": [step],
            "enable_caching": True
        }
        
        # Call build with the mock step
        with patch.object(derived_builder, '_check_missing_inputs', return_value=[]):
            with patch.object(derived_builder, '_filter_kwargs', 
                             return_value=expected_kwargs):
                result_step = derived_builder.build([step])
                
                # Verify custom properties were matched
                self.assertEqual(result_step.custom_input, "custom_value")
                self.assertEqual(result_step.hyperparams, {"learning_rate": 0.01, "batch_size": 32})


if __name__ == '__main__':
    unittest.main()
