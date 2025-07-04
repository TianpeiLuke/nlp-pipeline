#!/usr/bin/env python3
"""
Unit tests for the PyTorch Training Step Specification.

This module provides tests for the PyTorch training step specification,
including validation of dependencies, outputs, and node type.
"""

import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from src.pipeline_deps.base_specifications import (
    DependencyType, NodeType, SpecificationRegistry
)
from src.pipeline_step_specs.pytorch_training_spec import PYTORCH_TRAINING_SPEC


class TestPyTorchTrainingSpec(unittest.TestCase):
    """Test cases for the PyTorch Training Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("pytorch_training_step", PYTORCH_TRAINING_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("pytorch_training_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "PyTorchTraining")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PYTORCH_TRAINING_SPEC.node_type, NodeType.INTERNAL)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = PYTORCH_TRAINING_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that input_path is required
        self.assertIn("input_path", required_dep_names)
        
        # Check dependency type
        input_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("input_path")
        self.assertEqual(input_path_dep.dependency_type, DependencyType.TRAINING_DATA)
        self.assertTrue(input_path_dep.required)
    
    def test_optional_dependencies(self):
        """Test that the specification has the optional dependencies."""
        optional_deps = PYTORCH_TRAINING_SPEC.list_optional_dependencies()
        optional_dep_names = [dep.logical_name for dep in optional_deps]
        
        # Check that checkpoint_path is optional
        self.assertIn("checkpoint_path", optional_dep_names)
        
        # Check dependency type
        checkpoint_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("checkpoint_path")
        self.assertEqual(checkpoint_path_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
        self.assertFalse(checkpoint_path_dep.required)
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check primary output names
        self.assertIsNotNone(PYTORCH_TRAINING_SPEC.get_output("model_output"))
        self.assertIsNotNone(PYTORCH_TRAINING_SPEC.get_output("training_job_name"))
        self.assertIsNotNone(PYTORCH_TRAINING_SPEC.get_output("metrics_output"))
        
        # Check output types
        model_output = PYTORCH_TRAINING_SPEC.get_output("model_output")
        self.assertEqual(model_output.output_type, DependencyType.MODEL_ARTIFACTS)
        
        training_job_name = PYTORCH_TRAINING_SPEC.get_output("training_job_name")
        self.assertEqual(training_job_name.output_type, DependencyType.CUSTOM_PROPERTY)
        
        metrics_output = PYTORCH_TRAINING_SPEC.get_output("metrics_output")
        self.assertEqual(metrics_output.output_type, DependencyType.CUSTOM_PROPERTY)
        
        # Check property paths
        self.assertEqual(
            model_output.property_path,
            "properties.ModelArtifacts.S3ModelArtifacts"
        )
        self.assertEqual(
            training_job_name.property_path,
            "properties.TrainingJobName"
        )
        self.assertEqual(
            metrics_output.property_path,
            "properties.TrainingMetrics"
        )
    
    def test_output_aliases(self):
        """Test that the output aliases point to the same property paths."""
        # Model output aliases
        model_output = PYTORCH_TRAINING_SPEC.get_output("model_output")
        model_output_aliases = [
            "ModelArtifacts", "model_data", 
            "output_path", "model_input"
        ]
        
        for alias in model_output_aliases:
            alias_output = PYTORCH_TRAINING_SPEC.get_output(alias)
            self.assertEqual(
                model_output.property_path, 
                alias_output.property_path,
                f"Alias {alias} should have the same property path as model_output"
            )
        
        # Training job name alias
        training_job_name = PYTORCH_TRAINING_SPEC.get_output("training_job_name")
        job_name_alias = PYTORCH_TRAINING_SPEC.get_output("TrainingJobName")
        self.assertEqual(training_job_name.property_path, job_name_alias.property_path)
        
        # Metrics output alias
        metrics_output = PYTORCH_TRAINING_SPEC.get_output("metrics_output")
        metrics_alias = PYTORCH_TRAINING_SPEC.get_output("TrainingMetrics")
        self.assertEqual(metrics_output.property_path, metrics_alias.property_path)
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        input_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("input_path")
        checkpoint_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("checkpoint_path")
        
        # Check input_path compatible sources
        self.assertIn("TabularPreprocessing", input_path_dep.compatible_sources)
        self.assertIn("ProcessingStep", input_path_dep.compatible_sources)
        self.assertIn("DataLoad", input_path_dep.compatible_sources)
        
        # Check checkpoint_path compatible sources
        self.assertIn("PyTorchTraining", checkpoint_path_dep.compatible_sources)
        self.assertIn("ProcessingStep", checkpoint_path_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        input_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("input_path")
        checkpoint_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("checkpoint_path")
        
        # Check input_path semantic keywords
        self.assertIn("data", input_path_dep.semantic_keywords)
        self.assertIn("input", input_path_dep.semantic_keywords)
        self.assertIn("training", input_path_dep.semantic_keywords)
        self.assertIn("dataset", input_path_dep.semantic_keywords)
        self.assertIn("processed", input_path_dep.semantic_keywords)
        self.assertIn("train", input_path_dep.semantic_keywords)
        self.assertIn("pytorch", input_path_dep.semantic_keywords)
        
        # Check checkpoint_path semantic keywords
        self.assertIn("checkpoint", checkpoint_path_dep.semantic_keywords)
        self.assertIn("model", checkpoint_path_dep.semantic_keywords)
        self.assertIn("weights", checkpoint_path_dep.semantic_keywords)
        self.assertIn("pretrained", checkpoint_path_dep.semantic_keywords)
        self.assertIn("resume", checkpoint_path_dep.semantic_keywords)
    
    def test_data_types(self):
        """Test that the dependencies and outputs have the correct data types."""
        # Check dependency data types
        input_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("input_path")
        self.assertEqual(input_path_dep.data_type, "S3Uri")
        
        checkpoint_path_dep = PYTORCH_TRAINING_SPEC.get_dependency("checkpoint_path")
        self.assertEqual(checkpoint_path_dep.data_type, "S3Uri")
        
        # Check output data types
        model_output = PYTORCH_TRAINING_SPEC.get_output("model_output")
        self.assertEqual(model_output.data_type, "S3Uri")
        
        training_job_name = PYTORCH_TRAINING_SPEC.get_output("training_job_name")
        self.assertEqual(training_job_name.data_type, "String")
        
        metrics_output = PYTORCH_TRAINING_SPEC.get_output("metrics_output")
        self.assertEqual(metrics_output.data_type, "String")
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PYTORCH_TRAINING_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
