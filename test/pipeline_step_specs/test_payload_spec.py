#!/usr/bin/env python3
"""
Unit tests for the MIMS Payload Step Specification.

This module provides tests for the payload step specification,
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
from src.pipeline_step_specs.payload_spec import PAYLOAD_SPEC


class TestPayloadSpec(unittest.TestCase):
    """Test cases for the MIMS Payload Step Specification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry()
        self.registry.register("payload_step", PAYLOAD_SPEC)
    
    def test_spec_registration(self):
        """Test that the specification can be registered successfully."""
        spec = self.registry.get_specification("payload_step")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.step_type, "Payload")
    
    def test_node_type(self):
        """Test that the specification has the correct node type."""
        self.assertEqual(PAYLOAD_SPEC.node_type, NodeType.INTERNAL)
    
    def test_required_dependencies(self):
        """Test that the specification has the required dependencies."""
        required_deps = PAYLOAD_SPEC.list_required_dependencies()
        required_dep_names = [dep.logical_name for dep in required_deps]
        
        # Check that model_input is required
        self.assertIn("model_input", required_dep_names)
        
        # Check dependency type
        model_input_dep = PAYLOAD_SPEC.get_dependency("model_input")
        self.assertEqual(model_input_dep.dependency_type, DependencyType.MODEL_ARTIFACTS)
        self.assertTrue(model_input_dep.required)
    
    def test_outputs(self):
        """Test that the specification has the correct outputs."""
        # Check output names
        self.assertIsNotNone(PAYLOAD_SPEC.get_output("payload_sample"))
        self.assertIsNotNone(PAYLOAD_SPEC.get_output("GeneratedPayloadSamples"))
        self.assertIsNotNone(PAYLOAD_SPEC.get_output("payload_metadata"))
        self.assertIsNotNone(PAYLOAD_SPEC.get_output("PayloadMetadata"))
        
        # Check output types
        payload_sample = PAYLOAD_SPEC.get_output("payload_sample")
        self.assertEqual(payload_sample.output_type, DependencyType.PROCESSING_OUTPUT)
        
        payload_metadata = PAYLOAD_SPEC.get_output("payload_metadata")
        self.assertEqual(payload_metadata.output_type, DependencyType.PROCESSING_OUTPUT)
        
        # Check property paths
        self.assertEqual(
            payload_sample.property_path,
            "properties.ProcessingOutputConfig.Outputs['GeneratedPayloadSamples'].S3Output.S3Uri"
        )
        self.assertEqual(
            payload_metadata.property_path,
            "properties.ProcessingOutputConfig.Outputs['PayloadMetadata'].S3Output.S3Uri"
        )
    
    def test_output_aliases(self):
        """Test that the output aliases point to the same property paths."""
        payload_sample = PAYLOAD_SPEC.get_output("payload_sample")
        payload_sample_alias = PAYLOAD_SPEC.get_output("GeneratedPayloadSamples")
        self.assertEqual(payload_sample.property_path, payload_sample_alias.property_path)
        
        payload_metadata = PAYLOAD_SPEC.get_output("payload_metadata")
        payload_metadata_alias = PAYLOAD_SPEC.get_output("PayloadMetadata")
        self.assertEqual(payload_metadata.property_path, payload_metadata_alias.property_path)
    
    def test_compatible_sources(self):
        """Test that the dependencies have the correct compatible sources."""
        model_input_dep = PAYLOAD_SPEC.get_dependency("model_input")
        
        # Check model_input compatible sources
        self.assertIn("XGBoostTraining", model_input_dep.compatible_sources)
        self.assertIn("TrainingStep", model_input_dep.compatible_sources)
        self.assertIn("ModelStep", model_input_dep.compatible_sources)
    
    def test_semantic_keywords(self):
        """Test that the dependencies have the correct semantic keywords."""
        model_input_dep = PAYLOAD_SPEC.get_dependency("model_input")
        
        # Check model_input semantic keywords
        self.assertIn("model", model_input_dep.semantic_keywords)
        self.assertIn("artifacts", model_input_dep.semantic_keywords)
        self.assertIn("trained", model_input_dep.semantic_keywords)
        self.assertIn("output", model_input_dep.semantic_keywords)
        self.assertIn("modelartifacts", model_input_dep.semantic_keywords)
    
    def test_data_types(self):
        """Test that the dependencies and outputs have the correct data types."""
        # Check dependency data types
        model_input_dep = PAYLOAD_SPEC.get_dependency("model_input")
        self.assertEqual(model_input_dep.data_type, "S3Uri")
        
        # Check output data types
        payload_sample = PAYLOAD_SPEC.get_output("payload_sample")
        self.assertEqual(payload_sample.data_type, "S3Uri")
        
        payload_metadata = PAYLOAD_SPEC.get_output("payload_metadata")
        self.assertEqual(payload_metadata.data_type, "S3Uri")
    
    def test_validation(self):
        """Test that the specification passes validation."""
        errors = PAYLOAD_SPEC.validate()
        self.assertEqual(len(errors), 0, f"Validation failed with errors: {errors}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
