"""
Tests for SpecificationRegistry - atomized registry functionality.
"""

import unittest
from src.pipeline_deps.specification_registry import SpecificationRegistry
from src.pipeline_deps.base_specifications import (
    StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
)


class TestSpecificationRegistry(unittest.TestCase):
    """Test cases for SpecificationRegistry."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = SpecificationRegistry("test_context")
        
        # Create test specifications
        output_spec = OutputSpec(
            logical_name="raw_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['RawData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Raw data output"
        )
        
        self.data_loading_spec = StepSpecification(
            step_type="DataLoadingStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[output_spec]
        )
        
        # Create dependency and output specs separately
        dep_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataLoadingStep"],
            semantic_keywords=["data", "input"],
            data_type="S3Uri",
            description="Input data for preprocessing"
        )
        
        output_spec = OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed data output"
        )
        
        self.preprocessing_spec = StepSpecification(
            step_type="PreprocessingStep",
            node_type=NodeType.INTERNAL,
            dependencies=[dep_spec],
            outputs=[output_spec]
        )
        
    def tearDown(self):
        """Clean up after tests."""
        # Clear the global registry manager to avoid state leakage
        from src.pipeline_deps.registry_manager import registry_manager
        registry_manager.clear_all_contexts()
    
    def test_registry_initialization(self):
        """Test registry initialization with context name."""
        registry = SpecificationRegistry("test_pipeline")
        self.assertEqual(registry.context_name, "test_pipeline")
        self.assertEqual(len(registry.list_step_names()), 0)
        self.assertEqual(len(registry.list_step_types()), 0)
    
    def test_register_specification(self):
        """Test registering step specifications."""
        # Register data loading step
        self.registry.register("data_loading", self.data_loading_spec)
        
        # Verify registration
        self.assertIn("data_loading", self.registry.list_step_names())
        self.assertIn("DataLoadingStep", self.registry.list_step_types())
        
        # Verify retrieval
        retrieved_spec = self.registry.get_specification("data_loading")
        self.assertIsNotNone(retrieved_spec)
        self.assertEqual(retrieved_spec.step_type, "DataLoadingStep")
    
    def test_register_invalid_specification(self):
        """Test registering invalid specifications raises errors."""
        with self.assertRaises(ValueError):
            self.registry.register("invalid", "not_a_specification")
    
    def test_get_specifications_by_type(self):
        """Test retrieving specifications by step type."""
        # Register multiple steps
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        
        # Get by type
        data_loading_specs = self.registry.get_specifications_by_type("DataLoadingStep")
        self.assertEqual(len(data_loading_specs), 1)
        self.assertEqual(data_loading_specs[0].step_type, "DataLoadingStep")
        
        preprocessing_specs = self.registry.get_specifications_by_type("PreprocessingStep")
        self.assertEqual(len(preprocessing_specs), 1)
        self.assertEqual(preprocessing_specs[0].step_type, "PreprocessingStep")
        
        # Non-existent type
        nonexistent_specs = self.registry.get_specifications_by_type("NonExistentStep")
        self.assertEqual(len(nonexistent_specs), 0)
    
    def test_find_compatible_outputs(self):
        """Test finding compatible outputs for dependencies."""
        # Register steps
        self.registry.register("data_loading", self.data_loading_spec)
        self.registry.register("preprocessing", self.preprocessing_spec)
        
        # Create dependency spec to match
        dependency_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["DataLoadingStep"],
            data_type="S3Uri"
        )
        
        # Find compatible outputs
        compatible = self.registry.find_compatible_outputs(dependency_spec)
        
        # Should find the data loading output
        self.assertGreater(len(compatible), 0)
        
        # Check the best match
        best_match = compatible[0]
        step_name, output_name, output_spec, score = best_match
        
        self.assertEqual(step_name, "data_loading")
        self.assertEqual(output_name, "raw_data")
        self.assertEqual(output_spec.output_type, DependencyType.PROCESSING_OUTPUT)
        self.assertGreater(score, 0.5)  # Should have good compatibility score
    
    def test_compatibility_checking(self):
        """Test internal compatibility checking logic."""
        # Register data loading step
        self.registry.register("data_loading", self.data_loading_spec)
        
        # Compatible dependency
        compatible_dep = DependencySpec(
            logical_name="data_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
        
        # Incompatible dependency (wrong type)
        incompatible_dep = DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            data_type="S3Uri"
        )
        
        # Test compatibility
        compatible_outputs = self.registry.find_compatible_outputs(compatible_dep)
        incompatible_outputs = self.registry.find_compatible_outputs(incompatible_dep)
        
        self.assertGreater(len(compatible_outputs), 0)
        self.assertEqual(len(incompatible_outputs), 0)
    
    def test_context_isolation(self):
        """Test that different registry contexts are isolated."""
        # Create two registries with different contexts
        registry1 = SpecificationRegistry("pipeline_1")
        registry2 = SpecificationRegistry("pipeline_2")
        
        # Register different specs in each
        registry1.register("step1", self.data_loading_spec)
        registry2.register("step2", self.preprocessing_spec)
        
        # Verify isolation
        self.assertIn("step1", registry1.list_step_names())
        self.assertNotIn("step1", registry2.list_step_names())
        
        self.assertIn("step2", registry2.list_step_names())
        self.assertNotIn("step2", registry1.list_step_names())
        
        # Verify context names
        self.assertEqual(registry1.context_name, "pipeline_1")
        self.assertEqual(registry2.context_name, "pipeline_2")
    
    def test_registry_string_representation(self):
        """Test string representation of registry."""
        self.registry.register("data_loading", self.data_loading_spec)
        
        repr_str = repr(self.registry)
        self.assertIn("test_context", repr_str)
        self.assertIn("steps=1", repr_str)
    
    def test_empty_registry_operations(self):
        """Test operations on empty registry."""
        empty_registry = SpecificationRegistry("empty")
        
        # Test empty operations
        self.assertEqual(len(empty_registry.list_step_names()), 0)
        self.assertEqual(len(empty_registry.list_step_types()), 0)
        self.assertIsNone(empty_registry.get_specification("nonexistent"))
        
        # Test finding compatible outputs on empty registry
        dep_spec = DependencySpec(
            logical_name="test_dep",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri"
        )
        compatible = empty_registry.find_compatible_outputs(dep_spec)
        self.assertEqual(len(compatible), 0)


if __name__ == '__main__':
    unittest.main()
