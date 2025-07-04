"""
Unit tests for pipeline registry module.

Tests the pipeline-scoped registry implementation including:
- PipelineRegistry functionality
- RegistryManager functionality
- Utility functions for registry access
- Integration with pipeline builders
"""

import unittest
from unittest.mock import MagicMock, patch

from src.pipeline_deps.base_specifications import (
    DependencySpec, OutputSpec, StepSpecification,
    DependencyType, NodeType, SpecificationRegistry
)
from src.pipeline_deps.pipeline_registry import (
    PipelineRegistry, RegistryManager, registry_manager,
    get_pipeline_registry, get_default_registry, integrate_with_pipeline_builder
)


class TestPipelineRegistry(unittest.TestCase):
    """Test cases for PipelineRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.pipeline_registry = PipelineRegistry("test_pipeline")
        
        # Create a test specification
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri"
        )
        
        self.step_spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[self.output_spec]
        )
    
    def test_pipeline_registry_initialization(self):
        """Test PipelineRegistry initialization."""
        self.assertEqual(self.pipeline_registry.pipeline_name, "test_pipeline")
        self.assertEqual(len(self.pipeline_registry._specifications), 0)
    
    def test_register_specification(self):
        """Test registering a specification in a pipeline registry."""
        self.pipeline_registry.register("test_step", self.step_spec)
        
        # Test retrieval
        retrieved_spec = self.pipeline_registry.get_specification("test_step")
        self.assertIsNotNone(retrieved_spec)
        self.assertEqual(retrieved_spec.step_type, "TestStep")
    
    def test_find_compatible_outputs(self):
        """Test finding compatible outputs in a pipeline registry."""
        # Register the specification
        self.pipeline_registry.register("test_step", self.step_spec)
        
        # Create a dependency that should match
        dep_spec = DependencySpec(
            logical_name="input_data",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            data_type="S3Uri",
            compatible_sources=["TestStep"],
            semantic_keywords=["test", "output"]
        )
        
        compatible = self.pipeline_registry.find_compatible_outputs(dep_spec)
        self.assertEqual(len(compatible), 1)
        
        step_name, output_name, output_spec, score = compatible[0]
        self.assertEqual(step_name, "test_step")
        self.assertEqual(output_name, "test_output")
        self.assertGreater(score, 0.5)  # Should have a good compatibility score
    
    def test_string_representation(self):
        """Test string representation of PipelineRegistry."""
        repr_str = repr(self.pipeline_registry)
        self.assertIn("PipelineRegistry", repr_str)
        self.assertIn("test_pipeline", repr_str)
        self.assertIn("steps=0", repr_str)
        
        # Add a step and check again
        self.pipeline_registry.register("test_step", self.step_spec)
        repr_str = repr(self.pipeline_registry)
        self.assertIn("steps=1", repr_str)


class TestRegistryManager(unittest.TestCase):
    """Test cases for RegistryManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry_manager = RegistryManager()
        
        # Create a test specification
        self.output_spec = OutputSpec(
            logical_name="test_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri"
        )
        
        self.step_spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.SOURCE,
            dependencies=[],
            outputs=[self.output_spec]
        )
    
    def test_registry_manager_initialization(self):
        """Test RegistryManager initialization."""
        self.assertEqual(len(self.registry_manager._pipeline_registries), 0)
        self.assertIsInstance(self.registry_manager._default_registry, SpecificationRegistry)
    
    def test_get_pipeline_registry(self):
        """Test getting a pipeline registry."""
        # Get a non-existent registry (should create it)
        pipeline_registry = self.registry_manager.get_pipeline_registry("test_pipeline")
        self.assertIsNotNone(pipeline_registry)
        self.assertIsInstance(pipeline_registry, PipelineRegistry)
        self.assertEqual(pipeline_registry.pipeline_name, "test_pipeline")
        
        # Get the same registry again (should return the existing one)
        same_registry = self.registry_manager.get_pipeline_registry("test_pipeline")
        self.assertIs(same_registry, pipeline_registry)
        
        # Get a registry without creating if missing
        non_existent = self.registry_manager.get_pipeline_registry("non_existent", create_if_missing=False)
        self.assertIsNone(non_existent)
    
    def test_get_default_registry(self):
        """Test getting the default registry."""
        default_registry = self.registry_manager.get_default_registry()
        self.assertIsNotNone(default_registry)
        self.assertIsInstance(default_registry, SpecificationRegistry)
    
    def test_list_pipeline_registries(self):
        """Test listing pipeline registries."""
        # Initially empty
        self.assertEqual(len(self.registry_manager.list_pipeline_registries()), 0)
        
        # Create a few registries
        self.registry_manager.get_pipeline_registry("pipeline1")
        self.registry_manager.get_pipeline_registry("pipeline2")
        
        # Check the list
        registry_list = self.registry_manager.list_pipeline_registries()
        self.assertEqual(len(registry_list), 2)
        self.assertIn("pipeline1", registry_list)
        self.assertIn("pipeline2", registry_list)
    
    def test_clear_pipeline_registry(self):
        """Test clearing a pipeline registry."""
        # Create a registry
        self.registry_manager.get_pipeline_registry("test_pipeline")
        
        # Clear it
        result = self.registry_manager.clear_pipeline_registry("test_pipeline")
        self.assertTrue(result)
        
        # Verify it's gone
        self.assertNotIn("test_pipeline", self.registry_manager.list_pipeline_registries())
        
        # Try to clear a non-existent registry
        result = self.registry_manager.clear_pipeline_registry("non_existent")
        self.assertFalse(result)
    
    def test_isolation_between_registries(self):
        """Test that registries are isolated from each other."""
        # Create two registries
        registry1 = self.registry_manager.get_pipeline_registry("pipeline1")
        registry2 = self.registry_manager.get_pipeline_registry("pipeline2")
        
        # Register a step in registry1
        registry1.register("test_step", self.step_spec)
        
        # Verify it's in registry1 but not in registry2
        self.assertIsNotNone(registry1.get_specification("test_step"))
        self.assertIsNone(registry2.get_specification("test_step"))
    
    def test_string_representation(self):
        """Test string representation of RegistryManager."""
        repr_str = repr(self.registry_manager)
        self.assertIn("RegistryManager", repr_str)
        self.assertIn("pipelines=0", repr_str)
        
        # Add a pipeline and check again
        self.registry_manager.get_pipeline_registry("test_pipeline")
        repr_str = repr(self.registry_manager)
        self.assertIn("pipelines=1", repr_str)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a fresh registry manager for each test
        self.patcher = patch('src.pipeline_deps.pipeline_registry.registry_manager', new=RegistryManager())
        self.mock_registry_manager = self.patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
    
    def test_get_pipeline_registry(self):
        """Test get_pipeline_registry utility function."""
        # Get a pipeline registry
        pipeline_registry = get_pipeline_registry("test_pipeline")
        self.assertIsNotNone(pipeline_registry)
        self.assertIsInstance(pipeline_registry, PipelineRegistry)
        self.assertEqual(pipeline_registry.pipeline_name, "test_pipeline")
    
    def test_get_default_registry(self):
        """Test get_default_registry utility function."""
        default_registry = get_default_registry()
        self.assertIsNotNone(default_registry)
        self.assertIsInstance(default_registry, SpecificationRegistry)


class TestIntegrationWithPipelineBuilder(unittest.TestCase):
    """Test cases for integration with pipeline builders."""
    
    def test_integrate_with_pipeline_builder(self):
        """Test the integrate_with_pipeline_builder decorator."""
        # Create a mock pipeline builder class
        class MockPipelineBuilder:
            def __init__(self, base_config=None):
                self.base_config = base_config or MagicMock()
                self.base_config.pipeline_name = "test_pipeline"
        
        # Apply the decorator
        DecoratedBuilder = integrate_with_pipeline_builder(MockPipelineBuilder)
        
        # Create an instance
        builder = DecoratedBuilder()
        
        # Check that it has a registry property
        self.assertTrue(hasattr(builder, "registry"))
        self.assertIsInstance(builder.registry, PipelineRegistry)
        self.assertEqual(builder.registry.pipeline_name, "test_pipeline")
    
    def test_integration_with_custom_pipeline_name(self):
        """Test integration with a custom pipeline name."""
        # Create a mock pipeline builder class
        class MockPipelineBuilder:
            def __init__(self, base_config=None):
                self.base_config = base_config or MagicMock()
                self.base_config.pipeline_name = "custom_pipeline"
        
        # Apply the decorator
        DecoratedBuilder = integrate_with_pipeline_builder(MockPipelineBuilder)
        
        # Create an instance
        builder = DecoratedBuilder()
        
        # Check that it has a registry with the custom name
        self.assertEqual(builder.registry.pipeline_name, "custom_pipeline")
    
    def test_integration_with_missing_pipeline_name(self):
        """Test integration when pipeline name is missing."""
        # Create a mock pipeline builder class without pipeline_name
        class MockPipelineBuilder:
            def __init__(self):
                # Use an empty object instead of MagicMock to ensure pipeline_name is truly missing
                class EmptyConfig:
                    pass
                self.base_config = EmptyConfig()
        
        # Apply the decorator
        DecoratedBuilder = integrate_with_pipeline_builder(MockPipelineBuilder)
        
        # Create an instance
        builder = DecoratedBuilder()
        
        # Check that it has a registry with the default name
        self.assertEqual(builder.registry.pipeline_name, "default_pipeline")


if __name__ == "__main__":
    unittest.main()
