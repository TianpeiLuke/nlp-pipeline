"""
Tests for atomized import structure - verify all imports work correctly.
"""

import unittest
import sys
import importlib


class TestAtomizedImports(unittest.TestCase):
    """Test that all imports work correctly with the new atomized structure."""
    
    def test_base_specifications_imports(self):
        """Test imports from base_specifications module."""
        try:
            from src.pipeline_deps.base_specifications import (
                DependencyType, DependencySpec, OutputSpec, PropertyReference, 
                StepSpecification, NodeType
            )
            
            # Verify classes are importable and instantiable
            self.assertTrue(hasattr(DependencyType, 'PROCESSING_OUTPUT'))
            self.assertTrue(hasattr(NodeType, 'SOURCE'))
            
            # Test basic instantiation
            dep_spec = DependencySpec(
                logical_name="test_dep",
                dependency_type=DependencyType.PROCESSING_OUTPUT,
                data_type="S3Uri"
            )
            self.assertEqual(dep_spec.logical_name, "test_dep")
            
        except ImportError as e:
            self.fail(f"Failed to import from base_specifications: {e}")
    
    def test_specification_registry_imports(self):
        """Test imports from specification_registry module."""
        try:
            from src.pipeline_deps.specification_registry import SpecificationRegistry
            
            # Test instantiation
            registry = SpecificationRegistry("test_context")
            self.assertEqual(registry.context_name, "test_context")
            
        except ImportError as e:
            self.fail(f"Failed to import from specification_registry: {e}")
    
    def test_registry_manager_imports(self):
        """Test imports from registry_manager module."""
        try:
            from src.pipeline_deps.registry_manager import (
                RegistryManager, registry_manager, get_registry, 
                get_pipeline_registry, get_default_registry,
                list_contexts, clear_context, get_context_stats
            )
            
            # Test manager instantiation
            manager = RegistryManager()
            self.assertIsInstance(manager, RegistryManager)
            
            # Test convenience functions
            registry = get_registry("test")
            self.assertIsNotNone(registry)
            
        except ImportError as e:
            self.fail(f"Failed to import from registry_manager: {e}")
    
    def test_dependency_resolver_imports(self):
        """Test imports from dependency_resolver module."""
        try:
            from src.pipeline_deps.dependency_resolver import (
                UnifiedDependencyResolver, DependencyResolutionError, global_resolver
            )
            
            # Test resolver instantiation
            resolver = UnifiedDependencyResolver()
            self.assertIsInstance(resolver, UnifiedDependencyResolver)
            
            # Test global instance
            self.assertIsInstance(global_resolver, UnifiedDependencyResolver)
            
        except ImportError as e:
            self.fail(f"Failed to import from dependency_resolver: {e}")
    
    def test_main_module_imports(self):
        """Test imports from main pipeline_deps module."""
        try:
            from src.pipeline_deps import (
                DependencyType, DependencySpec, OutputSpec, PropertyReference, 
                StepSpecification, SpecificationRegistry, RegistryManager,
                get_registry, get_pipeline_registry, get_default_registry,
                UnifiedDependencyResolver, DependencyResolutionError
            )
            
            # Verify all expected classes are available
            self.assertTrue(hasattr(DependencyType, 'PROCESSING_OUTPUT'))
            self.assertIsNotNone(SpecificationRegistry)
            self.assertIsNotNone(RegistryManager)
            self.assertIsNotNone(UnifiedDependencyResolver)
            
        except ImportError as e:
            self.fail(f"Failed to import from main pipeline_deps module: {e}")
    
    def test_backward_compatibility_imports(self):
        """Test that backward compatibility imports still work."""
        try:
            # These should work for backward compatibility
            from src.pipeline_deps import get_pipeline_registry, get_default_registry
            
            # Test they return the expected types
            pipeline_registry = get_pipeline_registry("test_pipeline")
            default_registry = get_default_registry()
            
            from src.pipeline_deps.specification_registry import SpecificationRegistry
            self.assertIsInstance(pipeline_registry, SpecificationRegistry)
            self.assertIsInstance(default_registry, SpecificationRegistry)
            
        except ImportError as e:
            self.fail(f"Backward compatibility imports failed: {e}")
    
    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        try:
            # Import all modules in sequence to check for circular dependencies
            import src.pipeline_deps.base_specifications
            import src.pipeline_deps.specification_registry
            import src.pipeline_deps.registry_manager
            import src.pipeline_deps.dependency_resolver
            import src.pipeline_deps
            
            # If we get here, no circular imports
            self.assertTrue(True)
            
        except ImportError as e:
            self.fail(f"Circular import detected: {e}")
    
    def test_module_structure(self):
        """Test that modules have expected structure."""
        # Test base_specifications module
        import src.pipeline_deps.base_specifications as base_specs
        expected_base_attrs = [
            'DependencyType', 'NodeType', 'DependencySpec', 
            'OutputSpec', 'PropertyReference', 'StepSpecification'
        ]
        for attr in expected_base_attrs:
            self.assertTrue(hasattr(base_specs, attr), f"Missing {attr} in base_specifications")
        
        # Test specification_registry module
        import src.pipeline_deps.specification_registry as spec_registry
        self.assertTrue(hasattr(spec_registry, 'SpecificationRegistry'))
        
        # Test registry_manager module - use sys.modules to get actual module
        import sys
        import src.pipeline_deps.registry_manager
        reg_manager = sys.modules['src.pipeline_deps.registry_manager']
        expected_manager_attrs = [
            'RegistryManager', 'get_registry', 'get_pipeline_registry', 
            'get_default_registry', 'registry_manager'
        ]
        for attr in expected_manager_attrs:
            self.assertTrue(hasattr(reg_manager, attr), f"Missing {attr} in registry_manager")
    
    def test_public_api_completeness(self):
        """Test that the public API includes all expected components."""
        import src.pipeline_deps as pipeline_deps
        
        # Check __all__ contains expected items
        expected_public_api = [
            'DependencyType', 'DependencySpec', 'OutputSpec', 'PropertyReference',
            'StepSpecification', 'SpecificationRegistry', 'RegistryManager',
            'get_registry', 'get_pipeline_registry', 'get_default_registry',
            'UnifiedDependencyResolver', 'DependencyResolutionError'
        ]
        
        for item in expected_public_api:
            self.assertTrue(hasattr(pipeline_deps, item), 
                          f"Missing {item} in public API")
    
    def test_import_performance(self):
        """Test that imports don't take excessive time."""
        import time
        
        start_time = time.time()
        
        # Import the main module
        import src.pipeline_deps
        
        end_time = time.time()
        import_time = end_time - start_time
        
        # Should import quickly (less than 1 second)
        self.assertLess(import_time, 1.0, "Import took too long")
    
    def test_module_reloading(self):
        """Test that modules can be reloaded without issues."""
        try:
            # Import modules first
            import src.pipeline_deps.base_specifications
            import src.pipeline_deps.specification_registry
            import src.pipeline_deps.registry_manager
            import src.pipeline_deps.dependency_resolver
            
            # Now reload them using sys.modules
            importlib.reload(sys.modules['src.pipeline_deps.base_specifications'])
            importlib.reload(sys.modules['src.pipeline_deps.specification_registry'])
            importlib.reload(sys.modules['src.pipeline_deps.registry_manager'])
            importlib.reload(sys.modules['src.pipeline_deps.dependency_resolver'])
            
            # If we get here, reloading works
            self.assertTrue(True)
            
        except Exception as e:
            self.fail(f"Module reloading failed: {e}")


class TestIntegrationWithAtomizedStructure(unittest.TestCase):
    """Test integration scenarios with the atomized structure."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow using atomized imports."""
        # Import directly from modules to avoid any import conflicts
        from src.pipeline_deps.registry_manager import get_registry
        from src.pipeline_deps.base_specifications import StepSpecification, OutputSpec, NodeType, DependencyType
        
        # Create registry
        registry = get_registry("integration_test")
        
        # Create output spec separately
        output_spec = OutputSpec(
            logical_name="test_output",
            output_type="processing_output",  # Use string value instead of enum
            property_path="properties.ProcessingOutputConfig.Outputs['TestOutput'].S3Output.S3Uri",
            data_type="S3Uri"
        )
        
        # Create and register specification
        spec = StepSpecification(
            step_type="TestStep",
            node_type=NodeType.SOURCE,  # Use enum instance
            dependencies=[],
            outputs=[output_spec]
        )
        
        registry.register("test_step", spec)
        
        # Verify it works
        retrieved_spec = registry.get_specification("test_step")
        self.assertIsNotNone(retrieved_spec)
        self.assertEqual(retrieved_spec.step_type, "TestStep")
    
    def test_cross_module_compatibility(self):
        """Test that components from different modules work together."""
        from src.pipeline_deps.specification_registry import SpecificationRegistry
        from src.pipeline_deps.registry_manager import RegistryManager
        from src.pipeline_deps.base_specifications import (
            StepSpecification, DependencyType, NodeType, OutputSpec
        )
        
        # Create manager and registry
        manager = RegistryManager()
        registry = manager.get_registry("cross_module_test")
        
        # Should be SpecificationRegistry instance
        self.assertIsInstance(registry, SpecificationRegistry)
        
        # Create output spec separately
        output_spec = OutputSpec(
            logical_name="output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.test",
            data_type="S3Uri"
        )
        
        # Create spec using base_specifications
        spec = StepSpecification(
            step_type="CrossModuleTest",
            node_type=NodeType.SOURCE,
            outputs=[output_spec]
        )
        
        # Register using registry from manager
        registry.register("cross_test", spec)
        
        # Verify it works
        self.assertIn("cross_test", registry.list_step_names())


if __name__ == '__main__':
    unittest.main()
