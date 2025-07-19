"""
Tests for integrating the CircularReferenceTracker with TypeAwareConfigSerializer.

This module verifies that the integrated TypeAwareConfigSerializer with
CircularReferenceTracker properly handles circular references in complex
object graphs during deserialization.
"""

import unittest
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field

from src.config_field_manager.circular_reference_tracker import CircularReferenceTracker
from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
from src.config_field_manager.config_class_store import ConfigClassStore


class EnhancedSerializer(TypeAwareConfigSerializer):
    """
    Enhanced version of TypeAwareConfigSerializer using CircularReferenceTracker.
    
    This demonstration class shows how the CircularReferenceTracker can be
    integrated with the existing TypeAwareConfigSerializer to provide
    improved circular reference detection and error reporting.
    """
    
    def __init__(self, config_classes=None):
        """Initialize with an integrated CircularReferenceTracker."""
        super().__init__(config_classes)
        # Create a circular reference tracker
        self.ref_tracker = CircularReferenceTracker(max_depth=100)
        
    def deserialize(self, field_data: Any, field_name: Optional[str] = None, 
                    expected_type: Optional[Type] = None) -> Any:
        """
        Deserialize with improved circular reference tracking.
        
        This overrides the base deserialize method to use CircularReferenceTracker
        for detecting circular references.
        
        Args:
            field_data: The data to deserialize
            field_name: Optional name of the field
            expected_type: Optional expected type
            
        Returns:
            The deserialized object, or None if a circular reference is detected
        """
        # Skip non-dict objects (can't have circular refs)
        if not isinstance(field_data, dict):
            return field_data
            
        # Use the tracker to check for circular references
        is_circular, error = self.ref_tracker.enter_object(
            field_data, field_name, 
            context={'expected_type': expected_type.__name__ if expected_type else None}
        )
        
        if is_circular:
            # Log the detailed error message
            self.logger.warning(error)
            # Return None instead of the circular reference
            return None
            
        try:
            # Use standard deserialization logic from parent class
            # For this demonstration, we'll implement a simplified version
            result = {}
            
            # Process each field
            for k, v in field_data.items():
                # Skip metadata fields
                if k in (self.MODEL_TYPE_FIELD, self.MODEL_MODULE_FIELD):
                    result[k] = v
                    continue
                    
                # Recursively deserialize nested objects
                if isinstance(v, dict):
                    result[k] = self.deserialize(v, k)
                else:
                    result[k] = v
                    
            return result
        finally:
            # Always exit the object when done, even if an exception occurred
            self.ref_tracker.exit_object()


class SerializerWithTrackerTest(unittest.TestCase):
    """Test the integration of CircularReferenceTracker with TypeAwareConfigSerializer."""
    
    def setUp(self):
        """Set up test models for testing."""
        # Define Pydantic models to test with
        class Item(BaseModel):
            name: str
            value: int
            
        class Container(BaseModel):
            name: str
            item: Optional[Item] = None
            container: Optional['Container'] = None
            
        # Register models with ConfigClassStore for serializer to use
        ConfigClassStore.register(Item)
        ConfigClassStore.register(Container)
        
        # Save classes for tests to use
        self.Item = Item
        self.Container = Container
    
    def test_circular_reference_handling(self):
        """Test that circular references are properly handled."""
        # Create the enhanced serializer
        serializer = EnhancedSerializer()
        
        # Create objects with a circular reference
        obj_a = {
            "__model_type__": "ConfigA",
            "__model_module__": "test.module",
            "name": "config_a",
            "id": 1
        }
        
        obj_b = {
            "__model_type__": "ConfigB",
            "__model_module__": "test.module",
            "name": "config_b",
            "id": 2
        }
        
        # Create circular reference
        obj_a["ref_to_b"] = obj_b
        obj_b["ref_to_a"] = obj_a
        
        # Deserialize with circular reference handling
        result = serializer.deserialize(obj_a, "root")
        
        # Verify the result
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "config_a")
        self.assertEqual(result["id"], 1)
        
        # The nested object should be present
        self.assertIsNotNone(result["ref_to_b"])
        self.assertEqual(result["ref_to_b"]["name"], "config_b")
        self.assertEqual(result["ref_to_b"]["id"], 2)
        
        # But the circular reference should be None
        self.assertIsNone(result["ref_to_b"]["ref_to_a"])
        
    def test_deeply_nested_objects(self):
        """Test handling of deeply nested objects."""
        # Create the enhanced serializer with a low max depth
        serializer = EnhancedSerializer()
        serializer.ref_tracker.max_depth = 5
        
        # Create a deeply nested structure
        obj1 = {"__model_type__": "Level1", "name": "level1"}
        obj2 = {"__model_type__": "Level2", "name": "level2"}
        obj3 = {"__model_type__": "Level3", "name": "level3"}
        obj4 = {"__model_type__": "Level4", "name": "level4"}
        obj5 = {"__model_type__": "Level5", "name": "level5"}
        obj6 = {"__model_type__": "Level6", "name": "level6"}
        
        # Link them together
        obj1["next"] = obj2
        obj2["next"] = obj3
        obj3["next"] = obj4
        obj4["next"] = obj5
        obj5["next"] = obj6
        
        # Deserialize - should handle the depth limit gracefully
        result = serializer.deserialize(obj1, "root")
        
        # The first levels should be present
        self.assertEqual(result["name"], "level1")
        self.assertEqual(result["next"]["name"], "level2")
        self.assertEqual(result["next"]["next"]["name"], "level3")
        self.assertEqual(result["next"]["next"]["next"]["name"], "level4")
        self.assertEqual(result["next"]["next"]["next"]["next"]["name"], "level5")
        
        # But the last level should be None due to max depth
        self.assertIsNone(result["next"]["next"]["next"]["next"]["next"])
        
    def test_real_serializer_with_circular_refs(self):
        """Test the actual TypeAwareConfigSerializer implementation with circular references."""
        # Create a standard TypeAwareConfigSerializer
        serializer = TypeAwareConfigSerializer()
        
        # Create circular reference with Pydantic models
        item = self.Item(name="test-item", value=42)
        container1 = self.Container(name="container1", item=item)
        container2 = self.Container(name="container2", item=item)
        
        # Create circular references
        container1.container = container2
        container2.container = container1  # This creates the cycle
        
        # Serialize container1
        serialized = serializer.serialize(container1)
        
        # Verify serialized structure has expected type info
        self.assertIn("__model_type__", serialized)
        self.assertEqual(serialized["__model_type__"], "Container")
        self.assertIn("name", serialized)
        self.assertEqual(serialized["name"], "container1")
        
        # Nested item should be properly serialized
        self.assertIn("item", serialized)
        self.assertIn("__model_type__", serialized["item"])
        self.assertEqual(serialized["item"]["__model_type__"], "Item")
        
        # Nested container should be properly serialized
        self.assertIn("container", serialized)
        self.assertIn("__model_type__", serialized["container"])
        self.assertEqual(serialized["container"]["__model_type__"], "Container")
        
        # Now deserialize - circular ref should be detected and broken
        deserialized = serializer.deserialize(serialized)
        
        # Check basic structure is intact
        self.assertEqual(deserialized["name"], "container1")
        self.assertEqual(deserialized["item"]["name"], "test-item")
        self.assertEqual(deserialized["item"]["value"], 42)
        
        # Check that container2 is present
        self.assertIn("container", deserialized)
        self.assertEqual(deserialized["container"]["name"], "container2")
        
        # But container2's reference back to container1 should be None (circular ref broken)
        self.assertIsNone(deserialized["container"]["container"])
        
    def test_job_type_variant_handling(self):
        """Test serializer correctly handles job type variants in step names."""
        # Create models with job type variant fields
        class ConfigWithJobType(BaseModel):
            name: str
            job_type: str
            data_type: Optional[str] = None
            mode: Optional[str] = None
            
        # Register with ConfigClassStore
        ConfigClassStore.register(ConfigWithJobType)
        
        # Create configs with different job types
        config1 = ConfigWithJobType(name="config1", job_type="training")
        config2 = ConfigWithJobType(name="config2", job_type="evaluation", data_type="tabular")
        config3 = ConfigWithJobType(name="config3", job_type="inference", mode="batch")
        
        # Create serializer
        serializer = TypeAwareConfigSerializer()
        
        # Test step name generation - should include job type variants
        step_name1 = serializer.generate_step_name(config1)
        step_name2 = serializer.generate_step_name(config2)
        step_name3 = serializer.generate_step_name(config3)
        
        # Verify job type is included
        self.assertIn("_training", step_name1)
        
        # Verify both job_type and data_type are included
        self.assertIn("_evaluation", step_name2)
        self.assertIn("_tabular", step_name2)
        
        # Verify job_type and mode are included
        self.assertIn("_inference", step_name3)
        self.assertIn("_batch", step_name3)


if __name__ == '__main__':
    unittest.main()
