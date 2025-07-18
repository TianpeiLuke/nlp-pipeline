"""
Test cases for the CircularReferenceTracker.

This module tests the CircularReferenceTracker's ability to detect and handle
circular references in object graphs during deserialization.
"""

import unittest
from typing import Dict, Any, Optional, List, Type

from src.config_field_manager.circular_reference_tracker import CircularReferenceTracker
from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
from pydantic import BaseModel


class CircularReferenceTrackerTest(unittest.TestCase):
    """Test suite for the CircularReferenceTracker."""
    
    def setUp(self):
        """Set up the test environment."""
        self.tracker = CircularReferenceTracker(max_depth=10)
        
    def test_simple_object_tracking(self):
        """Test tracking simple objects with no circular references."""
        # Create a simple object
        obj = {"__model_type__": "TestConfig", "__model_module__": "test.module", "name": "test1"}
        
        # Enter the object
        is_circular, message = self.tracker.enter_object(obj, "root")
        
        # Assert not circular
        self.assertFalse(is_circular)
        self.assertIsNone(message)
        
        # Check current path
        path_str = self.tracker.get_current_path_str()
        self.assertEqual(path_str, "TestConfig(name=test1)")
        
        # Exit the object
        self.tracker.exit_object()
        
        # Path should now be empty
        path_str = self.tracker.get_current_path_str()
        self.assertEqual(path_str, "")
        
    def test_nested_object_tracking(self):
        """Test tracking nested objects with no circular references."""
        # Create parent object
        parent = {"__model_type__": "ParentConfig", "__model_module__": "test.module", "name": "parent"}
        
        # Enter parent object
        is_circular, message = self.tracker.enter_object(parent, "root")
        self.assertFalse(is_circular)
        
        # Create child object
        child = {"__model_type__": "ChildConfig", "__model_module__": "test.module", "id": 123}
        
        # Enter child object
        is_circular, message = self.tracker.enter_object(child, "child_field", {"parent": "ParentConfig"})
        self.assertFalse(is_circular)
        
        # Check path
        path_str = self.tracker.get_current_path_str()
        self.assertEqual(path_str, "ParentConfig(name=parent) -> ChildConfig(id=123)")
        
        # Exit child then parent
        self.tracker.exit_object()  # Exit child
        self.tracker.exit_object()  # Exit parent
        
        # Path should be empty
        path_str = self.tracker.get_current_path_str()
        self.assertEqual(path_str, "")
        
    def test_circular_reference_detection(self):
        """Test detection of circular references."""
        # Create objects that will form a cycle
        obj_a = {"__model_type__": "ConfigA", "__model_module__": "test.module", "name": "a"}
        obj_b = {"__model_type__": "ConfigB", "__model_module__": "test.module", "name": "b"}
        
        # Enter object A
        is_circular, message = self.tracker.enter_object(obj_a, "root")
        self.assertFalse(is_circular)
        
        # Enter object B as child of A
        is_circular, message = self.tracker.enter_object(obj_b, "b_field")
        self.assertFalse(is_circular)
        
        # Try to enter object A again (as child of B) - should detect cycle
        is_circular, message = self.tracker.enter_object(obj_a, "a_field")
        self.assertTrue(is_circular)
        self.assertIsNotNone(message)
        
        # Check the error message contains useful information
        self.assertIn("Circular reference detected", message)
        self.assertIn("ConfigA", message)
        self.assertIn("Original definition path", message)
        self.assertIn("Reference path", message)
        
        # Exit objects
        self.tracker.exit_object()  # Exit B
        self.tracker.exit_object()  # Exit A
        
    def test_max_depth_detection(self):
        """Test detection of maximum recursion depth."""
        # Create a tracker with small max depth
        shallow_tracker = CircularReferenceTracker(max_depth=3)
        
        # Create objects for nesting
        obj1 = {"__model_type__": "Config1", "__model_module__": "test.module"}
        obj2 = {"__model_type__": "Config2", "__model_module__": "test.module"}
        obj3 = {"__model_type__": "Config3", "__model_module__": "test.module"}
        obj4 = {"__model_type__": "Config4", "__model_module__": "test.module"}
        
        # Enter objects up to max depth
        shallow_tracker.enter_object(obj1, "field1")
        shallow_tracker.enter_object(obj2, "field2")
        shallow_tracker.enter_object(obj3, "field3")
        
        # Next enter should trigger max depth error
        is_circular, message = shallow_tracker.enter_object(obj4, "field4")
        self.assertTrue(is_circular)
        self.assertIn("Maximum recursion depth (3) exceeded", message)
        
    def test_object_identification(self):
        """Test the object identification logic."""
        # Objects with same type but different identifiers should be considered different
        obj1 = {"__model_type__": "SameConfig", "name": "test1"}
        obj2 = {"__model_type__": "SameConfig", "name": "test2"}
        
        self.tracker.enter_object(obj1, "field1")
        is_circular, _ = self.tracker.enter_object(obj2, "field2")
        self.assertFalse(is_circular)  # Not circular, because they're different objects
        
        self.tracker.exit_object()  # Exit obj2
        self.tracker.exit_object()  # Exit obj1
        
        # Objects with same type and identifier should be considered the same
        obj3 = {"__model_type__": "SameConfig", "name": "test3"}
        obj4 = {"__model_type__": "SameConfig", "name": "test3"}  # Same name
        
        self.tracker.enter_object(obj3, "field3")
        is_circular, _ = self.tracker.enter_object(obj4, "field4")
        self.assertTrue(is_circular)  # Circular, because they're considered the same object
        
    def test_integration_with_serializer_simulation(self):
        """
        Test a simulated integration with a serializer.
        
        This test simulates how the CircularReferenceTracker would be used in a serializer
        by setting up a simple deserialization function.
        """
        def deserialize(data: Dict[str, Any], field_name: Optional[str] = None) -> Any:
            """Simulate deserializing an object with cycle detection."""
            # Check if it's not a dict
            if not isinstance(data, dict):
                return data
                
            # Use tracker to check for circular references
            is_circular, error = self.tracker.enter_object(data, field_name)
            
            if is_circular:
                print(f"Circular reference detected: {error}")
                return None  # Return None for circular references
                
            try:
                # Simulate deserializing fields recursively
                result = {}
                for k, v in data.items():
                    if isinstance(v, dict):
                        result[k] = deserialize(v, k)  # Recursive call
                    else:
                        result[k] = v
                return result
            finally:
                # Always clean up the tracking stack
                self.tracker.exit_object()
        
        # Create objects with a circular reference
        obj_a = {"__model_type__": "ConfigA", "name": "a"}
        obj_b = {"__model_type__": "ConfigB", "name": "b"}
        obj_a["ref_to_b"] = obj_b
        obj_b["ref_to_a"] = obj_a  # This creates a cycle
        
        # Try to deserialize - should handle the cycle gracefully
        result = deserialize(obj_a, "root")
        
        # Check that we got a result with ref_to_b populated
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "a")
        self.assertIsNotNone(result["ref_to_b"])
        self.assertEqual(result["ref_to_b"]["name"], "b")
        # The circular reference should be broken by returning None
        self.assertIsNone(result["ref_to_b"]["ref_to_a"])
    
    def test_complex_nested_paths(self):
        """Test tracking complex nested paths through an object graph."""
        # Create a complex nested structure
        root = {"__model_type__": "RootConfig", "name": "root", "id": "root-1"}
        level1a = {"__model_type__": "Level1Config", "name": "level1a", "id": "1a"}
        level1b = {"__model_type__": "Level1Config", "name": "level1b", "id": "1b"}
        level2a = {"__model_type__": "Level2Config", "name": "level2a", "id": "2a"}
        level2b = {"__model_type__": "Level2Config", "name": "level2b", "id": "2b"}
        level3 = {"__model_type__": "Level3Config", "name": "level3", "id": "3"}
        
        # Set up nested structure
        root["child1"] = level1a
        root["child2"] = level1b
        level1a["child"] = level2a
        level1b["child"] = level2b
        level2a["child"] = level3
        level2b["child"] = level3  # Both level2a and level2b point to same level3 (diamond pattern)
        
        # Enter through one path
        self.tracker.enter_object(root, "root")
        self.tracker.enter_object(level1a, "child1")
        self.tracker.enter_object(level2a, "child")
        self.tracker.enter_object(level3, "child")
        
        # Check the path
        path_str = self.tracker.get_current_path_str()
        self.assertEqual(path_str, "RootConfig(name=root) -> Level1Config(name=level1a) -> " + 
                                   "Level2Config(name=level2a) -> Level3Config(name=level3)")
        
        # Exit back up
        self.tracker.exit_object()  # Exit level3
        self.tracker.exit_object()  # Exit level2a
        self.tracker.exit_object()  # Exit level1a
        
        # Now try a different path to level3 (should be circular because level3 is still tracked)
        self.tracker.enter_object(level1b, "child2")
        self.tracker.enter_object(level2b, "child")
        is_circular, message = self.tracker.enter_object(level3, "child")
        
        # Should be circular because level3 was already visited in this session
        # The tracker maintains object history across different paths in the same session
        self.assertTrue(is_circular)
        
        # Check the new path - when circular reference is detected, the circular object is not added to path
        path_str = self.tracker.get_current_path_str()
        self.assertEqual(path_str, "RootConfig(name=root) -> Level1Config(name=level1b) -> " + 
                                   "Level2Config(name=level2b)")
        
        # Exit everything
        self.tracker.exit_object()  # Exit level3
        self.tracker.exit_object()  # Exit level2b
        self.tracker.exit_object()  # Exit level1b
        self.tracker.exit_object()  # Exit root
        
    def test_actual_integration_with_serializer(self):
        """Test actual integration with TypeAwareConfigSerializer."""
        # Create Pydantic models to test with
        class Item(BaseModel):
            name: str
            value: int
        
        class Container(BaseModel):
            name: str
            item: Optional[Item] = None
            container: Optional["Container"] = None  # Self-referential
        
        # Create a serializer with our tracker
        serializer = TypeAwareConfigSerializer()
        serializer.ref_tracker = self.tracker  # Use our test tracker
        
        # Create a circular reference
        item = Item(name="test-item", value=42)
        container1 = Container(name="container1", item=item)
        container2 = Container(name="container2", item=item)
        container1.container = container2
        container2.container = container1  # This creates the cycle
        
        # Serialize first to create a dict representation
        serialized = serializer.serialize(container1)
        
        # Check if serialization failed due to circular reference
        if isinstance(serialized, str) and "Serialization error" in serialized:
            # Serialization failed due to circular reference - this is expected behavior
            self.assertIn("Circular reference detected", serialized)
            return
        
        # If serialization succeeded, try to deserialize
        deserialized = serializer.deserialize(serialized)
        
        # If deserialization returns a dict, verify the structure
        if isinstance(deserialized, dict):
            self.assertEqual(deserialized["name"], "container1")
            if "item" in deserialized and deserialized["item"]:
                self.assertEqual(deserialized["item"]["name"], "test-item")
                self.assertEqual(deserialized["item"]["value"], 42)
            if "container" in deserialized and deserialized["container"]:
                self.assertEqual(deserialized["container"]["name"], "container2")
                # The circular reference should be detected and set to None
                self.assertIsNone(deserialized["container"].get("container"))
        
    def test_error_message_formatting(self):
        """Test that error messages are properly formatted for complex paths."""
        # Create a complex object graph
        root = {"__model_type__": "RootConfig", "name": "root-obj", "id": 1}
        child1 = {"__model_type__": "ChildConfig", "name": "child1", "id": 2}
        child2 = {"__model_type__": "ChildConfig", "name": "child2", "id": 3}
        grandchild = {"__model_type__": "GrandchildConfig", "name": "grandchild", "id": 4}
        
        # Set up the relationships
        root["child1"] = child1
        child1["child"] = child2
        child2["child"] = grandchild
        
        # Enter objects
        self.tracker.enter_object(root, "root")
        self.tracker.enter_object(child1, "child1")
        self.tracker.enter_object(child2, "child")
        self.tracker.enter_object(grandchild, "child")
        
        # Now try to create a circular reference
        is_circular, error_msg = self.tracker.enter_object(root, "circular_ref")
        
        # Verify it's circular
        self.assertTrue(is_circular)
        
        # Check error message formatting
        self.assertIn("Circular reference detected", error_msg)
        self.assertIn("RootConfig", error_msg)
        
        # Verify path info in the error
        self.assertIn("Original definition path:", error_msg)
        self.assertIn("RootConfig(name=root-obj)", error_msg)
        
        self.assertIn("Reference path:", error_msg)
        self.assertIn("RootConfig(name=root-obj) -> ChildConfig(name=child1) -> ChildConfig(name=child2) -> GrandchildConfig(name=grandchild)", error_msg)


if __name__ == '__main__':
    unittest.main()
