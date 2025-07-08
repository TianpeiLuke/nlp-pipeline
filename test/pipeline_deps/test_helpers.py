"""
Test helpers for pipeline_deps tests to ensure proper isolation.

This module provides utility functions and base classes to reset global state
before and after tests, ensuring proper isolation between test cases.
"""

import unittest
from src.pipeline_deps.registry_manager import registry_manager
from src.pipeline_deps.dependency_resolver import global_resolver
from src.pipeline_deps.semantic_matcher import semantic_matcher


def reset_all_global_state():
    """
    Reset all global state for testing.
    
    This function resets the state of all global singletons used in the pipeline_deps
    module, ensuring that tests start with a clean state.
    """
    # Reset registry manager
    registry_manager.clear_all_contexts()
    
    # Reset global resolver cache
    global_resolver.clear_cache()
    
    # No explicit reset for semantic_matcher as it's stateless
    # If semantic_matcher ever becomes stateful, add reset here


class IsolatedTestCase(unittest.TestCase):
    """
    Base class for tests that need isolation from global state.
    
    This class automatically resets all global state before and after each test,
    ensuring that tests are properly isolated from each other.
    """
    
    def setUp(self):
        """Set up test fixtures, resetting global state."""
        reset_all_global_state()
    
    def tearDown(self):
        """Clean up after tests, resetting global state."""
        reset_all_global_state()


# Example usage:
"""
from test.pipeline_deps.test_helpers import IsolatedTestCase, reset_all_global_state

class TestMyFeature(IsolatedTestCase):
    def test_something(self):
        # This test starts with clean global state
        pass
        
    def test_something_else(self):
        # This test also starts with clean global state
        pass

# For existing test classes that can't inherit from IsolatedTestCase:
class ExistingTestClass(unittest.TestCase):
    def setUp(self):
        # Reset global state
        reset_all_global_state()
        # ... other setup code ...
    
    def tearDown(self):
        # ... other teardown code ...
        # Reset global state
        reset_all_global_state()
"""
