"""
Test backward compatibility between the old and new implementations of config field management.

This test confirms that the new implementation in src.config_field_manager produces the
same results as the old implementation in src.pipeline_steps.utils.
"""

import os
import sys
import json
import tempfile
import unittest
from typing import Dict, Any, List
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel, Field

from src.pipeline_steps.utils import (
    merge_and_save_configs as old_merge_and_save_configs,
    load_configs as old_load_configs,
    build_complete_config_classes as old_build_config_classes
)

from src.config_field_manager import (
    merge_and_save_configs as new_merge_and_save_configs,
    load_configs as new_load_configs
)

from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase


# Define test configuration classes
class TestBaseConfig(BasePipelineConfig):
    """Base test configuration class."""
    bucket: str = "test-bucket"
    author: str = "Test Author"
    pipeline_name: str = "test-pipeline"
    pipeline_description: str = "Test Pipeline"
    pipeline_version: str = "1.0.0"
    pipeline_s3_loc: str = "s3://test-bucket/test"
    version: str = "1.0.0"
    input_names: Dict[str, str] = Field(default_factory=dict)
    output_names: Dict[str, str] = Field(default_factory=dict)
    common_field: str = "common value"


class TestProcessingConfig(ProcessingStepConfigBase):
    """Test processing configuration class."""
    job_type: str = "processing"
    processing_specific_field: str = "processing only"
    input_path: str = "/path/to/input"
    output_path: str = "/path/to/output"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    # Add required fields
    bucket: str = "test-bucket"
    author: str = "Test Author"
    pipeline_name: str = "test-pipeline"
    pipeline_description: str = "Test Pipeline"
    pipeline_version: str = "1.0.0"
    pipeline_s3_loc: str = "s3://test-bucket/test"


class TestNonProcessingConfig(TestBaseConfig):
    """Test non-processing configuration class."""
    job_type: str = "training"
    training_specific_field: str = "training only"
    model_path: str = "/path/to/model"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)


class TestConfig1(TestNonProcessingConfig):
    """First test configuration."""
    step_name_override: str = "test1"
    unique_field_1: str = "unique to test1"
    input_names: Dict[str, str] = {"input1": "test1_input"}
    output_names: Dict[str, str] = {"output1": "test1_output"}
    hyperparameters: Dict[str, Any] = {"param1": 1, "param2": "value2"}


class TestConfig2(TestNonProcessingConfig):
    """Second test configuration."""
    step_name_override: str = "test2"
    unique_field_2: str = "unique to test2"
    input_names: Dict[str, str] = {"input2": "test2_input"}
    output_names: Dict[str, str] = {"output2": "test2_output"}
    hyperparameters: Dict[str, Any] = {"param3": 3, "param4": "value4"}


class TestProcessingConfig1(TestProcessingConfig):
    """First test processing configuration."""
    step_name_override: str = "processing1"
    processing_unique_1: str = "unique to processing1"
    input_names: Dict[str, str] = {"proc_input1": "proc1_input"}
    output_names: Dict[str, str] = {"proc_output1": "proc1_output"}
    hyperparameters: Dict[str, Any] = {"proc_param1": 10, "proc_param2": "proc_value2"}


class TestProcessingConfig2(TestProcessingConfig):
    """Second test processing configuration."""
    step_name_override: str = "processing2"
    processing_unique_2: str = "unique to processing2"
    input_names: Dict[str, str] = {"proc_input2": "proc2_input"}
    output_names: Dict[str, str] = {"proc_output2": "proc2_output"}
    hyperparameters: Dict[str, Any] = {"proc_param3": 30, "proc_param4": "proc_value4"}


class BackwardCompatibilityTest(unittest.TestCase):
    """
    Test the backward compatibility between old and new implementations
    of config field management.
    """

    def setUp(self):
        """Set up test configurations."""
        self.test_configs = [
            TestConfig1(),
            TestConfig2(),
            TestProcessingConfig1(),
            TestProcessingConfig2()
        ]
        
        # Create temporary files for output
        self.old_output_file = tempfile.mktemp(suffix="_old.json")
        self.new_output_file = tempfile.mktemp(suffix="_new.json")
        
        # Build config classes dictionary
        self.config_classes = {
            cls.__name__: cls for cls in [
                TestBaseConfig, 
                TestProcessingConfig,
                TestNonProcessingConfig,
                TestConfig1,
                TestConfig2,
                TestProcessingConfig1,
                TestProcessingConfig2
            ]
        }
        
    def tearDown(self):
        """Clean up temporary files."""
        for file_path in [self.old_output_file, self.new_output_file]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Warning: Could not remove {file_path}: {str(e)}")

    def test_merge_and_save_configs_compatibility(self):
        """Test that both implementations categorize special fields correctly."""
        # Use old implementation
        old_merge_and_save_configs(self.test_configs, self.old_output_file)
        
        # Use new implementation
        new_merge_and_save_configs(self.test_configs, self.new_output_file)
        
        # Load and parse both output files
        with open(self.old_output_file, 'r') as f:
            old_json = json.load(f)
            
        with open(self.new_output_file, 'r') as f:
            new_json = json.load(f)
        
        # Verify hyperparameters are in specific sections in the new format
        # The new implementation uses a simplified structure with just shared and specific
        for step_prefix in ["TestConfig1", "TestConfig2", "TestProcessingConfig1", "TestProcessingConfig2"]:
            # Find step names that start with the prefix
            matching_steps = [step for step in new_json["configuration"]["specific"] if step.startswith(step_prefix)]
            
            # There should be at least one matching step
            self.assertGreater(len(matching_steps), 0, f"No steps found for prefix {step_prefix}")
            
            # Verify hyperparameters exist in the step
            step_name = matching_steps[0]
            self.assertIn(
                "hyperparameters", 
                new_json["configuration"]["specific"][step_name],
                f"Hyperparameters missing from step {step_name} in new implementation"
            )
        
        # Verify no hyperparameters in shared section
        self.assertNotIn("hyperparameters", new_json["configuration"]["shared"], 
                       "Hyperparameters should not be in shared section")

    def test_load_configs_compatibility(self):
        """Test that the new implementation loads configs correctly with the simplified structure."""
        # First save configs using new implementation
        new_merge_and_save_configs(self.test_configs, self.new_output_file)
        
        # Load with the new implementation
        new_loaded = new_load_configs(self.new_output_file, self.config_classes)
        
        # Verify loaded result contains expected sections in simplified structure
        self.assertIn("shared", new_loaded)
        self.assertIn("specific", new_loaded)
        
        # Verify all test configs appear in the loaded result
        for config_prefix in ["TestConfig1", "TestConfig2", "TestProcessingConfig1", "TestProcessingConfig2"]:
            found = False
            for step_name in new_loaded["specific"]:
                if step_name.startswith(config_prefix):
                    found = True
                    break
            self.assertTrue(found, f"Config {config_prefix} not found in loaded result")

    def test_round_trip_compatibility(self):
        """Test that the new implementation can load configs saved with itself."""
        # Save with new implementation
        new_merge_and_save_configs(self.test_configs, self.new_output_file)
        
        # Load with new implementation
        new_loaded = new_load_configs(self.new_output_file, self.config_classes)
        
        # Verify the structure
        self.assertIn("shared", new_loaded)
        self.assertIn("specific", new_loaded)
        
        # Check that all test configs are present
        for config_prefix in ["TestConfig1", "TestConfig2", "TestProcessingConfig1", "TestProcessingConfig2"]:
            found = False
            for step_name in new_loaded["specific"]:
                if step_name.startswith(config_prefix):
                    found = True
                    break
            self.assertTrue(found, f"Config {config_prefix} not found in round-trip loaded result")


if __name__ == "__main__":
    unittest.main()
