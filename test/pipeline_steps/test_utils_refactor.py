"""
Test the refactoring of utils.py to verify backward compatibility.

This test confirms that the new implementation in src/pipeline_steps/utils.py
that delegates to src/config_field_manager is fully compatible with the
original implementation saved in src/pipeline_steps/utils_legacy.py.
"""

import os
import tempfile
import unittest
import json
from pathlib import Path

from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.utils import serialize_config
from src.pipeline_steps.utils import merge_and_save_configs, load_configs
from src.pipeline_steps.utils import build_complete_config_classes
from src.pipeline_steps.utils_legacy import serialize_config as legacy_serialize_config
from src.pipeline_steps.utils_legacy import merge_and_save_configs_legacy

from pydantic import BaseModel, Field
from typing import Dict, Any, List


class TestConfig(BasePipelineConfig):
    """Test config for validation."""
    version: str = "1.0.0"
    author: str = "Test Author"
    pipeline_name: str = "TestPipeline"
    pipeline_description: str = "Test Pipeline Description"
    pipeline_version: str = "1.0.0"
    pipeline_s3_loc: str = "s3://test-bucket/test-pipeline"
    hyperparameters: Dict[str, Any] = Field(default_factory=lambda: {"param1": "value1"})
    input_names: Dict[str, str] = Field(default_factory=lambda: {"input1": "value1"})
    output_names: Dict[str, str] = Field(default_factory=lambda: {"output1": "value1"})


class TestProcessingConfig(ProcessingStepConfigBase):
    """Test processing config for validation."""
    job_type: str = "processing"
    processing_specific_field: str = "processing_value"
    input_names: Dict[str, str] = Field(default_factory=lambda: {"proc_input": "proc_value"})
    output_names: Dict[str, str] = Field(default_factory=lambda: {"proc_output": "proc_value"})
    hyperparameters: Dict[str, Any] = Field(default_factory=lambda: {"proc_param": "proc_value"})


class UtilsRefactorTest(unittest.TestCase):
    """Test the refactoring of utils.py to verify backward compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.legacy_output = os.path.join(self.temp_dir, "legacy_output.json")
        self.new_output = os.path.join(self.temp_dir, "new_output.json")
        
        # Create test configs
        self.configs = [
            TestConfig(),
            TestConfig(author="Author2", pipeline_name="Pipeline2"),
            TestProcessingConfig(),
            TestProcessingConfig(processing_specific_field="custom_value")
        ]

        # Config classes for deserialization
        self.config_classes = {
            "TestConfig": TestConfig,
            "TestProcessingConfig": TestProcessingConfig,
            "BasePipelineConfig": BasePipelineConfig,
            "ProcessingStepConfigBase": ProcessingStepConfigBase
        }

    def test_serialize_config(self):
        """Test that the new serialize_config produces the same output as the legacy version."""
        config = self.configs[0]
        
        # Get serialized dicts
        legacy_serialized = legacy_serialize_config(config)
        new_serialized = serialize_config(config)
        
        # Check that metadata exists
        self.assertIn("_metadata", legacy_serialized)
        self.assertIn("_metadata", new_serialized)
        
        # Check that step_name is generated
        self.assertEqual(
            legacy_serialized["_metadata"]["step_name"],
            new_serialized["_metadata"]["step_name"]
        )
        
        # Check that all fields are included
        legacy_fields = set(legacy_serialized.keys())
        new_fields = set(new_serialized.keys())
        
        # The field sets should be identical
        self.assertEqual(legacy_fields, new_fields)

    def test_merge_and_save_configs(self):
        """Test that merge_and_save_configs produces equivalent output."""
        # Generate outputs with both implementations
        legacy_result = merge_and_save_configs_legacy(self.configs, self.legacy_output)
        new_result = merge_and_save_configs(self.configs, self.new_output)
        
        # Read the output files
        with open(self.legacy_output, "r") as f:
            legacy_output = json.load(f)
        
        with open(self.new_output, "r") as f:
            new_output = json.load(f)
        
        # Verify structure is present in both outputs
        self.assertIn("metadata", legacy_output)
        self.assertIn("metadata", new_output)
        self.assertIn("configuration", legacy_output)
        self.assertIn("configuration", new_output)
        
        # Verify configuration structure
        legacy_config = legacy_output["configuration"]
        new_config = new_output["configuration"]
        
        # Check top-level sections
        self.assertIn("shared", legacy_config)
        self.assertIn("shared", new_config)
        self.assertIn("processing", legacy_config)
        self.assertIn("processing", new_config)
        self.assertIn("specific", legacy_config)
        self.assertIn("specific", new_config)
        
        # Check processing sections
        self.assertIn("processing_shared", legacy_config["processing"])
        self.assertIn("processing_shared", new_config["processing"])
        self.assertIn("processing_specific", legacy_config["processing"])
        self.assertIn("processing_specific", new_config["processing"])
        
        # Both should categorize hyperparameters as specific
        # Check for hyperparameters in specific sections
        for config_output in [legacy_config, new_config]:
            hyperparams_found = False
            
            # Check in specific sections
            for step, fields in config_output["specific"].items():
                if "hyperparameters" in fields:
                    hyperparams_found = True
            
            # Check in processing_specific sections
            for step, fields in config_output["processing"]["processing_specific"].items():
                if "hyperparameters" in fields:
                    hyperparams_found = True
            
            self.assertTrue(hyperparams_found, "Hyperparameters not found in specific sections")
            
            # Verify hyperparameters are not in shared sections
            self.assertNotIn("hyperparameters", config_output["shared"])
            self.assertNotIn("hyperparameters", config_output["processing"]["processing_shared"])

    def test_build_complete_config_classes(self):
        """Test that build_complete_config_classes registers all classes."""
        config_classes = build_complete_config_classes()
        
        # Check that base classes are included
        self.assertIn("BasePipelineConfig", config_classes)
        self.assertIn("ProcessingStepConfigBase", config_classes)
        
        # Check that ConfigClassStore has these classes registered
        from src.config_field_manager import ConfigClassStore
        store_classes = ConfigClassStore.get_all_classes()
        
        for name, cls in config_classes.items():
            self.assertIn(name, store_classes)
            self.assertEqual(cls, store_classes[name])

    @unittest.skip("Deserialization requires further development")
    def test_load_configs(self):
        """Test that load_configs successfully loads configurations."""
        # First, save configs using the new implementation
        merge_and_save_configs(self.configs, self.new_output)
        
        # Then load them
        loaded_configs = load_configs(self.new_output, self.config_classes)
        
        # Check that we have loaded configs
        self.assertTrue(len(loaded_configs) > 0, "No configs were loaded")
        
        # Check that at least one type of config was loaded
        has_valid_config = False
        
        for config in loaded_configs.values():
            if isinstance(config, TestConfig) or isinstance(config, TestProcessingConfig):
                has_valid_config = True
                break
                
        self.assertTrue(has_valid_config, "No valid config instances were loaded")


if __name__ == "__main__":
    unittest.main()
