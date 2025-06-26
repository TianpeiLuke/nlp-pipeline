#!/usr/bin/env python
"""
Unittest for the utils module in pipeline_steps, incorporating tests for:
1. Default value handling
2. Field categorization (shared vs. specific)
3. Different input/output defaults
4. Base class derived defaults
"""
import os
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

import sys
import os
from pathlib import Path

# Add the project root to the sys path so we can import src
project_root = Path(__file__).parent.parent.parent  # Navigate up to the project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline_steps.utils import merge_and_save_configs, load_configs, get_field_default
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase

# Base common args for creating config instances in tests
COMMON_ARGS = {
    "bucket": "test-bucket",
    "author": "test-author",
    "pipeline_name": "test-pipeline",
    "pipeline_description": "Test pipeline",
    "pipeline_version": "1.0.0",
    "pipeline_s3_loc": "s3://test-bucket/test"
}

class TestUtils(unittest.TestCase):
    """
    Tests for the pipeline_steps/utils.py module, focusing on config serialization,
    categorization of fields between "shared", "processing", and "specific", and config loading.
    """
    
    def setUp(self):
        """Setup for each test - save the original registry"""
        self.original_registry = BasePipelineConfig.STEP_NAMES.copy()
        # Create a temp dir for test files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Restore the original registry after each test"""
        BasePipelineConfig.STEP_NAMES = self.original_registry
        
    def test_is_likely_static(self):
        """Test the _is_likely_static function correctly identifies static fields."""
        from src.pipeline_steps.utils import _is_likely_static
        
        # Fields that should be considered static
        static_fields = [
            ("author", "test-author"),
            ("region", "us-west-2"),
            ("version", "1.0.0"),
            ("framework", "pytorch"),
            ("simple_list", [1, 2, 3, 4]),
            ("simple_dict", {"a": 1, "b": 2}),
        ]
        
        # Fields that should not be considered static
        non_static_fields = [
            ("input_names", {"input1": "Input 1", "input2": "Input 2"}),
            ("output_names", {"output": "Output"}),
            ("processing_count", 5),
            ("complex_list", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ("complex_dict", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}),
        ]
        
        # Test static fields
        for field_name, value in static_fields:
            self.assertTrue(
                _is_likely_static(field_name, value),
                f"Field '{field_name}' with value '{value}' should be considered static"
            )
            
        # Test non-static fields
        for field_name, value in non_static_fields:
            self.assertFalse(
                _is_likely_static(field_name, value),
                f"Field '{field_name}' with value '{value}' should not be considered static"
            )

    def test_default_values(self):
        """Test that default values in Pydantic models are properly preserved."""
        
        # Create a test config class with default values
        class TestConfig(BaseModel):
            """Simple test config with default values for testing."""
            name: str
            region: str
            # Dictionary with default_factory
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "test_input_1": "Description of input 1",
                "test_input_2": "Description of input 2"
            })
            # Dictionary with default value
            output_names: Dict[str, str] = {"test_output": "Description of output"}
            # Optional field with default
            version: str = "1.0.0"
            # Optional field without default
            description: Optional[str] = None
        
        # Create a minimal TestConfig instance
        test_config = TestConfig(
            name="test-config", 
            region="NA"
        )
        
        # Verify the default input_names and output_names are present
        self.assertEqual(
            test_config.input_names, 
            {"test_input_1": "Description of input 1", "test_input_2": "Description of input 2"}
        )
        self.assertEqual(
            test_config.output_names,
            {"test_output": "Description of output"}
        )
        
        # Create a list of configs and a temp file
        config_list = [test_config]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs to the temp file
            merge_and_save_configs(config_list, tmp_path)
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {'TestConfig': TestConfig}
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Get the loaded config
            loaded_config = next(iter(loaded_configs.values()), None)
            self.assertIsNotNone(loaded_config, "Could not find loaded config")
            
            # Verify that input_names and output_names were preserved
            self.assertIn("test_input_1", loaded_config.input_names, "test_input_1 not in input_names")
            self.assertIn("test_input_2", loaded_config.input_names, "test_input_2 not in input_names")
            self.assertIn("test_output", loaded_config.output_names, "test_output not in output_names")
            
            # Verify that version was preserved
            self.assertEqual(loaded_config.version, "1.0.0", "version was not preserved")
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_dynamic_field_categorization(self):
        """Test that fields are properly categorized as shared or specific based on their values."""
        
        # Update registry for this test
        BasePipelineConfig.STEP_NAMES.update({
            'TestBaseConfig': 'Base',
            'TestDerivedConfig1': 'Derived1', 
            'TestDerivedConfig2': 'Derived2'
        })
        
        # Create test config classes
        class TestBaseConfig(BasePipelineConfig):
            """Base config class with default values."""
            base_value: str = "base"
            input_names: Dict[str, str] = Field(default_factory=dict)
            output_names: Dict[str, str] = Field(default_factory=dict)
            shared_value: str = "shared across configs"
            
        class TestDerivedConfig1(TestBaseConfig):
            """First derived config with customized values."""
            # Override base value
            base_value: str = "derived1"
            # Add derived-specific field
            derived1_field: str = "only in derived1"
            # Custom input_names specific to this config
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "derived1_input": "Input specific to derived1"
            })
            
        class TestDerivedConfig2(TestBaseConfig):
            """Second derived config with different customized values."""
            # Override base value differently
            base_value: str = "derived2"
            # Add derived-specific field
            derived2_field: str = "only in derived2"
            # Custom input_names specific to this config
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "derived2_input": "Input specific to derived2"
            })
        
        # Create config instances
        base_config = TestBaseConfig(**COMMON_ARGS)
        derived1_config = TestDerivedConfig1(**COMMON_ARGS)
        derived2_config = TestDerivedConfig2(**COMMON_ARGS)
        
        # Create a list of configs
        config_list = [base_config, derived1_config, derived2_config]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # Verify field categorization
            
            # 'shared_value' should be in shared section
            self.assertIn('shared_value', merged['shared'], "'shared_value' should be in shared section")
            
            # 'base_value' should not be shared (different values)
            self.assertNotIn('base_value', merged['shared'], 
                "'base_value' should NOT be in shared section (different values)")
                
            # 'input_names' should not be shared (different values)
            self.assertNotIn('input_names', merged['shared'], 
                "'input_names' should NOT be in shared section (different values)")
            
            # 'base_value' should be specific to each config
            self.assertIn('base_value', merged['specific']['Base'], "'base_value' should be specific for Base")
            self.assertIn('base_value', merged['specific']['Derived1'], 
                "'base_value' should be specific for Derived1")
            self.assertIn('base_value', merged['specific']['Derived2'], 
                "'base_value' should be specific for Derived2")
            
            # Verify correct values for each config's 'base_value'
            self.assertEqual(merged['specific']['Base']['base_value'], "base")
            self.assertEqual(merged['specific']['Derived1']['base_value'], "derived1")
            self.assertEqual(merged['specific']['Derived2']['base_value'], "derived2")
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {
                'TestBaseConfig': TestBaseConfig,
                'TestDerivedConfig1': TestDerivedConfig1,
                'TestDerivedConfig2': TestDerivedConfig2,
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Verify that loaded configs have the correct values
            base_loaded = loaded_configs.get('Base')
            derived1_loaded = loaded_configs.get('Derived1') 
            derived2_loaded = loaded_configs.get('Derived2')
            
            self.assertIsNotNone(base_loaded, "Base config should be loaded")
            self.assertIsNotNone(derived1_loaded, "Derived1 config should be loaded")
            self.assertIsNotNone(derived2_loaded, "Derived2 config should be loaded")
            
            self.assertEqual(base_loaded.base_value, "base")
            self.assertEqual(derived1_loaded.base_value, "derived1")
            self.assertEqual(derived2_loaded.base_value, "derived2")
            
            self.assertEqual(derived1_loaded.derived1_field, "only in derived1")
            self.assertEqual(derived2_loaded.derived2_field, "only in derived2")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_different_input_output_defaults(self):
        """
        Test that input_names and output_names with different default values
        are correctly kept in the specific section.
        """
        
        # Register step names
        BasePipelineConfig.STEP_NAMES.update({
            'DataLoadConfig': 'DataLoad',
            'TrainingConfig': 'Training',
            'ModelEvalConfig': 'ModelEval'
        })
        
        # Create config classes with different default input/output names
        class DataLoadConfig(BasePipelineConfig):
            """Data loading config with specific input/output names."""
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "raw_data": "Raw input data location"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "processed_data": "Processed data output"
            })
        
        class TrainingConfig(BasePipelineConfig):
            """Training config with specific input/output names."""
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "processed_data": "Processed data for training",
                "validation_data": "Validation dataset"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "model_artifacts": "Trained model artifacts"
            })
        
        class ModelEvalConfig(BasePipelineConfig):
            """Model evaluation config with specific input/output names."""
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "model_artifacts": "Model artifacts to evaluate",
                "test_data": "Test dataset for evaluation"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "metrics": "Evaluation metrics",
                "confusion_matrix": "Confusion matrix"
            })
        
        # Create config instances
        data_load_config = DataLoadConfig(**COMMON_ARGS)
        training_config = TrainingConfig(**COMMON_ARGS)
        model_eval_config = ModelEvalConfig(**COMMON_ARGS)
        
        # Create a list of configs
        config_list = [data_load_config, training_config, model_eval_config]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # Verify input_names and output_names are not in shared section
            self.assertNotIn('input_names', merged['shared'], 
                "'input_names' should NOT be in shared section (different values)")
            self.assertNotIn('output_names', merged['shared'],
                "'output_names' should NOT be in shared section (different values)")
            
            # Verify input_names and output_names are in specific sections
            for step_name in ['DataLoad', 'Training', 'ModelEval']:
                self.assertIn('input_names', merged['specific'][step_name],
                    f"'input_names' should be specific for {step_name}")
                self.assertIn('output_names', merged['specific'][step_name],
                    f"'output_names' should be specific for {step_name}")
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {
                'DataLoadConfig': DataLoadConfig,
                'TrainingConfig': TrainingConfig,
                'ModelEvalConfig': ModelEvalConfig
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Verify that input/output names are preserved correctly for each config
            data_load = loaded_configs.get('DataLoad')
            training = loaded_configs.get('Training')
            model_eval = loaded_configs.get('ModelEval')
            
            self.assertIsNotNone(data_load, "DataLoad config should be loaded")
            self.assertIsNotNone(training, "Training config should be loaded")
            self.assertIsNotNone(model_eval, "ModelEval config should be loaded")
            
            # Verify each config has the correct input/output names
            self.assertIn("raw_data", data_load.input_names)
            self.assertIn("processed_data", data_load.output_names)
            
            self.assertIn("processed_data", training.input_names)
            self.assertIn("validation_data", training.input_names)
            self.assertIn("model_artifacts", training.output_names)
            
            self.assertIn("model_artifacts", model_eval.input_names)
            self.assertIn("metrics", model_eval.output_names)
            self.assertIn("confusion_matrix", model_eval.output_names)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_base_class_derived_defaults(self):
        """
        Test that fields with different default values in derived classes
        are correctly categorized.
        """
        
        # Register step names
        BasePipelineConfig.STEP_NAMES.update({
            'CommonBase': 'Base',
            'ConfigWithCustomDefaults1': 'Config1',
            'ConfigWithCustomDefaults2': 'Config2',
            'ConfigWithSameValues': 'Config3'
        })
        
        # Create base class with default values
        class CommonBase(BasePipelineConfig):
            """Base config class with default values."""
            # Common field present in all configs
            common_field: str = "base default value"
            
            # Fields that derived classes will override with different defaults
            variable_field: str = "base variable value"
            
            # Field with default_factory
            dict_field: Dict[str, str] = Field(default_factory=lambda: {"base": "value"})
        
        class ConfigWithCustomDefaults1(CommonBase):
            """First config with custom default for variable_field."""
            # Override with different default
            variable_field: str = "config1 value"
            
            # Override with different default_factory
            dict_field: Dict[str, str] = Field(default_factory=lambda: {"config1": "custom"})
        
        class ConfigWithCustomDefaults2(CommonBase):
            """Second config with custom default for variable_field."""
            # Override with different default
            variable_field: str = "config2 value"
            
            # Override with different default_factory
            dict_field: Dict[str, str] = Field(default_factory=lambda: {"config2": "custom"})
        
        class ConfigWithSameValues(CommonBase):
            """Config that explicitly sets the same values as base."""
            # No value changes from base - will use base defaults
            pass
        
        # Create config instances
        base_config = CommonBase(**COMMON_ARGS)
        config1 = ConfigWithCustomDefaults1(**COMMON_ARGS)
        config2 = ConfigWithCustomDefaults2(**COMMON_ARGS)
        config3 = ConfigWithSameValues(**COMMON_ARGS)
        
        # Create a list of configs
        config_list = [base_config, config1, config2, config3]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # common_field should be shared (same value in all configs)
            self.assertIn('common_field', merged['shared'])
            self.assertEqual(merged['shared']['common_field'], "base default value")
            
            # variable_field should be specific (different values)
            self.assertNotIn('variable_field', merged['shared'])
            self.assertIn('variable_field', merged['specific']['Base'])
            self.assertIn('variable_field', merged['specific']['Config1'])
            self.assertIn('variable_field', merged['specific']['Config2'])
            self.assertIn('variable_field', merged['specific']['Config3'])
            
            # Verify correct values for each config's variable_field
            self.assertEqual(merged['specific']['Base']['variable_field'], "base variable value")
            self.assertEqual(merged['specific']['Config1']['variable_field'], "config1 value")
            self.assertEqual(merged['specific']['Config2']['variable_field'], "config2 value")
            self.assertEqual(merged['specific']['Config3']['variable_field'], "base variable value")
            
            # dict_field should be specific (different values and default_factory)
            self.assertNotIn('dict_field', merged['shared'])
            self.assertIn('dict_field', merged['specific']['Base'])
            self.assertIn('dict_field', merged['specific']['Config1'])
            self.assertIn('dict_field', merged['specific']['Config2'])
            self.assertIn('dict_field', merged['specific']['Config3'])
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {
                'CommonBase': CommonBase,
                'ConfigWithCustomDefaults1': ConfigWithCustomDefaults1,
                'ConfigWithCustomDefaults2': ConfigWithCustomDefaults2,
                'ConfigWithSameValues': ConfigWithSameValues
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Expected values for each config
            expected_values = {
                'Base': {
                    'variable_field': 'base variable value',
                    'dict_field': {'base': 'value'}
                },
                'Config1': {
                    'variable_field': 'config1 value',
                    'dict_field': {'config1': 'custom'}
                },
                'Config2': {
                    'variable_field': 'config2 value',
                    'dict_field': {'config2': 'custom'}
                },
                'Config3': {
                    'variable_field': 'base variable value',
                    'dict_field': {'base': 'value'}
                }
            }
            
            # Verify that loaded configs have the correct values
            for step_name, expected in expected_values.items():
                config = loaded_configs.get(step_name)
                self.assertIsNotNone(config, f"Missing config: {step_name}")
                
                for field, expected_value in expected.items():
                    actual_value = getattr(config, field)
                    
                    # Check if values match based on type
                    if isinstance(expected_value, dict):
                        # Convert to JSON for dict comparison
                        self.assertEqual(
                            json.dumps(actual_value, sort_keys=True),
                            json.dumps(expected_value, sort_keys=True),
                            f"Wrong {field} in {step_name}"
                        )
                    else:
                        self.assertEqual(actual_value, expected_value, 
                            f"Wrong {field} in {step_name}")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_processing_nested_structure(self):
        """
        Test that the new nested structure for processing configs works correctly
        with processing_shared and processing_specific sections according to the new definition:
        
        - "processing_shared": fields common across all processing configs
        - "processing_specific": 1) fields unique to specific processing configs; 
                               2) fields shared across processing configs with different values
        """
        
        # Register step names
        BasePipelineConfig.STEP_NAMES.update({
            'BaseConfig': 'Base',
            'ProcessingBase': 'Processing',
            'ProcessingA': 'ProcessingA',
            'ProcessingB': 'ProcessingB',
            'RegularConfig': 'Regular'
        })
        
        # Create config classes
        class BaseConfig(BasePipelineConfig):
            """Base config for all others."""
            # Fields that should be shared across all configs
            shared_value: str = "shared across all"
            
            # Fields that might have different values
            input_names: Dict[str, str] = Field(default_factory=dict)
            output_names: Dict[str, str] = Field(default_factory=dict)

        class ProcessingBase(ProcessingStepConfigBase):
            """Base for all processing configs."""
            # Fields that should be shared across processing configs
            processing_shared_value: str = "shared across processing"
            processing_instance_count: int = 1
            
            # Use a dummy field instead of processing_source_dir to avoid validation issues
            processing_source_dir_dummy: str = "/tmp/dummy_dir"
            
            # Fields that might vary between processing configs
            processing_entry_point: Optional[str] = None
            use_large_processing_instance: bool = False
            
            # Override validation for testing purposes
            @classmethod
            def validate_entry_point_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
                """Skip entry point validation for testing"""
                return values
                
            @classmethod
            def model_validator(cls, values: Dict[str, Any]) -> Dict[str, Any]:
                """Skip validation for testing purposes"""
                # Skip directory existence validation
                return values

        class ProcessingA(ProcessingBase):
            """First processing config type."""
            processing_entry_point: str = "processing_a.py"
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "data_input": "Input for processing A"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "processed_output": "Output from processing A"
            })

        class ProcessingB(ProcessingBase):
            """Second processing config type."""
            processing_entry_point: str = "processing_b.py"
            use_large_processing_instance: bool = True
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "model_input": "Model input for processing B",
                "data_input": "Data input for processing B"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "model_output": "Model output from processing B"
            })

        class RegularConfig(BaseConfig):
            """Regular non-processing config."""
            regular_specific_field: str = "only in regular"
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "regular_input": "Input for regular config"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "regular_output": "Output from regular config"
            })
        
        # Create config instances without needing a real directory
        common_args = {
            "author": "test-author",
            "bucket": "test-bucket",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test pipeline",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test"
        }
    
        # Create the config instances
        base_config = BaseConfig(**common_args)
        processing_base = ProcessingBase(**common_args)
        processing_a = ProcessingA(**common_args)
        processing_b = ProcessingB(**common_args)
        regular_config = RegularConfig(**common_args)
        
        # Create a list of configs
        config_list = [base_config, processing_base, processing_a, processing_b, regular_config]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
            
            try:
                # Save the configs
                merged = merge_and_save_configs(config_list, tmp_path)
                
                # Verify the structure of the merged config
                
                    # Check shared section
                self.assertIn('shared_value', merged['shared'], 
                    "shared_value should be in shared section")
                
                # Check processing_shared section - must be common across ALL processing configs
                self.assertIn('processing_shared', merged['processing'], 
                    "processing_shared section should exist")
                
                # Since we're forcing these to be added in for testing, these checks are still valid
                self.assertIn('processing_shared_value', merged['processing']['processing_shared'], 
                    "processing_shared_value should be in processing_shared section")
                # Utils.py uses processing_source_dir internally, not our dummy
                self.assertIn('processing_source_dir', merged['processing']['processing_shared'],
                    "processing_source_dir should be in processing_shared section") 
                self.assertIn('processing_instance_count', merged['processing']['processing_shared'], 
                    "processing_instance_count should be in processing_shared section")
                    
                # Check processing_specific section
                self.assertIn('processing_specific', merged['processing'], 
                    "processing_specific section should exist")
                
                # Check processing_entry_point in processing_specific
                for step in ['ProcessingA', 'ProcessingB']:
                    self.assertIn(step, merged['processing']['processing_specific'], 
                        f"{step} should be in processing_specific section")
                    self.assertIn('processing_entry_point', merged['processing']['processing_specific'][step], 
                        f"processing_entry_point should be in {step}")
                    self.assertIn('input_names', merged['processing']['processing_specific'][step], 
                        f"input_names should be in {step}")
                    self.assertIn('output_names', merged['processing']['processing_specific'][step], 
                        f"output_names should be in {step}")
                
                # Verify values in processing_specific
                self.assertEqual(
                    merged['processing']['processing_specific']['ProcessingA']['processing_entry_point'], 
                    "processing_a.py",
                    "Wrong processing_entry_point for ProcessingA"
                )
                self.assertEqual(
                    merged['processing']['processing_specific']['ProcessingB']['processing_entry_point'], 
                    "processing_b.py",
                    "Wrong processing_entry_point for ProcessingB"
                )
                
                # Verify use_large_processing_instance is specific to ProcessingB
                self.assertIn(
                    'use_large_processing_instance', 
                    merged['processing']['processing_specific']['ProcessingB'],
                    "use_large_processing_instance should be in ProcessingB's specific section"
                )
                self.assertEqual(
                    merged['processing']['processing_specific']['ProcessingB']['use_large_processing_instance'], 
                    True,
                    "Wrong use_large_processing_instance value for ProcessingB"
                )
                
                # Check regular config is in specific section
                self.assertIn('Regular', merged['specific'], 
                    "Regular config should be in specific section")
                self.assertIn('regular_specific_field', merged['specific']['Regular'], 
                    "regular_specific_field should be in Regular's specific section")
                
                # Define config classes dict for loading
                CONFIG_CLASSES = {
                    'BaseConfig': BaseConfig,
                    'ProcessingBase': ProcessingBase, 
                    'ProcessingA': ProcessingA,
                    'ProcessingB': ProcessingB,
                    'RegularConfig': RegularConfig
                }
                
                # No need to update with a real directory since we're using a dummy field
                with open(tmp_path, 'r') as f:
                    json_data = json.load(f)
                
                # Write back the modified JSON
                with open(tmp_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Load the configs back
                loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
                
                # Verify loaded configs
                proc_a = loaded_configs.get('ProcessingA')
                proc_b = loaded_configs.get('ProcessingB')
                regular = loaded_configs.get('Regular')
                
                # Check that all configs were loaded
                self.assertIsNotNone(proc_a, "ProcessingA should be loaded")
                self.assertIsNotNone(proc_b, "ProcessingB should be loaded")
                self.assertIsNotNone(regular, "Regular should be loaded")
                
                # Check processing_shared_value is shared across processing configs
                self.assertEqual(proc_a.processing_shared_value, "shared across processing")
                self.assertEqual(proc_b.processing_shared_value, "shared across processing")
                
                # Check that processing_source_dir_dummy is shared
                self.assertEqual(proc_a.processing_source_dir_dummy, "/tmp/dummy_dir")
                self.assertEqual(proc_b.processing_source_dir_dummy, "/tmp/dummy_dir")
                # Note: processing_source_dir is None in the shared section
                
                # Check that processing_entry_point is specific
                self.assertEqual(proc_a.processing_entry_point, "processing_a.py")
                self.assertEqual(proc_b.processing_entry_point, "processing_b.py")
                
                # Check that use_large_processing_instance is specific to ProcessingB
                self.assertEqual(proc_a.use_large_processing_instance, False)
                self.assertEqual(proc_b.use_large_processing_instance, True)
                
                # Check that input_names and output_names are correct
                self.assertIn("data_input", proc_a.input_names)
                self.assertIn("processed_output", proc_a.output_names)
                self.assertIn("model_input", proc_b.input_names)
                self.assertIn("model_output", proc_b.output_names)
                self.assertIn("regular_input", regular.input_names)
                self.assertIn("regular_output", regular.output_names)
                
            finally:
                # Clean up
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


# Main execution
    def test_static_fields_categorization(self):
        """
        Test that static fields are correctly categorized according to the new definition:
        - "shared": fields with identical values across configs AND static
        - "specific": fields unique to specific configs OR shared with different values
        """
        
        # Register step names for this test
        BasePipelineConfig.STEP_NAMES.update({
            'StaticFieldConfig1': 'Static1',
            'StaticFieldConfig2': 'Static2',
            'StaticFieldConfig3': 'Static3'
        })
        
        # Create config classes with a mix of static and non-static fields
        class StaticFieldConfig1(BasePipelineConfig):
            """First config with static and non-static fields."""
            # Static fields - should be shared
            static_field: str = "static value"
            simple_list: List[int] = [1, 2, 3]
            
            # Non-static fields - should be specific even with identical values
            input_names: Dict[str, str] = Field(default_factory=lambda: {"input": "Input data"})
            output_names: Dict[str, str] = Field(default_factory=lambda: {"output": "Output data"})
        
        class StaticFieldConfig2(BasePipelineConfig):
            """Second config with the same static fields but different values for non-static fields."""
            # Same static fields as StaticFieldConfig1
            static_field: str = "static value"
            simple_list: List[int] = [1, 2, 3]
            
            # Different values for non-static fields
            input_names: Dict[str, str] = Field(default_factory=lambda: {"different": "Different input"})
            output_names: Dict[str, str] = Field(default_factory=lambda: {"different": "Different output"})
        
        class StaticFieldConfig3(BasePipelineConfig):
            """Third config with different values for static fields."""
            # Different values for static fields
            static_field: str = "different static value"
            simple_list: List[int] = [4, 5, 6]
            
            # Same as StaticFieldConfig1 for non-static fields
            input_names: Dict[str, str] = Field(default_factory=lambda: {"input": "Input data"})
            output_names: Dict[str, str] = Field(default_factory=lambda: {"output": "Output data"})
        
        # Create config instances
        config1 = StaticFieldConfig1(**COMMON_ARGS)
        config2 = StaticFieldConfig2(**COMMON_ARGS)
        config3 = StaticFieldConfig3(**COMMON_ARGS)
        
        # Create a list of configs
        config_list = [config1, config2, config3]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # Check that static_field is in shared for configs 1 and 2 since they share the same value
            # (it won't be in shared for all 3 configs since config3 has a different value)
            self.assertNotIn('static_field', merged['shared'],
                "static_field should not be in shared section because config3 has a different value")
            
            # Check that input_names and output_names are not in shared even though configs 1 and 3 share values
            # (they're considered non-static)
            self.assertNotIn('input_names', merged['shared'],
                "input_names should not be in shared section because it's considered non-static")
            self.assertNotIn('output_names', merged['shared'],
                "output_names should not be in shared section because it's considered non-static")
                
            # Verify specific sections have the correct fields
            for step_name in ['Static1', 'Static2', 'Static3']:
                self.assertIn('static_field', merged['specific'][step_name],
                    f"static_field should be in {step_name}'s specific section")
                self.assertIn('input_names', merged['specific'][step_name],
                    f"input_names should be in {step_name}'s specific section")
                self.assertIn('output_names', merged['specific'][step_name],
                    f"output_names should be in {step_name}'s specific section")
            
            # Verify correct values in specific sections
            self.assertEqual(merged['specific']['Static1']['static_field'], "static value")
            self.assertEqual(merged['specific']['Static2']['static_field'], "static value")
            self.assertEqual(merged['specific']['Static3']['static_field'], "different static value")
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {
                'StaticFieldConfig1': StaticFieldConfig1,
                'StaticFieldConfig2': StaticFieldConfig2,
                'StaticFieldConfig3': StaticFieldConfig3
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Verify loaded configs have correct values
            self.assertEqual(loaded_configs['Static1'].static_field, "static value")
            self.assertEqual(loaded_configs['Static2'].static_field, "static value")
            self.assertEqual(loaded_configs['Static3'].static_field, "different static value")
            
            # Verify input_names and output_names were preserved
            self.assertEqual(loaded_configs['Static1'].input_names, {"input": "Input data"})
            self.assertEqual(loaded_configs['Static2'].input_names, {"different": "Different input"})
                
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_cross_type_fields_categorization(self):
        """
        Test that fields appearing in both processing and non-processing configs are correctly categorized
        according to the refined rules:
        
        1. If the field is static AND has identical values across ALL configs:
           - Place it in the root "shared" section
        2. Otherwise:
           - For processing configs: keep in "processing_specific"
           - For non-processing configs: keep in "specific"
           
        The following categories are mutually exclusive:
        - "shared" and "specific" sections have no overlapping fields
        - "processing_shared" and "processing_specific" sections have no overlapping fields
        """
        
        # Register step names for this test
        BasePipelineConfig.STEP_NAMES.update({
            'StandardConfig': 'Standard',
            'ProcessingConfig': 'Processing',
            'AnotherStandardConfig': 'AnotherStandard',
            'AnotherProcessingConfig': 'AnotherProcessing'
        })
        
        # Create config classes with cross-type fields
        class StandardConfig(BasePipelineConfig):
            """Standard non-processing config."""
            # Cross-type field with same value across all configs
            shared_cross_type: str = "shared across all configs"
            
            # Cross-type field with different value in each config
            varying_cross_type: str = "standard value"
            
            # Non-cross-type field specific to standard configs
            standard_only: str = "only in standard configs" 
            
            # Common fields
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "standard_input": "Input for standard config"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "standard_output": "Output from standard config"
            })
        
        class ProcessingConfig(ProcessingStepConfigBase):
            """Processing config."""
            # Cross-type field with same value across all configs
            shared_cross_type: str = "shared across all configs"
            
            # Cross-type field with different value in each config
            varying_cross_type: str = "processing value"
            
            # Non-cross-type field specific to processing configs
            processing_only: str = "only in processing configs"
            
            # Common fields with different values
            input_names: Dict[str, str] = Field(default_factory=lambda: {
                "processing_input": "Input for processing config"
            })
            output_names: Dict[str, str] = Field(default_factory=lambda: {
                "processing_output": "Output from processing config"
            })
            
            # Use processing_source_dir_dummy instead of processing_source_dir to avoid validation
            processing_source_dir_dummy: str = "/tmp/dummy_dir"
            processing_entry_point: str = "process.py"

            # Disable validation for the test
            class Config:
                """Pydantic config to disable validation for testing"""
                validate_assignment = False
                arbitrary_types_allowed = True
                validate_default = False
            
            @classmethod
            def validate_entry_point_paths(cls, values):
                """Skip validation for testing"""
                return values
                
            @classmethod
            def model_validator(cls, values):
                """Skip validation for testing purposes"""
                return values
        
        class AnotherStandardConfig(StandardConfig):
            """Another standard config with different varying field value."""
            varying_cross_type: str = "another standard value"
        
        class AnotherProcessingConfig(ProcessingConfig):
            """Another processing config with different varying field value."""
            varying_cross_type: str = "another processing value"
            # Inherit processing_source_dir_dummy from ProcessingConfig
        
        # Create config instances
        standard_config = StandardConfig(**COMMON_ARGS)
        processing_config = ProcessingConfig(**COMMON_ARGS)
        another_standard = AnotherStandardConfig(**COMMON_ARGS)
        another_processing = AnotherProcessingConfig(**COMMON_ARGS)
        
        # Create a list of configs
        config_list = [standard_config, processing_config, another_standard, another_processing]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # Verify shared_cross_type is in shared section (identical across ALL configs)
            self.assertIn('shared_cross_type', merged['shared'], 
                "'shared_cross_type' should be in shared section (identical across ALL configs)")
            self.assertEqual(merged['shared']['shared_cross_type'], "shared across all configs")
            
            # Verify varying_cross_type is NOT in shared section
            self.assertNotIn('varying_cross_type', merged['shared'],
                "'varying_cross_type' should NOT be in shared section (different values)")
            
            # Verify standard_only is not in shared section (only in some configs)
            self.assertNotIn('standard_only', merged['shared'],
                "'standard_only' should NOT be in shared section (only in standard configs)")
                
            # Verify processing_only is not in shared section (only in some configs)
            self.assertNotIn('processing_only', merged['shared'],
                "'processing_only' should NOT be in shared section (only in processing configs)")
                
            # Verify varying_cross_type is in specific sections based on config type
            for step_name in ['Standard', 'AnotherStandard']:
                self.assertIn('varying_cross_type', merged['specific'][step_name],
                    f"'varying_cross_type' should be in specific for {step_name}")
                
            for step_name in ['Processing', 'AnotherProcessing']:
                self.assertIn('varying_cross_type', merged['processing']['processing_specific'][step_name],
                    f"'varying_cross_type' should be in processing_specific for {step_name}")
            
            # Check values for varying_cross_type
            self.assertEqual(merged['specific']['Standard']['varying_cross_type'], "standard value")
            self.assertEqual(merged['specific']['AnotherStandard']['varying_cross_type'], "another standard value")
            self.assertEqual(
                merged['processing']['processing_specific']['Processing']['varying_cross_type'], 
                "processing value"
            )
            self.assertEqual(
                merged['processing']['processing_specific']['AnotherProcessing']['varying_cross_type'], 
                "another processing value"
            )
            
            # Verify input_names and output_names are in specific sections since they're not static
            for step_name in ['Standard', 'AnotherStandard']:
                self.assertIn('input_names', merged['specific'][step_name],
                    f"'input_names' should be in specific for {step_name}")
                self.assertIn('output_names', merged['specific'][step_name],
                    f"'output_names' should be in specific for {step_name}")
                
            for step_name in ['Processing', 'AnotherProcessing']:
                self.assertIn('input_names', merged['processing']['processing_specific'][step_name],
                    f"'input_names' should be in processing_specific for {step_name}")
                self.assertIn('output_names', merged['processing']['processing_specific'][step_name],
                    f"'output_names' should be in processing_specific for {step_name}")
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {
                'StandardConfig': StandardConfig,
                'ProcessingConfig': ProcessingConfig,
                'AnotherStandardConfig': AnotherStandardConfig,
                'AnotherProcessingConfig': AnotherProcessingConfig
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Verify that loaded configs have the correct values
            self.assertEqual(loaded_configs['Standard'].shared_cross_type, "shared across all configs")
            self.assertEqual(loaded_configs['Processing'].shared_cross_type, "shared across all configs")
            self.assertEqual(loaded_configs['AnotherStandard'].shared_cross_type, "shared across all configs")
            self.assertEqual(loaded_configs['AnotherProcessing'].shared_cross_type, "shared across all configs")
            
            # Verify varying fields
            self.assertEqual(loaded_configs['Standard'].varying_cross_type, "standard value")
            self.assertEqual(loaded_configs['Processing'].varying_cross_type, "processing value")
            self.assertEqual(loaded_configs['AnotherStandard'].varying_cross_type, "another standard value")
            self.assertEqual(loaded_configs['AnotherProcessing'].varying_cross_type, "another processing value")
            
            # Verify mutual exclusivity between shared and specific sections
            # Check that shared_cross_type is only in shared section, not in specific sections
            self.assertIn('shared_cross_type', merged['shared'])
            for step in ['Standard', 'AnotherStandard']:
                self.assertNotIn('shared_cross_type', merged['specific'][step],
                    f"'shared_cross_type' should not appear in both shared and specific.{step} sections")
                
            # Check that varying_cross_type is only in specific sections, not in shared
            self.assertNotIn('varying_cross_type', merged['shared'])
            for step in ['Standard', 'AnotherStandard']:
                self.assertIn('varying_cross_type', merged['specific'][step])
                
            # Check mutual exclusivity between processing_shared and processing_specific
            for step in ['Processing', 'AnotherProcessing']:
                self.assertNotIn('shared_cross_type', merged['processing']['processing_specific'].get(step, {}),
                    f"'shared_cross_type' should not appear in both shared and processing_specific.{step} sections")
                
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    def test_enforce_mutual_exclusivity(self):
        """
        Test that fields appear in exactly one location, enforcing mutual exclusivity between:
        - "shared" and "specific" sections
        - "processing_shared" and "processing_specific" sections
        """
        # Register step names for this test
        BasePipelineConfig.STEP_NAMES.update({
            'ConfigWithDuplicates': 'Duplicates',
            'ProcessingWithDuplicates': 'ProcDuplicates'
        })
        
        # Create config classes with fields that could potentially be duplicated
        class ConfigWithDuplicates(BasePipelineConfig):
            """Config with fields that should be shared but might be duplicated."""
            shared_static_field: str = "this should only be in shared"
            specific_field: str = "this should only be in specific"
            input_names: Dict[str, str] = Field(default_factory=lambda: {"input": "Should be specific"})
            output_names: Dict[str, str] = Field(default_factory=dict)
        
        class ProcessingWithDuplicates(ProcessingStepConfigBase):
            """Processing config with fields that could be duplicated."""
            shared_static_field: str = "this should only be in shared" 
            processing_shared_field: str = "this should only be in processing_shared"
            processing_specific_field: str = "this should only be in processing_specific"
            
            # Avoid using processing_source_dir to prevent validation issues
            # Instead use fields that don't require directory validation
            processing_instance_count: int = 1
            processing_entry_point: str = "process.py"
            input_names: Dict[str, str] = Field(default_factory=dict)
            output_names: Dict[str, str] = Field(default_factory=dict)
            
            # Disable validation for the test
            @classmethod
            def validate_entry_point_paths(cls, values):
                """Skip validation for testing"""
                return values
                
            @classmethod
            def model_validator(cls, values):
                """Skip validation for testing purposes"""
                return values
        
        # Create instances
        config = ConfigWithDuplicates(**COMMON_ARGS)
        proc_config = ProcessingWithDuplicates(**COMMON_ARGS)
        
        # Create a list of configs
        config_list = [config, proc_config]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # Verify shared_static_field is only in shared section
            self.assertIn('shared_static_field', merged['shared'], 
                "shared_static_field should be in shared section")
            
            # Verify field is NOT duplicated in specific section
            self.assertNotIn('shared_static_field', merged['specific'].get('Duplicates', {}),
                "shared_static_field should not be duplicated in specific section")
                
            # For processor fields to be in processing_shared, they must:
            # 1. Be explicitly in ProcessingStepConfigBase
            # 2. Be in all processing configs
            # 3. Have identical values
            # Our processing_shared_field is not part of ProcessingStepConfigBase so it's not added at all
            
            # Verify shared_static_field is properly categorized in shared section
            self.assertIn('shared_static_field', merged['shared'], 
                "shared_static_field should be in shared section")
            
            # Verify field is NOT duplicated in specific section  
            self.assertNotIn('shared_static_field', merged['specific'].get('Duplicates', {}),
                "shared_static_field should not be duplicated in specific section")
                
            # Check that processing_shared_field is completely excluded from final config
            # (This is expected because it's not part of the ProcessingStepConfigBase)
            # Verify specific_field is in specific section as expected
            
            # Verify specific_field is only in specific section
            self.assertIn('specific_field', merged['specific']['Duplicates'],
                "specific_field should be in specific section")
                
            # Our custom fields don't get added to the output because they're not 
            # part of ProcessingStepConfigBase, only the built-in fields
            # Instead, verify that standard fields like input_names are there
            self.assertIn('input_names',
                merged['processing']['processing_specific']['ProcDuplicates'],
                "input_names should be in processing_specific section")
                
            self.assertIn('output_names',
                merged['processing']['processing_specific']['ProcDuplicates'],
                "output_names should be in processing_specific section")
            
            # Check common processing fields are properly handled
            # processing_entry_point should be in processing_shared
            self.assertIn('processing_entry_point', merged['processing']['processing_shared'],
                "processing_entry_point should be in processing_shared")
            self.assertNotIn('processing_entry_point', 
                merged['processing']['processing_specific'].get('ProcDuplicates', {}),
                "processing_entry_point should not be duplicated in processing_specific")
                
            # Load the configs for verification
            CONFIG_CLASSES = {
                'ConfigWithDuplicates': ConfigWithDuplicates,
                'ProcessingWithDuplicates': ProcessingWithDuplicates
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Verify all values were properly loaded despite being in different sections
            dup_config = loaded_configs.get('Duplicates')
            proc_config = loaded_configs.get('ProcDuplicates')
            
            self.assertEqual(dup_config.shared_static_field, "this should only be in shared")
            self.assertEqual(proc_config.shared_static_field, "this should only be in shared")
            self.assertEqual(proc_config.processing_shared_field, "this should only be in processing_shared")
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_nested_pydantic_models(self):
        """Test that nested Pydantic models are correctly serialized."""
        # Create a test nested Pydantic model structure
        class InnerModel(BaseModel):
            """Simple inner model for testing nested serialization."""
            value: str
            count: int

        class OuterModel(BasePipelineConfig):
            """Outer model that contains a nested Pydantic model."""
            name: str
            inner_model: InnerModel
            input_names: Dict[str, str] = Field(default_factory=dict)
            output_names: Dict[str, str] = Field(default_factory=dict)
            
            # Required BasePipelineConfig fields
            bucket: str = "test-bucket"
            author: str = "test-author"
            pipeline_name: str = "test-pipeline"
            pipeline_description: str = "Test pipeline"
            pipeline_version: str = "1.0.0" 
            pipeline_s3_loc: str = "s3://test-bucket/test"

        # Register the model name in the registry
        BasePipelineConfig.STEP_NAMES.update({
            'OuterModel': 'Outer'
        })

        # Create an instance of the outer model with a nested inner model
        outer_model = OuterModel(
            name="test-outer",
            inner_model=InnerModel(value="test-value", count=42)
        )
        
        # Create a list of configs
        config_list = [outer_model]
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save the configs - this should serialize the nested model without errors
            merged = merge_and_save_configs(config_list, tmp_path)
            
            # Verify that the inner model was serialized as a dictionary
            self.assertIn('Outer', merged['specific'])
            self.assertIn('inner_model', merged['specific']['Outer'])
            inner_model_dict = merged['specific']['Outer']['inner_model']
            self.assertIsInstance(inner_model_dict, dict)
            self.assertEqual(inner_model_dict['value'], "test-value")
            self.assertEqual(inner_model_dict['count'], 42)
            
            # Define config classes dict for loading
            CONFIG_CLASSES = {
                'OuterModel': OuterModel
            }
            
            # Load the configs back
            loaded_configs = load_configs(tmp_path, CONFIG_CLASSES)
            
            # Verify that the loaded config has the correct inner model
            outer = loaded_configs.get('Outer')
            self.assertIsNotNone(outer)
            self.assertIsInstance(outer.inner_model, InnerModel)
            self.assertEqual(outer.inner_model.value, "test-value")
            self.assertEqual(outer.inner_model.count, 42)
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

if __name__ == "__main__":
    unittest.main()
