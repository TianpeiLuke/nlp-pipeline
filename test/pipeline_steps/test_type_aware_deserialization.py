import unittest
import json
import os
from pathlib import Path
import tempfile
import sys

# Add the repository root directory to the path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, repo_root)

import pytest
from unittest.mock import patch, MagicMock, Mock

# Import utilities for config serialization
from src.pipeline_steps.utils import merge_and_save_configs, load_configs, _serialize
from src.pipeline_steps.utils import build_complete_config_classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_dummy_training import DummyTrainingConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

# Try to import the BSM hyperparameters class if available
try:
    from src.pipeline_steps.hyperparameters_bsm import BSMModelHyperparameters
    BSM_AVAILABLE = True
except ImportError:
    BSM_AVAILABLE = False
    # Create a mock class for testing if the real one is not available
    class BSMModelHyperparameters(ModelHyperparameters):
        lr_decay: float = 0.05
        adam_epsilon: float = 1e-08
        text_name: str = "dialogue"


class TestTypeAwareDeserialization(unittest.TestCase):
    """
    Tests for the type-aware model serialization and deserialization.
    This tests the ability to correctly serialize and deserialize derived model classes.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        # Define paths
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.model_path = os.path.join(os.path.dirname(__file__), "model.tar.gz")
        self.pipeline_scripts_path = os.path.join(self.repo_root, "src/pipeline_scripts")
        
        # Check that required directories and files exist
        self.assertTrue(os.path.exists(self.model_path), f"Test model file missing: {self.model_path}")
        self.assertTrue(os.path.exists(self.pipeline_scripts_path), 
                        f"Required directory not found: {self.pipeline_scripts_path}")
        
        # Create a base hyperparameters object
        self.base_hyperparams = ModelHyperparameters(
            full_field_list=["field1", "field2", "field3"],
            cat_field_list=["field3"],
            tab_field_list=["field1", "field2"],
            input_tab_dim=2,
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1],
            class_weights=[1.0, 2.0]
        )
        
        # Skip BSM tests if the class is not available
        if not BSM_AVAILABLE:
            self.skipTest("BSMModelHyperparameters not available")
            
        # Create a derived BSM hyperparameters object with additional fields
        self.bsm_hyperparams = BSMModelHyperparameters(
            full_field_list=["field1", "field2", "field3"],
            cat_field_list=["field3"],
            tab_field_list=["field1", "field2"],
            input_tab_dim=2,
            is_binary=True,
            num_classes=2,
            multiclass_categories=[0, 1],
            class_weights=[1.0, 2.0],
            # BSM-specific fields
            lr_decay=0.05,
            adam_epsilon=1e-08,
            text_name="dialogue",
            chunk_trancate=True,
            max_total_chunks=3,
            tokenizer="bert-base-multilingual-uncased",
            max_sen_len=512
        )
        
        # Create configs using the hyperparameters objects
        # Add job_type to distinguish configs with the same class
        self.base_config = DummyTrainingConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline-base",
            pipeline_description="Test Pipeline Base",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline-base",
            pretrained_model_path=self.model_path,
            processing_source_dir=self.pipeline_scripts_path,
            hyperparameters=self.base_hyperparams,
            job_type="base"  # Add distinguishing attribute
        )
        
        self.bsm_config = DummyTrainingConfig(
            bucket="test-bucket",
            author="test-author",
            pipeline_name="test-pipeline-bsm",
            pipeline_description="Test Pipeline BSM",
            pipeline_version="1.0.0",
            pipeline_s3_loc="s3://test-bucket/test-pipeline-bsm",
            pretrained_model_path=self.model_path,
            processing_source_dir=self.pipeline_scripts_path,
            hyperparameters=self.bsm_hyperparams,
            job_type="bsm"  # Add distinguishing attribute
        )
        
    def test_type_preservation(self):
        """Test that derived class types are preserved during serialization and deserialization."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, self.bsm_config], output_path)
            
            # Read the output file
            with open(output_path, 'r') as f:
                output_json = json.load(f)
                
            # Get complete config classes
            config_classes = build_complete_config_classes()
            
            # Load the configs back
            loaded_configs = load_configs(output_path, config_classes)
            
            # Find the BSM config by job_type
            bsm_step = None
            for step_name, config in loaded_configs.items():
                if "bsm" in step_name:
                    bsm_step = step_name
                    break
                    
            self.assertIsNotNone(bsm_step, "BSM config not found in loaded configs")
            
            # Verify the BSM hyperparameters class type is preserved
            bsm_config_loaded = loaded_configs[bsm_step]
            self.assertIsInstance(bsm_config_loaded.hyperparameters, BSMModelHyperparameters, 
                                 "BSM hyperparameters type not preserved")
                                 
            # Verify BSM-specific fields are present and have correct values
            self.assertTrue(hasattr(bsm_config_loaded.hyperparameters, 'lr_decay'), 
                           "BSM-specific field 'lr_decay' missing")
            self.assertEqual(bsm_config_loaded.hyperparameters.lr_decay, 0.05,
                           "BSM-specific field 'lr_decay' has incorrect value")
                           
            self.assertTrue(hasattr(bsm_config_loaded.hyperparameters, 'text_name'), 
                           "BSM-specific field 'text_name' missing")
            self.assertEqual(bsm_config_loaded.hyperparameters.text_name, "dialogue",
                           "BSM-specific field 'text_name' has incorrect value")
                           
            # Check that the base config still has a ModelHyperparameters instance
            base_step = None
            for step_name, config in loaded_configs.items():
                if "base" in step_name:
                    base_step = step_name
                    break
                    
            self.assertIsNotNone(base_step, "Base config not found in loaded configs")
            
            # Verify the base hyperparameters class type is preserved
            base_config_loaded = loaded_configs[base_step]
            self.assertIsInstance(base_config_loaded.hyperparameters, ModelHyperparameters, 
                                 "Base hyperparameters type not preserved")
                                 
            # Verify BSM-specific fields are not present on the base hyperparameters
            self.assertFalse(hasattr(base_config_loaded.hyperparameters, 'lr_decay'), 
                            "Base hyperparameters should not have BSM-specific fields")
                            
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
                
    def test_type_metadata_in_serialized_output(self):
        """Test that type metadata is included in the serialized output."""
        # Serialize a BSM hyperparameters object
        serialized = _serialize(self.bsm_hyperparams)
        
        # Verify type metadata fields are present
        self.assertIn('__model_type__', serialized, "Type metadata field missing")
        self.assertEqual(serialized['__model_type__'], 'BSMModelHyperparameters', 
                        "Type metadata has incorrect value")
                        
        self.assertIn('__model_module__', serialized, "Module metadata field missing")
        self.assertTrue(serialized['__model_module__'].endswith('hyperparameters_bsm'), 
                       "Module metadata has incorrect value")
                       
        # Verify BSM-specific fields are present
        self.assertIn('lr_decay', serialized, "BSM-specific field missing in serialized output")
        self.assertEqual(serialized['lr_decay'], 0.05, "BSM-specific field has incorrect value")
                
    def test_fallback_behavior(self):
        """Test the fallback behavior when a derived class is not available."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            output_path = temp_file.name
            
        try:
            # Merge the configs and save to the temp file
            merge_and_save_configs([self.base_config, self.bsm_config], output_path)
            
            # Define a custom config_classes that doesn't include BSMModelHyperparameters
            limited_config_classes = {
                'BasePipelineConfig': BasePipelineConfig,
                'DummyTrainingConfig': DummyTrainingConfig,
                'ModelHyperparameters': ModelHyperparameters,
                # BSMModelHyperparameters intentionally omitted
            }
            
            # Load the configs back with the limited class registry
            loaded_configs = load_configs(output_path, limited_config_classes)
            
            # Find the BSM config by job_type
            bsm_step = None
            for step_name, config in loaded_configs.items():
                if "bsm" in step_name:
                    bsm_step = step_name
                    break
                    
            self.assertIsNotNone(bsm_step, "BSM config not found in loaded configs")
            
            # Verify the fallback to base ModelHyperparameters when BSM class is not available
            bsm_config_loaded = loaded_configs[bsm_step]
            self.assertIsInstance(bsm_config_loaded.hyperparameters, ModelHyperparameters, 
                                 "Should fallback to ModelHyperparameters")
            
            # Verify base fields are present
            self.assertTrue(hasattr(bsm_config_loaded.hyperparameters, 'full_field_list'), 
                           "Base field 'full_field_list' missing")
                           
        finally:
            # Clean up the temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
                

if __name__ == "__main__":
    unittest.main()
