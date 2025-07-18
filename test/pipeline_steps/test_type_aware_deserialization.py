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
from src.pipeline_steps.utils import merge_and_save_configs, load_configs, serialize_config
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
        # Use the serializer directly to test round-trip serialization
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer, deserialize_config
        
        # Create serializer with complete config classes
        config_classes = build_complete_config_classes()
        serializer = TypeAwareConfigSerializer(config_classes=config_classes)
        
        # Test BSM hyperparameters serialization and deserialization
        serialized_bsm = serializer.serialize(self.bsm_hyperparams)
        print("BSM serialized:", serialized_bsm)
        self.assertIn('__model_type__', serialized_bsm)
        self.assertEqual(serialized_bsm['__model_type__'], 'BSMModelHyperparameters')
        
        # Deserialize back
        deserialized_bsm = serializer.deserialize(serialized_bsm)
        self.assertIsInstance(deserialized_bsm, BSMModelHyperparameters)
        self.assertTrue(hasattr(deserialized_bsm, 'lr_decay'))
        self.assertEqual(deserialized_bsm.lr_decay, 0.05)
        
        # Test BSM-specific fields
        self.assertTrue(hasattr(deserialized_bsm, 'text_name'))
        self.assertEqual(deserialized_bsm.text_name, 'dialogue')
        
        # Test that base hyperparameters class doesn't have BSM-specific fields
        serialized_base = serializer.serialize(self.base_hyperparams)
        deserialized_base = serializer.deserialize(serialized_base)
        self.assertIsInstance(deserialized_base, ModelHyperparameters)
        self.assertFalse(hasattr(deserialized_base, 'lr_decay'))
                
    def test_type_metadata_in_serialized_output(self):
        """Test that type metadata is included in the serialized output."""
        # Create a serializer and use it directly
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
        serializer = TypeAwareConfigSerializer()
        
        # Serialize a BSM hyperparameters object
        serialized = serializer.serialize(self.bsm_hyperparams)
        
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
        # Use the serializer directly to test fallback behavior
        from src.config_field_manager.type_aware_config_serializer import TypeAwareConfigSerializer
        
        # Create serializer with limited config classes (no BSMModelHyperparameters)
        limited_config_classes = {
            'BasePipelineConfig': BasePipelineConfig,
            'DummyTrainingConfig': DummyTrainingConfig,
            'ModelHyperparameters': ModelHyperparameters
            # BSMModelHyperparameters intentionally omitted
        }
        
        # Create serializer with limited classes
        serializer = TypeAwareConfigSerializer(config_classes=limited_config_classes)
        
        # Test direct serialization and deserialization of BSM hyperparameters
        serialized_bsm = serializer.serialize(self.bsm_hyperparams)
        self.assertIn('__model_type__', serialized_bsm)
        self.assertEqual(serialized_bsm['__model_type__'], 'BSMModelHyperparameters')
        
        # Deserialize with limited class registry - should fallback to ModelHyperparameters
        deserialized_bsm = serializer.deserialize(serialized_bsm)
        self.assertIsInstance(deserialized_bsm, ModelHyperparameters, 
                             "Should fallback to ModelHyperparameters")
        
        # Verify base fields are still present
        self.assertTrue(hasattr(deserialized_bsm, 'full_field_list'))
        self.assertListEqual(deserialized_bsm.full_field_list, ["field1", "field2", "field3"])
                

if __name__ == "__main__":
    unittest.main()
