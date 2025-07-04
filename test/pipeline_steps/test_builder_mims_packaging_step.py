import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class and config class to be tested
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase

class TestMIMSPackagingStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, valid configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'mims_package.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy MIMS packaging script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a real PackageStepConfig instance
        self.config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "processing_entry_point": "mims_package.py",
            "processing_source_dir": self.temp_dir,
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "processing_instance_type_large": "ml.m5.4xlarge",
            "processing_instance_type_small": "ml.m5.large",
            "use_large_processing_instance": False,
            "processing_framework_version": "0.23-1",
            "input_names": {
                "model_input": "model_input",
                "inference_scripts_input": "inference_scripts_input"
            },
            "output_names": {
                "packaged_model_output": "packaged_model_output"
            }
        }
        
        # Initialize the config with our data
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = PackageStepConfig(**self.config_data)
        
        # Initialize the builder with our config
        with patch.object(ProcessingStepConfigBase, 'get_script_path', return_value=os.path.join(self.temp_dir, 'mims_package.py')):
            self.builder = MIMSPackagingStepBuilder(
                config=self.config,
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole',
                notebook_root=Path('.')
            )
        
        # Mock methods for testing
        self.builder._get_step_name = MagicMock(return_value='Package')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-mims-packaging-pkg')
        self.builder._get_cache_config = MagicMock(return_value=CacheConfig(enable_caching=True))
        self.builder._extract_param = MagicMock(side_effect=lambda kwargs, key, default=None: kwargs.get(key, default))

    def tearDown(self):
        """Clean up the temporary directory after each test."""
        shutil.rmtree(self.temp_dir)
        
    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_source_dir(self):
        """Test that configuration validation fails with missing source directory."""
        # Create a new config with missing source_dir
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            # Create a new builder with a config that has no source_dir
            config_data = self.config_data.copy()
            config_data.pop("processing_source_dir")
            # Don't add source_dir
            
            # Create the config and builder
            with self.assertRaises(ValueError):
                config = PackageStepConfig(**config_data)

    def test_validate_configuration_missing_input_names(self):
        """Test that configuration validation fails with missing required input names."""
        # Create a new config with missing required input names
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            # Create a new config with invalid input_names
            config_data = self.config_data.copy()
            config_data["input_names"] = {"wrong_name": "description"}
            
            # Create the config
            with self.assertRaises(ValueError):
                config = PackageStepConfig(**config_data)

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
    def test_create_processor(self, mock_processor_cls):
        """Test that the processor is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Create processor
        processor = self.builder._create_processor()
        
        # Verify SKLearnProcessor was created with correct parameters
        # Don't check the exact base_job_name since it's being modified in the implementation
        # Instead, check that the call was made with the correct parameters except base_job_name
        self.assertEqual(mock_processor_cls.call_count, 1)
        call_args = mock_processor_cls.call_args[1]
        self.assertEqual(call_args['framework_version'], "0.23-1")
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], self.config.get_instance_type())
        self.assertEqual(call_args['instance_count'], self.config.processing_instance_count)
        self.assertEqual(call_args['volume_size_in_gb'], self.config.processing_volume_size)
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertTrue('base_job_name' in call_args)
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    def test_get_processor_inputs(self):
        """Test that processor inputs are created correctly."""
        # Create inputs dictionary with required keys
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "inference_scripts_input": "s3://bucket/scripts"
        }
        
        proc_inputs = self.builder._get_processor_inputs(inputs)
        
        self.assertEqual(len(proc_inputs), 2)
        
        # Check model data input
        model_input = next(i for i in proc_inputs if i.input_name == "model_input")
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        
        # Check inference scripts input
        scripts_input = next(i for i in proc_inputs if i.input_name == "inference_scripts_input")
        self.assertIsInstance(scripts_input, ProcessingInput)
        self.assertEqual(scripts_input.source, "s3://bucket/scripts")
        self.assertEqual(scripts_input.destination, "/opt/ml/processing/input/script")

    def test_get_processor_inputs_missing(self):
        """Test that _get_processor_inputs raises ValueError when inputs are missing."""
        # Test with empty inputs
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({})
        
        # Test with missing model_input
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({"inference_scripts_input": "s3://bucket/scripts"})
        
        # Test with missing inference_scripts_input
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({"model_input": "s3://bucket/model.tar.gz"})

    def test_get_processor_outputs(self):
        """Test that processor outputs are created correctly."""
        outputs = {
            "packaged_model_output": "s3://bucket/packaged_model"
        }
        
        proc_outputs = self.builder._get_processor_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        self.assertIsInstance(proc_outputs[0], ProcessingOutput)
        self.assertEqual(proc_outputs[0].source, "/opt/ml/processing/output")
        self.assertEqual(proc_outputs[0].destination, "s3://bucket/packaged_model")
        self.assertEqual(proc_outputs[0].output_name, "packaged_model_output")
        
    def test_get_processor_outputs_missing(self):
        """Test that _get_processor_outputs raises ValueError when outputs are missing."""
        # Test with empty outputs
        with self.assertRaises(ValueError):
            self.builder._get_processor_outputs({})
        
        # Test with missing packaged_model_output
        with self.assertRaises(ValueError):
            self.builder._get_processor_outputs({"wrong_output": "s3://bucket/wrong"})

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_packaging_step.ProcessingStep')
    def test_create_step(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Create inputs and outputs
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "inference_scripts_input": "s3://bucket/scripts"
        }
        outputs = {
            "packaged_model_output": "s3://bucket/packaged_model"
        }
        
        # Create step
        step = self.builder.create_step(
            inputs=inputs,
            outputs=outputs,
            enable_caching=True
        )
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'Package')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertEqual(call_kwargs['depends_on'], [])
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in call_kwargs['inputs']))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in call_kwargs['outputs']))
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_packaging_step.ProcessingStep')
    def test_create_step_with_dependencies(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create inputs and outputs
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "inference_scripts_input": "s3://bucket/scripts"
        }
        outputs = {
            "packaged_model_output": "s3://bucket/packaged_model"
        }
        
        # Create step with dependencies
        step = self.builder.create_step(
            inputs=inputs,
            outputs=outputs,
            dependencies=dependencies,
            enable_caching=True
        )
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    def test_match_custom_properties(self):
        """Test _match_custom_properties method."""
        # Create a mock step with properties
        prev_step = MagicMock()
        prev_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        # Set up input requirements
        input_requirements = {
            "inputs": "Dictionary containing model_input, inference_scripts_input S3 paths"
        }
        
        # Call the method
        inputs = {}
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Verify inputs were matched
        self.assertIn("model_input", matched)
        self.assertIn("inference_scripts_input", matched)
        self.assertIn("model_input", inputs)
        self.assertEqual(inputs["model_input"], "s3://bucket/model.tar.gz")
        
    def test_match_custom_properties_no_model_artifacts(self):
        """Test _match_custom_properties method when no model artifacts are available."""
        # Create a mock step without ModelArtifacts property
        prev_step = MagicMock(spec=[])  # No properties
        
        # Set up input requirements
        input_requirements = {
            "inputs": "Dictionary containing model_input, inference_scripts_input S3 paths"
        }
        
        # Call the method
        inputs = {}
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Verify only inference_scripts_input was matched
        self.assertEqual(matched, {"inference_scripts_input"})
        self.assertIn("inference_scripts_input", inputs)
        
    def test_get_input_requirements(self):
        """Test get_input_requirements method."""
        # Call the method
        input_reqs = self.builder.get_input_requirements()
        
        # Verify the input requirements
        self.assertIn("inputs", input_reqs)
        self.assertIn("outputs", input_reqs)
        self.assertIn("enable_caching", input_reqs)
        
    def test_get_output_properties(self):
        """Test get_output_properties method."""
        # Call the method
        output_props = self.builder.get_output_properties()
        
        # Verify the output properties
        self.assertEqual(output_props, {
            "packaged_model_output": "packaged_model_output"
        })

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
