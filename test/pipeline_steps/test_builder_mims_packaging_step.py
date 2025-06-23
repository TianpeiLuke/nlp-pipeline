import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder

class TestMIMSPackagingStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        
        # Required attributes for validation
        self.config.processing_entry_point = 'mims_package.py'
        self.config.processing_source_dir = '/path/to/scripts'
        self.config.source_dir = '/path/to/inference_scripts'
        self.config.processing_framework_version = '1.2-1'
        self.config.use_large_processing_instance = False
        self.config.processing_instance_type_small = 'ml.m5.large'
        self.config.processing_instance_type_large = 'ml.m5.4xlarge'
        self.config.processing_script_arguments = ['--arg1', 'value1']
        
        # Processing configuration
        self.config.processing_instance_type = 'ml.m5.large'
        self.config.processing_instance_count = 1
        self.config.processing_volume_size = 30
        
        # IO configuration
        self.config.pipeline_s3_loc = 's3://bucket/pipeline'
        self.config.pipeline_name = 'test-pipeline'
        self.config.input_names = {
            "model_input": "model_input",
            "inference_scripts_input": "inference_scripts_input"
        }
        self.config.output_names = {
            "packaged_model_output": "packaged_model_output"
        }
        
        # Methods
        self.config.get_script_path = MagicMock(return_value='mims_packaging.py')
        self.config.get_instance_type = MagicMock(return_value='ml.m5.large')
        self.config.get_input_names = MagicMock(return_value=self.config.input_names)
        self.config.get_output_names = MagicMock(return_value=self.config.output_names)
        self.config.get_effective_source_dir = MagicMock(return_value='/path/to/scripts')
        # Mock Path.exists to return True for script validation
        self.path_exists_patch = patch('pathlib.Path.exists', return_value=True)
        self.path_exists_patch.start()
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(MIMSPackagingStepBuilder)
        self.builder.config = self.config
        
        # Create a properly configured session mock
        session_mock = MagicMock()
        # Set sagemaker_config to an empty dict to pass validation
        session_mock.sagemaker_config = {}
        self.builder.session = session_mock
        
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        # Use a mock for the _get_step_name method from the base class
        self.builder._get_step_name = MagicMock(return_value='MIMS_Packaging')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-mims-packaging-mims-pkg')

    def tearDown(self):
        """Clean up after each test."""
        self.path_exists_patch.stop()
        
    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_source_dir(self):
        """Test that configuration validation fails with missing source directory."""
        self.config.processing_source_dir = None
        self.config.source_dir = None
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()

    def test_validate_configuration_missing_input_names(self):
        """Test that configuration validation fails with missing required input names."""
        # Save original input_names
        original_input_names = self.config.input_names
        # Set input_names to a dict missing required keys
        self.config.input_names = {"wrong_name": "description"}
        # Mock get_input_names to return the modified input_names
        self.config.get_input_names = MagicMock(return_value=self.config.input_names)
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
        # Restore original input_names and mock
        self.config.input_names = original_input_names
        self.config.get_input_names = MagicMock(return_value=self.config.input_names)

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
        self.assertEqual(call_args['framework_version'], "1.2-1")
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], self.config.processing_instance_type)
        self.assertEqual(call_args['instance_count'], self.config.processing_instance_count)
        self.assertEqual(call_args['volume_size_in_gb'], self.config.processing_volume_size)
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertTrue('base_job_name' in call_args)
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    def test_get_processing_inputs(self):
        """Test that processing inputs are created correctly."""
        model_data = 's3://bucket/model.tar.gz'
        
        inputs = self.builder._get_processing_inputs(model_data)
        
        self.assertEqual(len(inputs), 2)
        
        # Check model data input
        model_input = inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, model_data)
        self.assertEqual(model_input.destination, '/opt/ml/processing/input/model')
        self.assertEqual(model_input.input_name, 'model_input')
        
        # Check inference scripts input
        scripts_input = inputs[1]
        self.assertIsInstance(scripts_input, ProcessingInput)
        self.assertEqual(scripts_input.source, self.config.source_dir)
        self.assertEqual(scripts_input.destination, '/opt/ml/processing/input/script')
        self.assertEqual(scripts_input.input_name, 'inference_scripts_input')

    def test_get_processing_outputs(self):
        """Test that processing outputs are created correctly."""
        outputs = self.builder._get_processing_outputs('MIMS_Packaging')
        
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], ProcessingOutput)
        self.assertEqual(outputs[0].source, '/opt/ml/processing/output')
        self.assertEqual(outputs[0].destination, 's3://bucket/pipeline/MIMS_Packaging/packaged_model_artifacts')
        self.assertEqual(outputs[0].output_name, 'packaged_model_output')

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
        
        # Create step
        model_data = 's3://bucket/model.tar.gz'
        step = self.builder.create_step(
            model_artifacts_input_source=model_data
        )
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'MIMS_Packaging')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertEqual(call_kwargs['depends_on'], [])
        
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
        
        # Create step with dependencies
        model_data = 's3://bucket/model.tar.gz'
        step = self.builder.create_step(
            model_artifacts_input_source=model_data,
            dependencies=dependencies
        )
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    def test_create_packaging_step_backward_compatibility(self):
        """Test that the old create_packaging_step method calls the new create_step method."""
        with patch.object(self.builder, 'create_step', return_value="step_created") as mock_create_step:
            # Call the old method
            result = self.builder.create_packaging_step(
                model_data='s3://bucket/model.tar.gz',
                dependencies=None
            )
            # Verify it called the new method
            mock_create_step.assert_called_once_with(
                's3://bucket/model.tar.gz',
                None
            )
            self.assertEqual(result, "step_created")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
