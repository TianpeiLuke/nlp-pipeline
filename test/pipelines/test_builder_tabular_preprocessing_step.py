import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class
from src.pipelines.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder

class TestTabularPreprocessingStepBuilder(unittest.TestCase):
    def setUp(self):
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        # BasePipelineConfig expected attributes
        self.config.region = 'NA'
        self.config.job_type = 'training'
        # Hyperparameters with label_name
        self.config.hyperparameters = SimpleNamespace(label_name='target')
        # Input/output channel names
        self.config.input_names = {'data_input': 'raw_data'}
        # FIX: The simplified builder now only requires and handles a single output.
        self.config.output_names = {
            'processed_data': 'ProcessedDataOutput'
        }
        # Ratios
        self.config.train_ratio = 0.7
        self.config.test_val_ratio = 0.3
        # Processing settings
        self.config.framework_version = '0.23-1'
        self.config.processing_framework_version = '0.23-1' # Match new config property
        self.config.processing_instance_count = 1
        self.config.processing_volume_size = 30
        self.config.get_instance_type = lambda: 'ml.m5.large'
        # Script path
        self.config.get_script_path = lambda: '/path/to/preprocess.py'
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(TabularPreprocessingStepBuilder)
        self.builder.config = self.config
        self.builder.session = MagicMock()
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        # Required by base
        self.builder.notebook_root = Path('.')
        # Use a mock for the _get_step_name method from the base class
        self.builder._get_step_name = MagicMock(return_value='ProcessingStep')
        self.builder.aws_region = self.builder.REGION_MAPPING['NA']

    def test_get_environment_variables(self):
        env = self.builder._get_environment_variables()
        self.assertEqual(env, {
            'LABEL_FIELD': 'target',
            'TRAIN_RATIO': '0.7',
            'TEST_VAL_RATIO': '0.3'
        })

    def test_get_processor_inputs_success(self):
        inputs = {'raw_data': 's3://bucket/raw'}
        proc_inputs = self.builder._get_processor_inputs(inputs)
        self.assertIsInstance(proc_inputs, list)
        self.assertIsInstance(proc_inputs[0], ProcessingInput)
        self.assertEqual(proc_inputs[0].input_name, 'raw_data')
        self.assertEqual(proc_inputs[0].source, 's3://bucket/raw')
        self.assertEqual(proc_inputs[0].destination, '/opt/ml/processing/input/data')

    def test_get_processor_inputs_missing(self):
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({})

    def test_get_processor_outputs_success(self):
        # FIX: The outputs dictionary now only needs to contain the single 'processed_data' key.
        outputs = {
            'ProcessedDataOutput': 's3://bucket/processed'
        }
        proc_outputs = self.builder._get_processor_outputs(outputs)
        self.assertIsInstance(proc_outputs, list)
        # FIX: Assert that only one output object is created.
        self.assertEqual(len(proc_outputs), 1)
        
        # Verify the processed_data output
        proc_output = proc_outputs[0]
        self.assertIsInstance(proc_output, ProcessingOutput)
        self.assertEqual(proc_output.output_name, 'ProcessedDataOutput')
        self.assertEqual(proc_output.destination, 's3://bucket/processed')
        self.assertEqual(proc_output.source, '/opt/ml/processing/output')

    def test_get_processor_outputs_missing(self):
        # FIX: Test that a ValueError is raised when the single required output is missing.
        with self.assertRaisesRegex(ValueError, "Must supply an S3 URI for 'ProcessedDataOutput'"):
            self.builder._get_processor_outputs({})

    def test_get_job_arguments(self):
        args = self.builder._get_job_arguments()
        self.assertEqual(args, ['--job_type', 'training'])

    @patch('src.pipelines.builder_tabular_preprocessing_step.SKLearnProcessor')
    def test_create_step(self, mock_processor_cls):
        # Stub processor instance
        dummy_processor = MagicMock()
        dummy_processor.env = {} # Mock the env attribute
        mock_processor_cls.return_value = dummy_processor
        
        # Provide inputs and the single required output
        inputs = {'raw_data': 's3://bucket/raw'}
        outputs = {'ProcessedDataOutput': 's3://bucket/processed'}
        
        # Create step with caching disabled
        step = self.builder.create_step(inputs=inputs, outputs=outputs, enable_caching=False)
        
        # Verify the ProcessingStep
        self.assertIsInstance(step, ProcessingStep)
        self.assertEqual(step.name, 'ProcessingStep-Training')
        self.assertIs(step.processor, dummy_processor)
        self.assertEqual(step.code, '/path/to/preprocess.py')
        self.assertEqual(step.job_arguments, ['--job_type', 'training'])
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in step.inputs))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in step.outputs))
        self.assertIsInstance(step.cache_config, CacheConfig)
        self.assertFalse(step.cache_config.enable_caching)
        
    def test_create_processing_step_backward_compatibility(self):
        """Test that the old create_processing_step method calls the new create_step method."""
        with patch.object(self.builder, 'create_step', return_value="step_created") as mock_create_step:
            # Call the old method
            result = self.builder.create_processing_step()
            # Verify it called the new method
            mock_create_step.assert_called_once()
            self.assertEqual(result, "step_created")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
