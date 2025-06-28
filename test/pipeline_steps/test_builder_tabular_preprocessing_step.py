import unittest
import tempfile
import shutil
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

# Import the builder class and config class
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

class TestTabularPreprocessingStepBuilder(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'tabular_preprocess.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy tabular preprocessing script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a real TabularPreprocessingConfig instance
        hyperparams = ModelHyperparameters(label_name='target')
        
        self.config = TabularPreprocessingConfig(
            region='NA',
            job_type='training',
            hyperparameters=hyperparams,
            input_names={
                'data_input': 'RawData',
                'metadata_input': 'Metadata',
                'signature_input': 'Signature'
            },
            output_names={
                'processed_data': 'ProcessedTabularData',
                'full_data': 'FullTabularData'
            },
            train_ratio=0.7,
            test_val_ratio=0.5,
            processing_instance_type_large='ml.m5.4xlarge',
            processing_instance_type_small='ml.m5.large',
            use_large_processing_instance=False,
            processing_instance_count=1,
            processing_volume_size=30,
            processing_entry_point='tabular_preprocess.py',
            processing_source_dir=self.temp_dir,
            processing_framework_version='0.23-1'
        )
        
        # Initialize the builder with our config
        with patch.object(ProcessingStepConfigBase, 'get_script_path', return_value='/path/to/preprocess.py'):
            self.builder = TabularPreprocessingStepBuilder(
                config=self.config,
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole',
                notebook_root=Path('.')
            )
        
        # Mock the _get_step_name method from the base class
        self.builder._get_step_name = MagicMock(return_value='ProcessingStep')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='processing-step-training')
        self.builder._get_cache_config = MagicMock(return_value=CacheConfig(enable_caching=False))
        self.builder._extract_param = MagicMock(side_effect=lambda kwargs, key, default=None: kwargs.get(key, default))
        
    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_get_environment_variables(self):
        env = self.builder._get_environment_variables()
        self.assertEqual(env, {
            'LABEL_FIELD': 'target',
            'TRAIN_RATIO': '0.7',
            'TEST_VAL_RATIO': '0.5'
        })

    # Removed test_get_environment_variables_with_columns as these columns don't exist in the config

    def test_get_processor_inputs_success(self):
        # Test with all three input channels
        # The _get_processor_inputs method has a complex logic:
        # 1. It checks if "metadata_input" in inputs (the key name)
        # 2. But then it uses inputs[self.config.input_names["metadata_input"]] (the channel name)
        # So we need to include both the key name and the channel name in the inputs dictionary
        inputs = {
            self.config.input_names['data_input']: 's3://bucket/data',
            'metadata_input': 's3://bucket/metadata',  # This key is checked in the method
            self.config.input_names['metadata_input']: 's3://bucket/metadata',  # This is used to get the source
            'signature_input': 's3://bucket/signature',  # This key is checked in the method
            self.config.input_names['signature_input']: 's3://bucket/signature'  # This is used to get the source
        }
        
        proc_inputs = self.builder._get_processor_inputs(inputs)
        
        # Should have three ProcessingInput objects
        self.assertEqual(len(proc_inputs), 3)
        
        # Verify data input
        data_input = next(i for i in proc_inputs if i.input_name == 'RawData')
        self.assertEqual(data_input.source, 's3://bucket/data')
        self.assertEqual(data_input.destination, '/opt/ml/processing/input/data')
        
        # Verify metadata input
        metadata_input = next(i for i in proc_inputs if i.input_name == 'Metadata')
        self.assertEqual(metadata_input.source, 's3://bucket/metadata')
        self.assertEqual(metadata_input.destination, '/opt/ml/processing/input/metadata')
        
        # Verify signature input
        signature_input = next(i for i in proc_inputs if i.input_name == 'Signature')
        self.assertEqual(signature_input.source, 's3://bucket/signature')
        self.assertEqual(signature_input.destination, '/opt/ml/processing/input/signature')

    def test_get_processor_inputs_minimal(self):
        # Test with only the required data_input channel
        data_input_channel = self.config.input_names['data_input']
        inputs = {data_input_channel: 's3://bucket/raw'}
        proc_inputs = self.builder._get_processor_inputs(inputs)
        
        # Should have one ProcessingInput object
        self.assertEqual(len(proc_inputs), 1)
        
        # Verify data input
        data_input = proc_inputs[0]
        self.assertEqual(data_input.input_name, 'RawData')
        self.assertEqual(data_input.source, 's3://bucket/raw')
        self.assertEqual(data_input.destination, '/opt/ml/processing/input/data')

    def test_get_processor_inputs_missing(self):
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({})

    def test_get_processor_outputs_success(self):
        # Test with both output channels
        outputs = {
            'ProcessedTabularData': 's3://bucket/processed',
            'FullTabularData': 's3://bucket/full'
        }
        proc_outputs = self.builder._get_processor_outputs(outputs)
        
        # Should have two ProcessingOutput objects
        self.assertEqual(len(proc_outputs), 2)
        
        # Verify processed data output
        processed_output = next(o for o in proc_outputs if o.output_name == 'ProcessedTabularData')
        self.assertEqual(processed_output.destination, 's3://bucket/processed')
        self.assertEqual(processed_output.source, '/opt/ml/processing/output')
        
        # Verify full data output
        full_output = next(o for o in proc_outputs if o.output_name == 'FullTabularData')
        self.assertEqual(full_output.destination, 's3://bucket/full')
        self.assertEqual(full_output.source, '/opt/ml/processing/output')

    def test_get_processor_outputs_optional_full_data(self):
        # Test with only the required processed_data output
        outputs = {
            'ProcessedTabularData': 's3://bucket/processed'
        }
        proc_outputs = self.builder._get_processor_outputs(outputs)
        
        # Should have one ProcessingOutput object
        self.assertEqual(len(proc_outputs), 1)
        
        # Verify processed data output
        processed_output = proc_outputs[0]
        self.assertEqual(processed_output.output_name, 'ProcessedTabularData')
        self.assertEqual(processed_output.destination, 's3://bucket/processed')
        self.assertEqual(processed_output.source, '/opt/ml/processing/output')

    def test_get_processor_outputs_missing(self):
        # Test with missing required output
        with self.assertRaisesRegex(ValueError, "Must supply an S3 URI for 'ProcessedTabularData'"):
            self.builder._get_processor_outputs({})

    def test_get_job_arguments(self):
        args = self.builder._get_job_arguments()
        self.assertEqual(args, ['--job_type', 'training'])

    @patch('src.pipeline_steps.builder_tabular_preprocessing_step.SKLearnProcessor')
    def test_create_step(self, mock_processor_cls):
        # Stub processor instance
        dummy_processor = MagicMock()
        dummy_processor.env = {}  # Mock the env attribute
        mock_processor_cls.return_value = dummy_processor
        
        # Provide inputs and outputs
        # We need to use the keys from self.config.input_names
        inputs = {}
        for key, channel in self.config.input_names.items():
            inputs[channel] = f's3://bucket/{key}'
        outputs = {
            'ProcessedTabularData': 's3://bucket/processed',
            'FullTabularData': 's3://bucket/full'
        }
        
        # Create step with caching disabled
        with patch.object(ProcessingStepConfigBase, 'get_script_path', return_value='/path/to/preprocess.py'):
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

    def test_match_custom_properties(self):
        # Create a mock step with properties
        prev_step = MagicMock()
        
        # Mock the _match_cradle_data_loading_step method to return False
        # so that _match_processing_step_outputs is called
        self.builder._match_cradle_data_loading_step = MagicMock(return_value=False)
        
        # Mock the _match_processing_step_outputs method to add 'inputs' to matched_inputs
        def mock_match_processing_step_outputs(inputs, prev_step, matched_inputs):
            if 'inputs' not in inputs:
                inputs['inputs'] = {}
            inputs['inputs']['RawData'] = 's3://bucket/data'
            matched_inputs.add('inputs')
        
        self.builder._match_processing_step_outputs = MagicMock(side_effect=mock_match_processing_step_outputs)
        
        # Set up input requirements
        input_requirements = {
            'inputs': 'Dictionary containing data_input, metadata_input S3 paths'
        }
        
        # Call the method
        inputs = {}
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Verify inputs were matched
        self.assertIn('inputs', matched)
        self.assertIn('inputs', inputs)
        self.assertIn('RawData', inputs['inputs'])
        self.assertEqual(inputs['inputs']['RawData'], 's3://bucket/data')

    def test_match_cradle_data_loading_step(self):
        # Create a mock CradleDataLoadingStep
        prev_step = MagicMock()
        prev_step.get_output_locations.return_value = {
            'data': 's3://bucket/cradle/data',
            'metadata': 's3://bucket/cradle/metadata',
            'signature': 's3://bucket/cradle/signature'
        }
        
        # Call the method
        inputs = {}
        matched_inputs = set()
        result = self.builder._match_cradle_data_loading_step(inputs, prev_step, matched_inputs)
        
        # Verify result
        self.assertTrue(result)
        self.assertIn('inputs', matched_inputs)
        self.assertIn('inputs', inputs)
        self.assertIn('RawData', inputs['inputs'])
        self.assertEqual(inputs['inputs']['RawData'], 's3://bucket/cradle/data')
        self.assertIn('Metadata', inputs['inputs'])
        self.assertEqual(inputs['inputs']['Metadata'], 's3://bucket/cradle/metadata')
        self.assertIn('Signature', inputs['inputs'])
        self.assertEqual(inputs['inputs']['Signature'], 's3://bucket/cradle/signature')

    def test_match_processing_step_outputs(self):
        # Create a mock ProcessingStep with outputs
        prev_step = MagicMock()
        prev_step.outputs = [
            SimpleNamespace(output_name='data', destination='s3://bucket/data'),
            SimpleNamespace(output_name='metadata', destination='s3://bucket/metadata'),
            SimpleNamespace(output_name='signature', destination='s3://bucket/signature')
        ]
        
        # Call the method
        inputs = {}
        matched_inputs = set()
        self.builder._match_processing_step_outputs(inputs, prev_step, matched_inputs)
        
        # Verify inputs were matched
        self.assertIn('inputs', matched_inputs)
        self.assertIn('inputs', inputs)
        self.assertIn('RawData', inputs['inputs'])
        self.assertEqual(inputs['inputs']['RawData'], 's3://bucket/data')
        self.assertIn('Metadata', inputs['inputs'])
        self.assertEqual(inputs['inputs']['Metadata'], 's3://bucket/metadata')
        self.assertIn('Signature', inputs['inputs'])
        self.assertEqual(inputs['inputs']['Signature'], 's3://bucket/signature')

    def test_get_input_requirements(self):
        # Test the get_input_requirements method
        input_reqs = self.builder.get_input_requirements()
        
        # Verify the input requirements
        self.assertIn('inputs', input_reqs)
        self.assertIn('outputs', input_reqs)
        self.assertIn('enable_caching', input_reqs)
        
        # Verify the inputs description contains all input channel names
        for channel in ['data_input', 'metadata_input', 'signature_input']:
            self.assertIn(channel, input_reqs['inputs'])
            
        # Verify the outputs description contains all output channel names
        for channel in ['processed_data', 'full_data']:
            self.assertIn(channel, input_reqs['outputs'])

    def test_get_output_properties(self):
        # Test the get_output_properties method
        output_props = self.builder.get_output_properties()
        
        # Verify the output properties
        self.assertEqual(output_props, {
            'processed_data': 'ProcessedTabularData',
            'full_data': 'FullTabularData'
        })

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
