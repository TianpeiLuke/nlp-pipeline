import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig

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
        self.config.output_names = {'processed_data': 'processed_data'}
        # Ratios
        self.config.train_ratio = 0.7
        self.config.test_val_ratio = 0.3
        # Processing settings
        self.config.framework_version = '0.23-1'
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
        self.builder.aws_region = TabularPreprocessingStepBuilder.REGION_MAPPING['NA']

    def test_get_environment_variables(self):
        env = TabularPreprocessingStepBuilder._get_environment_variables(self.builder)
        self.assertEqual(env, {
            'LABEL_FIELD': 'target',
            'TRAIN_RATIO': '0.7',
            'TEST_VAL_RATIO': '0.3'
        })

    def test_get_processor_inputs_success(self):
        inputs = {'raw_data': 's3://bucket/raw'}
        proc_inputs = TabularPreprocessingStepBuilder._get_processor_inputs(self.builder, inputs)
        self.assertIsInstance(proc_inputs, list)
        self.assertIsInstance(proc_inputs[0], ProcessingInput)
        self.assertEqual(proc_inputs[0].input_name, 'raw_data')
        self.assertEqual(proc_inputs[0].source, 's3://bucket/raw')
        self.assertEqual(proc_inputs[0].destination, '/opt/ml/processing/input/data')

    def test_get_processor_inputs_missing(self):
        with self.assertRaises(ValueError):
            TabularPreprocessingStepBuilder._get_processor_inputs(self.builder, {})

    def test_get_processor_outputs_success(self):
        outputs = {'processed_data': 's3://bucket/processed'}
        proc_outputs = TabularPreprocessingStepBuilder._get_processor_outputs(self.builder, outputs)
        self.assertIsInstance(proc_outputs, list)
        self.assertIsInstance(proc_outputs[0], ProcessingOutput)
        self.assertEqual(proc_outputs[0].output_name, 'processed_data')
        self.assertEqual(proc_outputs[0].destination, 's3://bucket/processed')
        self.assertEqual(proc_outputs[0].source, '/opt/ml/processing/output')

    def test_get_processor_outputs_missing(self):
        with self.assertRaises(ValueError):
            TabularPreprocessingStepBuilder._get_processor_outputs(self.builder, {})

    def test_get_job_arguments(self):
        args = TabularPreprocessingStepBuilder._get_job_arguments(self.builder)
        self.assertEqual(args, ['--job_type', 'training'])

    @patch('src.pipelines.builder_tabular_preprocessing_step.SKLearnProcessor')
    def test_create_step(self, mock_processor_cls):
        # Stub processor instance
        dummy_processor = MagicMock()
        mock_processor_cls.return_value = dummy_processor
        # Provide inputs and outputs
        inputs = {'raw_data': 's3://bucket/raw'}
        outputs = {'processed_data': 's3://bucket/processed'}
        # Create step with caching disabled
        step = self.builder.create_step(inputs=inputs, outputs=outputs, enable_caching=False)
        # Verify the ProcessingStep
        self.assertIsInstance(step, ProcessingStep)
        # Name should be <ProcessingStep>-Training
        self.assertEqual(step.name, 'ProcessingStep-Training')
        # Processor should be our dummy
        self.assertIs(step.processor, dummy_processor)
        # Code path
        self.assertEqual(step.code, '/path/to/preprocess.py')
        # Job arguments include job type
        self.assertEqual(step.job_arguments, ['--job_type', 'training'])
        # Inputs and outputs
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in step.inputs))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in step.outputs))
        # CacheConfig
        self.assertIsInstance(step.cache_config, CacheConfig)
        self.assertFalse(step.cache_config.enable_caching)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
