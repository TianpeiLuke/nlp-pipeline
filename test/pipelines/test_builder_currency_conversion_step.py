import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipelines.builder_currency_conversion_step import CurrencyConversionStepBuilder

class TestCurrencyConversionStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        
        # Required attributes for validation
        self.config.processing_entry_point = 'currency_conversion.py'
        self.config.enable_currency_conversion = True
        self.config.marketplace_id_col = 'marketplace_id'
        self.config.currency_col = None
        self.config.currency_conversion_var_list = ['price', 'cost']
        self.config.currency_conversion_dict = {'USD': 1.0, 'EUR': 0.85, 'GBP': 0.75}
        self.config.marketplace_info = {
            'US': {'currency_code': 'USD'},
            'UK': {'currency_code': 'GBP'},
            'DE': {'currency_code': 'EUR'}
        }
        self.config.skip_invalid_currencies = False
        self.config.default_currency = 'USD'
        
        # Processing configuration
        self.config.job_type = 'training'
        self.config.mode = 'per_split'
        self.config.train_ratio = 0.7
        self.config.test_val_ratio = 0.5
        self.config.label_field = 'target'
        self.config.use_large_processing_instance = False
        self.config.processing_instance_count = 1
        self.config.processing_volume_size = 30
        
        # IO configuration
        self.config.input_names = {'data_input': 'ProcessedTabularData'}
        self.config.output_names = {'converted_data': 'ConvertedCurrencyData'}
        self.config.pipeline_s3_loc = 's3://bucket/pipeline'
        self.config.pipeline_name = 'test-pipeline'
        
        # Methods
        self.config.get_script_path = MagicMock(return_value='currency_conversion.py')
        # Mock Path.exists to return True for script validation
        self.path_exists_patch = patch('pathlib.Path.exists', return_value=True)
        self.path_exists_patch.start()
        self.config.get_instance_type = MagicMock(return_value='ml.m5.large')
        self.config.get_script_arguments = MagicMock(return_value=[
            '--data-type', 'training',
            '--mode', 'per_split',
            '--marketplace-id-col', 'marketplace_id',
            '--default-currency', 'USD',
            '--enable-conversion', 'true'
        ])
        self.config.get_environment_variables = MagicMock(return_value={
            'CURRENCY_CONVERSION_VARS': '["price", "cost"]',
            'CURRENCY_CONVERSION_DICT': '{"USD": 1.0, "EUR": 0.85, "GBP": 0.75}',
            'MARKETPLACE_INFO': '{"US": {"currency_code": "USD"}, "UK": {"currency_code": "GBP"}, "DE": {"currency_code": "EUR"}}',
            'LABEL_FIELD': 'target',
            'TRAIN_RATIO': '0.7',
            'TEST_VAL_RATIO': '0.5'
        })
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(CurrencyConversionStepBuilder)
        self.builder.config = self.config
        
        # Create a properly configured session mock
        session_mock = MagicMock()
        # Set sagemaker_config to an empty dict to pass validation
        session_mock.sagemaker_config = {}
        self.builder.session = session_mock
        
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        # Use a mock for the _get_step_name method from the base class
        self.builder._get_step_name = MagicMock(return_value='Currency_Conversion')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-currency-conversion')

    def tearDown(self):
        """Clean up after each test."""
        self.path_exists_patch.stop()
        
    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_marketplace_info(self):
        """Test that configuration validation fails with missing marketplace info."""
        self.config.marketplace_info = {}
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()

    def test_validate_configuration_missing_conversion_rates(self):
        """Test that configuration validation fails with missing conversion rates."""
        # Add a currency to marketplace_info that's not in currency_conversion_dict
        self.config.marketplace_info['JP'] = {'currency_code': 'JPY'}
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
        
        # Should pass if skip_invalid_currencies is True
        self.config.skip_invalid_currencies = True
        self.builder.validate_configuration()  # Should not raise

    @patch('src.pipelines.builder_currency_conversion_step.SKLearnProcessor')
    def test_create_processor(self, mock_processor_cls):
        """Test that the processor is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Create processor
        processor = self.builder._create_processor()
        
        # Verify SKLearnProcessor was created with correct parameters
        mock_processor_cls.assert_called_once_with(
            framework_version="1.2-1",
            role=self.builder.role,
            instance_type='ml.m5.large',
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            sagemaker_session=self.builder.session,
            base_job_name='test-pipeline-currency-conversion'
        )
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    def test_get_processing_inputs(self):
        """Test that processing inputs are created correctly."""
        data_input = 's3://bucket/input'
        inputs = self.builder._get_processing_inputs(data_input)
        
        self.assertEqual(len(inputs), 1)
        self.assertIsInstance(inputs[0], ProcessingInput)
        self.assertEqual(inputs[0].source, data_input)
        self.assertEqual(inputs[0].destination, '/opt/ml/processing/input/data')
        self.assertEqual(inputs[0].input_name, 'ProcessedTabularData')

    def test_get_processing_outputs(self):
        """Test that processing outputs are created correctly."""
        outputs = self.builder._get_processing_outputs()
        
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0], ProcessingOutput)
        self.assertEqual(outputs[0].source, '/opt/ml/processing/output')
        self.assertEqual(outputs[0].destination, 's3://bucket/pipeline/currency_conversion')
        self.assertEqual(outputs[0].output_name, 'ConvertedCurrencyData')

    @patch('src.pipelines.builder_currency_conversion_step.SKLearnProcessor')
    @patch('src.pipelines.builder_currency_conversion_step.ProcessingStep')
    def test_create_step(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Create step
        data_input = 's3://bucket/input'
        step = self.builder.create_step(data_input=data_input)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'Currency_Conversion')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertEqual(call_kwargs['code'], self.config.get_script_path())
        self.assertEqual(call_kwargs['depends_on'], [])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    @patch('src.pipelines.builder_currency_conversion_step.SKLearnProcessor')
    @patch('src.pipelines.builder_currency_conversion_step.ProcessingStep')
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
        data_input = 's3://bucket/input'
        step = self.builder.create_step(data_input=data_input, dependencies=dependencies)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    def test_create_conversion_step_backward_compatibility(self):
        """Test that the old create_conversion_step method calls the new create_step method."""
        with patch.object(self.builder, 'create_step', return_value="step_created") as mock_create_step:
            # Call the old method
            result = self.builder.create_conversion_step('s3://bucket/input')
            # Verify it called the new method
            mock_create_step.assert_called_once_with('s3://bucket/input', None)
            self.assertEqual(result, "step_created")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
