import unittest
import tempfile
import shutil
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

# Import the builder class and config class to be tested
from src.pipeline_steps.builder_currency_conversion_step import CurrencyConversionStepBuilder
from src.pipeline_steps.config_currency_conversion_step import CurrencyConversionConfig

class TestCurrencyConversionStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'currency_conversion.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy currency conversion script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a valid config for the CurrencyConversionConfig
        self.valid_config_data = {
            "job_type": "training",
            "mode": "per_split",
            "train_ratio": 0.7,
            "test_val_ratio": 0.5,
            "label_field": "target",
            "processing_entry_point": "currency_conversion.py",
            "processing_source_dir": self.temp_dir,
            "processing_framework_version": "0.23-1",
            "processing_instance_type_large": "ml.m5.4xlarge",
            "processing_instance_type_small": "ml.m5.large",
            "use_large_processing_instance": False,
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "marketplace_id_col": "marketplace_id",
            "currency_conversion_var_list": ["price", "cost"],
            "currency_conversion_dict": {"USD": 1.0, "EUR": 0.85, "GBP": 0.75},
            "marketplace_info": {
                "US": {"currency_code": "USD"},
                "UK": {"currency_code": "GBP"},
                "DE": {"currency_code": "EUR"}
            },
            "default_currency": "USD",
            "enable_currency_conversion": True,
            "input_names": {"data_input": "ProcessedTabularData"},
            "output_names": {"converted_data": "ConvertedCurrencyData"},
            "pipeline_s3_loc": "s3://bucket/pipeline",
            "pipeline_name": "test-pipeline"
        }
        
        # Create a real CurrencyConversionConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = CurrencyConversionConfig(**self.valid_config_data)
        
        # Instantiate builder with the real config
        self.builder = CurrencyConversionStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.')
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._get_step_name = MagicMock(return_value='Currency_Conversion')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-currency-conversion')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())
        self.builder._extract_param = MagicMock(side_effect=lambda kwargs, key, default=None: kwargs.get(key, default))

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_fields(self):
        """Test that configuration validation fails with missing required fields."""
        # Test missing processing_instance_count
        with patch.object(self.config, 'processing_instance_count', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing processing_volume_size
        with patch.object(self.config, 'processing_volume_size', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing processing_entry_point
        with patch.object(self.config, 'processing_entry_point', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing processing_source_dir
        with patch.object(self.config, 'processing_source_dir', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing processing_framework_version
        with patch.object(self.config, 'processing_framework_version', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing job_type
        with patch.object(self.config, 'job_type', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()

    def test_validate_configuration_invalid_job_type(self):
        """Test that configuration validation fails with invalid job_type."""
        with patch.object(self.config, 'job_type', 'invalid_job_type'):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()

    @patch('src.pipeline_steps.builder_currency_conversion_step.SKLearnProcessor')
    def test_create_processor(self, mock_processor_cls):
        """Test that the processor is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Create processor
        processor = self.builder._create_processor()
        
        # Verify SKLearnProcessor was created with correct parameters
        mock_processor_cls.assert_called_once()
        call_args = mock_processor_cls.call_args[1]
        self.assertEqual(call_args['framework_version'], self.config.processing_framework_version)
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], self.config.processing_instance_type_small)  # Default small instance
        self.assertEqual(call_args['instance_count'], self.config.processing_instance_count)
        self.assertEqual(call_args['volume_size_in_gb'], self.config.processing_volume_size)
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertTrue('base_job_name' in call_args)
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)
        
    @patch('src.pipeline_steps.builder_currency_conversion_step.SKLearnProcessor')
    def test_create_processor_large_instance(self, mock_processor_cls):
        """Test that the processor is created with large instance when configured."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Set use_large_processing_instance to True
        with patch.object(self.config, 'use_large_processing_instance', True):
            # Create processor
            processor = self.builder._create_processor()
            
            # Verify large instance type was used
            call_args = mock_processor_cls.call_args[1]
            self.assertEqual(call_args['instance_type'], self.config.processing_instance_type_large)

    def test_get_processor_inputs(self):
        """Test that processor inputs are created correctly."""
        # Create inputs dictionary with required keys
        inputs = {
            self.config.input_names["data_input"]: "s3://bucket/input"
        }
        
        proc_inputs = self.builder._get_processor_inputs(inputs)
        
        self.assertEqual(len(proc_inputs), 1)
        self.assertIsInstance(proc_inputs[0], ProcessingInput)
        self.assertEqual(proc_inputs[0].source, "s3://bucket/input")
        self.assertEqual(proc_inputs[0].destination, "/opt/ml/processing/input/data")
        self.assertEqual(proc_inputs[0].input_name, "ProcessedTabularData")
        
    def test_get_processor_inputs_with_exchange_rates(self):
        """Test that processor inputs include exchange rates when provided."""
        # Add exchange_rates_input to config
        with patch.object(self.config, 'input_names', {
            "data_input": "ProcessedTabularData",
            "exchange_rates_input": "ExchangeRatesInput"
        }):
            # Create inputs dictionary with both keys
            inputs = {
                "ProcessedTabularData": "s3://bucket/input",
                "ExchangeRatesInput": "s3://bucket/exchange_rates"
            }
            
            proc_inputs = self.builder._get_processor_inputs(inputs)
            
            self.assertEqual(len(proc_inputs), 2)
            
            # Check data input
            data_input = next(i for i in proc_inputs if i.input_name == "ProcessedTabularData")
            self.assertEqual(data_input.source, "s3://bucket/input")
            self.assertEqual(data_input.destination, "/opt/ml/processing/input/data")
            
            # Check exchange rates input
            exchange_rates_input = next(i for i in proc_inputs if i.input_name == "ExchangeRatesInput")
            self.assertEqual(exchange_rates_input.source, "s3://bucket/exchange_rates")
            self.assertEqual(exchange_rates_input.destination, "/opt/ml/processing/input/exchange_rates")
    
    def test_get_processor_inputs_missing(self):
        """Test that _get_processor_inputs raises ValueError when inputs are missing."""
        # Test with empty inputs
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({})

    def test_get_processor_outputs(self):
        """Test that processor outputs are created correctly."""
        # Create outputs dictionary with required keys
        outputs = {
            "ConvertedCurrencyData": "s3://bucket/output"
        }
        
        proc_outputs = self.builder._get_processor_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        self.assertIsInstance(proc_outputs[0], ProcessingOutput)
        self.assertEqual(proc_outputs[0].source, "/opt/ml/processing/output")
        self.assertEqual(proc_outputs[0].destination, "s3://bucket/output")
        self.assertEqual(proc_outputs[0].output_name, "ConvertedCurrencyData")
    
    def test_get_processor_outputs_missing(self):
        """Test that _get_processor_outputs raises ValueError when outputs are missing."""
        # Test with empty outputs
        with self.assertRaises(ValueError):
            self.builder._get_processor_outputs({})
            
    def test_get_environment_variables(self):
        """Test that environment variables are set correctly."""
        env_vars = self.builder._get_environment_variables()
        
        # Verify required environment variables
        self.assertIn("JOB_TYPE", env_vars)
        self.assertEqual(env_vars["JOB_TYPE"], self.config.job_type)
        self.assertIn("CURRENCY_FIELD", env_vars)
        self.assertEqual(env_vars["CURRENCY_FIELD"], self.config.currency_field)
        self.assertIn("AMOUNT_FIELD", env_vars)
        self.assertEqual(env_vars["AMOUNT_FIELD"], self.config.amount_field)
        self.assertIn("TARGET_CURRENCY", env_vars)
        self.assertEqual(env_vars["TARGET_CURRENCY"], self.config.target_currency)

    @patch('src.pipeline_steps.builder_currency_conversion_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_currency_conversion_step.ProcessingStep')
    def test_create_step(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Create step
        inputs = {
            "inputs": {
                "ProcessedTabularData": "s3://bucket/input"
            }
        }
        outputs = {
            "outputs": {
                "ConvertedCurrencyData": "s3://bucket/output"
            }
        }
        step = self.builder.create_step(inputs=inputs, outputs=outputs)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'Currency_Conversion-Training')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in call_kwargs['inputs']))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in call_kwargs['outputs']))
        self.assertEqual(call_kwargs['depends_on'], [])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    @patch('src.pipeline_steps.builder_currency_conversion_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_currency_conversion_step.ProcessingStep')
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
        inputs = {
            "inputs": {
                "ProcessedTabularData": "s3://bucket/input"
            }
        }
        outputs = {
            "outputs": {
                "ConvertedCurrencyData": "s3://bucket/output"
            }
        }
        step = self.builder.create_step(inputs=inputs, outputs=outputs, dependencies=dependencies)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
    
    def test_get_input_requirements(self):
        """Test that input requirements are returned correctly."""
        input_reqs = self.builder.get_input_requirements()
        
        # Verify input requirements
        self.assertIn("inputs", input_reqs)
        self.assertIn("outputs", input_reqs)
        self.assertIn("enable_caching", input_reqs)
        
        # Verify inputs description contains input channel names
        self.assertIn("data_input", input_reqs["inputs"])
        
        # Verify outputs description contains output channel names
        self.assertIn("converted_data", input_reqs["outputs"])
    
    def test_get_output_properties(self):
        """Test that output properties are returned correctly."""
        output_props = self.builder.get_output_properties()
        
        # Verify output properties
        self.assertEqual(output_props, {"converted_data": "ConvertedCurrencyData"})

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
