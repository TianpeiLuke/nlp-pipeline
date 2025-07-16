import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import os
import sys

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
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
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "processing_source_dir": self.temp_dir,
            "processing_entry_point": "currency_conversion.py",
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "processing_instance_type_small": "ml.m5.large",
            "processing_instance_type_large": "ml.m5.xlarge",
            "use_large_processing_instance": False,
            "processing_framework_version": "0.23-1",
            "job_type": "training",
            "mode": "per_split",
            "train_ratio": 0.7,
            "test_val_ratio": 0.5,
            "label_field": "target",
            "marketplace_id_col": "marketplace_id",
            "currency_conversion_var_list": ["price", "cost"],
            "currency_conversion_dict": {"USD": 1.0, "EUR": 0.85, "GBP": 0.75},
            "marketplace_info": {
                "US": {"currency_code": "USD"},
                "UK": {"currency_code": "GBP"},
                "DE": {"currency_code": "EUR"}
            },
            "default_currency": "USD",
            "enable_currency_conversion": True
        }
        
        # Create a real CurrencyConversionConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = CurrencyConversionConfig(**self.valid_config_data)
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Instantiate builder with the mocked config
        self.builder = CurrencyConversionStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.'),
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-currency-conversion-training')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_with_invalid_job_type(self):
        """Test that config creation raises ValidationError with invalid job type."""
        config_data = self.valid_config_data.copy()
        config_data['job_type'] = 'invalid_job_type'
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            # The validation error should occur at config creation time
            with self.assertRaises(Exception) as context:
                config = CurrencyConversionConfig(**config_data)
            # Should be a Pydantic validation error
            self.assertIn("job_type", str(context.exception))

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_attrs(self):
        """Test that configuration validation fails with missing required attributes."""
        # Directly modify the config object to have empty processing_instance_count
        original_count = self.builder.config.processing_instance_count
        object.__setattr__(self.builder.config, 'processing_instance_count', None)  # Set to None
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("processing_instance_count", str(context.exception))
        
        # Restore original count
        object.__setattr__(self.builder.config, 'processing_instance_count', original_count)

    def test_validate_configuration_invalid_job_type(self):
        """Test that configuration validation fails with invalid job_type."""
        # Directly modify the config object to have invalid job_type
        original_job_type = self.builder.config.job_type
        object.__setattr__(self.builder.config, 'job_type', 'invalid_type')  # Set invalid job type
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("Invalid job_type", str(context.exception))
        
        # Restore original job type
        object.__setattr__(self.builder.config, 'job_type', original_job_type)

    def test_validate_configuration_currency_conversion_missing_marketplace_id(self):
        """Test that configuration validation fails when currency conversion is enabled but marketplace_id_col is missing."""
        # Directly modify the config object to have empty marketplace_id_col
        original_marketplace_id = self.builder.config.marketplace_id_col
        object.__setattr__(self.builder.config, 'marketplace_id_col', "")  # Set empty marketplace_id_col
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("marketplace_id_col must be provided", str(context.exception))
        
        # Restore original marketplace_id_col
        object.__setattr__(self.builder.config, 'marketplace_id_col', original_marketplace_id)

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
        self.assertEqual(call_args['framework_version'], "0.23-1")
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], "ml.m5.large")  # Small instance
        self.assertEqual(call_args['instance_count'], 1)
        self.assertEqual(call_args['volume_size_in_gb'], 30)
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertTrue('base_job_name' in call_args)
        self.assertTrue('env' in call_args)
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    @patch('src.pipeline_steps.builder_currency_conversion_step.SKLearnProcessor')
    def test_create_processor_large_instance(self, mock_processor_cls):
        """Test that the processor uses large instance when configured."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Set use_large_processing_instance to True
        self.builder.config.use_large_processing_instance = True
        
        # Create processor
        processor = self.builder._create_processor()
        
        # Verify large instance type was used
        call_args = mock_processor_cls.call_args[1]
        self.assertEqual(call_args['instance_type'], "ml.m5.xlarge")  # Large instance

    def test_get_environment_variables(self):
        """Test that environment variables are set correctly."""
        env_vars = self.builder._get_environment_variables()
        
        # Verify required environment variables
        self.assertIn("CURRENCY_CONVERSION_VARS", env_vars)
        self.assertIn("CURRENCY_CONVERSION_DICT", env_vars)
        self.assertIn("MARKETPLACE_INFO", env_vars)
        self.assertIn("LABEL_FIELD", env_vars)
        self.assertEqual(env_vars["LABEL_FIELD"], "target")
        self.assertIn("TRAIN_RATIO", env_vars)
        self.assertEqual(env_vars["TRAIN_RATIO"], "0.7")
        self.assertIn("TEST_VAL_RATIO", env_vars)
        self.assertEqual(env_vars["TEST_VAL_RATIO"], "0.5")

    def test_get_inputs_with_spec(self):
        """Test that inputs are created correctly using specification."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "data_input"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"data_input": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "data_input": "/opt/ml/processing/input/data"
        }
        
        # Create inputs dictionary
        inputs = {
            "data_input": "s3://bucket/input.csv"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should have 1 input: data_input
        self.assertEqual(len(proc_inputs), 1)
        
        # Check data input
        data_input = proc_inputs[0]
        self.assertIsInstance(data_input, ProcessingInput)
        self.assertEqual(data_input.input_name, "data_input")
        self.assertEqual(data_input.source, "s3://bucket/input.csv")
        self.assertEqual(data_input.destination, "/opt/ml/processing/input/data")

    def test_get_inputs_missing_required(self):
        """Test that _get_inputs raises ValueError when required inputs are missing."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "data_input"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"data_input": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "data_input": "/opt/ml/processing/input/data"
        }
        
        # Test with empty inputs
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs({})
        self.assertIn("Required input 'data_input' not provided", str(context.exception))

    def test_get_inputs_no_spec(self):
        """Test that _get_inputs raises ValueError when no specification is available."""
        self.builder.spec = None
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs({})
        self.assertIn("Step specification is required", str(context.exception))

    def test_get_outputs_with_spec(self):
        """Test that outputs are created correctly using specification."""
        # Mock the spec and contract
        mock_output = MagicMock()
        mock_output.logical_name = "converted_data"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"converted_data": mock_output}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_output_paths = {
            "converted_data": "/opt/ml/processing/output"
        }
        
        # Create outputs dictionary
        outputs = {
            "converted_data": "s3://bucket/output/"
        }
        
        proc_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        
        # Check converted data output
        converted_output = proc_outputs[0]
        self.assertIsInstance(converted_output, ProcessingOutput)
        self.assertEqual(converted_output.output_name, "converted_data")
        self.assertEqual(converted_output.source, "/opt/ml/processing/output")
        self.assertEqual(converted_output.destination, "s3://bucket/output/")

    def test_get_outputs_generated_destination(self):
        """Test that outputs use generated destination when not provided."""
        # Mock the spec and contract
        mock_output = MagicMock()
        mock_output.logical_name = "converted_data"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"converted_data": mock_output}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_output_paths = {
            "converted_data": "/opt/ml/processing/output"
        }
        
        # Create empty outputs dictionary
        outputs = {}
        
        proc_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        
        # Check converted data output with generated destination
        converted_output = proc_outputs[0]
        self.assertIsInstance(converted_output, ProcessingOutput)
        self.assertEqual(converted_output.output_name, "converted_data")
        self.assertEqual(converted_output.source, "/opt/ml/processing/output")
        expected_dest = f"{self.config.pipeline_s3_loc}/currency_conversion/{self.config.job_type}/converted_data"
        self.assertEqual(converted_output.destination, expected_dest)

    def test_get_outputs_no_spec(self):
        """Test that _get_outputs raises ValueError when no specification is available."""
        self.builder.spec = None
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_outputs({})
        self.assertIn("Step specification is required", str(context.exception))

    def test_get_job_arguments(self):
        """Test that job arguments are created correctly."""
        job_args = self.builder._get_job_arguments()
        
        # Verify required job arguments
        self.assertIsInstance(job_args, list)
        self.assertIn("--job-type", job_args)
        self.assertIn("training", job_args)
        self.assertIn("--mode", job_args)
        self.assertIn("per_split", job_args)
        self.assertIn("--marketplace-id-col", job_args)
        self.assertIn("marketplace_id", job_args)
        self.assertIn("--default-currency", job_args)
        self.assertIn("USD", job_args)
        self.assertIn("--enable-conversion", job_args)
        self.assertIn("true", job_args)

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
        
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "data_input"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "converted_data"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"data_input": mock_dependency}
        self.builder.spec.outputs = {"converted_data": mock_output}
        self.builder.spec.step_type = "CurrencyConversion-Training"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "data_input": "/opt/ml/processing/input/data"
        }
        self.builder.contract.expected_output_paths = {
            "converted_data": "/opt/ml/processing/output"
        }
        
        # Create step with data_input
        step = self.builder.create_step(inputs={"data_input": "s3://bucket/input.csv"})
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'CurrencyConversion-Training')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertEqual(call_kwargs['depends_on'], [])
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in call_kwargs['inputs']))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in call_kwargs['outputs']))
        
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
        
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "data_input"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "converted_data"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"data_input": mock_dependency}
        self.builder.spec.outputs = {"converted_data": mock_output}
        self.builder.spec.step_type = "CurrencyConversion-Training"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "data_input": "/opt/ml/processing/input/data"
        }
        self.builder.contract.expected_output_paths = {
            "converted_data": "/opt/ml/processing/output"
        }
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies and data_input
        step = self.builder.create_step(
            inputs={"data_input": "s3://bucket/input.csv"},
            dependencies=dependencies
        )
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_currency_conversion_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_currency_conversion_step.ProcessingStep')
    def test_create_step_with_dependency_extraction(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the step extracts inputs from dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "data_input"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "converted_data"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"data_input": mock_dependency}
        self.builder.spec.outputs = {"converted_data": mock_output}
        self.builder.spec.step_type = "CurrencyConversion-Training"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "data_input": "/opt/ml/processing/input/data"
        }
        self.builder.contract.expected_output_paths = {
            "converted_data": "/opt/ml/processing/output"
        }
        
        # Mock extract_inputs_from_dependencies
        self.builder.extract_inputs_from_dependencies = MagicMock(
            return_value={"data_input": "s3://bucket/extracted_input.csv"}
        )
        
        # Setup mock dependency
        dependency = MagicMock()
        
        # Create step with dependency but no direct inputs
        step = self.builder.create_step(dependencies=[dependency])
        
        # Verify extract_inputs_from_dependencies was called
        self.builder.extract_inputs_from_dependencies.assert_called_once_with([dependency])
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], [dependency])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

if __name__ == '__main__':
    unittest.main()
