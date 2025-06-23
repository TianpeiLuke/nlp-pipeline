import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

# Add the project root to the Python path to allow for absolute imports
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

class TestXGBoostModelEvalStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a dummy hyperparameters object
        self.hyperparameters = SimpleNamespace()
        self.hyperparameters.id_name = "id"
        self.hyperparameters.label_name = "label"
        
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        
        # Required attributes for validation
        self.config.processing_entry_point = 'model_evaluation_xgboost.py'
        self.config.processing_source_dir = '/path/to/scripts'
        self.config.processing_instance_count = 1
        self.config.processing_volume_size = 30
        self.config.pipeline_name = 'test-pipeline'
        self.config.job_type = 'validation'
        self.config.hyperparameters = self.hyperparameters
        self.config.xgboost_framework_version = '1.5-1'
        
        # Processing configuration
        self.config.use_large_processing_instance = False
        self.config.processing_instance_type_small = 'ml.m5.large'
        self.config.processing_instance_type_large = 'ml.m5.4xlarge'
        
        # Input/output channels
        self.config.INPUT_CHANNELS = {
            "model_input": "Model artifacts input",
            "eval_data_input": "Evaluation data input"
        }
        
        self.config.OUTPUT_CHANNELS = {
            "eval_output": "Output name for evaluation predictions",
            "metrics_output": "Output name for evaluation metrics"
        }
        
        # Add input_names and output_names attributes for compatibility with the builder
        self.config.input_names = self.config.INPUT_CHANNELS
        self.config.output_names = self.config.OUTPUT_CHANNELS
        
        # Methods
        self.config.get_input_names = MagicMock(return_value=self.config.INPUT_CHANNELS)
        self.config.get_output_names = MagicMock(return_value=self.config.OUTPUT_CHANNELS)
        self.config.get_instance_type = MagicMock(return_value='ml.m5.large')
        self.config.get_script_path = MagicMock(return_value='model_evaluation_xgboost.py')
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(XGBoostModelEvalStepBuilder)
        self.builder.config = self.config
        
        # Create a properly configured session mock
        session_mock = MagicMock()
        session_mock.sagemaker_config = {}
        self.builder.session = session_mock
        
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        
        # Mock methods from the base class
        self.builder._get_step_name = MagicMock(return_value='XGBoostModelEval')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-xgb-eval')

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_attribute(self):
        """Test that configuration validation fails with missing required attribute."""
        # Save original value
        original_value = self.config.processing_entry_point
        # Set to None to trigger validation error
        self.config.processing_entry_point = None
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original value
        self.config.processing_entry_point = original_value

    def test_validate_configuration_missing_input_names(self):
        """Test that configuration validation fails with missing required input names."""
        # Save original input_names
        original_input_names = self.config.INPUT_CHANNELS
        # Set input_names to a dict missing required keys
        self.config.INPUT_CHANNELS = {"wrong_name": "description"}
        # Also update the input_names attribute
        self.config.input_names = self.config.INPUT_CHANNELS
        # Mock get_input_names to return the modified input_names
        self.config.get_input_names = MagicMock(return_value=self.config.INPUT_CHANNELS)
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original input_names and mock
        self.config.INPUT_CHANNELS = original_input_names
        self.config.input_names = original_input_names
        self.config.get_input_names = MagicMock(return_value=self.config.INPUT_CHANNELS)

    def test_validate_configuration_missing_output_names(self):
        """Test that configuration validation fails with missing required output names."""
        # Save original output_names
        original_output_names = self.config.OUTPUT_CHANNELS
        # Set output_names to a dict missing required keys
        self.config.OUTPUT_CHANNELS = {"wrong_name": "description"}
        # Also update the output_names attribute
        self.config.output_names = self.config.OUTPUT_CHANNELS
        # Mock get_output_names to return the modified output_names
        self.config.get_output_names = MagicMock(return_value=self.config.OUTPUT_CHANNELS)
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original output_names and mock
        self.config.OUTPUT_CHANNELS = original_output_names
        self.config.output_names = original_output_names
        self.config.get_output_names = MagicMock(return_value=self.config.OUTPUT_CHANNELS)

    def test_get_environment_variables(self):
        """Test that environment variables are created correctly."""
        env_vars = self.builder._get_environment_variables()
        
        self.assertEqual(env_vars["ID_FIELD"], "id")
        self.assertEqual(env_vars["LABEL_FIELD"], "label")

    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.XGBoostProcessor')
    def test_create_processor(self, mock_processor_cls):
        """Test that the processor is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Create processor
        processor = self.builder._create_processor()
        
        # Verify XGBoostProcessor was created with correct parameters
        self.assertEqual(mock_processor_cls.call_count, 1)
        call_args = mock_processor_cls.call_args[1]
        self.assertEqual(call_args['framework_version'], "1.5-1")
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], 'ml.m5.large')
        self.assertEqual(call_args['instance_count'], 1)
        self.assertEqual(call_args['volume_size_in_gb'], 30)
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertEqual(call_args['base_job_name'], 'test-pipeline-xgb-eval-xgb-eval')
        
        # Verify environment variables
        env_vars = call_args['env']
        self.assertEqual(env_vars["ID_FIELD"], "id")
        self.assertEqual(env_vars["LABEL_FIELD"], "label")
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    def test_get_processing_inputs(self):
        """Test that processing inputs are created correctly."""
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "eval_data_input": "s3://bucket/eval_data"
        }
        
        processing_inputs = self.builder._get_processing_inputs(inputs)
        
        self.assertEqual(len(processing_inputs), 2)
        
        # Check model input
        model_input = processing_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        self.assertEqual(model_input.input_name, "model_input")
        
        # Check eval data input
        eval_input = processing_inputs[1]
        self.assertIsInstance(eval_input, ProcessingInput)
        self.assertEqual(eval_input.source, "s3://bucket/eval_data")
        self.assertEqual(eval_input.destination, "/opt/ml/processing/input/eval_data")
        self.assertEqual(eval_input.input_name, "eval_data_input")

    def test_get_processing_inputs_missing_required(self):
        """Test that _get_processing_inputs raises ValueError with missing required inputs."""
        # Missing eval_data_input
        inputs = {
            "model_input": "s3://bucket/model.tar.gz"
        }
        
        with self.assertRaises(ValueError):
            self.builder._get_processing_inputs(inputs)

    def test_get_processing_outputs(self):
        """Test that processing outputs are created correctly."""
        outputs = {
            "eval_output": "s3://bucket/eval_output",
            "metrics_output": "s3://bucket/metrics_output"
        }
        
        processing_outputs = self.builder._get_processing_outputs(outputs)
        
        self.assertEqual(len(processing_outputs), 2)
        
        # Check eval output
        eval_output = processing_outputs[0]
        self.assertIsInstance(eval_output, ProcessingOutput)
        self.assertEqual(eval_output.source, "/opt/ml/processing/output/eval")
        self.assertEqual(eval_output.destination, "s3://bucket/eval_output")
        self.assertEqual(eval_output.output_name, "eval_output")
        
        # Check metrics output
        metrics_output = processing_outputs[1]
        self.assertIsInstance(metrics_output, ProcessingOutput)
        self.assertEqual(metrics_output.source, "/opt/ml/processing/output/metrics")
        self.assertEqual(metrics_output.destination, "s3://bucket/metrics_output")
        self.assertEqual(metrics_output.output_name, "metrics_output")

    def test_get_processing_outputs_missing_required(self):
        """Test that _get_processing_outputs raises ValueError with missing required outputs."""
        # Missing metrics_output
        outputs = {
            "eval_output": "s3://bucket/eval_output"
        }
        
        with self.assertRaises(ValueError):
            self.builder._get_processing_outputs(outputs)

    def test_get_job_arguments(self):
        """Test that job arguments are created correctly."""
        job_args = self.builder._get_job_arguments()
        
        self.assertEqual(len(job_args), 2)
        self.assertEqual(job_args[0], "--job_type")
        self.assertEqual(job_args[1], "validation")

    def test_get_cache_config(self):
        """Test that cache config is created correctly."""
        # With caching enabled
        cache_config = self.builder._get_cache_config(True)
        self.assertTrue(cache_config.enable_caching)
        self.assertEqual(cache_config.expire_after, "30d")
        
        # With caching disabled
        cache_config = self.builder._get_cache_config(False)
        self.assertIsNone(cache_config)

    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.XGBoostProcessor')
    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.ProcessingStep')
    def test_create_step(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock step_args
        mock_step_args = MagicMock()
        mock_processor.run.return_value = mock_step_args
        
        # Create step
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "eval_data_input": "s3://bucket/eval_data"
        }
        
        outputs = {
            "eval_output": "s3://bucket/eval_output",
            "metrics_output": "s3://bucket/metrics_output"
        }
        
        step = self.builder.create_step(inputs, outputs)
        
        # Verify processor.run was called with correct parameters
        mock_processor.run.assert_called_once()
        run_args = mock_processor.run.call_args[1]
        self.assertEqual(run_args['code'], 'model_evaluation_xgboost.py')
        self.assertEqual(run_args['source_dir'], '/path/to/scripts')
        self.assertEqual(len(run_args['inputs']), 2)
        self.assertEqual(len(run_args['outputs']), 2)
        self.assertEqual(run_args['arguments'], ["--job_type", "validation"])
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'XGBoostModelEval-Validation')
        self.assertEqual(call_kwargs['step_args'], mock_step_args)
        self.assertEqual(call_kwargs['depends_on'], [])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.XGBoostProcessor')
    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.ProcessingStep')
    def test_create_step_with_dependencies(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock step_args
        mock_step_args = MagicMock()
        mock_processor.run.return_value = mock_step_args
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "eval_data_input": "s3://bucket/eval_data"
        }
        
        outputs = {
            "eval_output": "s3://bucket/eval_output",
            "metrics_output": "s3://bucket/metrics_output"
        }
        
        step = self.builder.create_step(inputs, outputs, dependencies=dependencies)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.XGBoostProcessor')
    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.ProcessingStep')
    def test_create_step_with_caching_disabled(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with caching disabled."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock step_args
        mock_step_args = MagicMock()
        mock_processor.run.return_value = mock_step_args
        
        # Create step with caching disabled
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "eval_data_input": "s3://bucket/eval_data"
        }
        
        outputs = {
            "eval_output": "s3://bucket/eval_output",
            "metrics_output": "s3://bucket/metrics_output"
        }
        
        step = self.builder.create_step(inputs, outputs, enable_caching=False)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertIsNone(call_kwargs['cache_config'])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
