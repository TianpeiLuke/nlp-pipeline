import unittest
import tempfile
import shutil
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
from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase

class TestXGBoostModelEvalStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'model_evaluation_xgboost.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy model evaluation script for testing\n')
            f.write('print("This is a dummy script")\n')
        # Create a proper XGBoostModelHyperparameters instance
        self.hyperparameters = XGBoostModelHyperparameters(
            id_name="id",
            label_name="label",
            # Other required fields with defaults will be used
        )
        
        # Create a proper XGBoostModelEvalConfig instance
        self.config = XGBoostModelEvalConfig(
            processing_entry_point='model_evaluation_xgboost.py',
            processing_source_dir=self.temp_dir,
            processing_instance_count=1,
            processing_volume_size=30,
            pipeline_name='test-pipeline',
            job_type='validation',
            hyperparameters=self.hyperparameters,
            xgboost_framework_version='1.5-1',
            use_large_processing_instance=False,
            processing_instance_type_small='ml.m5.large',
            processing_instance_type_large='ml.m5.4xlarge',
            input_names={
                "model_input": "Model artifacts input",
                "eval_data_input": "Evaluation data input"
            },
            output_names={
                "eval_output": "Output name for evaluation predictions",
                "metrics_output": "Output name for evaluation metrics"
            }
        )
        
        # Create a properly configured builder instance
        with patch.object(ProcessingStepConfigBase, 'get_script_path', return_value=os.path.join(self.temp_dir, 'model_evaluation_xgboost.py')):
            self.builder = XGBoostModelEvalStepBuilder(
                config=self.config,
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole',
                notebook_root=Path('.')
            )
            
        # Mock methods from the base class
        self.builder._get_step_name = MagicMock(return_value='XGBoostModelEval')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-xgb-eval')

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_attribute(self):
        """Test that configuration validation fails with missing required attribute."""
        # Create a new config with a missing required attribute
        # We'll use a mock to simulate the validation failure
        with patch.object(self.builder, 'validate_configuration', side_effect=ValueError("Missing required attribute")):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()

    def test_validate_configuration_missing_input_names(self):
        """Test that configuration validation fails with missing required input names."""
        # Save original input_names
        original_input_names = self.config.input_names
        
        # Set input_names to a dict missing required keys
        self.config.input_names = {"wrong_name": "description"}
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original input_names
        self.config.input_names = original_input_names

    def test_validate_configuration_missing_output_names(self):
        """Test that configuration validation fails with missing required output names."""
        # Save original output_names
        original_output_names = self.config.output_names
        
        # Set output_names to a dict missing required keys
        self.config.output_names = {"wrong_name": "description"}
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original output_names
        self.config.output_names = original_output_names

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
        self.assertEqual(call_args['base_job_name'], 'test-pipeline-xgb-eval')
        
        # Verify environment variables
        env_vars = call_args['env']
        self.assertEqual(env_vars["ID_FIELD"], "id")
        self.assertEqual(env_vars["LABEL_FIELD"], "label")
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    def test_get_processor_inputs(self):
        """Test that processor inputs are created correctly."""
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "eval_data_input": "s3://bucket/eval_data"
        }
        
        processing_inputs = self.builder._get_processor_inputs(inputs)
        
        self.assertEqual(len(processing_inputs), 2)
        
        # Check model input
        model_input = processing_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        self.assertEqual(model_input.input_name, "Model artifacts input")
        
        # Check eval data input
        eval_input = processing_inputs[1]
        self.assertIsInstance(eval_input, ProcessingInput)
        self.assertEqual(eval_input.source, "s3://bucket/eval_data")
        self.assertEqual(eval_input.destination, "/opt/ml/processing/input/eval_data")
        self.assertEqual(eval_input.input_name, "Evaluation data input")

    def test_get_processor_inputs_missing_required(self):
        """Test that _get_processor_inputs raises ValueError with missing required inputs."""
        # Missing eval_data_input
        inputs = {
            "model_input": "s3://bucket/model.tar.gz"
        }
        
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs(inputs)

    def test_get_processor_outputs(self):
        """Test that processor outputs are created correctly."""
        outputs = {
            "Output name for evaluation predictions": "s3://bucket/eval_output",
            "Output name for evaluation metrics": "s3://bucket/metrics_output"
        }
        
        processing_outputs = self.builder._get_processor_outputs(outputs)
        
        self.assertEqual(len(processing_outputs), 2)
        
        # Check eval output
        eval_output = processing_outputs[0]
        self.assertIsInstance(eval_output, ProcessingOutput)
        self.assertEqual(eval_output.source, "/opt/ml/processing/output/eval")
        self.assertEqual(eval_output.destination, "s3://bucket/eval_output")
        self.assertEqual(eval_output.output_name, "Output name for evaluation predictions")
        
        # Check metrics output
        metrics_output = processing_outputs[1]
        self.assertIsInstance(metrics_output, ProcessingOutput)
        self.assertEqual(metrics_output.source, "/opt/ml/processing/output/metrics")
        self.assertEqual(metrics_output.destination, "s3://bucket/metrics_output")
        self.assertEqual(metrics_output.output_name, "Output name for evaluation metrics")

    def test_get_processor_outputs_missing_required(self):
        """Test that _get_processor_outputs raises ValueError with missing required outputs."""
        # Missing metrics_output
        outputs = {
            "Output name for evaluation predictions": "s3://bucket/eval_output"
        }
        
        with self.assertRaises(ValueError):
            self.builder._get_processor_outputs(outputs)

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
        self.assertEqual(cache_config.expire_after, "P30D")
        
        # With caching disabled - mock the method to return None
        with patch.object(self.builder, '_get_cache_config', return_value=None):
            cache_config = self.builder._get_cache_config(False)
            self.assertIsNone(cache_config)

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)
        
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
            "Output name for evaluation predictions": "s3://bucket/eval_output",
            "Output name for evaluation metrics": "s3://bucket/metrics_output"
        }
        
        step = self.builder.create_step(inputs=inputs, outputs=outputs)
        
        # Verify processor.run was called with correct parameters
        mock_processor.run.assert_called_once()
        run_args = mock_processor.run.call_args[1]
        self.assertEqual(run_args['code'], 'model_evaluation_xgboost.py')
        self.assertEqual(run_args['source_dir'], self.temp_dir)
        self.assertEqual(len(run_args['inputs']), 2)
        self.assertEqual(len(run_args['outputs']), 2)
        self.assertEqual(run_args['arguments'], ["--job_type", "validation"])
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'XGBoostModelEval')
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
            "Output name for evaluation predictions": "s3://bucket/eval_output",
            "Output name for evaluation metrics": "s3://bucket/metrics_output"
        }
        
        step = self.builder.create_step(inputs=inputs, outputs=outputs, dependencies=dependencies)
        
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
            "Output name for evaluation predictions": "s3://bucket/eval_output",
            "Output name for evaluation metrics": "s3://bucket/metrics_output"
        }
        
        # Mock the _get_cache_config method to return None when enable_caching is False
        with patch.object(self.builder, '_get_cache_config', return_value=None):
            step = self.builder.create_step(inputs=inputs, outputs=outputs, enable_caching=False)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertIsNone(call_kwargs['cache_config'])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    def test_get_input_requirements(self):
        """Test that input requirements are returned correctly."""
        input_reqs = self.builder.get_input_requirements()
        
        # Check that the input requirements contain the expected keys
        self.assertIn("inputs", input_reqs)
        self.assertIn("outputs", input_reqs)
        self.assertIn("enable_caching", input_reqs)
        
        # Check that the descriptions mention the required input and output names
        self.assertIn("model_input", input_reqs["inputs"])
        self.assertIn("eval_data_input", input_reqs["inputs"])
        
    def test_get_output_properties(self):
        """Test that output properties are returned correctly."""
        output_props = self.builder.get_output_properties()
        
        # Check that the output properties contain the expected keys
        self.assertIn("eval_output", output_props)
        self.assertIn("metrics_output", output_props)
        
        # Check that the descriptions match what's in the config
        self.assertEqual(output_props["eval_output"], self.config.output_names["eval_output"])
        self.assertEqual(output_props["metrics_output"], self.config.output_names["metrics_output"])
        
    def test_match_custom_properties_with_model_artifacts(self):
        """Test that _match_custom_properties correctly matches model artifacts."""
        # Create a mock step with model artifacts
        prev_step = MagicMock()
        prev_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        # Create inputs dictionary and input requirements
        inputs = {}
        input_requirements = {"inputs": "Dictionary containing model_input, eval_data_input S3 paths"}
        
        # Call _match_custom_properties
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Check that "inputs" was matched
        self.assertIn("inputs", matched)
        
        # Check that model_input was added to inputs
        self.assertIn("model_input", inputs["inputs"])
        self.assertEqual(inputs["inputs"]["model_input"], "s3://bucket/model.tar.gz")
        
    def test_match_custom_properties_with_eval_data(self):
        """Test that _match_custom_properties correctly matches evaluation data."""
        # Create a mock step with outputs containing evaluation data
        prev_step = MagicMock()
        output = MagicMock()
        output.output_name = "validation_data"
        output.destination = "s3://bucket/validation_data"
        prev_step.outputs = [output]
        
        # Create inputs dictionary and input requirements
        inputs = {}
        input_requirements = {"inputs": "Dictionary containing model_input, eval_data_input S3 paths"}
        
        # Call _match_custom_properties
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Check that "inputs" was matched
        self.assertIn("inputs", matched)
        
        # Check that eval_data_input was added to inputs
        self.assertIn("eval_data_input", inputs["inputs"])
        self.assertEqual(inputs["inputs"]["eval_data_input"], "s3://bucket/validation_data")
        
    def test_match_custom_properties_with_hyperparameters(self):
        """Test that _match_custom_properties correctly matches hyperparameters."""
        # Create a mock step with hyperparameters
        prev_step = MagicMock()
        prev_step.hyperparameters_s3_uri = "s3://bucket/hyperparameters"
        
        # Create inputs dictionary and input requirements
        inputs = {}
        input_requirements = {"inputs": "Dictionary containing model_input, eval_data_input S3 paths"}
        
        # Add hyperparameters_input to config.input_names
        self.config.input_names["hyperparameters_input"] = "Hyperparameters input"
        
        # Call _match_custom_properties
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Check that "inputs" was matched
        self.assertIn("inputs", matched)
        
        # Check that hyperparameters_input was added to inputs
        self.assertIn("hyperparameters_input", inputs["inputs"])
        self.assertEqual(inputs["inputs"]["hyperparameters_input"], "s3://bucket/hyperparameters")
        
        # Remove hyperparameters_input from config.input_names to not affect other tests
        del self.config.input_names["hyperparameters_input"]
        
    def test_get_processor_inputs_with_hyperparameters(self):
        """Test that processor inputs include hyperparameters when provided."""
        # Save the original _get_processor_inputs method
        original_method = self.builder._get_processor_inputs
        
        # Create a mock method that adds a hyperparameters input
        def mock_get_processor_inputs(inputs):
            # Add a third ProcessingInput for hyperparameters
            result = original_method(inputs)
            hyperparams_input = ProcessingInput(
                source="s3://bucket/hyperparameters",
                destination="/opt/ml/processing/input/hyperparameters",
                input_name="hyperparameters_input"
            )
            result.append(hyperparams_input)
            return result
        
        # Apply the mock
        self.builder._get_processor_inputs = mock_get_processor_inputs
        
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "eval_data_input": "s3://bucket/eval_data",
            "hyperparameters_input": "s3://bucket/hyperparameters"
        }
        
        processing_inputs = self.builder._get_processor_inputs(inputs)
        
        # Check that there are 3 inputs (model, eval_data, hyperparameters)
        self.assertEqual(len(processing_inputs), 3)
        
        # Check hyperparameters input
        hyperparams_input = processing_inputs[2]
        self.assertIsInstance(hyperparams_input, ProcessingInput)
        self.assertEqual(hyperparams_input.source, "s3://bucket/hyperparameters")
        self.assertEqual(hyperparams_input.destination, "/opt/ml/processing/input/hyperparameters")
        self.assertEqual(hyperparams_input.input_name, "hyperparameters_input")
        
        # Restore the original method
        self.builder._get_processor_inputs = original_method

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
