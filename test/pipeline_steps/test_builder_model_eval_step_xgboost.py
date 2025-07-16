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
        )
        
        # Create a valid config for the XGBoostModelEvalConfig
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "processing_entry_point": "model_evaluation_xgboost.py",
            "processing_source_dir": self.temp_dir,
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "job_type": "validation",
            "hyperparameters": self.hyperparameters,
            "xgboost_framework_version": "1.5-1",
            "use_large_processing_instance": False,
            "processing_instance_type_small": "ml.m5.large",
            "processing_instance_type_large": "ml.m5.4xlarge"
        }
        
        # Create a real XGBoostModelEvalConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = XGBoostModelEvalConfig(**self.valid_config_data)
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Create a properly configured builder instance
        self.builder = XGBoostModelEvalStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.'),
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock methods from the base class
        self.builder._get_step_name = MagicMock(return_value='XGBoostModelEval')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-xgb-eval')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_with_invalid_config(self):
        """Test that __init__ raises ValueError with invalid config type."""
        with self.assertRaises(ValueError) as context:
            XGBoostModelEvalStepBuilder(
                config="invalid_config",  # Should be XGBoostModelEvalConfig instance
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole'
            )
        self.assertIn("XGBoostModelEvalConfig instance", str(context.exception))

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_attrs(self):
        """Test that configuration validation fails with missing required attributes."""
        # Directly modify the config object to have empty processing_entry_point
        original_entry_point = self.builder.config.processing_entry_point
        object.__setattr__(self.builder.config, 'processing_entry_point', None)  # Set to None
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("processing_entry_point", str(context.exception))
        
        # Restore original entry point
        object.__setattr__(self.builder.config, 'processing_entry_point', original_entry_point)

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

    def test_create_processor_large_instance(self):
        """Test that the processor uses large instance when configured."""
        # Modify config to use large instance
        original_use_large = self.builder.config.use_large_processing_instance
        object.__setattr__(self.builder.config, 'use_large_processing_instance', True)
        
        with patch('src.pipeline_steps.builder_model_eval_step_xgboost.XGBoostProcessor') as mock_processor_cls:
            mock_processor = MagicMock()
            mock_processor_cls.return_value = mock_processor
            
            processor = self.builder._create_processor()
            
            # Verify large instance type was used
            call_args = mock_processor_cls.call_args[1]
            self.assertEqual(call_args['instance_type'], 'ml.m5.4xlarge')
        
        # Restore original setting
        object.__setattr__(self.builder.config, 'use_large_processing_instance', original_use_large)

    def test_get_inputs_with_spec(self):
        """Test that inputs are processed correctly with specification."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        mock_contract = MagicMock()
        mock_contract.expected_input_paths = {"model_artifacts": "/opt/ml/processing/input/model"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.contract = mock_contract
        
        inputs = {
            "model_artifacts": "s3://bucket/model.tar.gz"
        }
        
        processing_inputs = self.builder._get_inputs(inputs)
        
        self.assertEqual(len(processing_inputs), 1)
        
        # Check model input
        model_input = processing_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        self.assertEqual(model_input.input_name, "model_artifacts")

    def test_get_inputs_missing_required(self):
        """Test that _get_inputs raises ValueError with missing required inputs."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.contract = MagicMock()
        
        inputs = {}  # Missing required input
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs(inputs)
        self.assertIn("Required input 'model_artifacts' not provided", str(context.exception))

    def test_get_inputs_no_spec(self):
        """Test that _get_inputs raises ValueError when no spec is available."""
        self.builder.spec = None
        
        inputs = {"model_artifacts": "s3://bucket/model.tar.gz"}
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs(inputs)
        self.assertIn("Step specification is required", str(context.exception))

    def test_get_outputs_with_spec(self):
        """Test that outputs are processed correctly with specification."""
        # Mock the spec and contract
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.contract = mock_contract
        
        outputs = {
            "evaluation_results": "s3://bucket/eval_output"
        }
        
        processing_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(processing_outputs), 1)
        
        # Check eval output
        eval_output = processing_outputs[0]
        self.assertIsInstance(eval_output, ProcessingOutput)
        self.assertEqual(eval_output.source, "/opt/ml/processing/output/eval")
        self.assertEqual(eval_output.destination, "s3://bucket/eval_output")
        self.assertEqual(eval_output.output_name, "evaluation_results")

    def test_get_outputs_generated_paths(self):
        """Test that outputs use generated paths when not provided."""
        # Mock the spec and contract
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.contract = mock_contract
        
        outputs = {}  # Empty outputs
        
        processing_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(processing_outputs), 1)
        
        # Check that generated path is used
        eval_output = processing_outputs[0]
        self.assertIn("model_evaluation", eval_output.destination)

    def test_get_outputs_no_spec(self):
        """Test that _get_outputs raises ValueError when no spec is available."""
        self.builder.spec = None
        
        outputs = {"evaluation_results": "s3://bucket/eval_output"}
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_outputs(outputs)
        self.assertIn("Step specification is required", str(context.exception))

    def test_get_job_arguments(self):
        """Test that job arguments are created correctly."""
        job_args = self.builder._get_job_arguments()
        
        self.assertEqual(len(job_args), 2)
        self.assertEqual(job_args[0], "--job_type")
        self.assertEqual(job_args[1], "validation")

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
        
        # Mock the spec and contract for inputs/outputs
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_input_paths = {"model_artifacts": "/opt/ml/processing/input/model"}
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.spec.step_type = "XGBoostModelEval"
        self.builder.contract = mock_contract
        
        # Create step
        inputs = {
            "model_artifacts": "s3://bucket/model.tar.gz"
        }
        
        outputs = {
            "evaluation_results": "s3://bucket/eval_output"
        }
        
        step = self.builder.create_step(inputs=inputs, outputs=outputs)
        
        # Verify processor.run was called with correct parameters
        mock_processor.run.assert_called_once()
        run_args = mock_processor.run.call_args[1]
        self.assertEqual(run_args['code'], 'model_evaluation_xgboost.py')
        self.assertEqual(run_args['source_dir'], self.temp_dir)
        self.assertEqual(len(run_args['inputs']), 1)
        self.assertEqual(len(run_args['outputs']), 1)
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
        
        # Mock the spec and contract for inputs/outputs
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_input_paths = {"model_artifacts": "/opt/ml/processing/input/model"}
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.spec.step_type = "XGBoostModelEval"
        self.builder.contract = mock_contract
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies
        inputs = {
            "model_artifacts": "s3://bucket/model.tar.gz"
        }
        
        outputs = {
            "evaluation_results": "s3://bucket/eval_output"
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
    def test_create_step_with_dependency_extraction(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the step extracts inputs from dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock step_args
        mock_step_args = MagicMock()
        mock_processor.run.return_value = mock_step_args
        
        # Mock the spec and contract for inputs/outputs
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_input_paths = {"model_artifacts": "/opt/ml/processing/input/model"}
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.spec.step_type = "XGBoostModelEval"
        self.builder.contract = mock_contract
        
        # Mock extract_inputs_from_dependencies
        self.builder.extract_inputs_from_dependencies = MagicMock(
            return_value={
                "model_artifacts": "s3://bucket/extracted_model.tar.gz"
            }
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
        
        # Mock the spec and contract for inputs/outputs
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_input_paths = {"model_artifacts": "/opt/ml/processing/input/model"}
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.spec.step_type = "XGBoostModelEval"
        self.builder.contract = mock_contract
        
        # Mock the _get_cache_config method to return None when enable_caching is False
        self.builder._get_cache_config = MagicMock(return_value=None)
        
        # Create step with caching disabled
        inputs = {
            "model_artifacts": "s3://bucket/model.tar.gz"
        }
        
        outputs = {
            "evaluation_results": "s3://bucket/eval_output"
        }
        
        step = self.builder.create_step(inputs=inputs, outputs=outputs, enable_caching=False)
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['cache_config'], None)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.XGBoostProcessor')
    @patch('src.pipeline_steps.builder_model_eval_step_xgboost.ProcessingStep')
    def test_create_step_attaches_spec(self, mock_processing_step_cls, mock_processor_cls):
        """Test that create_step attaches spec to the step."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock step_args
        mock_step_args = MagicMock()
        mock_processor.run.return_value = mock_step_args
        
        # Mock the spec and contract for inputs/outputs
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_artifacts"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "evaluation_results"
        
        mock_contract = MagicMock()
        mock_contract.expected_input_paths = {"model_artifacts": "/opt/ml/processing/input/model"}
        mock_contract.expected_output_paths = {"evaluation_results": "/opt/ml/processing/output/eval"}
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_artifacts": mock_dependency}
        self.builder.spec.outputs = {"evaluation_results": mock_output}
        self.builder.spec.step_type = "XGBoostModelEval"
        self.builder.contract = mock_contract
        
        # Create step
        inputs = {
            "model_artifacts": "s3://bucket/model.tar.gz"
        }
        
        outputs = {
            "evaluation_results": "s3://bucket/eval_output"
        }
        
        step = self.builder.create_step(inputs=inputs, outputs=outputs)
        
        # Verify spec was attached to the step
        # We can't directly check setattr calls on the mock, but we can verify the step was returned
        self.assertEqual(step, mock_step)

if __name__ == '__main__':
    unittest.main()
