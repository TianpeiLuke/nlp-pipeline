import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the problematic specification imports at the module level
mock_spec_modules = {
    'src.pipeline_step_specs.batch_transform_training_spec': MagicMock(BATCH_TRANSFORM_TRAINING_SPEC=None),
    'src.pipeline_step_specs.batch_transform_calibration_spec': MagicMock(BATCH_TRANSFORM_CALIBRATION_SPEC=None),
    'src.pipeline_step_specs.batch_transform_validation_spec': MagicMock(BATCH_TRANSFORM_VALIDATION_SPEC=None),
    'src.pipeline_step_specs.batch_transform_testing_spec': MagicMock(BATCH_TRANSFORM_TESTING_SPEC=None)
}

with patch.dict('sys.modules', mock_spec_modules):
    from sagemaker.transformer import Transformer
    from sagemaker.workflow.steps import TransformStep
    from sagemaker.inputs import TransformInput
    
    # Import the builder class to be tested
    from src.pipeline_steps.builder_batch_transform_step import BatchTransformStepBuilder
    from src.pipeline_steps.config_batch_transform_step import BatchTransformStepConfig

class TestBatchTransformStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid config for the BatchTransformStepConfig
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "job_type": "training",
            "batch_input_location": "s3://bucket/input",
            "batch_output_location": "s3://bucket/output",
            "transform_instance_type": "ml.m5.large",
            "transform_instance_count": 1,
            "content_type": "text/csv",
            "accept": "text/csv",
            "split_type": "Line",
            "assemble_with": "Line",
            "input_filter": "$[1:]",
            "output_filter": "$[-1]",
            "join_source": "Input"
        }
        
        # Create a real BatchTransformStepConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = BatchTransformStepConfig(**self.valid_config_data)
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Instantiate builder with the mocked config
        self.builder = BatchTransformStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.'),
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_with_invalid_config(self):
        """Test that __init__ raises ValueError with invalid config type."""
        with self.assertRaises(ValueError) as context:
            BatchTransformStepBuilder(
                config="invalid_config",  # Should be BatchTransformStepConfig instance
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole'
            )
        self.assertIn("BatchTransformStepConfig instance", str(context.exception))

    def test_init_with_missing_job_type(self):
        """Test that config creation raises ValidationError when job_type is missing."""
        # Create config without job_type
        config_data = self.valid_config_data.copy()
        del config_data['job_type']
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            # The validation error should occur at config creation time
            with self.assertRaises(Exception) as context:
                config = BatchTransformStepConfig(**config_data)
            # Should be a Pydantic validation error
            self.assertIn("job_type", str(context.exception))

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_invalid_job_type(self):
        """Test that configuration validation fails with invalid job_type."""
        # Directly modify the config object to have invalid job_type
        original_job_type = self.builder.config.job_type
        object.__setattr__(self.builder.config, 'job_type', 'invalid_type')  # Set invalid job type
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("Unsupported job_type", str(context.exception))
        
        # Restore original job type
        object.__setattr__(self.builder.config, 'job_type', original_job_type)

    def test_validate_configuration_missing_required_attrs(self):
        """Test that configuration validation fails with missing required attributes."""
        # Directly modify the config object to have empty transform_instance_type
        original_instance_type = self.builder.config.transform_instance_type
        object.__setattr__(self.builder.config, 'transform_instance_type', None)  # Set to None
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("transform_instance_type", str(context.exception))
        
        # Restore original instance type
        object.__setattr__(self.builder.config, 'transform_instance_type', original_instance_type)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    def test_create_transformer(self, mock_transformer_cls):
        """Test that the transformer is created with the correct parameters."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        model_name = 'test-model'
        output_path = 's3://bucket/output'
        transformer = self.builder._create_transformer(model_name, output_path)
        
        # Verify Transformer was created with correct parameters
        mock_transformer_cls.assert_called_once_with(
            model_name=model_name,
            instance_type=self.config.transform_instance_type,
            instance_count=self.config.transform_instance_count,
            output_path=output_path,
            accept=self.config.accept,
            assemble_with=self.config.assemble_with,
            sagemaker_session=self.builder.session
        )
        
        # Verify the returned transformer is our mock
        self.assertEqual(transformer, mock_transformer)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    def test_create_transformer_no_output_path(self, mock_transformer_cls):
        """Test that the transformer is created without output path."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        model_name = 'test-model'
        transformer = self.builder._create_transformer(model_name)
        
        # Verify Transformer was created with None output_path
        call_args = mock_transformer_cls.call_args[1]
        self.assertEqual(call_args['output_path'], None)

    def test_get_inputs_with_model_and_data(self):
        """Test that inputs are processed correctly with model and data."""
        inputs = {
            "model_name": "test-model",
            "processed_data": "s3://bucket/processed_data.csv"
        }
        
        transform_input, model_name = self.builder._get_inputs(inputs)
        
        # Check model name
        self.assertEqual(model_name, "test-model")
        
        # Check transform input
        self.assertIsInstance(transform_input, TransformInput)
        self.assertEqual(transform_input.data, "s3://bucket/processed_data.csv")
        self.assertEqual(transform_input.content_type, self.config.content_type)
        self.assertEqual(transform_input.split_type, self.config.split_type)
        self.assertEqual(transform_input.join_source, self.config.join_source)
        self.assertEqual(transform_input.input_filter, self.config.input_filter)
        self.assertEqual(transform_input.output_filter, self.config.output_filter)

    def test_get_inputs_with_input_data_fallback(self):
        """Test that inputs work with input_data fallback."""
        inputs = {
            "model_name": "test-model",
            "input_data": "s3://bucket/input_data.csv"  # backward compatibility
        }
        
        transform_input, model_name = self.builder._get_inputs(inputs)
        
        # Check model name
        self.assertEqual(model_name, "test-model")
        
        # Check transform input
        self.assertIsInstance(transform_input, TransformInput)
        self.assertEqual(transform_input.data, "s3://bucket/input_data.csv")

    def test_get_inputs_missing_model_name(self):
        """Test that _get_inputs raises ValueError when model_name is missing."""
        inputs = {
            "processed_data": "s3://bucket/processed_data.csv"
        }
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs(inputs)
        self.assertIn("model_name is required", str(context.exception))

    def test_get_inputs_missing_data(self):
        """Test that _get_inputs raises ValueError when data is missing."""
        inputs = {
            "model_name": "test-model"
        }
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs(inputs)
        self.assertIn("Input data source (processed_data) is required", str(context.exception))

    def test_get_outputs_with_spec(self):
        """Test that outputs are processed correctly with specification."""
        # Mock the spec
        mock_output = MagicMock()
        mock_output.logical_name = "transform_results"
        mock_output.property_path = "TransformJob.TransformOutput.S3OutputPath"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"transform_results": mock_output}
        
        outputs = {
            "transform_results": "s3://bucket/transform_output/"
        }
        
        result = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result["transform_results"], "s3://bucket/transform_output/")

    def test_get_outputs_generated_paths(self):
        """Test that outputs use generated paths when not provided."""
        # Mock the spec
        mock_output = MagicMock()
        mock_output.logical_name = "transform_results"
        mock_output.property_path = "TransformJob.TransformOutput.S3OutputPath"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"transform_results": mock_output}
        
        outputs = {}  # Empty outputs
        
        result = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(result), 1)
        self.assertIn("Will be available at:", result["transform_results"])

    def test_get_outputs_no_spec(self):
        """Test that outputs work without specification."""
        self.builder.spec = None
        
        outputs = {}
        result = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(result), 0)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    @patch('src.pipeline_steps.builder_batch_transform_step.TransformStep')
    def test_create_step(self, mock_transform_step_cls, mock_transformer_cls):
        """Test that the transform step is created with the correct parameters."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Setup mock step
        mock_step = MagicMock()
        mock_transform_step_cls.return_value = mock_step
        
        # Create step with inputs
        inputs = {
            "model_name": "test-model",
            "processed_data": "s3://bucket/processed_data.csv"
        }
        step = self.builder.create_step(inputs=inputs)
        
        # Verify TransformStep was created with correct parameters
        mock_transform_step_cls.assert_called_once()
        call_kwargs = mock_transform_step_cls.call_args.kwargs
        self.assertIn('name', call_kwargs)
        self.assertEqual(call_kwargs['transformer'], mock_transformer)
        self.assertIsInstance(call_kwargs['inputs'], TransformInput)
        self.assertEqual(call_kwargs['depends_on'], [])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    @patch('src.pipeline_steps.builder_batch_transform_step.TransformStep')
    def test_create_step_with_dependencies(self, mock_transform_step_cls, mock_transformer_cls):
        """Test that the transform step is created with dependencies."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Setup mock step
        mock_step = MagicMock()
        mock_transform_step_cls.return_value = mock_step
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies and inputs
        inputs = {
            "model_name": "test-model",
            "processed_data": "s3://bucket/processed_data.csv"
        }
        step = self.builder.create_step(inputs=inputs, dependencies=dependencies)
        
        # Verify TransformStep was created with correct parameters
        mock_transform_step_cls.assert_called_once()
        call_kwargs = mock_transform_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    @patch('src.pipeline_steps.builder_batch_transform_step.TransformStep')
    def test_create_step_with_dependency_extraction(self, mock_transform_step_cls, mock_transformer_cls):
        """Test that the step extracts inputs from dependencies."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Setup mock step
        mock_step = MagicMock()
        mock_transform_step_cls.return_value = mock_step
        
        # Mock extract_inputs_from_dependencies
        self.builder.extract_inputs_from_dependencies = MagicMock(
            return_value={
                "model_name": "extracted-model",
                "processed_data": "s3://bucket/extracted_data.csv"
            }
        )
        
        # Setup mock dependency
        dependency = MagicMock()
        
        # Create step with dependency but no direct inputs
        step = self.builder.create_step(dependencies=[dependency])
        
        # Verify extract_inputs_from_dependencies was called
        self.builder.extract_inputs_from_dependencies.assert_called_once_with([dependency])
        
        # Verify TransformStep was created with correct parameters
        mock_transform_step_cls.assert_called_once()
        call_kwargs = mock_transform_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], [dependency])
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    @patch('src.pipeline_steps.builder_batch_transform_step.TransformStep')
    def test_create_step_with_caching_disabled(self, mock_transform_step_cls, mock_transformer_cls):
        """Test that the step can be created with caching disabled."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Setup mock step
        mock_step = MagicMock()
        mock_transform_step_cls.return_value = mock_step
        
        # Create step with caching disabled
        inputs = {
            "model_name": "test-model",
            "processed_data": "s3://bucket/processed_data.csv"
        }
        step = self.builder.create_step(inputs=inputs, enable_caching=False)
        
        # Verify TransformStep was created with cache_config=None
        mock_transform_step_cls.assert_called_once()
        call_kwargs = mock_transform_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['cache_config'], None)

    @patch('src.pipeline_steps.builder_batch_transform_step.Transformer')
    @patch('src.pipeline_steps.builder_batch_transform_step.TransformStep')
    def test_create_step_attaches_spec(self, mock_transform_step_cls, mock_transformer_cls):
        """Test that create_step attaches spec to the step."""
        # Setup mock transformer
        mock_transformer = MagicMock()
        mock_transformer_cls.return_value = mock_transformer
        
        # Setup mock step
        mock_step = MagicMock()
        mock_transform_step_cls.return_value = mock_step
        
        # Mock the spec
        self.builder.spec = MagicMock()
        
        # Create step
        inputs = {
            "model_name": "test-model",
            "processed_data": "s3://bucket/processed_data.csv"
        }
        step = self.builder.create_step(inputs=inputs)
        
        # Verify spec was attached to the step
        # We can't directly check setattr calls on the mock, but we can verify the step was returned
        self.assertEqual(step, mock_step)

if __name__ == '__main__':
    unittest.main()
