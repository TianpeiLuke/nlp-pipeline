import unittest
import json
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
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
from src.pipeline_steps.config_mims_payload_step import PayloadConfig, VariableType

class TestMIMSPayloadStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'mims_payload.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy MIMS payload script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a valid config for the PayloadConfig
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "model_owner": "test-team",
            "model_registration_domain": "BuyerSellerMessaging",
            "model_registration_objective": "TestObjective",
            "source_model_inference_content_types": ["text/csv"],
            "source_model_inference_response_types": ["application/json"],
            "source_model_inference_output_variable_list": {"score": VariableType.NUMERIC},
            "source_model_inference_input_variable_list": {
                "feature1": VariableType.NUMERIC, 
                "feature2": VariableType.TEXT
            },
            "processing_source_dir": self.temp_dir,
            "processing_entry_point": "mims_payload.py",
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "processing_instance_type_small": "ml.m5.large",
            "processing_instance_type_large": "ml.m5.xlarge",
            "use_large_processing_instance": False,
            "processing_framework_version": "1.0-1"
        }
        
        # Create a real PayloadConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = PayloadConfig(**self.valid_config_data)
        
        # Mock the generate_and_upload_payloads method at the module level
        self.patcher = patch('src.pipeline_steps.config_mims_payload_step.PayloadConfig.generate_and_upload_payloads')
        self.mock_gen_upload = self.patcher.start()
        self.mock_gen_upload.return_value = 's3://test-bucket/mods/payload/payload_test-pipeline_1.0.0_TestObjective.tar.gz'
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Instantiate builder with the mocked config
        self.builder = MIMSPayloadStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.'),
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-payload-test')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())

    def tearDown(self):
        """Clean up after each test."""
        # Stop the patcher if it's active
        if hasattr(self, 'patcher'):
            self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_with_invalid_config(self):
        """Test that __init__ raises ValueError with invalid config type."""
        with self.assertRaises(ValueError) as context:
            MIMSPayloadStepBuilder(
                config="invalid_config",  # Should be PayloadConfig instance
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole'
            )
        self.assertIn("PayloadConfig instance", str(context.exception))

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_bucket(self):
        """Test that configuration validation fails with missing bucket."""
        # Directly modify the config object to have empty bucket using object.__setattr__ to bypass Pydantic validation
        original_bucket = self.builder.config.bucket
        object.__setattr__(self.builder.config, 'bucket', "")  # Set empty bucket
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("bucket", str(context.exception))
        
        # Restore original bucket
        object.__setattr__(self.builder.config, 'bucket', original_bucket)

    def test_validate_configuration_missing_required_attrs(self):
        """Test that configuration validation fails with missing required attributes."""
        # Directly modify the config object to have empty pipeline_name using object.__setattr__ to bypass Pydantic validation
        original_pipeline_name = self.builder.config.pipeline_name
        object.__setattr__(self.builder.config, 'pipeline_name', "")  # Set empty pipeline_name
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("pipeline_name", str(context.exception))
        
        # Restore original pipeline_name
        object.__setattr__(self.builder.config, 'pipeline_name', original_pipeline_name)

    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
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
        self.assertEqual(call_args['framework_version'], "1.0-1")
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], "ml.m5.large")  # Small instance
        self.assertEqual(call_args['instance_count'], 1)
        self.assertEqual(call_args['volume_size_in_gb'], 30)
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertTrue('base_job_name' in call_args)
        self.assertTrue('env' in call_args)
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)

    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
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
        self.assertIn("PIPELINE_NAME", env_vars)
        self.assertEqual(env_vars["PIPELINE_NAME"], "test-pipeline")
        self.assertIn("REGION", env_vars)
        self.assertIn("CONTENT_TYPES", env_vars)
        self.assertEqual(env_vars["CONTENT_TYPES"], "text/csv")
        self.assertIn("DEFAULT_NUMERIC_VALUE", env_vars)
        self.assertEqual(env_vars["DEFAULT_NUMERIC_VALUE"], str(self.config.default_numeric_value))
        self.assertIn("DEFAULT_TEXT_VALUE", env_vars)
        self.assertEqual(env_vars["DEFAULT_TEXT_VALUE"], self.config.default_text_value)
        self.assertIn("BUCKET_NAME", env_vars)
        self.assertEqual(env_vars["BUCKET_NAME"], "test-bucket")

    def test_get_inputs_with_spec(self):
        """Test that inputs are created correctly using specification."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_input"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_input": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model"
        }
        
        # Create inputs dictionary
        inputs = {
            "model_input": "s3://bucket/model.tar.gz"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        self.assertEqual(len(proc_inputs), 1)
        
        # Check model data input
        model_input = proc_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.input_name, "model_input")
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")

    def test_get_inputs_missing_required(self):
        """Test that _get_inputs raises ValueError when required inputs are missing."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "model_input"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_input": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model"
        }
        
        # Test with empty inputs
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs({})
        self.assertIn("Required input 'model_input' not provided", str(context.exception))

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
        mock_output.logical_name = "payload_sample"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"payload_sample": mock_output}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_output_paths = {
            "payload_sample": "/opt/ml/processing/output"
        }
        
        # Create outputs dictionary
        outputs = {
            "payload_sample": "s3://bucket/payload/"
        }
        
        proc_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        
        # Check payload output
        payload_output = proc_outputs[0]
        self.assertIsInstance(payload_output, ProcessingOutput)
        self.assertEqual(payload_output.output_name, "payload_sample")
        self.assertEqual(payload_output.source, "/opt/ml/processing/output")
        self.assertEqual(payload_output.destination, "s3://bucket/payload/")

    def test_get_outputs_generated_destination(self):
        """Test that outputs use generated destination when not provided."""
        # Mock the spec and contract
        mock_output = MagicMock()
        mock_output.logical_name = "payload_sample"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"payload_sample": mock_output}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_output_paths = {
            "payload_sample": "/opt/ml/processing/output"
        }
        
        # Create empty outputs dictionary
        outputs = {}
        
        proc_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        
        # Check payload output with generated destination
        payload_output = proc_outputs[0]
        self.assertIsInstance(payload_output, ProcessingOutput)
        self.assertEqual(payload_output.output_name, "payload_sample")
        self.assertEqual(payload_output.source, "/opt/ml/processing/output")
        expected_dest = f"{self.config.pipeline_s3_loc}/payload/payload_sample"
        self.assertEqual(payload_output.destination, expected_dest)

    def test_get_outputs_no_spec(self):
        """Test that _get_outputs raises ValueError when no specification is available."""
        self.builder.spec = None
        
        with self.assertRaises(ValueError) as context:
            self.builder._get_outputs({})
        self.assertIn("Step specification is required", str(context.exception))

    def test_get_job_arguments_default(self):
        """Test that job arguments return default values."""
        job_args = self.builder._get_job_arguments()
        
        # Verify default job arguments
        self.assertIsInstance(job_args, list)
        self.assertEqual(len(job_args), 2)
        self.assertEqual(job_args, ["--mode", "standard"])

    def test_get_job_arguments_custom(self):
        """Test that job arguments use custom script arguments when provided."""
        # Set custom script arguments
        self.builder.config.processing_script_arguments = ["--custom", "arg"]
        
        job_args = self.builder._get_job_arguments()
        
        # Verify custom job arguments
        self.assertEqual(job_args, ["--custom", "arg"])

    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_payload_step.ProcessingStep')
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
        mock_dependency.logical_name = "model_input"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "payload_sample"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_input": mock_dependency}
        self.builder.spec.outputs = {"payload_sample": mock_output}
        self.builder.spec.step_type = "PayloadGeneration"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model"
        }
        self.builder.contract.expected_output_paths = {
            "payload_sample": "/opt/ml/processing/output"
        }
        self.builder.contract.entry_point = "mims_payload.py"
        
        # Create step with model_input
        step = self.builder.create_step(inputs={"model_input": "s3://bucket/model.tar.gz"})
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'PayloadGeneration')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertEqual(call_kwargs['depends_on'], [])
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in call_kwargs['inputs']))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in call_kwargs['outputs']))
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_payload_step.ProcessingStep')
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
        mock_dependency.logical_name = "model_input"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "payload_sample"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_input": mock_dependency}
        self.builder.spec.outputs = {"payload_sample": mock_output}
        self.builder.spec.step_type = "PayloadGeneration"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model"
        }
        self.builder.contract.expected_output_paths = {
            "payload_sample": "/opt/ml/processing/output"
        }
        self.builder.contract.entry_point = "mims_payload.py"
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies and model_input
        step = self.builder.create_step(
            inputs={"model_input": "s3://bucket/model.tar.gz"},
            dependencies=dependencies
        )
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], dependencies)
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_payload_step.ProcessingStep')
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
        mock_dependency.logical_name = "model_input"
        mock_dependency.required = True
        
        mock_output = MagicMock()
        mock_output.logical_name = "payload_sample"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"model_input": mock_dependency}
        self.builder.spec.outputs = {"payload_sample": mock_output}
        self.builder.spec.step_type = "PayloadGeneration"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model"
        }
        self.builder.contract.expected_output_paths = {
            "payload_sample": "/opt/ml/processing/output"
        }
        self.builder.contract.entry_point = "mims_payload.py"
        
        # Mock extract_inputs_from_dependencies
        self.builder.extract_inputs_from_dependencies = MagicMock(
            return_value={"model_input": "s3://bucket/extracted_model.tar.gz"}
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

    def test_get_script_path_from_config(self):
        """Test that get_script_path returns path from config."""
        script_path = self.builder.config.get_script_path()
        
        # Should combine source dir with entry point
        expected_path = str(Path(self.temp_dir) / "mims_payload.py")
        self.assertEqual(script_path, expected_path)

    def test_get_script_path_s3_source(self):
        """Test that get_script_path handles S3 source directory."""
        # Set S3 source directory
        self.builder.config.processing_source_dir = "s3://bucket/scripts/"
        
        script_path = self.builder.config.get_script_path()
        
        # Should combine S3 path with entry point
        expected_path = "s3://bucket/scripts/mims_payload.py"
        self.assertEqual(script_path, expected_path)

    def test_get_script_contract(self):
        """Test that get_script_contract returns the MIMS payload contract."""
        contract = self.builder.config.get_script_contract()
        
        # Should return the MIMS payload contract
        self.assertIsNotNone(contract)
        self.assertEqual(contract.entry_point, "mims_payload.py")
        self.assertIn("model_input", contract.expected_input_paths)
        self.assertIn("payload_sample", contract.expected_output_paths)

if __name__ == '__main__':
    unittest.main()
