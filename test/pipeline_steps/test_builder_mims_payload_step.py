import unittest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
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
            "processing_entry_point": "mims_payload.py"
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
        
        # Instantiate builder with the mocked config
        self.builder = MIMSPayloadStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.')
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._get_step_name = MagicMock(return_value='PayloadTestStep')
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-payload-test')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())
        self.builder._extract_param = MagicMock(side_effect=lambda kwargs, key, default=None: kwargs.get(key, default))

    def tearDown(self):
        """Clean up after each test."""
        # Stop the patcher if it's active
        if hasattr(self, 'patcher'):
            self.patcher.stop()
        
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_fields(self):
        """Test that configuration validation fails with missing required fields."""
        # Test missing expected_tps
        with patch.object(self.config, 'expected_tps', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing max_latency_in_millisecond
        with patch.object(self.config, 'max_latency_in_millisecond', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing model_registration_domain
        with patch.object(self.config, 'model_registration_domain', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()
        
        # Test missing bucket
        with patch.object(self.config, 'bucket', None):
            with self.assertRaises(ValueError):
                self.builder.validate_configuration()

    @patch('src.pipeline_steps.builder_mims_payload_step.MIMSPayloadStepBuilder.validate_configuration')
    def test_init_calls_validate_configuration(self, mock_validate):
        """Test that __init__ calls validate_configuration."""
        # Create a new builder instance
        builder = MIMSPayloadStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.')
        )
        
        # Verify validate_configuration was called
        mock_validate.assert_called_once()

    def test_get_input_requirements(self):
        """Test that input requirements are returned correctly."""
        input_reqs = self.builder.get_input_requirements()
        self.assertIn("model_input", input_reqs)
        self.assertIn("dependencies", input_reqs)
        self.assertIn("enable_caching", input_reqs)

    def test_get_output_properties(self):
        """Test that output properties are returned correctly."""
        output_props = self.builder.get_output_properties()
        self.assertIn("payload_sample", output_props)
        self.assertIn("payload_metadata", output_props)

    def test_match_custom_properties(self):
        """Test _match_custom_properties method."""
        # Create a mock step with properties
        prev_step = MagicMock()
        prev_step.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        # Set up input requirements
        input_requirements = {
            "model_input": "S3 URI of the model artifacts from training"
        }
        
        # Call the method
        inputs = {}
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Verify inputs were matched
        self.assertIn("inputs", matched)
        self.assertIn("inputs", inputs)
        model_key = self.config.input_names.get("model_input", "model_input")
        self.assertIn(model_key, inputs["inputs"])
        self.assertEqual(inputs["inputs"][model_key], "s3://bucket/model.tar.gz")
        
    def test_match_custom_properties_no_model_artifacts(self):
        """Test _match_custom_properties method when no model artifacts are available."""
        # Create a mock step without ModelArtifacts property
        prev_step = MagicMock(spec=[])  # No properties
        
        # Set up input requirements
        input_requirements = {
            "model_input": "S3 URI of the model artifacts from training"
        }
        
        # Call the method
        inputs = {}
        matched = self.builder._match_custom_properties(inputs, input_requirements, prev_step)
        
        # Verify no inputs were matched
        self.assertEqual(matched, set())
        self.assertEqual(inputs, {})

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
        self.assertEqual(call_args['framework_version'], "0.23-1")
        self.assertEqual(call_args['role'], self.builder.role)
        self.assertEqual(call_args['instance_type'], "ml.m5.large")  # Default
        self.assertEqual(call_args['instance_count'], 1)
        self.assertEqual(call_args['volume_size_in_gb'], 30)  # Default
        self.assertEqual(call_args['sagemaker_session'], self.builder.session)
        self.assertTrue('base_job_name' in call_args)
        
        # Verify the returned processor is our mock
        self.assertEqual(processor, mock_processor)
        
    def test_get_processor_inputs(self):
        """Test that processor inputs are created correctly."""
        # Create inputs dictionary with required keys
        inputs = {
            "inputs": {
                "model_input": "s3://bucket/model.tar.gz"
            }
        }
        
        proc_inputs = self.builder._get_processor_inputs(inputs)
        
        self.assertEqual(len(proc_inputs), 1)
        
        # Check model data input
        model_input = proc_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model/model.tar.gz")

    def test_get_processor_inputs_missing(self):
        """Test that _get_processor_inputs raises ValueError when inputs are missing."""
        # Test with empty inputs
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({})
        
        # Test with missing model_input
        with self.assertRaises(ValueError):
            self.builder._get_processor_inputs({"inputs": {}})

    def test_get_processor_outputs(self):
        """Test that processor outputs are created correctly."""
        proc_outputs = self.builder._get_processor_outputs({})
        
        self.assertEqual(len(proc_outputs), 2)
        
        # Check payload_sample output
        payload_sample = next(o for o in proc_outputs if o.output_name == "payload_sample")
        self.assertIsInstance(payload_sample, ProcessingOutput)
        self.assertEqual(payload_sample.source, "/opt/ml/processing/output/payload_sample")
        self.assertTrue(payload_sample.destination.startswith("s3://"))
        
        # Check payload_metadata output
        payload_metadata = next(o for o in proc_outputs if o.output_name == "payload_metadata")
        self.assertIsInstance(payload_metadata, ProcessingOutput)
        self.assertEqual(payload_metadata.source, "/opt/ml/processing/output/payload_metadata")
        self.assertTrue(payload_metadata.destination.startswith("s3://"))

    def test_get_environment_variables(self):
        """Test that environment variables are set correctly."""
        env_vars = self.builder._get_environment_variables()
        
        # Verify required environment variables
        self.assertIn("CONTENT_TYPES", env_vars)
        self.assertEqual(env_vars["CONTENT_TYPES"], "text/csv")
        self.assertIn("DEFAULT_NUMERIC_VALUE", env_vars)
        self.assertEqual(env_vars["DEFAULT_NUMERIC_VALUE"], str(self.config.default_numeric_value))
        self.assertIn("DEFAULT_TEXT_VALUE", env_vars)
        self.assertEqual(env_vars["DEFAULT_TEXT_VALUE"], self.config.default_text_value)
        
    def test_get_environment_variables_with_special_fields(self):
        """Test that environment variables include special field values."""
        # Add special field values
        self.config.special_field_values = {
            "feature2": "special_value"
        }
        
        env_vars = self.builder._get_environment_variables()
        
        # Verify special field environment variables
        self.assertIn("SPECIAL_FIELD_FEATURE2", env_vars)
        self.assertEqual(env_vars["SPECIAL_FIELD_FEATURE2"], "special_value")
        
    def test_get_job_arguments(self):
        """Test that job arguments are created correctly."""
        job_args = self.builder._get_job_arguments()
        
        # Verify job arguments
        self.assertIsInstance(job_args, list)
        self.assertEqual(len(job_args), 2)  # Dummy arg and value
        self.assertEqual(job_args[0], "--dummy-arg")
        
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
        
        # Create step with model_input
        step = self.builder.create_step(model_input="s3://bucket/model.tar.gz")
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'PayloadTestStep')
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
        
        # Setup mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies and model_input
        step = self.builder.create_step(
            model_input="s3://bucket/model.tar.gz",
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
    def test_create_step_with_auto_detect_inputs(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the step auto-detects inputs from dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Setup mock dependency with model artifacts
        dependency = MagicMock()
        dependency.properties.ModelArtifacts.S3ModelArtifacts = "s3://bucket/model.tar.gz"
        
        # Create step with dependency but no direct model_input
        step = self.builder.create_step(dependencies=[dependency])
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], [dependency])
        
        # Verify inputs were auto-detected
        self.assertTrue(any(isinstance(i, ProcessingInput) for i in call_kwargs['inputs']))
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)
        
    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_payload_step.ProcessingStep')
    def test_create_step_missing_model_input(self, mock_processing_step_cls, mock_processor_cls):
        """Test that create_step raises ValueError when model_input is missing."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Create step with no model_input and no dependencies
        with self.assertRaises(ValueError):
            self.builder.create_step()
            
    @patch('src.pipeline_steps.builder_mims_payload_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_payload_step.ProcessingStep')
    def test_create_step_constructs_s3_key_if_none(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the step sets sample_payload_s3_key if it's None."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Set sample_payload_s3_key to None
        self.config.sample_payload_s3_key = None
        
        # Create step
        self.builder.create_step(model_input="s3://bucket/model.tar.gz")
        
        # Verify sample_payload_s3_key is no longer None
        self.assertIsNotNone(self.config.sample_payload_s3_key)
        self.assertTrue(self.config.sample_payload_s3_key.startswith('mods/payload/'))
        
    def test_get_script_path(self):
        """Test that get_script_path returns the correct path."""
        # Mock the config's get_script_path method
        with patch.object(self.config, 'get_script_path', return_value=os.path.join(self.temp_dir, 'mims_payload.py')):
            script_path = self.builder.config.get_script_path()
            
            # Verify script path
            self.assertTrue(script_path.endswith('mims_payload.py'))
            
    def test_get_script_path_fallback(self):
        """Test that get_script_path falls back to default path when config method returns None."""
        # Mock the config's get_script_path method to return None
        with patch.object(self.config, 'get_script_path', return_value=None):
            # Mock os.path.join to return a predictable path
            with patch('os.path.join', return_value='/path/to/mims_payload.py'):
                with patch('os.path.dirname', return_value='/path/to'):
                    with patch('os.path.abspath', return_value='/path/to'):
                        script_path = self.builder._get_script_path()
                        
                        # Verify script path
                        self.assertEqual(script_path, '/path/to/mims_payload.py')

if __name__ == '__main__':
    unittest.main()
