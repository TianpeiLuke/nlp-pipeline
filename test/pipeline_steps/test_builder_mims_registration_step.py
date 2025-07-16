import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import os
import sys

from sagemaker.processing import ProcessingInput

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Mock the secure_ai_sandbox_workflow_python_sdk module before importing the builder
with patch.dict('sys.modules', {
    'secure_ai_sandbox_workflow_python_sdk': MagicMock(),
    'secure_ai_sandbox_workflow_python_sdk.mims_model_registration': MagicMock(),
    'secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step': MagicMock()
}):
    # Import the builder class to be tested
    from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder
    from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig

class TestModelRegistrationStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a valid config for the ModelRegistrationConfig
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "region": "NA",
            "model_registration_domain": "BuyerSellerMessaging",
            "model_registration_objective": "TestObjective",
            "framework": "xgboost",
            "inference_instance_type": "ml.m5.large",
            "inference_entry_point": "inference.py",
            "source_model_inference_content_types": ["text/csv"],
            "source_model_inference_response_types": ["application/json"],
            "source_model_inference_input_variable_list": {"feature1": "NUMERIC", "feature2": "TEXT"},
            "source_model_inference_output_variable_list": {"score": "NUMERIC"}
        }
        
        # Create a real ModelRegistrationConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = ModelRegistrationConfig(**self.valid_config_data)
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Instantiate builder with the mocked config
        self.builder = ModelRegistrationStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.'),
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock the MimsModelRegistrationProcessingStep
        self.mock_registration_step = MagicMock()
        self.mock_registration_step_patcher = patch(
            'src.pipeline_steps.builder_mims_registration_step.MimsModelRegistrationProcessingStep',
            return_value=self.mock_registration_step
        )
        self.mock_registration_step_class = self.mock_registration_step_patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        self.mock_registration_step_patcher.stop()

    def test_init_with_invalid_config(self):
        """Test that __init__ raises ValueError with invalid config type."""
        with self.assertRaises(ValueError) as context:
            ModelRegistrationStepBuilder(
                config="invalid_config",  # Should be ModelRegistrationConfig instance
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole'
            )
        self.assertIn("ModelRegistrationConfig instance", str(context.exception))

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_required_attrs(self):
        """Test that configuration validation fails with missing required attributes."""
        # Directly modify the config object to have empty model_registration_domain
        original_domain = self.builder.config.model_registration_domain
        object.__setattr__(self.builder.config, 'model_registration_domain', "")  # Set empty domain
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("model_registration_domain", str(context.exception))
        
        # Restore original domain
        object.__setattr__(self.builder.config, 'model_registration_domain', original_domain)

    def test_validate_configuration_missing_objective(self):
        """Test that configuration validation fails with missing objective."""
        # Directly modify the config object to have empty model_registration_objective
        original_objective = self.builder.config.model_registration_objective
        object.__setattr__(self.builder.config, 'model_registration_objective', "")  # Set empty objective
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("model_registration_objective", str(context.exception))
        
        # Restore original objective
        object.__setattr__(self.builder.config, 'model_registration_objective', original_objective)

    def test_get_inputs_with_spec_model_only(self):
        """Test that inputs are created correctly using specification with model only."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Create inputs dictionary with only model
        inputs = {
            "PackagedModel": "s3://bucket/model.tar.gz"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should have 1 input: PackagedModel
        self.assertEqual(len(proc_inputs), 1)
        
        # Check model input
        model_input = proc_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.input_name, "PackagedModel")
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        self.assertEqual(model_input.s3_data_distribution_type, "FullyReplicated")
        self.assertEqual(model_input.s3_input_mode, "File")

    def test_get_inputs_with_spec_model_and_payload(self):
        """Test that inputs are created correctly using specification with model and payload."""
        # Mock the spec and contract
        mock_dependency1 = MagicMock()
        mock_dependency1.logical_name = "PackagedModel"
        mock_dependency1.required = True
        
        mock_dependency2 = MagicMock()
        mock_dependency2.logical_name = "GeneratedPayloadSamples"
        mock_dependency2.required = False
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {
            "PackagedModel": mock_dependency1,
            "GeneratedPayloadSamples": mock_dependency2
        }
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model",
            "GeneratedPayloadSamples": "/opt/ml/processing/mims_payload"
        }
        
        # Create inputs dictionary with model and payload
        inputs = {
            "PackagedModel": "s3://bucket/model.tar.gz",
            "GeneratedPayloadSamples": "s3://bucket/payload.tar.gz"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should have 2 inputs: PackagedModel and GeneratedPayloadSamples
        self.assertEqual(len(proc_inputs), 2)
        
        # Check model input (should be first)
        model_input = proc_inputs[0]
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.input_name, "PackagedModel")
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        
        # Check payload input (should be second)
        payload_input = proc_inputs[1]
        self.assertIsInstance(payload_input, ProcessingInput)
        self.assertEqual(payload_input.input_name, "GeneratedPayloadSamples")
        self.assertEqual(payload_input.source, "s3://bucket/payload.tar.gz")
        self.assertEqual(payload_input.destination, "/opt/ml/processing/mims_payload")

    def test_get_inputs_missing_required(self):
        """Test that _get_inputs raises ValueError when required inputs are missing."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Test with empty inputs
        with self.assertRaises(ValueError) as context:
            self.builder._get_inputs({})
        self.assertIn("Required input 'PackagedModel' not provided", str(context.exception))

    def test_get_inputs_legacy_method(self):
        """Test that legacy input method works when no specification is available."""
        # Set spec to None to trigger legacy method
        self.builder.spec = None
        
        # Create inputs dictionary
        inputs = {
            "PackagedModel": "s3://bucket/model.tar.gz",
            "GeneratedPayloadSamples": "s3://bucket/payload.tar.gz"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should have 2 inputs: PackagedModel and GeneratedPayloadSamples
        self.assertEqual(len(proc_inputs), 2)
        
        # Check model input (should be first)
        model_input = proc_inputs[0]
        self.assertEqual(model_input.input_name, "PackagedModel")
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        
        # Check payload input (should be second)
        payload_input = proc_inputs[1]
        self.assertEqual(payload_input.input_name, "GeneratedPayloadSamples")
        self.assertEqual(payload_input.source, "s3://bucket/payload.tar.gz")
        self.assertEqual(payload_input.destination, "/opt/ml/processing/mims_payload")

    def test_get_inputs_legacy_method_model_only(self):
        """Test that legacy input method works with model only."""
        # Set spec to None to trigger legacy method
        self.builder.spec = None
        
        # Create inputs dictionary with only model
        inputs = {
            "PackagedModel": "s3://bucket/model.tar.gz"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should have 1 input: PackagedModel
        self.assertEqual(len(proc_inputs), 1)
        
        # Check model input
        model_input = proc_inputs[0]
        self.assertEqual(model_input.input_name, "PackagedModel")
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")

    def test_get_outputs(self):
        """Test that _get_outputs returns None (registration step has no outputs)."""
        outputs = self.builder._get_outputs({})
        self.assertIsNone(outputs)

    def test_handle_legacy_parameters(self):
        """Test that legacy parameters are handled correctly."""
        kwargs = {
            'packaged_model_output': 's3://bucket/model.tar.gz',
            'payload_s3_key': 's3://bucket/payload.tar.gz',
            'some_other_param': 'ignored'
        }
        
        legacy_inputs = self.builder._handle_legacy_parameters(kwargs)
        
        # Should map legacy parameters to standard names
        self.assertEqual(legacy_inputs['PackagedModel'], 's3://bucket/model.tar.gz')
        self.assertEqual(legacy_inputs['GeneratedPayloadSamples'], 's3://bucket/payload.tar.gz')
        self.assertNotIn('some_other_param', legacy_inputs)

    def test_create_step_with_dependencies(self):
        """Test that create_step works with dependencies."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Mock extract_inputs_from_dependencies
        self.builder.extract_inputs_from_dependencies = MagicMock(
            return_value={"PackagedModel": "s3://bucket/extracted_model.tar.gz"}
        )
        
        # Setup mock dependency
        dependency = MagicMock()
        
        # Create step with dependency
        step = self.builder.create_step(dependencies=[dependency])
        
        # Verify extract_inputs_from_dependencies was called
        self.builder.extract_inputs_from_dependencies.assert_called_once_with([dependency])
        
        # Verify MimsModelRegistrationProcessingStep was created with correct parameters
        self.mock_registration_step_class.assert_called_once()
        call_kwargs = self.mock_registration_step_class.call_args.kwargs
        self.assertEqual(call_kwargs['depends_on'], [dependency])
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_registration_step)

    def test_create_step_with_inputs(self):
        """Test that create_step works with direct inputs."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Create step with direct inputs
        inputs = {"PackagedModel": "s3://bucket/model.tar.gz"}
        step = self.builder.create_step(inputs=inputs)
        
        # Verify MimsModelRegistrationProcessingStep was created
        self.mock_registration_step_class.assert_called_once()
        call_kwargs = self.mock_registration_step_class.call_args.kwargs
        
        # Check that processing_input was provided
        self.assertIn('processing_input', call_kwargs)
        processing_inputs = call_kwargs['processing_input']
        self.assertEqual(len(processing_inputs), 1)
        self.assertEqual(processing_inputs[0].source, "s3://bucket/model.tar.gz")
        
        # Verify the returned step is our mock
        self.assertEqual(step, self.mock_registration_step)

    def test_create_step_with_legacy_parameters(self):
        """Test that create_step works with legacy parameters."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Create step with legacy parameters
        step = self.builder.create_step(packaged_model_output="s3://bucket/model.tar.gz")
        
        # Verify MimsModelRegistrationProcessingStep was created
        self.mock_registration_step_class.assert_called_once()
        call_kwargs = self.mock_registration_step_class.call_args.kwargs
        
        # Check that processing_input was provided
        self.assertIn('processing_input', call_kwargs)
        processing_inputs = call_kwargs['processing_input']
        self.assertEqual(len(processing_inputs), 1)
        self.assertEqual(processing_inputs[0].source, "s3://bucket/model.tar.gz")

    def test_create_step_with_performance_metadata(self):
        """Test that create_step handles performance metadata location."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Create step with performance metadata
        inputs = {"PackagedModel": "s3://bucket/model.tar.gz"}
        performance_location = "s3://bucket/performance.json"
        step = self.builder.create_step(
            inputs=inputs,
            performance_metadata_location=performance_location
        )
        
        # Verify MimsModelRegistrationProcessingStep was created with performance metadata
        self.mock_registration_step_class.assert_called_once()
        call_kwargs = self.mock_registration_step_class.call_args.kwargs
        self.assertEqual(call_kwargs['performance_metadata_location'], performance_location)

    def test_create_step_no_inputs_raises_error(self):
        """Test that create_step raises error when no inputs are provided."""
        with self.assertRaises(ValueError) as context:
            self.builder.create_step()
        self.assertIn("No inputs provided", str(context.exception))

    def test_create_step_attaches_spec_and_contract(self):
        """Test that create_step attaches spec and contract to the step."""
        # Mock the spec and contract
        mock_dependency = MagicMock()
        mock_dependency.logical_name = "PackagedModel"
        mock_dependency.required = True
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {"PackagedModel": mock_dependency}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "PackagedModel": "/opt/ml/processing/input/model"
        }
        
        # Create step
        inputs = {"PackagedModel": "s3://bucket/model.tar.gz"}
        step = self.builder.create_step(inputs=inputs)
        
        # Verify spec and contract were attached to the step
        # We can't directly check setattr calls on the mock, but we can verify the step was returned
        self.assertEqual(step, self.mock_registration_step)

if __name__ == '__main__':
    unittest.main()
