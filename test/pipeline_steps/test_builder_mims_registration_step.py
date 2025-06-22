import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch
import os
import sys

# Add the project root to the Python path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the builder class to be tested
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig, VariableType

class TestModelRegistrationStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a dummy config object with required attributes
        self.config = SimpleNamespace()
        
        # Required attributes for validation
        self.config.model_owner = "test-team"
        self.config.model_registration_domain = "BuyerSellerMessaging"
        self.config.model_registration_objective = "TestObjective"
        self.config.source_model_inference_content_types = ["text/csv"]
        self.config.source_model_inference_response_types = ["application/json"]
        self.config.source_model_inference_output_variable_list = {"score": "NUMERIC"}
        self.config.source_model_inference_input_variable_list = {"feature1": "NUMERIC", "feature2": "TEXT"}
        
        # Region configuration
        self.config.region = "us-east-1"
        self.config.REGION_MAPPING = {
            "us-east-1": "IAD",
            "us-west-2": "PDX",
            "eu-west-1": "DUB"
        }
        
        # S3 configuration
        self.config.bucket = "test-bucket"
        
        # Instantiate builder without running __init__ (to bypass type checks)
        self.builder = object.__new__(ModelRegistrationStepBuilder)
        self.builder.config = self.config
        
        # Create a properly configured session mock
        session_mock = MagicMock()
        session_mock.sagemaker_config = {}
        self.builder.session = session_mock
        
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        
        # Mock the MimsModelRegistrationProcessingStep
        self.mock_registration_step = MagicMock()
        self.mock_registration_step_patcher = patch(
            'src.pipeline_steps.builder_mims_registration_step.MimsModelRegistrationProcessingStep',
            return_value=self.mock_registration_step
        )
        self.mock_registration_step_class = self.mock_registration_step_patcher.start()
        
    def tearDown(self):
        """Clean up after each test."""
        self.mock_registration_step_patcher.stop()
        
    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()
        
    def test_validate_configuration_missing_required_attribute(self):
        """Test that configuration validation fails with missing required attribute."""
        # Save original value
        original_value = self.config.model_owner
        # Set to None to trigger validation error
        self.config.model_owner = None
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original value
        self.config.model_owner = original_value
        
    def test_validate_configuration_missing_output_variables(self):
        """Test that configuration validation fails with missing output variables."""
        # Save original value
        original_value = self.config.source_model_inference_output_variable_list
        # Set to empty dict to trigger validation error
        self.config.source_model_inference_output_variable_list = {}
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original value
        self.config.source_model_inference_output_variable_list = original_value
        
    def test_validate_configuration_invalid_content_type(self):
        """Test that configuration validation fails with invalid content type."""
        # Save original value
        original_value = self.config.source_model_inference_content_types
        # Set to invalid value to trigger validation error
        self.config.source_model_inference_content_types = ["invalid/type"]
        
        with self.assertRaises(ValueError):
            self.builder.validate_configuration()
            
        # Restore original value
        self.config.source_model_inference_content_types = original_value
        
    def test_get_processing_inputs_model_only(self):
        """Test that processing inputs are created correctly with model only."""
        model_path = "s3://bucket/model.tar.gz"
        
        inputs = self.builder._get_processing_inputs(model_path)
        
        # Should have one input for the model
        self.assertEqual(len(inputs), 1)
        
        # Check model input
        model_input = inputs[0]
        self.assertEqual(model_input.source, model_path)
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        self.assertEqual(model_input.s3_data_distribution_type, "FullyReplicated")
        self.assertEqual(model_input.s3_input_mode, "File")
        
    def test_get_processing_inputs_with_payload(self):
        """Test that processing inputs are created correctly with model and payload."""
        model_path = "s3://bucket/model.tar.gz"
        payload_key = "payloads/test-payload.json"
        
        inputs = self.builder._get_processing_inputs(model_path, payload_key)
        
        # Should have two inputs: model and payload
        self.assertEqual(len(inputs), 2)
        
        # Check model input
        model_input = inputs[0]
        self.assertEqual(model_input.source, model_path)
        
        # Check payload input
        payload_input = inputs[1]
        self.assertEqual(payload_input.source, f"s3://{self.config.bucket}/{payload_key}")
        self.assertEqual(payload_input.destination, "/opt/ml/processing/mims_payload")
        
    def test_validate_regions_success(self):
        """Test that region validation succeeds with valid regions."""
        # Should not raise any exceptions
        self.builder._validate_regions(["us-east-1", "us-west-2"])
        
    def test_validate_regions_invalid(self):
        """Test that region validation fails with invalid regions."""
        with self.assertRaises(ValueError):
            self.builder._validate_regions(["us-east-1", "invalid-region"])
            
    def test_create_step_single_region(self):
        """Test that create_step creates a registration step for a single region."""
        model_path = "s3://bucket/model.tar.gz"
        
        # Create step for a single region
        step = self.builder.create_step(model_path)
        
        # Verify MimsModelRegistrationProcessingStep was called once
        self.mock_registration_step_class.assert_called_once()
        
        # Verify the step was returned
        self.assertEqual(step, self.mock_registration_step)
        
    def test_create_step_multiple_regions(self):
        """Test that create_step creates registration steps for multiple regions."""
        model_path = "s3://bucket/model.tar.gz"
        regions = ["us-east-1", "us-west-2"]
        
        # Create steps for multiple regions
        steps = self.builder.create_step(model_path, regions=regions)
        
        # Verify MimsModelRegistrationProcessingStep was called twice (once for each region)
        self.assertEqual(self.mock_registration_step_class.call_count, 2)
        
        # Verify the steps dictionary contains both regions
        self.assertEqual(len(steps), 2)
        self.assertIn("us-east-1", steps)
        self.assertIn("us-west-2", steps)
        
    def test_create_step_with_dependencies(self):
        """Test that create_step handles dependencies correctly."""
        model_path = "s3://bucket/model.tar.gz"
        
        # Create mock dependencies
        dependency1 = MagicMock()
        dependency2 = MagicMock()
        dependencies = [dependency1, dependency2]
        
        # Create step with dependencies
        step = self.builder.create_step(model_path, dependencies=dependencies)
        
        # Verify MimsModelRegistrationProcessingStep was called with dependencies
        call_kwargs = self.mock_registration_step_class.call_args.kwargs
        self.assertEqual(call_kwargs["depends_on"], dependencies)
        
    def test_create_step_with_payload(self):
        """Test that create_step handles payload correctly."""
        model_path = "s3://bucket/model.tar.gz"
        payload_key = "payloads/test-payload.json"
        
        # Create step with payload
        step = self.builder.create_step(model_path, payload_s3_key=payload_key)
        
        # Verify MimsModelRegistrationProcessingStep was called with correct inputs
        call_kwargs = self.mock_registration_step_class.call_args.kwargs
        processing_inputs = call_kwargs["processing_input"]
        
        # Should have two inputs: model and payload
        self.assertEqual(len(processing_inputs), 2)
        
        # Check payload input
        payload_input = processing_inputs[1]
        self.assertEqual(payload_input.source, f"s3://{self.config.bucket}/{payload_key}")
        
    def test_create_registration_steps_backward_compatibility(self):
        """Test that the old create_registration_steps method calls the new create_step method."""
        model_path = "s3://bucket/model.tar.gz"
        
        with patch.object(self.builder, 'create_step') as mock_create_step:
            # Set up mock to return a dictionary for multiple regions
            mock_create_step.return_value = {
                "us-east-1": MagicMock(),
                "us-west-2": MagicMock()
            }
            
            # Call the old method with multiple regions
            result = self.builder.create_registration_steps(
                packaging_step_output=model_path,
                dependencies=None,
                payload_s3_key=None,
                regions=["us-east-1", "us-west-2"]
            )
            
            # Verify it called the new method with correct parameters
            mock_create_step.assert_called_once_with(
                packaging_step_output=model_path,
                dependencies=None,
                payload_s3_key=None,
                regions=["us-east-1", "us-west-2"]
            )
            
            # Verify the result is the dictionary returned by create_step
            self.assertEqual(result, mock_create_step.return_value)
            
        # Test with single region that returns a step instead of dict
        with patch.object(self.builder, 'create_step') as mock_create_step:
            # Set up mock to return a single step
            mock_step = MagicMock()
            mock_create_step.return_value = mock_step
            
            # Call the old method with single region
            result = self.builder.create_registration_steps(
                packaging_step_output=model_path
            )
            
            # Verify the result is a dictionary with the region as key
            self.assertIsInstance(result, dict)
            self.assertEqual(len(result), 1)
            self.assertIn(self.config.region, result)
            self.assertEqual(result[self.config.region], mock_step)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
