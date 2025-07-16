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
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig

class TestMIMSPackagingStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up a minimal, mocked configuration and builder instance for each test."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create the entry point script in the temporary directory
        entry_point = 'mims_package.py'
        entry_point_path = os.path.join(self.temp_dir, entry_point)
        with open(entry_point_path, 'w') as f:
            f.write('# Dummy MIMS packaging script for testing\n')
            f.write('print("This is a dummy script")\n')
        
        # Create a valid config for the PackageStepConfig
        self.valid_config_data = {
            "bucket": "test-bucket",
            "author": "test-author",
            "pipeline_name": "test-pipeline",
            "pipeline_description": "Test Pipeline Description",
            "pipeline_version": "1.0.0",
            "pipeline_s3_loc": "s3://test-bucket/test-pipeline",
            "source_dir": self.temp_dir,  # Use source_dir instead of processing_source_dir
            "processing_entry_point": "mims_package.py",
            "processing_instance_count": 1,
            "processing_volume_size": 30,
            "processing_instance_type_small": "ml.m5.large",
            "processing_instance_type_large": "ml.m5.xlarge",
            "use_large_processing_instance": False,
            "processing_framework_version": "1.0-1",
            "model_type": "xgboost",
            "model_registration_objective": "TestObjective"
        }
        
        # Create a real PackageStepConfig instance
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_file', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            self.config = PackageStepConfig(**self.valid_config_data)
        
        # Mock registry manager and dependency resolver
        self.mock_registry_manager = MagicMock()
        self.mock_dependency_resolver = MagicMock()
        
        # Instantiate builder with the mocked config
        self.builder = MIMSPackagingStepBuilder(
            config=self.config,
            sagemaker_session=MagicMock(),
            role='arn:aws:iam::000000000000:role/DummyRole',
            notebook_root=Path('.'),
            registry_manager=self.mock_registry_manager,
            dependency_resolver=self.mock_dependency_resolver
        )
        
        # Mock the methods that interact with SageMaker
        self.builder._sanitize_name_for_sagemaker = MagicMock(return_value='test-pipeline-packaging-test')
        self.builder._get_cache_config = MagicMock(return_value=MagicMock())

    def tearDown(self):
        """Clean up after each test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_init_with_invalid_config(self):
        """Test that __init__ raises ValueError with invalid config type."""
        with self.assertRaises(ValueError) as context:
            MIMSPackagingStepBuilder(
                config="invalid_config",  # Should be PackageStepConfig instance
                sagemaker_session=MagicMock(),
                role='arn:aws:iam::000000000000:role/DummyRole'
            )
        self.assertIn("PackageStepConfig instance", str(context.exception))

    def test_validate_configuration_success(self):
        """Test that configuration validation succeeds with valid config."""
        # Should not raise any exceptions
        self.builder.validate_configuration()

    def test_validate_configuration_missing_entry_point(self):
        """Test that configuration validation fails with missing entry point."""
        # Directly modify the config object to have empty processing_entry_point
        original_entry_point = self.builder.config.processing_entry_point
        object.__setattr__(self.builder.config, 'processing_entry_point', "")  # Set empty entry point
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("processing_entry_point", str(context.exception))
        
        # Restore original entry point
        object.__setattr__(self.builder.config, 'processing_entry_point', original_entry_point)

    def test_validate_configuration_missing_required_attrs(self):
        """Test that configuration validation fails with missing required attributes."""
        # Directly modify the config object to have empty pipeline_name
        original_pipeline_name = self.builder.config.pipeline_name
        object.__setattr__(self.builder.config, 'pipeline_name', "")  # Set empty pipeline_name
        
        with self.assertRaises(ValueError) as context:
            self.builder.validate_configuration()
        self.assertIn("pipeline_name", str(context.exception))
        
        # Restore original pipeline_name
        object.__setattr__(self.builder.config, 'pipeline_name', original_pipeline_name)

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
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

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
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
        
        # The packaging script only uses basic environment variables
        self.assertIn("PIPELINE_NAME", env_vars)
        self.assertEqual(env_vars["PIPELINE_NAME"], "test-pipeline")
        self.assertIn("REGION", env_vars)
        self.assertEqual(env_vars["REGION"], "NA")
        
        # Only these two environment variables should be set
        self.assertEqual(len(env_vars), 2)

    def test_get_inputs_with_spec(self):
        """Test that inputs are created correctly using specification."""
        # Mock the spec and contract
        mock_dependency1 = MagicMock()
        mock_dependency1.logical_name = "model_input"
        mock_dependency1.required = True
        
        mock_dependency2 = MagicMock()
        mock_dependency2.logical_name = "inference_scripts_input"
        mock_dependency2.required = False
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {
            "model_input": mock_dependency1,
            "inference_scripts_input": mock_dependency2
        }
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model",
            "inference_scripts_input": "/opt/ml/processing/input/script"
        }
        
        # Create inputs dictionary
        inputs = {
            "model_input": "s3://bucket/model.tar.gz",
            "inference_scripts_input": "s3://bucket/scripts/"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should have 2 inputs: model_input and inference_scripts_input (local override)
        self.assertEqual(len(proc_inputs), 2)
        
        # Check model data input
        model_input = next(i for i in proc_inputs if i.input_name == "model_input")
        self.assertIsInstance(model_input, ProcessingInput)
        self.assertEqual(model_input.source, "s3://bucket/model.tar.gz")
        self.assertEqual(model_input.destination, "/opt/ml/processing/input/model")
        
        # Check inference scripts input - should use local path from config, not the provided S3 path
        scripts_input = next(i for i in proc_inputs if i.input_name == "inference_scripts_input")
        self.assertIsInstance(scripts_input, ProcessingInput)
        # Should use config.source_dir (temp_dir) instead of the provided S3 path
        # The builder uses config.source_dir which is set to temp_dir in our test setup
        expected_source = self.config.source_dir or self.temp_dir
        self.assertEqual(scripts_input.source, expected_source)
        self.assertEqual(scripts_input.destination, "/opt/ml/processing/input/script")

    def test_get_inputs_inference_scripts_local_override(self):
        """Test that inference_scripts_input always uses local path from config."""
        # Mock the spec and contract
        mock_dependency1 = MagicMock()
        mock_dependency1.logical_name = "model_input"
        mock_dependency1.required = True
        
        mock_dependency2 = MagicMock()
        mock_dependency2.logical_name = "inference_scripts_input"
        mock_dependency2.required = False
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {
            "model_input": mock_dependency1,
            "inference_scripts_input": mock_dependency2
        }
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model",
            "inference_scripts_input": "/opt/ml/processing/input/script"
        }
        
        # Create inputs dictionary with only model_input (no inference_scripts_input)
        inputs = {
            "model_input": "s3://bucket/model.tar.gz"
        }
        
        proc_inputs = self.builder._get_inputs(inputs)
        
        # Should still have 2 inputs because inference_scripts_input is added from local config
        self.assertEqual(len(proc_inputs), 2)
        
        # Check inference scripts input uses local path
        scripts_input = next(i for i in proc_inputs if i.input_name == "inference_scripts_input")
        self.assertIsInstance(scripts_input, ProcessingInput)
        self.assertEqual(scripts_input.source, self.temp_dir)
        self.assertEqual(scripts_input.destination, "/opt/ml/processing/input/script")

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
        mock_output.logical_name = "packaged_model"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"packaged_model": mock_output}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_output_paths = {
            "packaged_model": "/opt/ml/processing/output"
        }
        
        # Create outputs dictionary
        outputs = {
            "packaged_model": "s3://bucket/packaged/"
        }
        
        proc_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        
        # Check packaged model output
        packaged_output = proc_outputs[0]
        self.assertIsInstance(packaged_output, ProcessingOutput)
        self.assertEqual(packaged_output.output_name, "packaged_model")
        self.assertEqual(packaged_output.source, "/opt/ml/processing/output")
        self.assertEqual(packaged_output.destination, "s3://bucket/packaged/")

    def test_get_outputs_generated_destination(self):
        """Test that outputs use generated destination when not provided."""
        # Mock the spec and contract
        mock_output = MagicMock()
        mock_output.logical_name = "packaged_model"
        
        self.builder.spec = MagicMock()
        self.builder.spec.outputs = {"packaged_model": mock_output}
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_output_paths = {
            "packaged_model": "/opt/ml/processing/output"
        }
        
        # Create empty outputs dictionary
        outputs = {}
        
        proc_outputs = self.builder._get_outputs(outputs)
        
        self.assertEqual(len(proc_outputs), 1)
        
        # Check packaged model output with generated destination
        packaged_output = proc_outputs[0]
        self.assertIsInstance(packaged_output, ProcessingOutput)
        self.assertEqual(packaged_output.output_name, "packaged_model")
        self.assertEqual(packaged_output.source, "/opt/ml/processing/output")
        expected_dest = f"{self.config.pipeline_s3_loc}/packaging/packaged_model"
        self.assertEqual(packaged_output.destination, expected_dest)

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

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_packaging_step.ProcessingStep')
    def test_create_step(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with the correct parameters."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Mock the spec and contract
        mock_dependency1 = MagicMock()
        mock_dependency1.logical_name = "model_input"
        mock_dependency1.required = True
        
        mock_dependency2 = MagicMock()
        mock_dependency2.logical_name = "inference_scripts_input"
        mock_dependency2.required = False
        
        mock_output = MagicMock()
        mock_output.logical_name = "packaged_model"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {
            "model_input": mock_dependency1,
            "inference_scripts_input": mock_dependency2
        }
        self.builder.spec.outputs = {"packaged_model": mock_output}
        self.builder.spec.step_type = "ModelPackaging"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model",
            "inference_scripts_input": "/opt/ml/processing/input/script"
        }
        self.builder.contract.expected_output_paths = {
            "packaged_model": "/opt/ml/processing/output"
        }
        self.builder.contract.entry_point = "mims_package.py"
        
        # Create step with model_input
        step = self.builder.create_step(inputs={"model_input": "s3://bucket/model.tar.gz"})
        
        # Verify ProcessingStep was created with correct parameters
        mock_processing_step_cls.assert_called_once()
        call_kwargs = mock_processing_step_cls.call_args.kwargs
        self.assertEqual(call_kwargs['name'], 'ModelPackaging')
        self.assertEqual(call_kwargs['processor'], mock_processor)
        self.assertEqual(call_kwargs['depends_on'], [])
        self.assertTrue(all(isinstance(i, ProcessingInput) for i in call_kwargs['inputs']))
        self.assertTrue(all(isinstance(o, ProcessingOutput) for o in call_kwargs['outputs']))
        
        # Verify the returned step is our mock
        self.assertEqual(step, mock_step)

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_packaging_step.ProcessingStep')
    def test_create_step_with_dependencies(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the processing step is created with dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Mock the spec and contract
        mock_dependency1 = MagicMock()
        mock_dependency1.logical_name = "model_input"
        mock_dependency1.required = True
        
        mock_dependency2 = MagicMock()
        mock_dependency2.logical_name = "inference_scripts_input"
        mock_dependency2.required = False
        
        mock_output = MagicMock()
        mock_output.logical_name = "packaged_model"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {
            "model_input": mock_dependency1,
            "inference_scripts_input": mock_dependency2
        }
        self.builder.spec.outputs = {"packaged_model": mock_output}
        self.builder.spec.step_type = "ModelPackaging"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model",
            "inference_scripts_input": "/opt/ml/processing/input/script"
        }
        self.builder.contract.expected_output_paths = {
            "packaged_model": "/opt/ml/processing/output"
        }
        self.builder.contract.entry_point = "mims_package.py"
        
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

    @patch('src.pipeline_steps.builder_mims_packaging_step.SKLearnProcessor')
    @patch('src.pipeline_steps.builder_mims_packaging_step.ProcessingStep')
    def test_create_step_with_dependency_extraction(self, mock_processing_step_cls, mock_processor_cls):
        """Test that the step extracts inputs from dependencies."""
        # Setup mock processor
        mock_processor = MagicMock()
        mock_processor_cls.return_value = mock_processor
        
        # Setup mock step
        mock_step = MagicMock()
        mock_processing_step_cls.return_value = mock_step
        
        # Mock the spec and contract
        mock_dependency1 = MagicMock()
        mock_dependency1.logical_name = "model_input"
        mock_dependency1.required = True
        
        mock_dependency2 = MagicMock()
        mock_dependency2.logical_name = "inference_scripts_input"
        mock_dependency2.required = False
        
        mock_output = MagicMock()
        mock_output.logical_name = "packaged_model"
        
        self.builder.spec = MagicMock()
        self.builder.spec.dependencies = {
            "model_input": mock_dependency1,
            "inference_scripts_input": mock_dependency2
        }
        self.builder.spec.outputs = {"packaged_model": mock_output}
        self.builder.spec.step_type = "ModelPackaging"
        
        self.builder.contract = MagicMock()
        self.builder.contract.expected_input_paths = {
            "model_input": "/opt/ml/processing/input/model",
            "inference_scripts_input": "/opt/ml/processing/input/script"
        }
        self.builder.contract.expected_output_paths = {
            "packaged_model": "/opt/ml/processing/output"
        }
        self.builder.contract.entry_point = "mims_package.py"
        
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
        expected_path = str(Path(self.temp_dir) / "mims_package.py")
        self.assertEqual(script_path, expected_path)

    def test_get_script_path_s3_source(self):
        """Test that get_script_path handles S3 source directory."""
        # Set S3 source directory
        self.builder.config.processing_source_dir = "s3://bucket/scripts/"
        
        script_path = self.builder.config.get_script_path()
        
        # Should combine S3 path with entry point
        expected_path = "s3://bucket/scripts/mims_package.py"
        self.assertEqual(script_path, expected_path)

    def test_get_script_contract(self):
        """Test that get_script_contract returns the MIMS package contract."""
        contract = self.builder.config.get_script_contract()
        
        # Should return the MIMS package contract
        self.assertIsNotNone(contract)
        self.assertEqual(contract.entry_point, "mims_package.py")
        self.assertIn("model_input", contract.expected_input_paths)
        self.assertIn("inference_scripts_input", contract.expected_input_paths)
        self.assertIn("packaged_model", contract.expected_output_paths)

if __name__ == '__main__':
    unittest.main()
