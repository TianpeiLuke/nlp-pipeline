import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from sagemaker.workflow.steps import CreateModelStep

# Import the builder class
from src.pipeline_steps.builder_model_step_pytorch import PyTorchModelStepBuilder

class TestPyTorchModelStepBuilder(unittest.TestCase):
    def setUp(self):
        # Build a minimal config namespace
        self.config = SimpleNamespace()
        self.config.region = 'NA'
        self.config.current_date = '20250610'
        
        # Required attributes
        self.config.entry_point = 'inference.py'
        self.config.source_dir = 'src'
        self.config.instance_type = 'ml.m5.large'
        self.config.framework_version = '1.12.0'
        self.config.py_version = 'py38'
        self.config.use_pytorch_framework = True
        
        # Add input_names and output_names
        self.config.input_names = {"model_data": "ModelArtifacts"}
        self.config.output_names = {
            "model": "ModelName",
            "model_artifacts_path": "ModelArtifactsPath"
        }
        
        # Model name generation
        self.config.get_model_name = lambda: 'test-model'

        # Instantiate builder bypassing __init__
        self.builder = object.__new__(PyTorchModelStepBuilder)
        self.builder.config = self.config
        self.builder.session = MagicMock()
        self.builder.role = 'arn:aws:iam::000000000000:role/DummyRole'
        self.builder.notebook_root = Path('.')
        self.builder.aws_region = PyTorchModelStepBuilder.REGION_MAPPING['NA']
        self.builder.log_info = MagicMock()

    def test_validate_configuration_missing_attr(self):
        # Missing required attrs should raise
        cfg2 = SimpleNamespace(region='NA')
        builder2 = object.__new__(PyTorchModelStepBuilder)
        builder2.config = cfg2
        builder2.session = None
        builder2.role = None
        builder2.notebook_root = Path('.')
        builder2.aws_region = builder2.REGION_MAPPING['NA']
        with self.assertRaises(ValueError):
            builder2.validate_configuration()

    @patch('src.pipeline_steps.builder_model_step_pytorch.PyTorchModel')
    def test_create_pytorch_model(self, mock_pytorch_model_cls):
        # Setup mock
        mock_model = MagicMock()
        mock_pytorch_model_cls.return_value = mock_model
        
        # Call _create_pytorch_model
        model = self.builder._create_pytorch_model('s3://bucket/model.tar.gz')
        
        # Verify PyTorchModel was called with correct args
        mock_pytorch_model_cls.assert_called_once()
        args, kwargs = mock_pytorch_model_cls.call_args
        self.assertEqual(kwargs['model_data'], 's3://bucket/model.tar.gz')
        self.assertEqual(kwargs['role'], self.builder.role)
        self.assertEqual(kwargs['entry_point'], self.config.entry_point)
        self.assertEqual(kwargs['source_dir'], self.config.source_dir)
        self.assertEqual(kwargs['framework_version'], self.config.framework_version)
        self.assertEqual(kwargs['py_version'], self.config.py_version)
        
        # Verify the returned model
        self.assertEqual(model, mock_model)

    def test_match_custom_properties(self):
        # Create a mock TrainingStep
        step = MagicMock()
        step.name = 'PyTorchTrainingStep'
        
        # Setup properties for the step
        step.properties = SimpleNamespace()
        step.properties.ModelArtifacts = SimpleNamespace()
        step.properties.ModelArtifacts.S3ModelArtifacts = 's3://bucket/model.tar.gz'
        
        # Call _match_custom_properties
        inputs = {}
        input_requirements = {'model_data': 'S3 URI of the model artifacts'}
        matched = self.builder._match_custom_properties(inputs, input_requirements, step)
        
        # Check that inputs are correctly matched
        self.assertIn('inputs', matched)
        self.assertIn('model_data', inputs.get('inputs', {}))
        self.assertEqual(inputs['inputs']['model_data'], 's3://bucket/model.tar.gz')

    @patch('src.pipeline_steps.builder_model_step_pytorch.PyTorchModel')
    def test_create_step(self, mock_pytorch_model_cls):
        # Setup mocks
        mock_model = MagicMock()
        mock_pytorch_model_cls.return_value = mock_model
        
        # Setup model.create to return step_args
        mock_model.create.return_value = {'ModelName': 'test-model'}
        
        # Call create_step
        step = self.builder.create_step(model_data='s3://bucket/model.tar.gz')
        
        # Verify the step is created correctly
        self.assertIsInstance(step, CreateModelStep)
        self.assertEqual(step.step_args, {'ModelName': 'test-model'})
        
        # Verify PyTorchModel.create was called with correct args
        mock_model.create.assert_called_once()
        args, kwargs = mock_model.create.call_args
        self.assertEqual(kwargs['instance_type'], self.config.instance_type)
        self.assertEqual(kwargs['model_name'], 'test-model')

    @patch('src.pipeline_steps.builder_model_step_pytorch.PyTorchModel')
    def test_create_step_with_dependencies(self, mock_pytorch_model_cls):
        # Setup mocks
        mock_model = MagicMock()
        mock_pytorch_model_cls.return_value = mock_model
        
        # Setup model.create to return step_args
        mock_model.create.return_value = {'ModelName': 'test-model'}
        
        # Create a mock dependency step
        dep_step = MagicMock()
        dep_step.name = 'PyTorchTrainingStep'
        
        # Setup properties for the dependency step
        dep_step.properties = SimpleNamespace()
        dep_step.properties.ModelArtifacts = SimpleNamespace()
        dep_step.properties.ModelArtifacts.S3ModelArtifacts = 's3://bucket/model.tar.gz'
        
        # Call create_step with the dependency
        step = self.builder.create_step(dependencies=[dep_step])
        
        # Verify the step is created correctly
        self.assertIsInstance(step, CreateModelStep)
        self.assertEqual(step.step_args, {'ModelName': 'test-model'})
        self.assertEqual(step.depends_on, [dep_step])
        
        # Verify PyTorchModel was called with correct model_data
        args, kwargs = mock_pytorch_model_cls.call_args
        self.assertEqual(kwargs['model_data'], 's3://bucket/model.tar.gz')

    @patch('src.pipeline_steps.builder_model_step_pytorch.PyTorchModel')
    def test_create_step_with_custom_container(self, mock_pytorch_model_cls):
        # Setup config for custom container
        self.config.use_pytorch_framework = False
        self.config.image_uri = 'custom-image-uri'
        
        # Setup mocks
        mock_model = MagicMock()
        
        # Patch the _create_model method
        with patch.object(self.builder, '_create_model', return_value=mock_model) as mock_create_model:
            # Setup model.create to return step_args
            mock_model.create.return_value = {'ModelName': 'test-model'}
            
            # Call create_step
            step = self.builder.create_step(model_data='s3://bucket/model.tar.gz')
            
            # Verify _create_model was called instead of _create_pytorch_model
            mock_create_model.assert_called_once_with('s3://bucket/model.tar.gz')
            mock_pytorch_model_cls.assert_not_called()
            
            # Verify the step is created correctly
            self.assertIsInstance(step, CreateModelStep)
            self.assertEqual(step.step_args, {'ModelName': 'test-model'})

    def test_get_input_requirements(self):
        # Test that input requirements are correctly generated
        input_reqs = self.builder.get_input_requirements()
        
        # Check that model_data is included
        self.assertIn('model_data', input_reqs)
        self.assertEqual(input_reqs['model_data'], 'S3 path for ModelArtifacts')
        
        # Check that common properties are included
        self.assertIn('dependencies', input_reqs)
        self.assertIn('enable_caching', input_reqs)

    def test_get_output_properties(self):
        # Test that output properties are correctly generated
        output_props = self.builder.get_output_properties()
        
        # Check that output properties match config.output_names
        self.assertIn('model', output_props)
        self.assertEqual(output_props['model'], 'ModelName')
        self.assertIn('model_artifacts_path', output_props)
        self.assertEqual(output_props['model_artifacts_path'], 'ModelArtifactsPath')

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
