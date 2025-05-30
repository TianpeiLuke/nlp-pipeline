import unittest
from unittest.mock import Mock, patch
import os
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep


from pipelines.builder_training_step_pytorch import PyTorchTrainingStepBuilder


class TestPyTorchTrainingStepBuilder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        # Mock ModelConfig
        self.mock_config = Mock()
        self.mock_config.region = "us-east-1"
        self.mock_config.role = "arn:aws:iam::123456789012:role/SageMakerRole"
        self.mock_config.bucket = "bucket"
        self.mock_config.input_path = "s3://bucket/input"
        self.mock_config.output_path = "s3://bucket/output"
        self.mock_config.instance_type = "ml.g5.12xlarge"
        self.mock_config.framework_version = "2.1.0"
        self.mock_config.py_version = "py310"
        self.mock_config.volume_size = 100
        self.mock_config.entry_point = "train.py"
        self.mock_config.source_dir = "./source"
        self.mock_config.instance_count = 1
        self.mock_config.current_date = "2025-05-20"
        self.mock_config.has_checkpoint.return_value = False
        
        # Mock ModelHyperparameters
        self.mock_hyperparams = Mock()
        self.mock_hyperparams.get_config.return_value = {"batch_size": 4}
        self.mock_hyperparams.serialize_config.return_value = {"batch_size": "4"}
        
        # Mock SageMaker session
        self.mock_session = Mock()
        self.mock_session.sagemaker_config = {
            "SchemaVersion": "1.0",
            "SageMaker": {
                "PythonSDK": {
                    "Modules": {
                        "Session": {
                            "DefaultS3Bucket": "my-bucket"
                        }
                    }
                }
            }
        }
        
        # Create builder instance
        self.builder = PyTorchTrainingStepBuilder(
            config=self.mock_config,
            hyperparams=self.mock_hyperparams,
            sagemaker_session=self.mock_session,
            role="arn:aws:iam::123456789012:role/SageMakerRole"
        )

    def test_create_profiler_config(self):
        """Test profiler configuration creation"""
        profiler_config = self.builder._create_profiler_config()
        self.assertIsInstance(profiler_config, ProfilerConfig)
        self.assertEqual(profiler_config.system_monitor_interval_millis, 1000)

    def test_get_metric_definitions(self):
        """Test metric definitions"""
        metrics = self.builder._get_metric_definitions()
        self.assertEqual(len(metrics), 4)
        self.assertIn('Name', metrics[0])
        self.assertIn('Regex', metrics[0])
        self.assertEqual(metrics[0]['Name'], 'Train Loss')

    @patch('pipelines.builder_training_step_pytorch.PyTorch')
    def test_create_estimator(self, mock_pytorch_class):
        """Test PyTorch estimator creation"""
        # Configure mock
        mock_estimator = Mock()
        mock_pytorch_class.return_value = mock_estimator
        
        # Create estimator
        estimator = self.builder.create_estimator()
        
        # Verify PyTorch constructor was called
        mock_pytorch_class.assert_called_once()
        
        # Get the call arguments
        args, kwargs = mock_pytorch_class.call_args
        
        # Verify the arguments
        self.assertEqual(kwargs['entry_point'], self.mock_config.entry_point)
        self.assertEqual(kwargs['source_dir'], self.mock_config.source_dir)
        self.assertEqual(kwargs['role'], "arn:aws:iam::123456789012:role/SageMakerRole")
        self.assertEqual(kwargs['instance_count'], self.mock_config.instance_count)
        self.assertEqual(kwargs['instance_type'], self.mock_config.instance_type)
        self.assertEqual(kwargs['framework_version'], self.mock_config.framework_version)
        self.assertEqual(kwargs['py_version'], self.mock_config.py_version)
        self.assertEqual(kwargs['volume_size'], self.mock_config.volume_size)
        self.assertEqual(kwargs['max_run'], 4 * 24 * 60 * 60)
        self.assertEqual(kwargs['output_path'], self.mock_config.output_path)
        self.assertIsInstance(kwargs['profiler_config'], ProfilerConfig)
        
        # Verify the returned estimator
        self.assertEqual(estimator, mock_estimator)

    @patch('pipelines.builder_training_step_pytorch.PyTorch')
    def test_create_training_step(self, mock_pytorch_class):
        """Test training step creation"""
        # Configure mock
        mock_estimator = Mock()
        mock_pytorch_class.return_value = mock_estimator

        # Case 1: pipeline_name is set
        self.mock_config.pipeline_name = "MyPipeline"
        training_step = self.builder.create_training_step()
        expected_name = "MyPipeline-Training"
        self.assertEqual(training_step.name, expected_name[:80])  # Verify the name

        # Case 2: pipeline_name is not set
        self.mock_config.pipeline_name = None
        training_step = self.builder.create_training_step()
        self.assertEqual(training_step.name, "DefaultModelTraining")  # Verify the default name

        # Verify input paths
        expected_train_path = os.path.join(self.mock_config.input_path, "train", "train.parquet")
        expected_val_path = os.path.join(self.mock_config.input_path, "val", "val.parquet")
        expected_test_path = os.path.join(self.mock_config.input_path, "test", "test.parquet")

        inputs = training_step.inputs
        self.assertIsInstance(inputs['train'], TrainingInput)
        self.assertEqual(inputs['train'].config['DataSource']['S3DataSource']['S3Uri'], 
                     expected_train_path)
        self.assertEqual(inputs['val'].config['DataSource']['S3DataSource']['S3Uri'], 
                     expected_val_path)
        self.assertEqual(inputs['test'].config['DataSource']['S3DataSource']['S3Uri'], 
                     expected_test_path)
        

    @patch('pipelines.builder_training_step_pytorch.PyTorch')
    def test_checkpoint_uri_handling(self, mock_pytorch_class):
        """Test checkpoint URI handling"""
        # Configure mock
        mock_estimator = Mock()
        mock_pytorch_class.return_value = mock_estimator
        
        # Test with checkpoint from config
        self.mock_config.has_checkpoint.return_value = True
        self.mock_config.get_checkpoint_uri.return_value = "s3://bucket/checkpoint"
        
        estimator = self.builder.create_estimator()
        args, kwargs = mock_pytorch_class.call_args
        self.assertEqual(kwargs['checkpoint_s3_uri'], "s3://bucket/checkpoint")
        
        # Test without checkpoint (default path)
        self.mock_config.has_checkpoint.return_value = False
        estimator = self.builder.create_estimator()
        args, kwargs = mock_pytorch_class.call_args
        expected_checkpoint_path = os.path.join(
            self.mock_config.output_path,
            "checkpoints",
            self.mock_config.current_date
        )
        self.assertEqual(kwargs['checkpoint_s3_uri'], expected_checkpoint_path)
        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)