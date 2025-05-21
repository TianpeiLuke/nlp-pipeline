from sagemaker.pytorch import PyTorch
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession # Crucial import

from typing import Optional, Dict, Tuple, List
import os

from .workflow_config import ModelConfig, ModelHyperparameters


import logging
logger = logging.getLogger(__name__)


class PyTorchTrainingStepBuilder:
    """PyTorch model builder"""
    def __init__(self, 
                 config: ModelConfig, 
                 hyperparams: ModelHyperparameters,
                 sagemaker_session: Optional[PipelineSession] = None,
                 role: str = None):
        """
        Initialize PyTorch model builder
        
        Args:
            config: Pydantic ModelConfig instance
            hyperparams: Pydantic ModelHyperparameters instance
            sagemaker_session: SageMaker session
            role: IAM role ARN
        """
        self.config = config
        self.hyperparams = hyperparams
        self.session = sagemaker_session
        self.role = role
        logger.info(f"Initialized PyTorchModelBuilder with hyperparams: {hyperparams.get_config()}")
        
    def _create_profiler_config(self) -> ProfilerConfig:
        """Create profiler configuration"""
        return ProfilerConfig(
            system_monitor_interval_millis=1000
        )

    def _get_metric_definitions(self) -> List[Dict[str, str]]:
        """Get metric definitions for training monitoring"""
        return [
            {'Name': 'Train Loss', 'Regex': 'train_loss=([0-9\\.]+)'},
            {'Name': 'Validation Loss', 'Regex': 'val_loss=([0-9\\.]+)'},
            {'Name': 'Validation F1 Score', 'Regex': 'val/f1_score=([0-9\\.]+)'},
            {'Name': 'Validation AUC ROC', 'Regex': 'val/auroc=([0-9\\.]+)'},
        ]
    
    def _create_pytorch_estimator(self, checkpoint_s3_uri) -> PyTorch:
        return PyTorch(
            entry_point=self.config.entry_point,
            source_dir=self.config.source_dir,
            role=self.role,
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            volume_size=self.config.volume_size,
            max_run=4 * 24 * 60 * 60,
            output_path=self.config.output_path,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
            sagemaker_session=self.session,
            hyperparameters=self.hyperparams.serialize_config(),
            profiler_config=self._create_profiler_config(),
            metric_definitions=self._get_metric_definitions()
        )

    
    def create_estimator(self) -> PyTorch:
        """Create PyTorch estimator"""
        # Use checkpoint path from config if available
        checkpoint_s3_uri = None
        if self.config.has_checkpoint():
            checkpoint_s3_uri = self.config.get_checkpoint_uri()
        else:
            # Create default checkpoint path
            checkpoint_s3_uri = os.path.join(
                self.config.output_path,
                "checkpoints",
                self.config.current_date
            )
    
        logger.info(
            f"Creating PyTorch estimator:"
            f"\n\tCheckpoint URI: {checkpoint_s3_uri}"
            f"\n\tInstance Type: {self.config.instance_type}"
            f"\n\tFramework Version: {self.config.framework_version}"
            f"\n\tPython Version: {self.config.py_version}"
        )
        
        return self._create_pytorch_estimator(checkpoint_s3_uri)

    def _sanitize_name_for_sagemaker(self, name: str, max_length: int = 63) -> str:
        """Sanitize a string to be a valid SageMaker resource name component."""
        if not name:
            return "default-name"
        # Replace non-alphanumeric characters (excluding hyphens) with a hyphen
        sanitized = "".join(c if c.isalnum() else '-' for c in str(name))
        # Remove leading/trailing hyphens and collapse multiple hyphens
        sanitized = '-'.join(filter(None, sanitized.split('-')))
        return sanitized[:max_length].rstrip('-')
    
    def create_training_step(self) -> TrainingStep:
        """Create training step with dataset inputs"""
        # Validate input path structure
        train_path = os.path.join(self.config.input_path, "train", "train.parquet")
        val_path = os.path.join(self.config.input_path, "val", "val.parquet")
        test_path = os.path.join(self.config.input_path, "test", "test.parquet")

        # Create training inputs
        inputs = {
            "train": TrainingInput(train_path),
            "val": TrainingInput(val_path),
            "test": TrainingInput(test_path)
        }

        estimator = self.create_estimator()
        
        step_name = "DefaultModelTraining" # Default if pipeline_name is not available
        # Max length for step names is 80 characters.
        # Pattern: ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,79}$
        if hasattr(self.config, 'pipeline_name') and self.config.pipeline_name:
            sanitized_pipeline_name = self._sanitize_name_for_sagemaker(self.config.pipeline_name, max_length=60)
            step_name = f"{sanitized_pipeline_name}-Training"
        else:
            logger.warning("pipeline_name not found in ModelConfig. Using default name 'DefaultModelTraining' for TrainingStep.")
        
        return TrainingStep(
            name=step_name[:80], # Ensure name does not exceed max length
            estimator=estimator,
            inputs=inputs,
            depends_on=[]
        )