from sagemaker.pytorch import PyTorch
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from pathlib import Path

from typing import Optional, Dict, List
import os
import logging

from .config_training_step_pytorch import PytorchTrainingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

class PyTorchTrainingStepBuilder(StepBuilderBase):
    """PyTorch model builder"""
    
    def __init__(
        self, 
        config: PytorchTrainingConfig, 
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize PyTorch model builder
        
        Args:
            config: Pydantic ModelConfig instance with hyperparameters
            sagemaker_session: SageMaker session
            role: IAM role ARN
            notebook_root: Root directory of notebook
        """
        super().__init__(config, sagemaker_session, role, notebook_root)
        
        if not self.config.hyperparameters:
            raise ValueError("ModelConfig must include hyperparameters for training")
            
        logger.info(f"Initialized PyTorchTrainingStepBuilder with hyperparams: {self.config.hyperparameters.get_config()}")

    def validate_configuration(self) -> None:
        """Validate configuration requirements"""
        required_attrs = [
            'training_entry_point',
            'source_dir',
            'training_instance_type',
            'training_instance_count',
            'training_volume_size',
            'framework_version',
            'py_version',
            'input_path',
            'output_path'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"ModelConfig missing required attribute: {attr}")
        
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
    
    def _create_pytorch_estimator(self, checkpoint_s3_uri: str) -> PyTorch:
        """Create PyTorch estimator"""
        return PyTorch(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.source_dir,
            role=self.role,
            instance_count=self.config.training_instance_count,
            instance_type=self.config.training_instance_type,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            volume_size=self.config.training_volume_size,
            max_run=4 * 24 * 60 * 60,
            output_path=self.config.output_path,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
            sagemaker_session=self.session,
            hyperparameters=self.config.hyperparameters.serialize_config(),
            profiler_config=self._create_profiler_config(),
            metric_definitions=self._get_metric_definitions()
        )

    def _get_checkpoint_uri(self) -> str:
        """Get checkpoint URI for training"""
        if self.config.has_checkpoint():
            return self.config.get_checkpoint_uri()
        
        return os.path.join(
            self.config.output_path,
            "checkpoints",
            self.config.current_date
        )

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        input_reqs = {k: v for k, v in self.config.input_names.items()}
        input_reqs["dependencies"] = self.COMMON_PROPERTIES["dependencies"]
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        return {k: v for k, v in self.config.output_names.items()}
    
    def create_step(self, dependencies: Optional[List] = None) -> Step:
        """
        Create training step with dataset inputs.
        
        Args:
            dependencies: List of dependent steps
            
        Returns:
            TrainingStep instance
        """
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

        # Get checkpoint URI and create estimator
        checkpoint_uri = self._get_checkpoint_uri()
        logger.info(
            f"Creating PyTorch estimator:"
            f"\n\tCheckpoint URI: {checkpoint_uri}"
            f"\n\tInstance Type: {self.config.instance_type}"
            f"\n\tFramework Version: {self.config.framework_version}"
            f"\n\tPython Version: {self.config.py_version}"
        )
        estimator = self._create_pytorch_estimator(checkpoint_uri)
        
        # Get step name
        step_name = self._get_step_name('PytorchTraining')
        
        return TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=inputs,
            depends_on=dependencies or []
        )

    # Maintain backwards compatibility
    def create_training_step(self, dependencies: Optional[List] = None) -> TrainingStep:
        """Backwards compatible method for creating training step"""
        return self.create_step(dependencies)
