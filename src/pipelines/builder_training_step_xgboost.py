from sagemaker.xgboost.estimator import XGBoost
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join # Make sure to import
from sagemaker.s3 import S3Uploader
from pathlib import Path
import os
import json
from typing import Optional, List, Dict
import logging
import tempfile

from .hyperparameters_xgboost import XGBoostModelHyperparameters
from .config_training_step_xgboost import XGBoostTrainingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


# --- XGBoost Training Step Builder ---
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """XGBoost model training step builder that uses a config file for hyperparameters."""

    def __init__(
        self,
        config: XGBoostTrainingConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        super().__init__(config, sagemaker_session, role, notebook_root)
        self.config: XGBoostTrainingConfig  # For type hinting

        if not isinstance(self.config.hyperparameters, XGBoostModelHyperparameters):
            raise ValueError("Config for XGBoostTrainingStepBuilder must include XGBoostModelHyperparameters.")
            
        self.validate_configuration()


    def validate_configuration(self) -> None:
        """Validate configuration requirements for XGBoost training"""
        required_attrs = [
            'training_entry_point', 'source_dir', 'training_instance_type',
            'training_instance_count', 'training_volume_size', 'framework_version',
            'py_version', 'input_path', 'output_path'
        ]
        
        missing_attrs = [attr for attr in required_attrs if not getattr(self.config, attr, None)]
        if missing_attrs:
            raise ValueError(f"XGBoostTrainingConfig missing required attributes: {', '.join(missing_attrs)}")

    def _prepare_hyperparameters_file(self) -> str:
        """
        Serializes the hyperparameters to a local JSON file and uploads it to S3.
        
        Returns:
            The S3 URI of the uploaded hyperparameters file.
        """
        hyperparams_dict = self.config.hyperparameters.model_dump()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp:
            json.dump(hyperparams_dict, tmp, indent=2)
            local_path = tmp.name
        
        s3_uri = self.config.hyperparameters_s3_uri
        logger.info(f"Uploading hyperparameters from {local_path} to {s3_uri}...")
        uploaded_uri = S3Uploader.upload(local_path, s3_uri, sagemaker_session=self.session)
        os.remove(local_path) # Clean up the temporary local file
        
        logger.info(f"Hyperparameters successfully uploaded to {uploaded_uri}")
        return uploaded_uri
    
    def _create_xgboost_estimator(self) -> XGBoost:
        """Create SageMaker XGBoost Estimator with an empty hyperparameters dictionary."""
        return XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.source_dir,
            role=self.role,
            instance_count=self.config.training_instance_count,
            instance_type=self.config.training_instance_type,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            # No hyperparameters are passed here directly
            hyperparameters={},
            sagemaker_session=self.session,
            output_path=self.config.output_path,
        )

    def create_step(self, dependencies: Optional[List[Step]] = None) -> TrainingStep:
        """
        Create XGBoost training step with data and config inputs.
        """
        # 1. Upload the hyperparameters to S3 and get the URI
        hparam_s3_uri = self._prepare_hyperparameters_file()
        
        # Define the S3 prefixes for each data channel using Join
        train_data_prefix = Join(on='/', values=[self.config.input_path, "train/"])
        val_data_prefix = Join(on='/', values=[self.config.input_path, "val/"])
        test_data_prefix = Join(on='/', values=[self.config.input_path, "test/"])

        # FIX: Use .expr to log the expression of dynamic pipeline variables
        logger.info(f"Train data path expression: {train_data_prefix.expr}")
        logger.info(f"Validation data path expression: {val_data_prefix.expr}")
        logger.info(f"Test data path expression: {test_data_prefix.expr}")

        # 2. Define the input channels
        inputs = {
            "train": TrainingInput(s3_data=train_data_prefix),
            "val": TrainingInput(s3_data=val_data_prefix),
            "test": TrainingInput(s3_data=test_data_prefix),
            # Add a channel to provide the hyperparameter file to the training job
            "config": TrainingInput(s3_data=hparam_s3_uri)
        }

        logger.info(f"Training inputs configured: {inputs}")

        # 3. Create the estimator (without hyperparameters)
        estimator = self._create_xgboost_estimator()
        
        step_name = self._get_step_name('XGBoostTraining')
        
        # 4. Create the TrainingStep
        return TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=inputs,
            depends_on=dependencies or []
        )

    def create_training_step(self, dependencies: Optional[List[Step]] = None) -> TrainingStep:
        """Backwards compatible method for creating training step"""
        logger.warning("create_training_step is deprecated, use create_step instead.")
        return self.create_step(dependencies)