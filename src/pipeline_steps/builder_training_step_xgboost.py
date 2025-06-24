from sagemaker.xgboost.estimator import XGBoost
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import Join  # Make sure to import
from sagemaker.s3 import S3Uploader
from pathlib import Path
import os
import json
import tempfile
import shutil
from botocore.exceptions import ClientError
from typing import Optional, List, Dict
import logging

from .hyperparameters_xgboost import XGBoostModelHyperparameters
from .config_training_step_xgboost import XGBoostTrainingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


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

    # The _prepare_hyperparameters_file method has been removed and replaced by a separate step


    def _create_xgboost_estimator(self) -> XGBoost:
        """Create SageMaker XGBoost Estimator with an empty hyperparameters dictionary."""
        
        # use secure-pypi
        env = {
             "CA_REPOSITORY_ARN": "arn:aws:codeartifact:us-west-2:149122183214:repository/amazon/secure-pypi"
        }

        logger.info(f"Use Secure-PyPI with CA_REPOSITORY_ARN = {env['CA_REPOSITORY_ARN']}")
        
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
            environment=env
        )

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        input_reqs = {k: v for k, v in self.config.input_names.items()}
        input_reqs["dependencies"] = self.COMMON_PROPERTIES["dependencies"]
        input_reqs["hyperparameters_s3_uri"] = "S3 URI to the hyperparameters.json file"
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        return {k: v for k, v in self.config.output_names.items()}
    
    def create_step(self, dependencies: Optional[List[Step]] = None, hyperparameters_s3_uri: Optional[str] = None) -> TrainingStep:
        """
        Create XGBoost training step with data and config inputs.
        
        Args:
            dependencies: Optional list of steps this step depends on
            hyperparameters_s3_uri: S3 URI to the hyperparameters.json file
        """
        # Validate hyperparameters_s3_uri
        if not hyperparameters_s3_uri:
            raise ValueError("hyperparameters_s3_uri must be provided")
        
        # Define the S3 prefixes for each data channel using Join
        train_data_prefix = Join(on='/', values=[self.config.input_path, "train/"])
        val_data_prefix = Join(on='/', values=[self.config.input_path, "val/"])
        test_data_prefix = Join(on='/', values=[self.config.input_path, "test/"])

        # Log expressions
        logger.info(f"Train data path expression: {train_data_prefix.expr}")
        logger.info(f"Validation data path expression: {val_data_prefix.expr}")
        logger.info(f"Test data path expression: {test_data_prefix.expr}")
        logger.info(f"Hyperparameters S3 URI: {hyperparameters_s3_uri}")

        # Define the input channels
        inputs = {
            "train": TrainingInput(s3_data=train_data_prefix),
            "val": TrainingInput(s3_data=val_data_prefix),
            "test": TrainingInput(s3_data=test_data_prefix),
            # Channel for the hyperparameter file
            "config": TrainingInput(s3_data=hyperparameters_s3_uri)
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
