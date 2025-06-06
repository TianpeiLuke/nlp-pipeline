import os
import logging
from pathlib import Path
from typing import Optional, List, Dict

from sagemaker.xgboost.estimator import XGBoost
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep, Step  # Step used for typing
from sagemaker.workflow.pipeline_context import PipelineSession

from .hyperparameters_xgboost import XGBoostModelHyperparameters
from .config_training_step_xgboost import XGBoostTrainingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XGBoostTrainingStepBuilder(StepBuilderBase):
    """XGBoost model training step builder."""

    def __init__(
        self,
        config: XGBoostTrainingConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize XGBoost model builder.

        Args:
            config: XGBoostTrainingConfig instance
            sagemaker_session: SageMaker PipelineSession
            role: IAM role ARN
            notebook_root: Root directory of notebook (optional, often unused)
        """
        if not isinstance(config, XGBoostTrainingConfig):
            raise ValueError("XGBoostTrainingStepBuilder requires an XGBoostTrainingConfig instance.")
        super().__init__(config=config, sagemaker_session=sagemaker_session, role=role, notebook_root=notebook_root)
        self.config: XGBoostTrainingConfig = config

        if not isinstance(self.config.hyperparameters, XGBoostModelHyperparameters):
            raise ValueError("Config for XGBoostTrainingStepBuilder must include XGBoostModelHyperparameters.")

        logger.info(f"Initialized XGBoostTrainingStepBuilder with hyperparameters: "
                    f"{self.config.hyperparameters.get_config()}")
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate configuration requirements for XGBoost training."""
        required_attrs = [
            "training_entry_point",
            "source_dir",
            "training_instance_type",
            "training_instance_count",
            "training_volume_size",
            "framework_version",
            "py_version",
            "input_path",
            "output_path",
        ]

        missing = [attr for attr in required_attrs if not getattr(self.config, attr, None)]
        if missing:
            raise ValueError(f"XGBoostTrainingConfig missing required attributes: {', '.join(missing)}")

        # If source_dir is local, ensure the entry point exists
        src = self.config.get_effective_source_dir()
        if src and not src.startswith("s3://"):
            script_path = Path(src) / self.config.training_entry_point
            if not script_path.is_file():
                logger.warning(
                    f"Entry point script '{self.config.training_entry_point}' not found under source_dir='{src}'."
                )

        # We assume preprocessing has written to <input_path>/train and <input_path>/val
        train_prefix = f"{self.config.input_path.rstrip('/')}/train"
        val_prefix = f"{self.config.input_path.rstrip('/')}/val"
        if not train_prefix.startswith("s3://") or not val_prefix.startswith("s3://"):
            raise ValueError(
                f"Constructed train/val prefixes must be valid S3 URIs: {train_prefix}, {val_prefix}"
            )

        logger.info("XGBoostTrainingConfig validation succeeded.")

    def _create_profiler_config(self) -> ProfilerConfig:
        """Create profiler configuration (optional for XGBoost)."""
        return ProfilerConfig(system_monitor_interval_millis=1000)

    def _get_metric_definitions(self) -> List[Dict[str, str]]:
        """
        Get metric definitions for XGBoost training monitoring.
        Adjust Regex patterns to match your train_xgb.py logging.
        """
        metrics = [
            {"Name": "validation:rmse", "Regex": r"\[\d+\]\s*validation-rmse:([0-9\.]+)"},
            {"Name": "validation:auc", "Regex": r"\[\d+\]\s*validation-auc:([0-9\.]+)"},
            {"Name": "validation:logloss", "Regex": r"\[\d+\]\s*validation-logloss:([0-9\.]+)"},
            {"Name": "validation:error", "Regex": r"\[\d+\]\s*validation-error:([0-9\.]+)"},
            {"Name": "validation:merror", "Regex": r"\[\d+\]\s*validation-merror:([0-9\.]+)"},
            {"Name": "train:auc", "Regex": r"\[\d+\]\s*train-auc:([0-9\.]+)"},
            {"Name": "train:rmse", "Regex": r"\[\d+\]\s*train-rmse:([0-9\.]+)"},
        ]

        hp_eval = self.config.hyperparameters.eval_metric
        if hp_eval:
            if isinstance(hp_eval, str):
                hp_eval = [hp_eval]
            for metric in hp_eval:
                val_name = f"validation:{metric}"
                train_name = f"train:{metric}"
                if not any(m["Name"] == val_name for m in metrics):
                    metrics.append({
                        "Name": val_name,
                        "Regex": rf"\[\d+\]\s*validation-{metric}:([0-9\.]+)"
                    })
                if not any(m["Name"] == train_name for m in metrics):
                    metrics.append({
                        "Name": train_name,
                        "Regex": rf"\[\d+\]\s*train-{metric}:([0-9\.]+)"
                    })

        return metrics

    def _create_xgboost_estimator(self, checkpoint_s3_uri: Optional[str]) -> XGBoost:
        """Create SageMaker XGBoost Estimator in script mode."""
        hyperparameters = self.config.hyperparameters.serialize_config()

        return XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.get_effective_source_dir(),
            role=self.role,
            instance_count=self.config.training_instance_count,
            instance_type=self.config.training_instance_type,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            volume_size=self.config.training_volume_size,
            output_path=self.config.output_path,
            sagemaker_session=self.session,
            hyperparameters=hyperparameters,
            profiler_config=self._create_profiler_config(),
            metric_definitions=self._get_metric_definitions(),
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
        )

    def _get_checkpoint_uri(self) -> Optional[str]:
        """Return S3 URI for checkpoints (if enabled)."""
        if self.config.has_checkpoint():
            return self.config.get_checkpoint_uri()

        # Default to a subfolder under output_path
        return os.path.join(
            self.config.output_path.rstrip("/"),
            "checkpoints",
            self.config.pipeline_name,
            self.config.current_date,
        )

    def create_step(self, dependencies: Optional[List[Step]] = None) -> TrainingStep:
        """
        Create and return a SageMaker Pipeline TrainingStep for XGBoost.

        Expects:
          • S3 data under <input_path>/train  and <input_path>/val
          • Data format (CSV or Parquet) consistent with train_xgb.py’s parsing
        """
        # Build S3 prefixes for train and validation channels
        train_data_prefix = f"{self.config.input_path.rstrip('/')}/train"
        val_data_prefix = f"{self.config.input_path.rstrip('/')}/val"

        logger.info(f"Using training data prefix: {train_data_prefix}")
        logger.info(f"Using validation data prefix: {val_data_prefix}")

        inputs = {
            "train": TrainingInput(
                s3_data=train_data_prefix,
                content_type="text/csv",       # or "application/x-parquet" if using Parquet
                s3_data_type="S3Prefix",
                distribution="FullyReplicated"
            ),
            "validation": TrainingInput(
                s3_data=val_data_prefix,
                content_type="text/csv",
                s3_data_type="S3Prefix",
                distribution="FullyReplicated"
            )
        }

        checkpoint_uri = self._get_checkpoint_uri()
        logger.info(
            f"Creating XGBoost estimator with checkpoint URI: {checkpoint_uri}, "
            f"instance_type: {self.config.training_instance_type}, "
            f"framework_version: {self.config.framework_version}, "
            f"entry_point: {self.config.training_entry_point}, "
            f"source_dir: {self.config.get_effective_source_dir()}"
        )
        estimator = self._create_xgboost_estimator(checkpoint_uri)

        step_name = self._get_step_name("Training")
        return TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=inputs,
            depends_on=dependencies or [],
        )

    def create_training_step(self, dependencies: Optional[List[Step]] = None) -> TrainingStep:
        """Backwards compatibility alias for create_step."""
        logger.warning("create_training_step is deprecated; use create_step instead.")
        return self.create_step(dependencies)
