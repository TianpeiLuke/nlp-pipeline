from typing import Dict, Optional, List
from pathlib import Path
import os
import logging

from sagemaker.xgboost.estimator import XGBoost
from sagemaker.debugger import ProfilerConfig
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession

from .hyperparameters_xgboost import XGBoostModelHyperparameters
from .config_training_step_xgboost import XGBoostTrainingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XGBoostTrainingStepBuilder(StepBuilderBase):
    """XGBoost model training step builder, aligned to train_xgb.py expectations."""

    def __init__(
        self,
        config: XGBoostTrainingConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        super().__init__(config, sagemaker_session, role, notebook_root)
        self.config: XGBoostTrainingConfig = config

        # Sanity‐check: hyperparameters must be XGBoostModelHyperparameters
        if not isinstance(self.config.hyperparameters, XGBoostModelHyperparameters):
            raise ValueError("Config.hyperparameters must be an instance of XGBoostModelHyperparameters.")

        logger.info(f"Initialized XGBoostTrainingStepBuilder with hyperparams: "
                    f"{self.config.hyperparameters.model_dump()}")
        # Note: StepBuilderBase.__init__ already calls validate_configuration()

    def validate_configuration(self) -> None:
        """Ensure required fields are present and paths exist (locally or conceptually)."""
        required = [
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
        missing = [r for r in required if not getattr(self.config, r, None)]
        if missing:
            raise ValueError(f"XGBoostTrainingConfig is missing required attributes: {missing}")

        # Warn if source_dir or entry_point is not found locally (we assume it'll exist in S3 during actual run)
        src_dir = Path(self.config.source_dir or "")
        if not src_dir.exists() or not src_dir.is_dir():
            logger.warning(f"Local source_dir '{self.config.source_dir}' not found. "
                           "Make sure it’s packaged for SageMaker.")
        else:
            ep = src_dir / self.config.training_entry_point
            if not ep.exists():
                logger.warning(f"Entry point '{self.config.training_entry_point}' not found under source_dir '{src_dir}'.")


    def _create_profiler_config(self) -> ProfilerConfig:
        return ProfilerConfig(system_monitor_interval_millis=1000)

    def _get_metric_definitions(self) -> List[Dict[str, str]]:
        """
        Common XGBoost metric definitions (regex patterns). We include
        any eval_metric names passed in hyperparameters as well.
        """
        base_metrics = [
            {"Name": "train:rmse",       "Regex": r"\[\d+\]\s*train-rmse:([0-9\.]+)"},
            {"Name": "validation:rmse",  "Regex": r"\[\d+\]\s*validation-rmse:([0-9\.]+)"},
            {"Name": "train:auc",        "Regex": r"\[\d+\]\s*train-auc:([0-9\.]+)"},
            {"Name": "validation:auc",   "Regex": r"\[\d+\]\s*validation-auc:([0-9\.]+)"},
            {"Name": "train:logloss",    "Regex": r"\[\d+\]\s*train-logloss:([0-9\.]+)"},
            {"Name": "validation:logloss","Regex": r"\[\d+\]\s*validation-logloss:([0-9\.]+)"},
            {"Name": "train:error",      "Regex": r"\[\d+\]\s*train-error:([0-9\.]+)"},
            {"Name": "validation:error", "Regex": r"\[\d+\]\s*validation-error:([0-9\.]+)"},
        ]

        # If the user specified eval_metric(s) in hyperparams, ensure we include them
        hp_eval = self.config.hyperparameters.eval_metric
        if hp_eval:
            if isinstance(hp_eval, str):
                hp_eval = [hp_eval]
            for metric in hp_eval:
                # add training + validation forms if not already there
                name_val   = f"validation:{metric}"
                regex_val  = rf"\[\d+\]\s*validation-{metric}:([0-9\.]+)"
                name_train = f"train:{metric}"
                regex_train= rf"\[\d+\]\s*train-{metric}:([0-9\.]+)"

                if not any(m["Name"] == name_val for m in base_metrics):
                    base_metrics.append({"Name": name_val, "Regex": regex_val})
                if not any(m["Name"] == name_train for m in base_metrics):
                    base_metrics.append({"Name": name_train, "Regex": regex_train})

        return base_metrics


    def _create_xgboost_estimator(self, checkpoint_s3_uri: Optional[str]) -> XGBoost:
        """
        Instantiate a SageMaker XGBoost Estimator in script mode, passing in:
        - entry_point, source_dir, hyperparameters
        - instance count/type, volume, framework_version, py_version
        - profiler_config, metric_definitions, checkpoint locations
        """
        # Convert our Pydantic hyperparams to the flat dict that SageMaker XGBoost expects
        hyperparams = self.config.hyperparameters.serialize_config()

        xgb_estimator = XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.source_dir,
            role=self.role,
            instance_count=self.config.training_instance_count,
            instance_type=self.config.training_instance_type,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            volume_size=self.config.training_volume_size,
            output_path=self.config.output_path,
            sagemaker_session=self.session,
            hyperparameters=hyperparams,
            profiler_config=self._create_profiler_config(),
            metric_definitions=self._get_metric_definitions(),
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path="/opt/ml/checkpoints",
            # You can add use_spot_instances, max_wait, etc. from config if desired
        )
        return xgb_estimator


    def _get_checkpoint_uri(self) -> Optional[str]:
        """Return the configured checkpoint S3 URI (or build a default)."""
        if self.config.has_checkpoint():
            return self.config.get_checkpoint_uri()

        # Otherwise build a default under output_path/checkpoints/... 
        return os.path.join(
            self.config.output_path,
            "checkpoints",
            self.config.pipeline_name,
            self.config.current_date
        )


    def create_step(self, dependencies: Optional[List[Step]] = None) -> TrainingStep:
        """
        Build and return a SageMaker `TrainingStep` that matches train_xgb.py’s
        expectation of “/opt/ml/input/data/train”, “/opt/ml/input/data/val”, (optionally “/opt/ml/input/data/test”).
        """
        conf = self.config
        # --- Build S3 prefixes for train/val/test ---
        #  (We do NOT point at a single file; train_xgb.py does its own "combine_shards" under that folder.)
        train_prefix = os.path.join(conf.input_path, "train")
        val_prefix   = os.path.join(conf.input_path, "val")
        test_prefix  = os.path.join(conf.input_path, "test")

        # Always require “train” and “val” to exist in S3—otherwise script will crash at runtime
        # (We trust the pipeline to have created or copied data under these exact locations.)
        logger.info(f"Using train prefix: {train_prefix}")
        logger.info(f"Using val   prefix: {val_prefix}")

        inputs: Dict[str, TrainingInput] = {
            # "train" channel → mounted to /opt/ml/input/data/train
            "train": TrainingInput(
                s3_data=train_prefix,
                distribution="FullyReplicated",
                content_type="application/x-parquet",  # train_xgb.py will detect format automatically
                s3_data_type="S3Prefix"
            ),
            # "val" channel → mounted to /opt/ml/input/data/val
            "val": TrainingInput(
                s3_data=val_prefix,
                distribution="FullyReplicated",
                content_type="application/x-parquet", 
                s3_data_type="S3Prefix"
            ),
        }

        # Optionally add a “test” channel if that prefix exists (not strictly required by train_xgb.py,
        # but train_xgb.py will check and load it if present).
        # We assume the user has put data under conf.input_path/test/… if they want “test”.
        # If that S3 prefix is empty or nonexistent, we skip it.
        from botocore.exceptions import ClientError
        import boto3

        s3 = boto3.resource("s3")
        parsed = conf.input_path.replace("s3://", "").split("/", 1)
        bucket = parsed[0]
        prefix = parsed[1] if len(parsed) > 1 else ""
        test_full_prefix = f"{prefix}/test"
        # Check “does this prefix exist?” by listing 1 object
        try:
            bucket_obj = s3.Bucket(bucket)
            objs = list(bucket_obj.objects.filter(Prefix=test_full_prefix).limit(1))
            if len(objs) > 0:
                # We do see at least one object under “…/test/”
                inputs["test"] = TrainingInput(
                    s3_data=test_prefix,
                    distribution="FullyReplicated",
                    content_type="application/x-parquet",
                    s3_data_type="S3Prefix"
                )
                logger.info(f"Detected and adding 'test' channel: {test_prefix}")
        except ClientError as e:
            # Could not list (e.g. no permission or bucket doesn’t exist) → skip “test”
            logger.warning(f"Unable to verify test prefix '{test_prefix}': {e}. Skipping 'test' channel.")

        # --- Instantiate the XGBoost estimator (script mode) ---
        checkpoint_uri = self._get_checkpoint_uri()
        estimator = self._create_xgboost_estimator(checkpoint_uri)

        step_name = self._get_step_name("XGBoostTraining")
        training_step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=inputs,
            depends_on=dependencies or []
        )

        return training_step


    def create_training_step(self, dependencies: Optional[List[Step]] = None) -> TrainingStep:
        """Deprecated alias"""
        logger.warning("create_training_step() is deprecated; use create_step() instead.")
        return self.create_step(dependencies)
