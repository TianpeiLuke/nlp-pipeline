from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.sklearn import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from .config_model_eval_step_xgboost import XGBoostModelEvalConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class XGBoostModelEvalStepBuilder(StepBuilderBase):
    """
    Builder for XGBoost model evaluation ProcessingStep.
    """

    def __init__(
        self,
        config: XGBoostModelEvalConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        if not isinstance(config, XGBoostModelEvalConfig):
            raise ValueError("XGBoostModelEvalStepBuilder requires a XGBoostModelEvalConfig instance.")
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: XGBoostModelEvalConfig = config

    def validate_configuration(self) -> None:
        logger.info("Validating XGBoostModelEvalConfig…")
        if not getattr(self.config, 'job_type', None):
            raise ValueError("job_type must be provided (e.g. 'training','calibration','validation','test')")
        if not self.config.hyperparameters:
            raise ValueError("hyperparameters must be provided and non-empty")
        logger.info("XGBoostModelEvalConfig validation succeeded.")

    def _get_environment_variables(self) -> Dict[str, str]:
        # Use id_name and label_name from hyperparameters
        env_vars = {
            "ID_FIELD": str(self.config.hyperparameters.id_name),
            "LABEL_FIELD": str(self.config.hyperparameters.label_name),
        }
        logger.info(f"Evaluation environment variables: {env_vars}")
        return env_vars

    def _create_processor(self) -> SKLearnProcessor:
        instance_type = self.config.get_instance_type(
            'large' if self.config.use_large_processing_instance else 'small'
        )
        logger.info(f"Using processing instance type for evaluation: {instance_type}")

        base_job_name_prefix = self._sanitize_name_for_sagemaker(self.config.pipeline_name, 30)

        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            command=["python3"],
            role=self.role,
            instance_count=self.config.processing_instance_count,
            instance_type=instance_type,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=f"{base_job_name_prefix}-xgb-eval",
            sagemaker_session=self.session,
            env=self._get_environment_variables(),
        )

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        # Expect keys: 'model_input', 'eval_data_input'
        input_names = self.config.get_input_names()
        if not inputs or "model_input" not in inputs or "eval_data_input" not in inputs:
            raise ValueError("Must supply S3 URIs for 'model_input' and 'eval_data_input' in 'inputs'")
        return [
            ProcessingInput(
                input_name=input_names["model_input"],
                source=inputs["model_input"],
                destination="/opt/ml/processing/input/model"
            ),
            ProcessingInput(
                input_name=input_names["eval_data_input"],
                source=inputs["eval_data_input"],
                destination="/opt/ml/processing/input/eval_data"
            )
        ]

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        # Expect keys: 'eval_output', 'metrics_output'
        output_names = self.config.get_output_names()
        if not outputs or "eval_output" not in outputs or "metrics_output" not in outputs:
            raise ValueError("Must supply S3 URIs for 'eval_output' and 'metrics_output' in 'outputs'")
        return [
            ProcessingOutput(
                output_name=output_names["eval_output"],
                source="/opt/ml/processing/output/eval",
                destination=outputs["eval_output"]
            ),
            ProcessingOutput(
                output_name=output_names["metrics_output"],
                source="/opt/ml/processing/output/metrics",
                destination=outputs["metrics_output"]
            )
        ]

    def _get_job_arguments(self) -> List[str]:
        return ["--job_type", self.config.job_type]

    def _get_cache_config(self, enable_caching: bool = True):
        if not enable_caching:
            return None
        from sagemaker.workflow.steps import CacheConfig
        expire_after = "30d"
        return CacheConfig(
            enable_caching=enable_caching,
            expire_after=expire_after
        )

    def create_step(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> ProcessingStep:
        logger.info("Creating XGBoost Model Evaluation ProcessingStep...")

        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        step_name = f"{self._get_step_name('XGBoostModelEval')}-{self.config.job_type.capitalize()}"

        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            job_arguments=job_args,
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
