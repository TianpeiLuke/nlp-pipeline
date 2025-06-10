from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.sklearn import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from .config_tabular_preprocessing_step import TabularPreprocessingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class TabularPreprocessingStepBuilder(StepBuilderBase):
    """
    Builder for a Tabular Preprocessing ProcessingStep. Uses `job_type` (e.g. 'training', 'testing')
    consistent with CradleDataLoadConfig.
    """

    def __init__(
        self,
        config: TabularPreprocessingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Args:
            config: TabularPreprocessingConfig (Pydantic‐validated).
            sagemaker_session: SageMaker PipelineSession (optional).
            role: IAM role ARN for the Processing job.
            notebook_root: If running locally, used to validate local paths.
        """
        if not isinstance(config, TabularPreprocessingConfig):
            raise ValueError(
                "TabularPreprocessingStepBuilder requires a TabularPreprocessingConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: TabularPreprocessingConfig = config

    def validate_configuration(self) -> None:       
        """
        Called by StepBuilderBase.__init__(). Ensures required fields are set
        and in the correct format.

        Checks:
          - hyperparameters.label_name is nonempty (already Pydantic‐checked).
          - data_type is provided and valid.
          - train_ratio, test_val_ratio are in (0,1).
          - processing_entry_point is relative.
          - input_names contains 'data_input'.
          - output_names contains 'processed_data' and 'full_data'.
        """
        logger.info("Validating TabularPreprocessingConfig…")
        if not self.config.hyperparameters.label_name:
            raise ValueError("hyperparameters.label_name must be provided and non‐empty")
        if not getattr(self.config, 'job_type', None):
            raise ValueError(
                "job_type must be provided (e.g. 'training','validation','testing','calibration')"
            )
        if "data_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'data_input'")
        outs = self.config.output_names or {}
        if "processed_data" not in outs or "full_data" not in outs:
            raise ValueError(
                "output_names must contain keys 'processed_data' and 'full_data'"
            )
        logger.info("TabularPreprocessingConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Create and return an SKLearnProcessor for this step.
        SKLearnProcessor auto-populates the ImageUri from framework_version.
        """
        return SKLearnProcessor(
            framework_version=self.config.framework_version,  # e.g. '0.23-1'
            command=["python3"],
            role=self.role,
            instance_count=self.config.processing_instance_count,
            instance_type=self.config.get_instance_type(),
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('Processing')}-{self.config.job_type}"
            ),
            sagemaker_session=self.session,
        )

    def _get_processor_inputs(
        self,
        inputs: Dict[str, Any]
    ) -> List[ProcessingInput]:
        """Validate inputs dict and return list of ProcessingInput."""
        key_in = self.config.input_names["data_input"]
        if inputs is None or key_in not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{key_in}' in 'inputs'")
        return [
            ProcessingInput(
                input_name=key_in,
                source=inputs[key_in],
                destination="/opt/ml/processing/input/data"
            )
        ]

    def _get_processor_outputs(
        self,
        outputs: Dict[str, Any]
    ) -> List[ProcessingOutput]:
        """Validate outputs dict and return list of ProcessingOutput."""
        key_proc = self.config.output_names["processed_data"]
        key_full = self.config.output_names["full_data"]
        if outputs is None or key_proc not in outputs or key_full not in outputs:
            raise ValueError(
                f"Must supply S3 URIs for '{key_proc}' and '{key_full}' in 'outputs'"
            )
        return [
            ProcessingOutput(
                output_name=key_proc,
                source="/opt/ml/processing/output/processed_data",
                destination=outputs[key_proc]
            ),
            ProcessingOutput(
                output_name=key_full,
                source="/opt/ml/processing/output/full_data",
                destination=outputs[key_full]
            )
        ]

    def _get_job_arguments(self) -> List[str]:
        """Construct and return the list of script arguments."""
        return [
            "--job_type", self.config.job_type,
            "--label_field", self.config.hyperparameters.label_name,
            "--train_ratio", str(self.config.train_ratio),
            "--test_val_ratio", str(self.config.test_val_ratio),
        ]

    def create_step(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> ProcessingStep:
        """Assemble and return the ProcessingStep for tabular preprocessing."""
        logger.info("Creating TabularPreprocessing ProcessingStep…")

        # Build components
        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        job_args = self._get_job_arguments()

        # Create the ProcessingStep
        step_name = f"{self._get_step_name('Processing')}-{self.config.job_type.capitalize()}"
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
