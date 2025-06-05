from typing import Optional, Dict, Any
from pathlib import Path
import logging

from sagemaker.sklearn import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession

from .config_tabular_preprocessing_step import TabularPreprocessingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class TabularPreprocessingStepBuilder(StepBuilderBase):
    """
    Builder for a tabular preprocessing ProcessingStep. 
    Takes a TabularPreprocessingConfig, validates it, and returns a ProcessingStep
    that runs preprocess.py on SageMaker.
    """

    def __init__(
        self,
        config: TabularPreprocessingConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        if not isinstance(config, TabularPreprocessingConfig):
            raise ValueError("TabularPreprocessingStepBuilder requires a TabularPreprocessingConfig instance.")
        super().__init__(config=config, sagemaker_session=sagemaker_session, role=role, notebook_root=notebook_root)
        self.config: TabularPreprocessingConfig = config

    def validate_configuration(self) -> None:
        """
        Ensures required fields are set and valid. Most validation is done by Pydantic.
        Additional checks:
          - processing_source_dir must exist (handled by base class)
          - processing_entry_point must be provided
          - hyperparameters.tab_field_list and hyperparameters.cat_field_list are non-empty
          - label_name is not in tab_field_list or cat_field_list (handled by config validator)
        """
        logger.info("Validating TabularPreprocessingConfig…")

        # 1) Ensure entry point is set
        if not self.config.processing_entry_point:
            raise ValueError("processing_entry_point must be provided for tabular preprocessing")

        # 2) Hyperparameters must include at least one tab or categorical field
        hp = self.config.hyperparameters
        if not hp.tab_field_list and not hp.cat_field_list:
            raise ValueError("At least one of tab_field_list or cat_field_list must be non‐empty in hyperparameters")

        # 3) Label name must not overlap (already enforced by Pydantic model_validator)
        # 4) Input / Output names keys
        inp = self.config.input_names
        out = self.config.output_names
        if "data_input" not in inp or "config_input" not in inp:
            raise ValueError("input_names must contain 'data_input' and 'config_input'")
        if "processed_data" not in out or "full_data" not in out:
            raise ValueError("output_names must contain 'processed_data' and 'full_data'")

        logger.info("TabularPreprocessingConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Create an SKLearnProcessor to run the preprocessing script.
        """
        instance_type = self.config.get_instance_type(
            'large' if self.config.use_large_processing_instance else 'small'
        )
        base_job_name = self._sanitize_name_for_sagemaker(
            f"{self.config.pipeline_name}-tabular-preprocessing",
            max_length=30
        )
        logger.info(f"Creating SKLearnProcessor with instance_type={instance_type}, count={self.config.processing_instance_count}")
        return SKLearnProcessor(
            framework_version="1.2-1",
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            sagemaker_session=self.session,
            base_job_name=base_job_name
        )

    def _get_processing_inputs(
        self,
        data_input: str,
        config_input: str
    ) -> list[ProcessingInput]:
        """
        Returns ProcessingInput definitions for:
          - raw data channel
          - configuration/artifacts channel
        """
        inputs = [
            ProcessingInput(
                source=data_input,
                destination="/opt/ml/processing/input/data",
                input_name=self.config.input_names["data_input"]
            ),
            ProcessingInput(
                source=config_input,
                destination="/opt/ml/processing/input/config",
                input_name=self.config.input_names["config_input"]
            ),
        ]
        logger.info(f"Configured processing inputs: {[inp.source for inp in inputs]}")
        return inputs

    def _get_processing_outputs(self) -> list[ProcessingOutput]:
        """
        Returns ProcessingOutput definitions for:
          - processed_data (only features + label)
          - full_data (all columns)
        """
        base_dest = f"{self.config.pipeline_s3_loc}/tabular_preprocessing"
        processed_dest = f"{base_dest}/{self.config.hyperparameters.label_name}/processed"
        full_dest = f"{base_dest}/{self.config.hyperparameters.label_name}/full"

        outputs = [
            ProcessingOutput(
                output_name=self.config.output_names["processed_data"],
                source="/opt/ml/processing/output/processed",
                destination=processed_dest
            ),
            ProcessingOutput(
                output_name=self.config.output_names["full_data"],
                source="/opt/ml/processing/output/full",
                destination=full_dest
            ),
        ]
        logger.info(f"Configured processing outputs: {[out.destination for out in outputs]}")
        return outputs

    def _get_environment(self) -> Dict[str, str]:
        """
        Build environment variables for preprocess.py from hyperparameters:
          - TAB_FIELDS: comma‐separated tabular feature names
          - CAT_FIELDS: comma‐separated categorical feature names
          - LABEL_FIELD: label name
          - N_WORKERS: number of parallel workers
        """
        hp = self.config.hyperparameters
        tab_fields = ",".join(hp.tab_field_list)
        cat_fields = ",".join(hp.cat_field_list)
        env = {
            "TAB_FIELDS": tab_fields,
            "CAT_FIELDS": cat_fields,
            "LABEL_FIELD": hp.label_name,
            "N_WORKERS": str(self.config.n_workers)
        }
        return env

    def create_step(
        self,
        data_input: str,
        config_input: str,
        dependencies: Optional[list[Step]] = None
    ) -> ProcessingStep:
        """
        Build the ProcessingStep for tabular preprocessing.

        Args:
          data_input: S3 URI or Properties object for raw data channel
          config_input: S3 URI or Properties object for config/artifacts channel
          dependencies: Optional list of prior Steps this depends on
        """
        logger.info("Creating TabularPreprocessing ProcessingStep…")

        # (a) Create processor
        processor = self._create_processor()

        # (b) Define inputs and outputs
        processing_inputs = self._get_processing_inputs(data_input, config_input)
        processing_outputs = self._get_processing_outputs()

        # (c) Build environment variables
        environment = self._get_environment()

        # (d) Build script arguments (if any) – none in this case
        script_args: list[str] = []

        # (e) Create the ProcessingStep
        step_name = f"{self._get_step_name('Processing')}-{self.config.hyperparameters.label_name}"
        step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=processing_inputs,
            outputs=processing_outputs,
            code=self.config.get_script_path(),
            job_arguments=script_args,
            environment=environment,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(True)
        )

        logger.info(f"Created ProcessingStep with name: {step_name}")
        return step

    def get_environment(self) -> Dict[str, str]:
        """
        Expose environment dict for external use (e.g., for testing).
        """
        return self._get_environment()
