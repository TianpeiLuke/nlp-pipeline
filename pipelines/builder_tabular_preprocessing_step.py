# File: pipelines/builder_tabular_preprocessing.py

from typing import Dict, Optional, Any
from pathlib import Path
import logging

from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from .config_tabular_preprocessing_step import TabularPreprocessingConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class TabularPreprocessingStepBuilder(StepBuilderBase):
    """
    Builder for a Tabular Preprocessing ProcessingStep. Takes a TabularPreprocessingConfig,
    validates it, constructs a ProcessingStep which runs tabular_preprocess.py, and returns it.
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
            raise ValueError("TabularPreprocessingStepBuilder requires a TabularPreprocessingConfig instance.")
        super().__init__(config=config, sagemaker_session=sagemaker_session, role=role, notebook_root=notebook_root)
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

        # (1) label_name (already validated in Pydantic)
        if not self.config.hyperparameters.label_name:
            raise ValueError("hyperparameters.label_name must be provided and non‐empty")

        # (2) data_type must exist
        if not self.config.data_type:
            raise ValueError("data_type must be provided (e.g. 'training','validation','testing','calibration')")

        # (3) train_ratio/test_val_ratio were validated by Pydantic

        # (4) Check input_names / output_names
        if "data_input" not in self.config.input_names:
            raise ValueError("input_names must contain key 'data_input'")
        if "processed_data" not in self.config.output_names or "full_data" not in self.config.output_names:
            raise ValueError("output_names must contain keys 'processed_data' and 'full_data'")

        logger.info("TabularPreprocessingConfig validation succeeded.")

    def create_step(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> ProcessingStep:
        """
        Build and return a SageMaker ProcessingStep that runs tabular_preprocess.py.

        Args:
          inputs:  Dict mapping "RawData" → S3 URI where shards live.
          outputs: Dict mapping "ProcessedTabularData" and "FullTabularData" → S3 URIs.
          enable_caching: whether to enable caching.

        Returns:
          A fully‐configured ProcessingStep.
        """
        logger.info("Creating TabularPreprocessing ProcessingStep…")

        # (A) Validate that inputs/outputs dicts contain the required channel names:
        if inputs is None or self.config.input_names["data_input"] not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{self.config.input_names['data_input']}' in 'inputs'")
        if outputs is None \
           or self.config.output_names["processed_data"] not in outputs \
           or self.config.output_names["full_data"] not in outputs:
            raise ValueError(
                f"Must supply S3 URIs for '{self.config.output_names['processed_data']}' "
                f"and '{self.config.output_names['full_data']}' in 'outputs'"
            )

        s3_input_uri    = inputs[self.config.input_names["data_input"]]
        s3_out_processed= outputs[self.config.output_names["processed_data"]]
        s3_out_full     = outputs[self.config.output_names["full_data"]]

        # (B) Create a ScriptProcessor that will run `tabular_preprocess.py`
        script_processor = ScriptProcessor(
            role=self.role,
            instance_count=self.config.processing_instance_count,
            instance_type=self.config.get_instance_type(),
            sagemaker_session=self.session,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('Processing')}-{self.config.data_type}"
            )
        )

        # (C) Define ProcessingInput / ProcessingOutput
        processor_inputs = [
            ProcessingInput(
                input_name=self.config.input_names["data_input"],
                source=s3_input_uri,
                destination="/opt/ml/processing/input/data"
            )
        ]

        processor_outputs = [
            ProcessingOutput(
                output_name=self.config.output_names["processed_data"],
                source="/opt/ml/processing/output",  # script writes under /opt/ml/processing/output/<split>/
                destination=s3_out_processed
            ),
            ProcessingOutput(
                output_name=self.config.output_names["full_data"],
                source="/opt/ml/processing/output",
                destination=s3_out_full
            )
        ]

        # (D) Build the ProcessingStep
        step_name = f"{self._get_step_name('Processing')}-{self.config.data_type.capitalize()}"

        processing_step = ProcessingStep(
            name=step_name,
            processor=script_processor,
            inputs=processor_inputs,
            outputs=processor_outputs,
            code=self.config.get_script_path(),
            job_arguments=["--data_type", self.config.data_type],
            environment={
                "LABEL_FIELD":     self.config.hyperparameters.label_name,
                "TRAIN_RATIO":     str(self.config.train_ratio),
                "TEST_VAL_RATIO":  str(self.config.test_val_ratio)
            },
            cache_config=self._get_cache_config(enable_caching)
        )

        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step

