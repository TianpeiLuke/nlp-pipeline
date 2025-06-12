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

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Get environment variables for the preprocessing script.

        Returns:
            Dict[str, str]: Dictionary of environment variables needed by the script.
                Keys:
                    - LABEL_FIELD: Name of the label column
                    - TRAIN_RATIO: Fraction of data for training
                    - TEST_VAL_RATIO: Fraction of holdout for test vs validation

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        if not self.config.hyperparameters.label_name:
            raise ValueError("Label field name must be set in hyperparameters")

        if not (0 < self.config.train_ratio < 1):
            raise ValueError(f"train_ratio must be between 0 and 1, got {self.config.train_ratio}")

        if not (0 < self.config.test_val_ratio < 1):
            raise ValueError(f"test_val_ratio must be between 0 and 1, got {self.config.test_val_ratio}")

        env_vars = {
            "LABEL_FIELD": str(self.config.hyperparameters.label_name),
            "TRAIN_RATIO": str(self.config.train_ratio),
            "TEST_VAL_RATIO": str(self.config.test_val_ratio)
        }

        logger.info("Processing environment variables:")
        for key, value in env_vars.items():
            logger.info(f"- {key}: {value}")

        return env_vars

    def _create_processor(self) -> SKLearnProcessor:
        """
        Create and return an SKLearnProcessor for this step.
        Includes environment variables in the processor configuration.
        """
        # Get environment variables
        environment = self._get_environment_variables()

        return SKLearnProcessor(
            framework_version=self.config.framework_version,
            command=["python3"],
            role=self.role,
            instance_count=self.config.processing_instance_count,
            instance_type=self.config.get_instance_type(),
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('Processing')}-{self.config.job_type}"
            ),
            sagemaker_session=self.session,
            env=environment,  # Set environment variables here
        )

    def _get_processor_inputs(
        self,
        inputs: Dict[str, Any]
    ) -> List[ProcessingInput]:
        """
        Validate inputs dict and return list of ProcessingInput.
        Handles both direct S3 paths and CradleDataLoadingStep outputs.
        """
        key_in = self.config.input_names["data_input"]
        if inputs is None or key_in not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{key_in}' in 'inputs'")

        processing_inputs = []
        
        # Main data input
        processing_inputs.append(
            ProcessingInput(
                input_name=key_in,
                source=inputs[key_in],
                destination="/opt/ml/processing/input/data"
            )
        )

        # Optional metadata input
        if "metadata_input" in self.config.input_names:
            metadata_key = self.config.input_names["metadata_input"]
            if metadata_key in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name=metadata_key,
                        source=inputs[metadata_key],
                        destination="/opt/ml/processing/input/metadata"
                    )
                )

        # Optional signature input
        if "signature_input" in self.config.input_names:
            signature_key = self.config.input_names["signature_input"]
            if signature_key in inputs:
                processing_inputs.append(
                    ProcessingInput(
                        input_name=signature_key,
                        source=inputs[signature_key],
                        destination="/opt/ml/processing/input/signature"
                    )
                )

        return processing_inputs

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
        """
        Get script arguments for the preprocessing job.
        
        Only job_type is passed as an argument since other parameters
        are passed through environment variables.

        Returns:
            List[str]: List of command-line arguments for the preprocessing script

        Example:
            For training job: ["--job_type", "training"]
            For testing job: ["--job_type", "testing"]
        """
        if not self.config.job_type:
            raise ValueError("job_type must be set in config")

        if self.config.job_type not in {'training', 'validation', 'testing', 'calibration'}:
            raise ValueError(
                f"job_type must be one of ['training', 'validation', 'testing', 'calibration'], "
                f"got: {self.config.job_type}"
            )

        args = ["--job_type", self.config.job_type]
        
        logger.info(f"Job arguments: {args}")
        return args

    def create_step(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True
    ) -> ProcessingStep:
        """
        Create the preprocessing ProcessingStep.

        Args:
            inputs: Dictionary mapping input names to S3 URIs
            outputs: Dictionary mapping output names to S3 URIs
            enable_caching: Whether to enable step caching

        Returns:
            ProcessingStep: The configured preprocessing step
        """
        logger.info("Creating TabularPreprocessing ProcessingStep...")

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
        logger.info(f"- Job arguments: {job_args}")
        logger.info(f"- Environment variables: {processor.env}")
        return processing_step
