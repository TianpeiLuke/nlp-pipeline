from typing import Optional, List, Union
from pathlib import Path
import os
import logging


from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import CacheConfig  # Add this import

from .config_mims_packaging_step import PackageStepConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class MIMSPackagingStepBuilder(StepBuilderBase):
    """Builder for MIMS packaging processing step"""

    def __init__(
        self, 
        config: PackageStepConfig, 
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

    def validate_configuration(self) -> None:
        """Validate required configuration settings for MIMS packaging."""
        logger.info(f"Running {self.__class__.__name__} specific configuration validation.")
        
        # Required attributes that must always be present
        required_attrs = [
            'source_dir', 
            'packaging_entry_point',
            'processing_instance_count', 
            'processing_volume_size',
            'pipeline_name', 
            'pipeline_s3_loc'
        ]
        
        # Check instance type based on configuration
        if self.config.use_large_processing_instance:
            required_attrs.append('processing_instance_type_large')
        else:
            required_attrs.append('processing_instance_type_small')

        # Validate required attributes
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"PackageStepConfig missing required attribute: {attr}")

        try:
            # Validate main packaging script
            self._get_resolved_script_entry_point_for_processor()
            
            # Validate inference_scripts_input_path if provided
            if self.config.inference_code_input_path:
                self._resolve_local_or_s3_path(
                    self.config.inference_code_input_path, 
                    is_dir_check=True
                )
                logger.info("Inference code input path validated.")
            else:
                logger.info("No inference code input path provided (optional).")
                
        except FileNotFoundError as e:
            raise ValueError(f"Configuration error for script/source paths: {e}")
            
        logger.info("MIMS Packaging configuration paths validated.")

    def _create_processor(self) -> SKLearnProcessor:
        """Create SKLearn processor for MIMS packaging."""
        if self.config.use_large_processing_instance:
            instance_type = self.config.processing_instance_type_large
        else:
            instance_type = self.config.processing_instance_type_small
        logger.info(f"Using processing instance type for packaging: {instance_type}")

        base_job_name_prefix = self._sanitize_name_for_sagemaker(self.config.pipeline_name, 30)

        return SKLearnProcessor(
            framework_version="1.2-1",
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            sagemaker_session=self.session,
            base_job_name=f"{base_job_name_prefix}-mims-pkg"
        )

    def _get_processing_inputs(
        self, model_artifacts_input_source: Union[str, Properties]
    ) -> List[ProcessingInput]:
        inputs = [
            ProcessingInput(
                source=model_artifacts_input_source,
                destination="/opt/ml/processing/input/model",
                input_name=self.config.model_input_name_in_job
            ),
            ProcessingInput(
                source=self.config.inference_code_source_dir, # Directly use the validated & resolved path
                destination="/opt/ml/processing/input/script",
                input_name=self.config.scripts_input_name_in_job
            )
        ]
        logger.info(f"Processing inputs: {[str(i.source) for i in inputs]}")
        return inputs

    def _get_processing_outputs(self, step_name_for_s3_path: str) -> List[ProcessingOutput]:
        """Defines the list of ProcessingOutput objects for the MIMS packaging step."""
        # The packaging script is expected to write its output (e.g., model.tar.gz for MIMS)
        # into /opt/ml/processing/output directory.
        output_s3_destination = os.path.join(
            str(self.config.pipeline_s3_loc).rstrip('/'), # Ensure pipeline_s3_loc is string
            step_name_for_s3_path,
            "packaged_model_artifacts"
        )

        outputs = [
            ProcessingOutput(
                output_name=self.config.packaged_model_output_name_from_job,
                source="/opt/ml/processing/output", # Capture everything in /opt/ml/processing/output
                destination=output_s3_destination
            )
        ]
        logger.info(f"Processing outputs defined: {outputs}")
        return outputs

    def _get_cache_config(self, enable_caching: bool = True) -> Optional[CacheConfig]:
        """
        Get cache configuration for the step.
        
        Args:
            enable_caching: Whether to enable caching
            
        Returns:
            CacheConfig object or None
        """
        if not enable_caching:
            return None
        
        expire_after = "30d"  # Format: "<number>d" for days

        return CacheConfig(
            enable_caching=enable_caching,
            expire_after=expire_after  # Cache expires after 30 days
        )

    def create_step(
        self,
        model_artifacts_input_source: Union[str, Properties],
        # common_artifacts_source removed as mims_package.py doesn't seem to use a third distinct input like that
        dependencies: Optional[List[Step]] = None
    ) -> ProcessingStep:
        """
        Creates a ProcessingStep for MIMS model packaging.
        Args:
            model_artifacts_input_source: S3 URI (str) or Properties object from a previous step
                                          pointing to the primary model artifacts.
            dependencies: Optional list of SageMaker Pipeline steps that this step depends on.
        Returns:
            A SageMaker ProcessingStep instance.
        """
        step_name = self._get_step_name('Package') # Uses 'PackagingStep' from STEP_NAMES

        # Construct the code path for SKLearnProcessor.
        # self.config.source_dir is the resolved absolute path to the directory.
        # self.config.packaging_entry_point is the script name within it.
        if str(self.config.source_dir).startswith('s3://'):
            # This scenario needs careful handling for SKLearnProcessor's 'code' argument.
            # Typically, for S3, 'code' is a tar.gz. If source_dir is an S3 prefix and
            # packaging_entry_point is a script within it, the processor needs to be
            # able to fetch and run it. This might require packaging source_dir into a tar.gz first
            # or ensuring the S3 path points to an executable script in a way the processor expects.
            # For now, we assume if source_dir is S3, packaging_entry_point forms a direct S3 URI to the script.
            main_packaging_script_uri = f"{self.config.source_dir.rstrip('/')}/{self.config.packaging_entry_point}"
            logger.warning(f"SKLearnProcessor 'code' argument is an S3 URI: {main_packaging_script_uri}. Ensure processor supports this for non-tar.gz S3 scripts or it's a tar.gz.")
        else: # Local path, should be absolute
            main_packaging_script_uri = str(Path(self.config.source_dir) / self.config.packaging_entry_point)
        
        logger.info(f"Defining {step_name} with main script: {main_packaging_script_uri}")

        processor = self._create_processor() # Uses instance types from config
        processing_inputs = self._get_processing_inputs(model_artifacts_input_source)
        processing_outputs = self._get_processing_outputs(step_name) # Pass step_name for S3 path
        job_arguments = self.config.packaging_script_arguments
        # Create cache config properly
        cache_config = self._get_cache_config(
            getattr(self.config, 'enable_caching_package_step', True)
        )

        return ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=processing_inputs,
            outputs=processing_outputs,
            code=main_packaging_script_uri, # Absolute local path to the entry script
            job_arguments=job_arguments,
            depends_on=dependencies or [],
            cache_config=cache_config  # Use the proper CacheConfig object
        )

    # Maintain backwards compatibilityity
    def create_packaging_step(
        self,
        model_data: str,
        dependencies: Optional[List] = None
    ) -> ProcessingStep:
        """Backwards compatible method for creating packaging step"""
        return self.create_step(model_data, dependencies)
