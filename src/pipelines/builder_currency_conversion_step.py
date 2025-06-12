from typing import Optional, List, Union, Dict
from pathlib import Path
import logging
import json

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import Properties

from .config_currency_conversion_step import CurrencyConversionConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class CurrencyConversionStepBuilder(StepBuilderBase):
    """Builder for currency conversion processing step"""

    def __init__(
        self,
        config: CurrencyConversionConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize currency conversion step builder.
        
        Args:
            config: Currency conversion configuration
            sagemaker_session: SageMaker session
            role: IAM role
            notebook_root: Root directory of notebook
        """
        super().__init__(config, sagemaker_session, role, notebook_root)
        self.config: CurrencyConversionConfig = config  # Type hint for IDE support

    def validate_configuration(self) -> None:
        """Validate required configuration settings"""
        logger.info("Validating currency conversion configuration...")
        
        # Validate processing-specific configuration
        if not self.config.processing_entry_point:
            raise ValueError("processing_entry_point must be provided")
        
        script_path = self.config.get_script_path()
        if script_path and not script_path.startswith('s3://'):
            if not Path(script_path).exists():
                raise ValueError(f"Processing script not found: {script_path}")

        # Validate currency conversion configuration
        if self.config.enable_currency_conversion:
            if not self.config.marketplace_info:
                raise ValueError("marketplace_info must be provided when currency conversion is enabled")
            
            # Check currency consistency
            marketplace_currencies = {
                info.get("currency_code") 
                for info in self.config.marketplace_info.values() 
                if "currency_code" in info
            }
            conversion_currencies = set(self.config.currency_conversion_dict.keys())
            
            missing_currencies = marketplace_currencies - conversion_currencies
            if missing_currencies and not self.config.skip_invalid_currencies:
                raise ValueError(
                    f"Missing conversion rates for currencies: {missing_currencies}. "
                    "Either add rates or set skip_invalid_currencies=True"
                )

        logger.info("Currency conversion configuration validation completed.")

    def _create_processor(self) -> SKLearnProcessor:
        """Create SKLearn processor for currency conversion"""
        instance_type = self.config.get_instance_type(
            'large' if self.config.use_large_processing_instance else 'small'
        )
        
        logger.info(f"Creating processor with instance type: {instance_type}")
        
        base_job_name = self._sanitize_name_for_sagemaker(
            f"{self.config.pipeline_name}-currency-conversion",
            max_length=30
        )

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
        data_input: Union[str, Properties]
    ) -> List[ProcessingInput]:
        """
        Get processing inputs for currency conversion.
        
        Args:
            data_input: Input data source
            
        Returns:
            List of ProcessingInput objects
        """
        inputs = [
            ProcessingInput(
                source=data_input,
                destination="/opt/ml/processing/input/data",
                input_name=self.config.input_names["data_input"]
            )
        ]
        
        logger.info(f"Processing inputs configured: {[str(i.source) for i in inputs]}")
        return inputs

    def _get_processing_outputs(self) -> List[ProcessingOutput]:
        """
        Get processing outputs for currency conversion.
        
        Returns:
            List of ProcessingOutput objects
        """
        output_path = f"{self.config.pipeline_s3_loc}/currency_conversion"
        
        outputs = [
            ProcessingOutput(
                output_name=self.config.output_names["converted_data"],
                source="/opt/ml/processing/output",
                destination=output_path
            )
        ]
        
        logger.info(f"Processing outputs configured with destination: {output_path}")
        return outputs

    def create_step(
        self,
        data_input: Union[str, Properties],
        dependencies: Optional[List[Step]] = None
    ) -> ProcessingStep:
        """
        Create currency conversion processing step.
        
        Args:
            data_input: Input data source
            dependencies: Optional list of dependent steps
            
        Returns:
            ProcessingStep for currency conversion
        """
        logger.info("Creating currency conversion processing step")

        # Create processor
        processor = self._create_processor()

        # Get inputs and outputs
        processing_inputs = self._get_processing_inputs(data_input)
        processing_outputs = self._get_processing_outputs()

        # Get script arguments and environment variables
        script_args = self.config.get_script_arguments()
        environment = self.config.get_environment_variables()

        # Create step
        step = ProcessingStep(
            name=self._get_step_name('Currency_Conversion'),
            processor=processor,
            inputs=processing_inputs,
            outputs=processing_outputs,
            code=self.config.get_script_path(),
            job_arguments=script_args,
            environment=environment,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(
                getattr(self.config, 'enable_caching', True)
            )
        )

        logger.info(f"Created currency conversion step with {len(processing_inputs)} inputs "
                   f"and {len(processing_outputs)} outputs")
        return step

    def create_conversion_step(
        self,
        data_input: str,
        dependencies: Optional[List] = None
    ) -> ProcessingStep:
        """
        Backwards compatible method for creating conversion step.
        
        Args:
            data_input: Input data path
            dependencies: Optional list of dependencies
            
        Returns:
            ProcessingStep
        """
        return self.create_step(data_input, dependencies)
