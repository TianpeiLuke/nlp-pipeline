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
        input_name = "data_input"
        if self.config.input_names:
            input_name = list(self.config.input_names.keys())[0] if self.config.input_names else "data_input"
            
        inputs = [
            ProcessingInput(
                source=data_input,
                destination="/opt/ml/processing/input/data",
                input_name=input_name
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
        
        output_name = "converted_data"
        if self.config.output_names:
            output_name = list(self.config.output_names.keys())[0] if self.config.output_names else "converted_data"
            
        outputs = [
            ProcessingOutput(
                output_name=output_name,
                source="/opt/ml/processing/output",
                destination=output_path
            )
        ]
        
        logger.info(f"Processing outputs configured with destination: {output_path}")
        return outputs

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        return {
            "data_input": "Input data source (S3 path or Properties object)",
            "dependencies": self.COMMON_PROPERTIES["dependencies"]
        }
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        return {
            "processed_data_output": "S3 URI of the processed data output"
        }
    
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps.
        
        This method extracts the inputs required by the CurrencyConversionStep from the dependency steps.
        Specifically, it looks for:
        1. data_input from a DataLoadStep or other processing steps
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            Dictionary of inputs extracted from dependency steps
        """
        inputs = {}
        
        # Look for data_input from a DataLoadStep or other processing steps
        for prev_step in dependency_steps:
            if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                try:
                    # Try to get the data output
                    if hasattr(prev_step.properties.ProcessingOutputConfig.Outputs, "__getitem__"):
                        # Try string keys (dict-like)
                        for key in prev_step.properties.ProcessingOutputConfig.Outputs:
                            output = prev_step.properties.ProcessingOutputConfig.Outputs[key]
                            if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                # If this is a data load step, use it as data_input
                                if key == "DATA" or key == "RawData" or key == "ProcessedTabularData":
                                    inputs["data_input"] = output.S3Output.S3Uri
                                    logger.info(f"Found data_input from step: {prev_step.name}")
                                    break
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Could not extract data output from step: {e}")
        
        # Add enable_caching
        inputs["enable_caching"] = getattr(self.config, 'enable_caching', True)
        
        return inputs
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Create currency conversion processing step.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - data_input: Input data source (S3 path or Properties object)
                - dependencies: Optional list of dependent steps
                - enable_caching: Whether to enable caching for this step (default: True)
            
        Returns:
            ProcessingStep for currency conversion
        """
        # Extract parameters
        data_input = self._extract_param(kwargs, 'data_input')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', 
                                            getattr(self.config, 'enable_caching', True))
        
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

    def create_conversion_step(self, **kwargs) -> ProcessingStep:
        """
        Backwards compatible method for creating conversion step.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - data_input: Input data source (S3 path or Properties object)
                - dependencies: Optional list of dependent steps
                - enable_caching: Whether to enable caching for this step (default: True)
            
        Returns:
            ProcessingStep
        """
        logger.warning("create_conversion_step is deprecated, use create_step instead.")
        return self.create_step(**kwargs)
