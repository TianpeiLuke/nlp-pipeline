from typing import Dict, Optional, Any, Set
from pathlib import Path
import logging
import boto3
import json

from sagemaker.workflow.steps import Step
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import LambdaFunction

from .config_mims_payload_step import PayloadConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class MIMSPayloadStepBuilder(StepBuilderBase):
    """
    Builder for a MIMS Payload LambdaStep.
    This class is responsible for configuring and creating a SageMaker Lambda Step
    that generates and uploads payload samples for MIMS model registration.
    """

    def __init__(
        self,
        config: PayloadConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the MIMS payload step.

        Args:
            config: A PayloadConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the Lambda.
            notebook_root: The root directory of the notebook environment (not used for Lambda steps).
        """
        if not isinstance(config, PayloadConfig):
            raise ValueError(
                "MIMSPayloadStepBuilder requires a PayloadConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: PayloadConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating PayloadConfig...")
        
        # Validate required attributes
        required_attrs = [
            'expected_tps', 
            'max_latency_in_millisecond',
            'max_acceptable_error_rate',
            'model_registration_domain',
            'model_registration_objective',
            'source_model_inference_content_types',
            'source_model_inference_response_types',
            'source_model_inference_input_variable_list',
            'source_model_inference_output_variable_list'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PayloadConfig missing required attribute: {attr}")
        
        # Validate bucket and ensure S3 path for payload exists
        if not hasattr(self.config, 'bucket') or not self.config.bucket:
            raise ValueError("PayloadConfig must have a 'bucket' attribute")
        
        # Make sure sample_payload_s3_key is set or can be constructed
        if not hasattr(self.config, 'sample_payload_s3_key') or not self.config.sample_payload_s3_key:
            try:
                self.config.ensure_payload_path()
            except Exception as e:
                raise ValueError(f"Could not construct payload path: {e}")
        
        logger.info("PayloadConfig validation succeeded.")

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # This step doesn't require any inputs from previous steps
        input_reqs = {
            "dependencies": self.COMMON_PROPERTIES["dependencies"],
            "enable_caching": self.COMMON_PROPERTIES["enable_caching"]
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        return {
            "payload_s3_uri": "S3 URI of the generated payload",
            "payload_s3_key": "S3 key of the generated payload"
        }
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to MIMSPayload step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        # No custom properties to match for this step
        return set()
    
    def _generate_payload(self):
        """
        Generate payload file and upload to S3 directly.
        
        This is a lightweight alternative that generates the payload directly 
        rather than through a Lambda function, which simplifies the implementation.
        
        Returns:
            tuple: (s3_uri, s3_key) of uploaded payload
        """
        # Generate payload and upload to S3
        s3_uri = self.config.generate_and_upload_payloads()
        
        # Return the S3 URI and key
        s3_key = self.config.sample_payload_s3_key
        
        return s3_uri, s3_key

    def _create_parameter_lambda_step(self, s3_uri, s3_key):
        """
        Create a simple Lambda step that just returns static values.
        This is much simpler than creating a full Lambda function for payload generation.
        
        Args:
            s3_uri: The S3 URI of the generated payload
            s3_key: The S3 key of the generated payload
            
        Returns:
            LambdaStep: A LambdaStep that returns the provided values
        """
        from sagemaker.workflow.parameters import ParameterString
        from sagemaker.workflow.functions import Join
        
        # Create parameter strings for output values
        payload_s3_uri = ParameterString(
            name=f"payload_s3_uri_{self._sanitize_name_for_sagemaker(self.config.pipeline_name)}",
            default_value=s3_uri
        )
        
        payload_s3_key = ParameterString(
            name=f"payload_s3_key_{self._sanitize_name_for_sagemaker(self.config.pipeline_name)}",
            default_value=s3_key
        )
        
        # Return parameter values directly
        return payload_s3_uri, payload_s3_key
    
    def create_step(self, **kwargs) -> LambdaStep:
        """
        Creates a lightweight SageMaker LambdaStep for payload generation.
        This implementation generates the payload directly and creates a simple
        step that just returns the file locations.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - dependencies: Optional list of steps that this step depends on.
                
        Returns:
            A configured sagemaker.workflow.steps.LambdaStep instance.
        """
        logger.info("Creating MIMSPayload LambdaStep...")
        
        # Extract parameters
        dependencies = self._extract_param(kwargs, 'dependencies')
        
        # Generate payload directly
        s3_uri, s3_key = self._generate_payload()
        
        # Create parameters for the outputs
        s3_uri_param, s3_key_param = self._create_parameter_lambda_step(s3_uri, s3_key)
        
        # Create a simple function that returns the values
        def return_payload_info(event):
            return {
                "payload_s3_uri": s3_uri,
                "payload_s3_key": s3_key
            }
        
        # Create the LambdaStep with the simple function
        step_name = self._get_step_name('MIMSPayload')
        
        payload_step = LambdaStep(
            name=step_name,
            lambda_func=return_payload_info,  # Simple function that returns static values
            inputs={},  # No inputs needed
            outputs=[
                LambdaOutput(output_name="payload_s3_uri", output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name="payload_s3_key", output_type=LambdaOutputTypeEnum.String)
            ],
            depends_on=dependencies or []
        )
        
        logger.info(f"Created LambdaStep with name: {payload_step.name}")
        return payload_step
