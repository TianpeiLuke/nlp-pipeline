from typing import Dict, Optional, Any, Set
from pathlib import Path
import logging
import boto3
import json
import tempfile
import os

from sagemaker.workflow.steps import Step
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda

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
    

    def create_step(self, **kwargs) -> LambdaStep:
        """
        Creates a SageMaker LambdaStep for the pipeline.
        
        Instead of creating a real Lambda function, this generates payloads directly 
        during pipeline construction and creates a lightweight dummy Lambda step
        that simply returns the pre-generated values.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - dependencies: Optional list of steps that this step depends on.
                
        Returns:
            A configured LambdaStep instance with necessary outputs.
        """
        # Extract common parameters
        dependencies = self._extract_param(kwargs, 'dependencies', None)
        logger.info("Creating MIMSPayload Step...")

        step_name = self._get_step_name('Payload')
        
        # Generate the payload directly during pipeline construction
        s3_uri = self.config.generate_and_upload_payloads()
        s3_key = self.config.sample_payload_s3_key
        
        logger.info(f"Generated payload at: {s3_uri}")
                
        # Create a simple dummy function that returns pre-generated values
        # This avoids complex Lambda creation and deployment
        def dummy_function(event, context=None):
            return {
                'payload_s3_uri': s3_uri,
                'payload_s3_key': s3_key
            }
        
        # Create Lambda step with the dummy function
        lambda_step = LambdaStep(
            name=step_name,
            lambda_func=dummy_function,
            outputs=[
                LambdaOutput(output_name="payload_s3_uri", output_type=LambdaOutputTypeEnum.String),
                LambdaOutput(output_name="payload_s3_key", output_type=LambdaOutputTypeEnum.String)
            ],
            depends_on=dependencies
        )
        
        # Monkey patch to avoid Lambda client access entirely
        lambda_step._get_function_arn = lambda: "arn:aws:lambda:us-east-1:123456789012:function:dummy-function"
        
        logger.info(f"Created dummy LambdaStep with name: {lambda_step.name}")
        return lambda_step
