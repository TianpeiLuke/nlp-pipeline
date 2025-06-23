from typing import Dict, Optional, List, Union
from pathlib import Path
import logging

from sagemaker.workflow.steps import Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.lambda_step import LambdaStep
from sagemaker.lambda_helper import Lambda
import json

from .config_mims_payload_step import PayloadConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class MIMSPayloadStepBuilder(StepBuilderBase):
    """Builder for MIMS payload test step"""

    def __init__(
        self, 
        config: PayloadConfig, 
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        """
        Initialize MIMS payload test step builder
        
        Args:
            config: PayloadConfig instance with configuration parameters
            sagemaker_session: SageMaker session
            role: IAM role ARN
            notebook_root: Root directory of notebook
        """
        super().__init__(config, sagemaker_session, role, notebook_root)
        self.config: PayloadConfig = config

    def validate_configuration(self) -> None:
        """Validate required configuration settings for MIMS payload test."""
        logger.info(f"Running {self.__class__.__name__} specific configuration validation.")
        
        required_attrs = [
            'pipeline_name',
            'pipeline_version',
            'bucket',
            'source_model_inference_input_variable_list',
            'source_model_inference_content_types'
        ]
        
        # Validate required attributes
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PayloadConfig missing required attribute for builder: {attr}")

        # Validate content types
        valid_content_types = ["text/csv", "application/json"]
        for content_type in self.config.source_model_inference_content_types:
            if content_type not in valid_content_types:
                raise ValueError(f"Invalid content type: {content_type}. Must be one of {valid_content_types}")

        logger.info(f"{self.__class__.__name__} configuration attributes presence check passed.")

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        input_reqs = {
            "dependencies": self.COMMON_PROPERTIES["dependencies"]
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        output_props = {
            "payload_s3_uri": "S3 URI of the generated payload archive",
            "payload_s3_key": "S3 key of the generated payload archive"
        }
        # Add any output names from config if they exist
        if hasattr(self.config, "output_names"):
            output_props.update({k: v for k, v in self.config.output_names.items()})
        return output_props
    

    def create_step(
        self,
        dependencies: Optional[List[Step]] = None
    ) -> LambdaStep:
        """
        Creates a LambdaStep that calls generate_and_upload_payloads from the config.
        
        Args:
            dependencies: Optional list of step dependencies
            
        Returns:
            LambdaStep object
        """
        step_name = self._get_step_name('Payload')
        logger.info(f"Creating {step_name} step")

        # Ensure S3 key is constructed
        if not self.config.sample_payload_s3_key:
            self.config.ensure_payload_path()
            logger.info(f"Constructed S3 key: {self.config.sample_payload_s3_key}")

        # Check if we have a script path or if we should use the embedded methods
        if hasattr(self.config, 'payload_script_path') and self.config.payload_script_path:
            # If a script path is provided, we would use it (but this is not implemented here)
            logger.info(f"Script path provided: {self.config.payload_script_path}")
            logger.warning("Using custom script path is not implemented, falling back to embedded methods")
        
        # Generate and upload payloads using the embedded methods in the config
        s3_uri = self.config.generate_and_upload_payloads()
        logger.info(f"Generated and uploaded payloads to: {s3_uri}")

        # Create a simple pass-through Lambda step that just returns the S3 URI and key
        # This allows us to integrate with the pipeline while actually executing the
        # payload generation directly in the builder
        
        def pass_through_function(event):
            """Simple pass-through function that returns the S3 URI and key"""
            return {
                "payload_s3_uri": event["payload_s3_uri"],
                "payload_s3_key": event["payload_s3_key"]
            }
        
        lambda_function = Lambda(
            function_name=f"{self.config.pipeline_name}-payload-reference",
            execution_role_arn=self.role,
            script=pass_through_function,
            handler="pass_through_function",
            session=self.session
        )
        
        step = LambdaStep(
            name=step_name,
            lambda_func=lambda_function,
            inputs={
                "payload_s3_uri": s3_uri,
                "payload_s3_key": self.config.sample_payload_s3_key
            },
            depends_on=dependencies or []
        )
        
        # Add properties for downstream steps
        step.properties.payload_s3_uri = f"s3://{self.config.bucket}/{self.config.sample_payload_s3_key}"
        step.properties.payload_s3_key = self.config.sample_payload_s3_key
        
        return step

    def create_payload_step(
        self,
        dependencies: Optional[List[Step]] = None
    ) -> LambdaStep:
        """
        Convenience method for creating payload step.
        
        Args:
            dependencies: Optional list of step dependencies
            
        Returns:
            LambdaStep object
        """
        return self.create_step(dependencies)
