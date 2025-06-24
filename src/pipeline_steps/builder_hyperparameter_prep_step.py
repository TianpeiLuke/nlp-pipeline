from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import json
import tempfile
import boto3
import os

from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda

from .config_hyperparameter_prep_step import HyperparameterPrepConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class HyperparameterPrepStepBuilder(StepBuilderBase):
    """
    Builder for a Hyperparameter Preparation LambdaStep.
    This class is responsible for configuring and creating a SageMaker LambdaStep
    that serializes hyperparameters to JSON and uploads them to S3.
    """

    def __init__(
        self,
        config: HyperparameterPrepConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the hyperparameter preparation step.

        Args:
            config: A HyperparameterPrepConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the Lambda function.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, HyperparameterPrepConfig):
            raise ValueError(
                "HyperparameterPrepStepBuilder requires a HyperparameterPrepConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: HyperparameterPrepConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating HyperparameterPrepConfig...")
        if not self.config.hyperparameters:
            raise ValueError("hyperparameters must be provided and non-empty")
        if not self.config.hyperparameters_s3_uri:
            raise ValueError("hyperparameters_s3_uri must be provided and non-empty")
        if "hyperparameters_output" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'hyperparameters_output'")
        logger.info("HyperparameterPrepConfig validation succeeded.")
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # This step doesn't require any inputs from previous steps
        input_reqs = {
            "enable_caching": self.COMMON_PROPERTIES["enable_caching"]
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Get output properties from config's output_names
        output_props = {k: v for k, v in (self.config.output_names or {}).items()}
        # Add the hyperparameters_s3_uri as an output property
        output_props["hyperparameters_s3_uri"] = "S3 URI of the hyperparameters.json file"
        return output_props

    def create_step(
        self,
        enable_caching: bool = True,
        dependencies: Optional[List[Any]] = None
    ) -> LambdaStep:
        """
        Creates a SageMaker LambdaStep for the pipeline.
        This method creates a Lambda function that serializes hyperparameters to JSON
        and uploads them to S3.

        Args:
            enable_caching: A boolean indicating whether to cache the results of this step
                            to speed up subsequent pipeline runs with the same inputs.
            dependencies: A list of steps that this step depends on.

        Returns:
            A configured sagemaker.workflow.lambda_step.LambdaStep instance.
        """
        logger.info("Creating HyperparameterPrep LambdaStep...")

        step_name = self._get_step_name('HyperparameterPrep')
        
        # Define the Lambda function code
        lambda_function_code = """
import json
import boto3
import os
from urllib.parse import urlparse

def lambda_handler(event, context):
    # Extract hyperparameters from the event
    hyperparameters = event['hyperparameters']
    s3_uri = event['hyperparameters_s3_uri']
    
    # Parse S3 URI
    parsed_uri = urlparse(s3_uri)
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    
    # Ensure the key ends with 'hyperparameters.json'
    if not key.endswith('/'):
        key += '/'
    key += 'hyperparameters.json'
    
    # Upload hyperparameters to S3
    s3_client = boto3.client('s3')
    s3_client.put_object(
        Body=json.dumps(hyperparameters, indent=2),
        Bucket=bucket,
        Key=key,
        ContentType='application/json'
    )
    
    # Return the full S3 URI to the hyperparameters file
    hyperparameters_s3_uri = f"s3://{bucket}/{key}"
    print(f"Uploaded hyperparameters to {hyperparameters_s3_uri}")
    
    return {
        'hyperparameters_s3_uri': hyperparameters_s3_uri
    }
"""

        # Create the Lambda function
        lambda_function = Lambda(
            function_name=f"HyperparameterPrep-{self.config.pipeline_name}",
            execution_role_arn=self.role,
            script=lambda_function_code,
            handler="index.lambda_handler",
            timeout=self.config.lambda_timeout,
            memory_size=self.config.lambda_memory_size,
            session=self.session.boto_session
        )
        
        # Serialize hyperparameters to a dictionary
        hyperparams_dict = self.config.hyperparameters.model_dump()
        
        # Create the LambdaStep
        lambda_step = LambdaStep(
            name=step_name,
            lambda_func=lambda_function,
            inputs={
                "hyperparameters": hyperparams_dict,
                "hyperparameters_s3_uri": self.config.hyperparameters_s3_uri
            },
            outputs=[
                LambdaOutput(output_name="hyperparameters_s3_uri", output_type=LambdaOutputTypeEnum.String)
            ],
            depends_on=dependencies
        )
        
        # Add the hyperparameters_s3_uri as a property of the step
        lambda_step.hyperparameters_s3_uri = lambda_step.properties.Outputs["hyperparameters_s3_uri"]
        
        logger.info(f"Created LambdaStep with name: {lambda_step.name}")
        return lambda_step
