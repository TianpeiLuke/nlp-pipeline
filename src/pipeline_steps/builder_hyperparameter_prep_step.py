from typing import Dict, Optional, Any, List
from pathlib import Path
import logging
import json
import tempfile
import boto3
import shutil
from botocore.exceptions import ClientError

from sagemaker.s3 import S3Uploader

from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.steps import Step

from .config_hyperparameter_prep_step import HyperparameterPrepConfig
from .builder_step_base import StepBuilderBase

# Try to import hyperparameter preparation specification, but don't fail if it doesn't exist
try:
    from ..pipeline_step_specs.hyperparameter_prep_spec import HYPERPARAMETER_PREP_SPEC
    SPEC_AVAILABLE = True
except ImportError:
    HYPERPARAMETER_PREP_SPEC = None
    SPEC_AVAILABLE = False

logger = logging.getLogger(__name__)


class _DummyLambdaRef:
    """A stand-in for an already-deployed Lambda; no code is ever inlined or zipped."""
    def __init__(self, arn: str):
        self.function_arn = arn
        # Must be None so the SDK never tries to zip or upsert anything
        self.script = None
        self.zipped_code_dir = None


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
        
        # Use the hyperparameter preparation specification if available (likely not)
        spec = HYPERPARAMETER_PREP_SPEC if SPEC_AVAILABLE else None
        
        super().__init__(
            config=config,
            spec=spec,
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
        self.log_info("Validating HyperparameterPrepConfig...")
        if not self.config.hyperparameters:
            raise ValueError("hyperparameters must be provided and non-empty")
        if not self.config.hyperparameters_s3_uri:
            raise ValueError("hyperparameters_s3_uri must be provided and non-empty")
        if "hyperparameters_s3_uri" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'hyperparameters_s3_uri'")
        self.log_info("HyperparameterPrepConfig validation succeeded.")
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get inputs for the step. In this case, there are no inputs from previous steps,
        as the hyperparameters are directly provided in the configuration.
        
        Args:
            inputs: A dictionary of inputs (unused in this step)
            
        Returns:
            A dictionary of inputs for the Lambda function
        """
        # This step has no inputs from previous steps
        return {}

    def _get_outputs(self, outputs: Dict[str, Any]) -> List[LambdaOutput]:
        """
        Get outputs for the step.
        
        Args:
            outputs: Output configuration (unused in this step as outputs are predetermined)
            
        Returns:
            A list of LambdaOutput objects
        """
        # The outputs are fixed for this step
        return [
            LambdaOutput(output_name="hyperparameters_s3_uri", output_type=LambdaOutputTypeEnum.String)
        ]
    
    def _prepare_hyperparameters_file(self) -> str:
        """
        Serializes the hyperparameters to JSON, uploads it as
        `<hyperparameters_s3_uri>/hyperparameters.json`, and
        returns that full S3 URI.
        """
        hyperparams_dict = self.config.hyperparameters.model_dump()
        local_dir = Path(tempfile.mkdtemp())
        local_file = local_dir / "hyperparameters.json"
        
        try:
            local_file.write_text(json.dumps(hyperparams_dict, indent=2))

            prefix = self.config.hyperparameters_s3_uri or ""
            prefix = prefix.rstrip("/")
            target_s3_uri = f"{prefix}/hyperparameters.json"

            s3_parts = target_s3_uri.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
            
            s3_client = self.session.boto_session.client('s3')
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                self.log_info("Found existing hyperparameters file at %s, deleting it...", target_s3_uri)
                s3_client.delete_object(Bucket=bucket, Key=key)
                self.log_info("Existing hyperparameters file deleted successfully")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    self.log_info("No existing hyperparameters file found at %s", target_s3_uri)
                else:
                    self.log_warning("Error checking/deleting existing file: %s", str(e))

            self.log_info("Uploading hyperparameters from %s to %s", local_file, target_s3_uri)
            S3Uploader.upload(str(local_file), target_s3_uri, sagemaker_session=self.session)
            
            self.log_info("Hyperparameters successfully uploaded to %s", target_s3_uri)
            return target_s3_uri
            
        finally:
            shutil.rmtree(local_dir)

    def create_step(self, **kwargs) -> LambdaStep:
        """
        Creates a SageMaker LambdaStep for the pipeline that serializes hyperparameters
        to JSON and uploads them to S3.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - enable_caching: Whether to enable caching for this step (default: True)
                - dependencies: Optional list of steps that this step depends on

        Returns:
            A configured sagemaker.workflow.lambda_step.LambdaStep instance.
        """
        # Extract common parameters
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        dependencies = self._extract_param(kwargs, 'dependencies', None)
        self.log_info("Creating HyperparameterPrep LambdaStep...")

        step_name = self._get_step_name()

        # Save Hyperparameter to dict
        target_s3_uri = self._prepare_hyperparameters_file()
        
        # Create a simple dummy function that returns pre-generated values
        # This avoids complex Lambda creation and deployment
        def dummy_function(event, context=None):
            return {
                'hyperparameters_s3_uri': target_s3_uri
            }

        # Build your LambdaStep
        # Create Lambda step with the dummy function
        lambda_step = LambdaStep(
            name=step_name,
            lambda_func=dummy_function,
            inputs={
                "hyperparameters_s3_uri": target_s3_uri,
            },
            outputs=[
                LambdaOutput(output_name="hyperparameters_s3_uri", output_type=LambdaOutputTypeEnum.String)
            ],
            depends_on=dependencies
        )

        lambda_step._get_function_arn = lambda: "arn:aws:lambda:us-east-1:123456789012:function:dummy-hyperprep"

        # Set the hyperparameters_s3_uri as a direct property for backward compatibility,
        # but primarily this will be accessed via standard step.properties.Outputs["hyperparameters_s3_uri"]
        # which is the pattern we're standardizing on
        lambda_step.hyperparameters_s3_uri = target_s3_uri

        self.log_info("Created LambdaStep with name: %s", lambda_step.name)
        return lambda_step
