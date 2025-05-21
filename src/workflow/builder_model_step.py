import os
import tarfile
import tempfile
from typing import Dict, Optional # Ensure these are imported
import boto3


from sagemaker.pytorch import PyTorchModel
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.properties import Properties
from sagemaker import image_uris
import logging

from .builder_workflow.workflow_config import ModelConfig

logger = logging.getLogger(__name__)


class PytorchModelStepBuilder:
    """Model step builder for PyTorch models"""

    REGION_MAPPING: Dict[str, str] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }

    def __init__(
        self,
        config: ModelConfig
        sagemaker_session: Optional[Session] = None,
        role: Optional[str] = None
    ):
        self.config = config
        self.session = sagemaker_session
        self.role = role

        self.aws_region = self.REGION_MAPPING.get(self.config.region)
        if not self.aws_region:
            raise ValueError(
                f"Invalid region code: {self.config.region}. "
                f"Must be one of: {', '.join(self.REGION_MAPPING.keys())}"
            )

        logger.info(f"Initializing PytorchModelStepBuilder with region code: {self.config.region} "
                    f"(AWS region: {self.aws_region})")

    def _sanitize_name_for_sagemaker(self, name: str, max_length: int = 63) -> str:
        """Sanitize a string to be a valid SageMaker resource name component."""
        if not name:
            return "default-name"
        sanitized = "".join(c if c.isalnum() else '-' for c in str(name))
        sanitized = '-'.join(filter(None, sanitized.split('-'))) # Remove multiple hyphens
        return sanitized[:max_length].rstrip('-')
        
    def create_model_step(self, model_data: Properties) -> ModelStep:
        """Create model step for deployment"""
        logger.info(f"Creating model step with instance type: {self.config.inference_instance_type} "
                    f"in region: {self.aws_region}")

        instance_type_param = Parameter(
            name="InferenceInstanceType",
            default_value=self.config.inference_instance_type
        )

        safe_date_string = self.config.current_date.replace(":", "-").replace("T", "-").replace("Z", "")
        model_name = f"bsm-rnr-model-{safe_date_string}"[:63]

        # PyTorchModel will handle packaging and uploading the source_dir.
        # The _package_and_upload_model_code method is generally not needed here.
        # Ensure self.config.source_dir points to your inference scripts (e.g., directory containing inference.py)
        # and self.config.inference_entry_point is the name of your entry script (e.g., "inference.py").

        model = PyTorchModel(
            name=model_name,
            model_data=model_data,  # This is the Properties object from the training step
            role=self.role,
            entry_point=self.config.inference_entry_point, # From your _create_env_config, e.g., "inference.py"
            source_dir=self.config.source_dir,          # Path to your inference code directory
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            sagemaker_session=self.session,
            env=self._create_env_config(),
            image_uri=self._get_image_uri()
            # You can add other PyTorchModel parameters if needed (e.g., dependencies)
        )

        # The step_args should come from model.create()
        step_creation_args = model.create(
            instance_type=instance_type_param,
            accelerator_type=None # Or configure as needed
        )
        
        step_name = "DefaultCreateModelStep" # Default if pipeline_name is not available
        # Max length for step names is 80 characters.
        # Pattern: ^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,79}$
        if hasattr(self.config, 'pipeline_name') and self.config.pipeline_name:
            sanitized_pipeline_name = self._sanitize_name_for_sagemaker(self.config.pipeline_name, max_length=60)
            step_name = f"{sanitized_pipeline_name}-CreateModel"
        else:
            logger.warning("pipeline_name not found in ModelConfig. Using default name 'DefaultCreateModelStep' for ModelStep.")


        return ModelStep(
            name=step_name[:80], # Ensure name does not exceed max length
            step_args=step_creation_args
        )

    def _get_image_uri(self) -> str:
        """Get the PyTorch inference image URI"""
        return image_uris.retrieve(
            framework="pytorch",
            region=self.aws_region,
            version=self.config.framework_version,
            py_version=self.config.py_version,
            instance_type=self.config.inference_instance_type, # This instance type is for retrieving the image
            image_scope="inference"
        )

    def _create_env_config(self) -> dict:
        """Create and validate environment configuration"""
        # Ensure self.config has an 'inference_entry_point' attribute
        if not hasattr(self.config, 'inference_entry_point') or not self.config.inference_entry_point:
            raise ValueError("ModelConfig must have 'inference_entry_point' defined for creating environment configuration.")

        env_config = {
            'MMS_DEFAULT_RESPONSE_TIMEOUT': str(self.config.container_startup_health_check_timeout),
            'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
            'SAGEMAKER_PROGRAM': self.config.inference_entry_point, # Use the entry_point from config
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code', # Standard path where source_dir content is placed
            'SAGEMAKER_CONTAINER_MEMORY_LIMIT': str(self.config.container_memory_limit),
            'SAGEMAKER_MODEL_DATA_DOWNLOAD_TIMEOUT': str(self.config.data_download_timeout),
            'SAGEMAKER_INFERENCE_MEMORY_LIMIT': str(self.config.inference_memory_limit),
            'SAGEMAKER_MAX_CONCURRENT_INVOCATIONS': str(self.config.max_concurrent_invocations),
            'SAGEMAKER_MAX_PAYLOAD_IN_MB': str(self.config.max_payload_size),
            'AWS_REGION': self.aws_region
        }

        for key, value in env_config.items():
            # Allow 0 for numeric-like fields if they are explicitly set, but not empty strings for critical paths.
            if value is None or (isinstance(value, str) and not value.strip() and key in ['SAGEMAKER_PROGRAM']):
                raise ValueError(f"Missing or empty environment variable value for critical key: {key}")
            elif value is None or (isinstance(value, str) and not value.strip()):
                 logger.warning(f"Environment variable {key} has an effectively empty value: '{value}'. This might be acceptable depending on the variable.")


        return env_config