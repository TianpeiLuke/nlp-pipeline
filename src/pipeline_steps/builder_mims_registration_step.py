from typing import Dict, Optional, List, Union, Any, Set
from pathlib import Path
import os
import importlib
import logging
from dotenv import load_dotenv # For loading .env file

from sagemaker.processing import ProcessingInput
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import Properties

# Load environment variables from .env file
# Ensure load_dotenv() is called early in your application's entry point.
# If .env is in a specific location relative to the script, you might use:
# from pathlib import Path
# dotenv_path = Path(__file__).resolve().parent / '.env' # Assuming .env is in the same dir as script
# load_dotenv(dotenv_path=dotenv_path)
# Or, if .env is in the project root and scripts are in subdirectories:
# dotenv_path = Path(__file__).resolve().parent.parent / '.env' # Go up one level for project root
# load_dotenv(dotenv_path=dotenv_path)
# For SageMaker, environment variables are typically set directly in the job configuration.
load_dotenv()

logger = logging.getLogger(__name__)

# Helper function for dynamic imports
def import_from_env(env_var_name_for_module: str, class_or_object_name: str):
    module_path = os.environ.get(env_var_name_for_module)
    if not module_path:
        logger.error(f"Environment variable '{env_var_name_for_module}' not set. Cannot import '{class_or_object_name}'.")
        return None
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_or_object_name)
    except ImportError:
        logger.error(f"Could not import module '{module_path}' for '{class_or_object_name}'.")
        return None
    except AttributeError:
        logger.error(f"'{class_or_object_name}' not found in module '{module_path}'.")
        return None
    

MimsModelRegistrationProcessingStep = import_from_env(
    "SECUREAI_MIMS_REGISTRATION_STEP_MODULE",
    "MimsModelRegistrationProcessingStep"
)
if MimsModelRegistrationProcessingStep:
    logger.info("MimsModelRegistrationProcessingStep class is available.")
else:
    logger.warning("MimsModelRegistrationProcessingStep class is not available.")

from .config_mims_registration_step import ModelRegistrationConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class ModelRegistrationStepBuilder(StepBuilderBase):
    """Builder for model registration steps"""

    def __init__(
        self,
        config: ModelRegistrationConfig,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None
    ):
        super().__init__(config, sagemaker_session, role, notebook_root)
        self.config: ModelRegistrationConfig = config

    def validate_configuration(self) -> None:
        """Validate required configuration settings"""
        logger.info("Validating model registration configuration...")

        # Validate required attributes
        required_attrs = [
            'model_owner',
            'model_registration_domain',
            'model_registration_objective',
            'source_model_inference_content_types',
            'source_model_inference_response_types'
        ]

        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelRegistrationConfig missing required attribute: {attr}")

        # Validate variable lists
        if not self.config.source_model_inference_output_variable_list:
            raise ValueError("At least one output variable must be defined")

        if not self.config.source_model_inference_input_variable_list:
            logger.warning("No input variables defined for model registration")

        # Validate content types
        valid_content_types = ["text/csv", "application/json"]
        for content_type in self.config.source_model_inference_content_types:
            if content_type not in valid_content_types:
                raise ValueError(f"Invalid content type: {content_type}. Must be one of {valid_content_types}")

        for response_type in self.config.source_model_inference_response_types:
            if response_type not in valid_content_types:
                raise ValueError(f"Invalid response type: {response_type}. Must be one of {valid_content_types}")

        logger.info(
            f"Validated registration configuration for region {self.config.region}\n"
            f"Domain: {self.config.model_registration_domain}\n"
            f"Objective: {self.config.model_registration_objective}\n"
            f"Input variables: {len(self.config.source_model_inference_input_variable_list)}\n"
            f"Output variables: {len(self.config.source_model_inference_output_variable_list)}"
        )

    def _get_processing_inputs(
        self, 
        packaging_step_output: Union[str, Properties],
        payload_s3_key: Optional[str] = None
    ) -> List[ProcessingInput]:
        """Get processing inputs for registration step."""
        model_input = ProcessingInput(
            source=packaging_step_output,
            destination="/opt/ml/processing/input/model",
            s3_data_distribution_type="FullyReplicated",
            s3_input_mode="File"
        )
        
        inputs = [model_input]

        if payload_s3_key:
            payload_url = f"s3://{self.config.bucket}/{payload_s3_key}"
            inputs.append(
                ProcessingInput(
                    source=payload_url,
                    destination="/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )

        return inputs

    def _validate_regions(self, regions: List[str]) -> None:
        """Validate region codes"""
        invalid_regions = [r for r in regions if r not in self.config.REGION_MAPPING]
        if invalid_regions:
            raise ValueError(
                f"Invalid region(s): {invalid_regions}. "
                f"Must be one of: {', '.join(self.config.REGION_MAPPING.keys())}"
            )

    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get base input requirements and add additional ones
        input_reqs = super().get_input_requirements()
        input_reqs["dependencies"] = self.COMMON_PROPERTIES["dependencies"]
        input_reqs["regions"] = "Optional list of regions for registration"
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Define the output properties for model registration
        output_props = {
            "model_package_arn": "ARN of the registered model package",
            "model_name": "Name of the registered model"
        }
        # Add any output names from config if they exist
        if hasattr(self.config, "output_names"):
            output_props.update({k: v for k, v in self.config.output_names.items()})
        return output_props
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to MIMS registration step.
        
        This method looks for:
        1. packaging_step_output from a PackagingStep
        2. payload_s3_key from a PayloadStep
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Look for packaging_step_output from a PackagingStep
        if (hasattr(prev_step, "properties") and 
            hasattr(prev_step.properties, "ProcessingOutputConfig") and
            hasattr(prev_step.properties.ProcessingOutputConfig, "Outputs") and
            hasattr(prev_step.properties.ProcessingOutputConfig.Outputs, "__getitem__")):
            
            try:
                # Try string keys (dict-like)
                if "packaged_model_output" in prev_step.properties.ProcessingOutputConfig.Outputs:
                    output = prev_step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"]
                    if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                        s3_uri = output.S3Output.S3Uri
                        if "packaging_step_output" in input_requirements:
                            inputs["packaging_step_output"] = s3_uri
                            matched_inputs.add("packaging_step_output")
                            logger.info(f"Found packaging_step_output from PackagingStep: {getattr(prev_step, 'name', str(prev_step))}")
            except (AttributeError, IndexError, KeyError) as e:
                logger.warning(f"Could not extract packaged model output from step: {e}")
        
        # Look for payload_s3_key from a PayloadStep
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "payload_s3_key"):
            try:
                payload_s3_key = prev_step.properties.payload_s3_key
                if "payload_s3_key" in input_requirements:
                    inputs["payload_s3_key"] = payload_s3_key
                    matched_inputs.add("payload_s3_key")
                    logger.info(f"Found payload_s3_key from PayloadStep: {getattr(prev_step, 'name', str(prev_step))}")
            except AttributeError as e:
                logger.warning(f"Could not extract payload_s3_key from step: {e}")
                
        return matched_inputs
        
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps.
        
        This method uses the base class implementation and adds enable_caching.
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            Dictionary of inputs extracted from dependency steps
        """
        # Use the base class implementation to extract inputs
        inputs = super().extract_inputs_from_dependencies(dependency_steps)
        
        # Add enable_caching
        inputs["enable_caching"] = True
        
        return inputs
    
    def create_step(self, **kwargs) -> Union[Step, Dict[str, Step]]:
        """
        Create registration steps for specified regions.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - packaging_step_output: Output from the packaging step (required)
                - dependencies: Optional list of steps this step depends on
                - payload_s3_key: Optional S3 key for the payload
                - regions: Optional list of regions for registration
                - enable_caching: Whether to enable caching for this step (default: True)
        
        Returns:
            A single registration step or a dictionary of registration steps by region
        """
        # Extract parameters
        packaging_step_output = self._extract_param(kwargs, 'packaging_step_output')
        dependencies = self._extract_param(kwargs, 'dependencies')
        payload_s3_key = self._extract_param(kwargs, 'payload_s3_key')
        regions = self._extract_param(kwargs, 'regions', [self.config.region])
        
        # Validate required parameters
        if not packaging_step_output:
            raise ValueError("packaging_step_output must be provided")
            
        self._validate_regions(regions)

        if isinstance(packaging_step_output, Properties):
            logger.info("Creating registration steps with packaging output from Properties")
        else:
            logger.info(f"Creating registration steps with packaging output path: {packaging_step_output}")

        registration_steps = {}
        for region in regions:
            step_name = f"Registration_{region}"
            logger.info(f"Creating registration step for region {region}")
            
            try:
                registration_inputs = self._get_processing_inputs(
                    packaging_step_output,
                    payload_s3_key
                )

                step = MimsModelRegistrationProcessingStep(
                    step_name=step_name,
                    role=self.role,
                    sagemaker_session=self.session,
                    processing_input=registration_inputs,
                    depends_on=dependencies or [],
                )
                
                # Add output properties for downstream steps
                if hasattr(step, 'properties'):
                    # These are placeholder properties that might be available in the actual implementation
                    # The actual implementation would need to expose these properties
                    pass
                
                registration_steps[region] = step
                logger.info(f"Created registration step for {region}")
            except Exception as e:
                logger.error(f"Failed to create registration step for {region}: {str(e)}")
                raise

        # If only one region, return the step directly
        if len(regions) == 1:
            return registration_steps[regions[0]]
        return registration_steps
    
    # Maintain backwards compatibility
    def create_registration_steps(
        self,
        packaging_step_output: str,
        dependencies: Optional[List[Step]] = None,
        payload_s3_key: Optional[str] = None,
        regions: Optional[List[str]] = None
    ) -> Dict[str, ProcessingStep]:
        """Backwards compatible method for creating registration steps"""
        result = self.create_step(
            packaging_step_output=packaging_step_output,
            dependencies=dependencies,
            payload_s3_key=payload_s3_key,
            regions=regions
        )
        # Ensure we always return a dictionary for backwards compatibility
        if isinstance(result, dict):
            return result
        return {self.config.region: result}
