from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging
import tempfile
import json
import shutil
import boto3
from botocore.exceptions import ClientError

from sagemaker.workflow.steps import TrainingStep, Step
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost import XGBoost
from sagemaker.s3 import S3Uploader
from sagemaker.workflow.functions import Join

from .config_training_step_xgboost import XGBoostTrainingConfig
from .builder_step_base import StepBuilderBase
from .s3_utils import S3PathHandler

# Register property paths for XGBoost Training outputs
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep",
    "model_output",                                # Logical name in output_names
    "properties.ModelArtifacts.S3ModelArtifacts"   # Runtime property path
)

# Register path to training metrics
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep",
    "metrics_output",
    "properties.TrainingMetrics"
)

# Register path to training job name
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep", 
    "training_job_name",
    "properties.TrainingJobName"
)

# Register path to model data for compatibility with different naming patterns
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep", 
    "model_data",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Register path to output directory with both logical names
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep",
    "output_path",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Critical fix - Register ModelOutputPath specifically to match the descriptor used in pattern matching
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep",
    "ModelOutputPath",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Add more registrations for the model artifacts with names that might be used by different steps
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep",
    "ModelArtifacts",
    "properties.ModelArtifacts.S3ModelArtifacts"
)

# Add mapping for common model input key names
StepBuilderBase.register_property_path(
    "XGBoostTrainingStep",
    "model_input",  # Common key name used by many step builders
    "properties.ModelArtifacts.S3ModelArtifacts"
)

logger = logging.getLogger(__name__)


class XGBoostTrainingStepBuilder(StepBuilderBase):
    """
    Builder for an XGBoost Training Step.
    This class is responsible for configuring and creating a SageMaker TrainingStep
    that trains an XGBoost model.
    """

    def __init__(
        self,
        config: XGBoostTrainingConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the training step.

        Args:
            config: A XGBoostTrainingConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Training Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, XGBoostTrainingConfig):
            raise ValueError(
                "XGBoostTrainingStepBuilder requires a XGBoostTrainingConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: XGBoostTrainingConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating XGBoostTrainingConfig...")
        
        # Validate required attributes
        required_attrs = [
            'training_instance_type',
            'training_instance_count',
            'training_volume_size',
            'training_entry_point',
            'source_dir',
            'framework_version',
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostTrainingConfig missing required attribute: {attr}")
        
        # Validate input and output names
        required_input_keys = ["input_path", "config"]
        missing_input_keys = [key for key in required_input_keys if key not in (self.config.input_names or {})]
        if missing_input_keys:
            raise ValueError(f"input_names must contain keys: {', '.join(required_input_keys)}. Missing: {', '.join(missing_input_keys)}")
            
        logger.info("XGBoostTrainingConfig validation succeeded.")

    def _create_estimator(self, output_path=None) -> XGBoost:
        """
        Creates and configures the XGBoost estimator for the SageMaker Training Job.
        This defines the execution environment for the training script, including the instance
        type, framework version, and environment variables.
        
        Args:
            output_path: Optional override for model output path. If provided, this will be used
                         instead of self.config.output_path.

        Returns:
            An instance of sagemaker.xgboost.XGBoost.
        """
        # Note: We don't pass hyperparameters directly here because they are passed
        # through the "config" input channel instead
        
        # Use provided output_path or fall back to config
        actual_output_path = output_path or self.config.output_path
        
        return XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.source_dir,
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('XGBoostTraining')}"
            ),
            sagemaker_session=self.session,
            output_path=actual_output_path,
            checkpoint_s3_uri=self.config.get_checkpoint_uri(),
            environment=self._get_environment_variables(),
        )

    def _get_environment_variables(self) -> Dict[str, str]:
        """
        Constructs a dictionary of environment variables to be passed to the training job.
        These variables are used to control the behavior of the training script
        without needing to pass them as hyperparameters.

        Returns:
            A dictionary of environment variables.
        """
        env_vars = {}
        
        # Add environment variables from config if they exist
        if hasattr(self.config, "env") and self.config.env:
            env_vars.update(self.config.env)
            
        logger.info(f"Training environment variables: {env_vars}")
        return env_vars
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        # Note: hyperparameters_s3_uri is no longer a required input since we can generate it internally
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
            "dependencies": self.COMMON_PROPERTIES["dependencies"],
            "enable_caching": self.COMMON_PROPERTIES["enable_caching"]
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides using VALUES from output_names.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Use values from output_names as property names
        output_props = {}
        
        # Get the model artifacts property name from output_names
        model_artifacts_key = None
        for key, value in (self.config.output_names or {}).items():
            if "model" in key.lower():
                model_artifacts_key = value  # Use the VALUE here
                output_props[value] = "S3 URI of the model artifacts"
                
                # Add alias property for pipeline template compatibility
                output_props["ModelOutputPath"] = "S3 URI for model output directory"
                break
        
        # If no model key found, use a default
        if not output_props:
            output_props["ModelArtifacts"] = "S3 URI of the model artifacts"
            output_props["ModelOutputPath"] = "S3 URI for model output directory"
        
        # Also add the standard SageMaker property for backward compatibility
        output_props["ModelArtifacts.S3ModelArtifacts"] = "S3 URI for model artifacts"
        
        return output_props

        
    def _normalize_s3_uri(self, uri: str, description: str = "S3 URI") -> str:
        """
        Normalizes an S3 URI to ensure it has no trailing slashes and is properly formatted.
        Uses S3PathHandler for consistent path handling.
        
        Args:
            uri: The S3 URI to normalize
            description: Description for logging purposes
            
        Returns:
            Normalized S3 URI
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
        
        # Handle Pipeline step references with Get key - return as is
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference during normalization: %s", uri)
            return uri
        
        return S3PathHandler.normalize(uri, description)
        
    def _get_s3_directory_path(self, uri: str, filename: str = None) -> str:
        """
        Gets the directory part of an S3 URI, handling special cases correctly.
        Uses S3PathHandler for consistent path handling.
        
        Args:
            uri: The S3 URI which may or may not contain a filename
            filename: Optional filename to check for at the end of the URI
            
        Returns:
            The directory part of the URI without trailing slash
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            uri = str(uri.expr)
            
        # Handle Pipeline step references with Get key - return as is
        if isinstance(uri, dict) and 'Get' in uri:
            self.log_info("Found Pipeline step reference in directory path: %s", uri)
            return uri
            
        return S3PathHandler.ensure_directory(uri, filename)

    def _get_training_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """
        Constructs a dictionary of TrainingInput objects from the provided inputs dictionary.
        This defines the data channels for the training job, mapping S3 locations
        to input channels for the training container.
        
        The training script expects two input channels:
        - "data": containing train/val/test subdirectories
        - "config": containing hyperparameters.json

        Args:
            inputs: A dictionary mapping logical input channel names to their S3 URIs or dynamic Step properties.
                   Can be either:
                   - Flat format: {"input_path": "s3://path/...", "hyperparameters_s3_uri": "s3://path/..."}
                   - Nested format: {"inputs": {"input_path": "s3://path/...", "config": "s3://path/..."}}

        Returns:
            A dictionary of channel names to sagemaker.inputs.TrainingInput objects.
        """
        training_inputs = {}
        
        # Get channel names from config
        input_path_key = next(iter(self.config.input_names.keys()), "input_path")
        config_key = "config"  # Name for the hyperparameters config channel
        data_key = "data"      # The SageMaker channel name for input data
        
        # Handle different input structures
        if not isinstance(inputs, dict):
            logger.warning(f"Expected inputs to be a dictionary, got {type(inputs)}")
            return training_inputs
            
        # Use the base class helper to normalize inputs
        normalized_inputs = self._normalize_inputs(inputs)
        
        # Check if input_path is set directly on the config object - use that instead
        # This is the pattern used in the backup implementation
        if hasattr(self.config, 'input_path') and self.config.input_path:
            input_base_path = self.config.input_path
            # Use expr attribute if it exists, otherwise safely convert to string
            logger.info("Using input_path from config")
            
            # Create train/val/test channels using Join
            train_path = Join(on='/', values=[input_base_path, "train/"])
            val_path = Join(on='/', values=[input_base_path, "val/"])
            test_path = Join(on='/', values=[input_base_path, "test/"])
            
            # Log the path expressions (safely handling Pipeline variables)
            logger.info("Created training data paths using config input_path")
            
            # Create separate channels for each data split
            training_inputs["train"] = TrainingInput(s3_data=train_path)
            training_inputs["val"] = TrainingInput(s3_data=val_path)
            training_inputs["test"] = TrainingInput(s3_data=test_path)
            
        # Fallback to input dictionary if input_path is not in config
        elif input_path_key in normalized_inputs:
            base_path = normalized_inputs[input_path_key]
            # Normalize the base path URI
            base_path = self._normalize_s3_uri(base_path, "base input path")
            
            if self._validate_s3_uri(base_path, "base input path"):
                # Handle Pipeline step references with Get key differently
                if isinstance(base_path, dict) and 'Get' in base_path:
                    # For step references in dictionary format, use a different approach
                    # Extract the step reference parts
                    step_parts = base_path['Get'].split('.')
                    step_name = step_parts[0] if step_parts[0].startswith('Steps.') else f"Steps.{step_parts[0]}" 
                    
                    # Create separate channel references by appending the paths
                    train_path = {'Get': f"{step_name}.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri/train"}
                    val_path = {'Get': f"{step_name}.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri/val"}
                    test_path = {'Get': f"{step_name}.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri/test"}
                    
                    # Log the references
                    logger.info("Created Step reference paths for train/val/test")
                    logger.info(f"Train path reference: {train_path}")
                    logger.info(f"Val path reference: {val_path}")
                    logger.info(f"Test path reference: {test_path}")
                else:
                    # Regular S3 paths or PipelineVariable objects
                    train_path = Join(on='/', values=[base_path, "train/"])
                    val_path = Join(on='/', values=[base_path, "val/"])
                    test_path = Join(on='/', values=[base_path, "test/"])
                    
                    # Log the path expressions
                    self.log_info("Train data path expression: %s", train_path)
                    self.log_info("Validation data path expression: %s", val_path)
                    self.log_info("Test data path expression: %s", test_path)
                
                # Create separate channels for each data split
                training_inputs["train"] = TrainingInput(s3_data=train_path)
                training_inputs["val"] = TrainingInput(s3_data=val_path)
                training_inputs["test"] = TrainingInput(s3_data=test_path)
        else:
            logger.warning(f"No input path found for train/val/test channels")
        
        # Process config channel for hyperparameters - use the FULL file path rather than just the directory
        if "hyperparameters_s3_uri" in inputs:
            s3_uri = inputs["hyperparameters_s3_uri"]
            
            # Detailed logging for debugging S3 path issues
            if hasattr(s3_uri, 'expr'):
                original_uri = str(s3_uri.expr)
            else:
                original_uri = str(s3_uri)
                
            # Ensure we're using the full file path, not just the directory
            hyperparameters_file_uri = s3_uri
            
            # If the URI doesn't already end with hyperparameters.json, append it
            if not S3PathHandler.get_name(s3_uri) == "hyperparameters.json":
                hyperparameters_file_uri = S3PathHandler.join(s3_uri, "hyperparameters.json")
                
            logger.info(f"Processing hyperparameters S3 URI:")
            logger.info(f"  - Original URI: {original_uri}")
            logger.info(f"  - Using full file path: {hyperparameters_file_uri}")
            
            if self._validate_s3_uri(hyperparameters_file_uri, "hyperparameter file path"):
                # Use the FULL file path as s3_data, not just the directory
                training_inputs[config_key] = TrainingInput(s3_data=hyperparameters_file_uri)
                logger.info(f"Added config channel: {config_key} using file: {hyperparameters_file_uri}")
        
        # Check if config is provided directly in normalized_inputs
        elif config_key in normalized_inputs:
            s3_uri = normalized_inputs[config_key]
            if self._validate_s3_uri(s3_uri, "config path"):
                training_inputs[config_key] = TrainingInput(s3_data=s3_uri)
                logger.info(f"Adding config channel: {config_key} from {s3_uri.expr if hasattr(s3_uri, 'expr') else s3_uri}")
                
        return training_inputs
    
    def _prepare_hyperparameters_file(self) -> str:
        """
        Serializes the hyperparameters to JSON, uploads it to S3, and
        returns that full S3 URI. This eliminates the need for a separate
        HyperparameterPrepStep in the pipeline.
        """
        hyperparams_dict = self.config.hyperparameters.model_dump()
        local_dir = Path(tempfile.mkdtemp())
        local_file = local_dir / "hyperparameters.json"
        
        try:
            # Write JSON locally
            with open(local_file, "w") as f:
                json.dump(hyperparams_dict, indent=2, fp=f)
            logger.info(f"Created hyperparameters JSON file at {local_file}")

            # Construct S3 URI for the config directory
            prefix = self.config.hyperparameters_s3_uri if hasattr(self.config, 'hyperparameters_s3_uri') else None
            if not prefix:
                # Fallback path construction
                bucket = self.config.bucket if hasattr(self.config, 'bucket') else "sandboxdependency-abuse-secureaisandboxteamshare-1l77v9am252um"
                pipeline_name = self.config.pipeline_name if hasattr(self.config, 'pipeline_name') else "xgboost-model"
                current_date = getattr(self.config, 'current_date', "2025-06-02")
                prefix = f"s3://{bucket}/{pipeline_name}/training_config/{current_date}" # No trailing slash
            
            # Use our helper methods for consistent path handling
            config_dir = self._normalize_s3_uri(prefix, "hyperparameters prefix")
            logger.info(f"Normalized hyperparameters prefix: {config_dir}")
            
            # Check if hyperparameters.json is already in the path
            if S3PathHandler.get_name(config_dir) == "hyperparameters.json":
                # Use path as is if it already includes the filename
                target_s3_uri = config_dir
                logger.info(f"Using existing hyperparameters path: {target_s3_uri}")
            else:
                # Otherwise append the filename using S3PathHandler.join for proper path handling
                target_s3_uri = S3PathHandler.join(config_dir, "hyperparameters.json")
                logger.info(f"Constructed hyperparameters path: {target_s3_uri}")
                
            logger.info(f"Using hyperparameters S3 target URI: {target_s3_uri}")

            # Check if file exists and handle appropriately
            s3_parts = target_s3_uri.replace('s3://', '').split('/', 1)
            bucket = s3_parts[0]
            key = s3_parts[1]
            
            s3_client = self.session.boto_session.client('s3')
            try:
                s3_client.head_object(Bucket=bucket, Key=key)
                logger.info(f"Found existing hyperparameters file at {target_s3_uri}")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.info(f"No existing hyperparameters file found at {target_s3_uri}")
                else:
                    logger.warning(f"Error checking existing file: {str(e)}")

            # Upload the file
            logger.info(f"Uploading hyperparameters from {local_file} to {target_s3_uri}")
            S3Uploader.upload(str(local_file), target_s3_uri, sagemaker_session=self.session)
            
            logger.info(f"Hyperparameters successfully uploaded to {target_s3_uri}")
            return target_s3_uri
        
        finally:
            # Clean up temporary files
            shutil.rmtree(local_dir)
            
    def _validate_s3_uri(self, uri: str, description: str = "data") -> bool:
        """
        Validates that a string is a properly formatted S3 URI.
        Uses S3PathHandler for consistent path validation.
        
        Args:
            uri: The URI to validate
            description: Description of what the URI is for (used in error messages)
            
        Returns:
            True if valid, False otherwise
        """
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            # For PipelineVariables, we trust they'll resolve to valid URIs at execution time
            return True
            
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            # For Get expressions, we also trust they'll resolve properly at execution time
            logger.info(f"Found Pipeline step reference: {uri}")
            return True
        
        if not isinstance(uri, str):
            logger.warning(f"Invalid {description} URI: type {type(uri).__name__}")
            return False
        
        # Use S3PathHandler for validation
        valid = S3PathHandler.is_valid(uri)
        if not valid:
            logger.warning(f"Invalid {description} URI format: {uri}")
        
        return valid
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to XGBoostTraining step.
        This method dispatches to specialized handlers based on the type of step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        step_name = getattr(prev_step, 'name', str(prev_step))
        logger.info(f"Matching inputs from dependency step: {step_name}")
        
        # First check for TabularPreprocessingStep
        if hasattr(prev_step, 'name') and 'tabularpreprocessing' in prev_step.name.lower():
            matched_inputs = self._match_tabular_preprocessing_outputs(inputs, prev_step)
            if matched_inputs:
                logger.info(f"Matched inputs from TabularPreprocessingStep: {step_name}")
                return matched_inputs
        
        # Then check for HyperparameterPrepStep
        if hasattr(prev_step, 'name') and 'hyperparameterprep' in prev_step.name.lower():
            matched_inputs = self._match_hyperparameter_outputs(inputs, prev_step)
            if matched_inputs:
                logger.info(f"Matched inputs from HyperparameterPrepStep: {step_name}")
                return matched_inputs
        
        # Fall back to generic output matching
        matched_inputs = self._match_generic_outputs(inputs, prev_step)
        if matched_inputs:
            logger.info(f"Matched inputs from generic step: {step_name}")
                
        return matched_inputs
        
    def _match_tabular_preprocessing_outputs(self, inputs: Dict[str, Any], prev_step: Step) -> Set[str]:
        """
        Match outputs from a TabularPreprocessingStep.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Get the configured input path key from config
        input_path_key = self.config.input_names.get("input_path", "input_path")
        
        # Check if this step has the expected output structure
        if not hasattr(prev_step, "outputs") or not prev_step.outputs:
            return matched_inputs
            
        try:
            # Find the processed_data output
            processed_data_output = None
            for output in prev_step.outputs:
                if (hasattr(output, "output_name") and 
                    "processed_data" in output.output_name.lower()):
                    processed_data_output = output
                    break
                    
            if not processed_data_output:
                return matched_inputs
                
            # TabularPreprocessingStep output is the base path that contains train/val/test subdirs
            base_path = processed_data_output.destination
            base_path = base_path.rstrip("/")
            
            # Initialize inputs dict if needed
            if "inputs" not in inputs:
                inputs["inputs"] = {}
                
            # Just use the base path directly - it contains all subdirectories
            # that the training script expects (train, val, test)
            if input_path_key not in inputs.get("inputs", {}):
                inputs["inputs"][input_path_key] = base_path
                matched_inputs.add("inputs")
                logger.info(f"Added input path: {base_path}")
                
        except Exception as e:
            logger.warning(f"Error matching TabularPreprocessingStep outputs: {e}")
            
        return matched_inputs
        
    def _match_hyperparameter_outputs(self, inputs: Dict[str, Any], prev_step: Step) -> Set[str]:
        """
        Match outputs from a HyperparameterPrepStep.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Check if this step has the expected output structure
        if not hasattr(prev_step, "properties") or not hasattr(prev_step.properties, "ProcessingOutputConfig"):
            return matched_inputs
            
        try:
            # Try to get the hyperparameters output
            hyperparameters_s3_uri = prev_step.properties.ProcessingOutputConfig.Outputs["hyperparameters"].S3Output.S3Uri
            
            # Initialize inputs dict if needed
            if "inputs" not in inputs:
                inputs["inputs"] = {}
                
            # Add hyperparameters S3 URI
            inputs["hyperparameters_s3_uri"] = hyperparameters_s3_uri
            matched_inputs.add("hyperparameters_s3_uri")
            logger.info(f"Added hyperparameters from HyperparameterPrepStep (reference)")
            
        except (KeyError, AttributeError) as e:
            logger.warning(f"Error matching hyperparameter outputs: {e}")
            
        return matched_inputs
        
    def _match_generic_outputs(self, inputs: Dict[str, Any], prev_step: Step) -> Set[str]:
        """
        Match generic outputs from any step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Get input path key from config
        input_path_key = self.config.input_names.get("input_path", "input_path")
        
        try:
            # Try to find a generic output path that might contain training data
            if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                outputs = prev_step.properties.ProcessingOutputConfig.Outputs
                
                # Log the type of outputs object to help with debugging
                logger.info(f"Processing outputs of type: {outputs.__class__.__name__ if hasattr(outputs, '__class__') else type(outputs)}")
                
                # Special handling for PropertiesList type
                if hasattr(outputs, "__class__") and outputs.__class__.__name__ == "PropertiesList":
                    logger.info("Detected PropertiesList object - using direct attribute access")
                    
                    # Try common output names that might contain processed data
                    common_names = ["ProcessedTabularData", "Data", "OutputData"]
                    for name in common_names:
                        if hasattr(outputs, name):
                            try:
                                output_uri = outputs[name].S3Output.S3Uri
                                
                                # Initialize inputs dict if needed
                                if "inputs" not in inputs:
                                    inputs["inputs"] = {}
                                    
                                # Add as input_path, which maps to TrainingDataDirectory
                                if input_path_key not in inputs.get("inputs", {}):
                                    inputs["inputs"][input_path_key] = output_uri
                                    matched_inputs.add("inputs")
                                    logger.info(f"Added input path from PropertiesList attribute {name}: {output_uri}")
                                    return matched_inputs
                            except (AttributeError, KeyError) as e:
                                logger.debug(f"Error accessing PropertiesList attribute {name}: {e}")
                
                # Safe iteration approach instead of using len()
                try:
                    # Try to get the first item safely
                    output_name = next(iter(outputs), None)
                    if output_name is not None:
                        output_uri = outputs[output_name].S3Output.S3Uri
                        
                        # Initialize inputs dict if needed
                        if "inputs" not in inputs:
                            inputs["inputs"] = {}
                            
                        # Add as input_path, which maps to TrainingDataDirectory
                        if input_path_key not in inputs.get("inputs", {}):
                            inputs["inputs"][input_path_key] = output_uri
                            matched_inputs.add("inputs")
                            logger.info(f"Added input path from generic step: {output_uri} (reference)")
                except (TypeError, StopIteration) as e:
                    logger.debug(f"Error iterating through outputs: {e}")
                    
        except (AttributeError, KeyError, IndexError) as e:
            logger.warning(f"Error matching generic outputs: {e}")
            
        return matched_inputs

    def create_step(self, **kwargs) -> TrainingStep:
        """
        Creates a SageMaker TrainingStep for the pipeline.
        
        This method creates the XGBoost estimator, sets up training inputs from the input data,
        uploads hyperparameters, and creates the SageMaker TrainingStep.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary mapping input channel names to their S3 locations,
                  or a nested dictionary with input_path and hyperparameters_s3_uri
                - input_path: Direct parameter for training data input path (for backward compatibility)
                - output_path: Direct parameter for model output path (for backward compatibility)
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: Whether to enable caching for this step.
                
        Returns:
            A configured TrainingStep instance.
        """
        # Extract common parameters
        inputs_raw = self._extract_param(kwargs, 'inputs', {})
        input_path = self._extract_param(kwargs, 'input_path')
        output_path = self._extract_param(kwargs, 'output_path')
        dependencies = self._extract_param(kwargs, 'dependencies', [])
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        logger.info("Creating XGBoost TrainingStep...")
        
        # Get the step name
        step_name = self._get_step_name('XGBoostTraining')
        
        # Construct inputs dictionary - handle both nested and flat structures
        inputs = self._normalize_inputs(inputs_raw)
        
        # Add direct input_path parameter if provided
        if input_path is not None:
            inputs["input_path"] = input_path
            self.log_info("Using directly provided input_path: %s", input_path)
            
        # Ensure we have hyperparameters - either generate them or use provided ones
        if "hyperparameters_s3_uri" not in inputs:
            # Generate hyperparameters file
            hyperparameters_s3_uri = self._prepare_hyperparameters_file()
            inputs["hyperparameters_s3_uri"] = hyperparameters_s3_uri
            logger.info(f"Generated hyperparameters at: {hyperparameters_s3_uri}")
            
        # Look for inputs from dependencies if we don't have what we need
        if "input_path" not in inputs and dependencies:
            input_requirements = self.get_input_requirements()
            
            # Extract inputs from dependencies
            for dep_step in dependencies:
                # Temporary dictionary to collect inputs from matching
                temp_inputs = {}
                matched = self._match_custom_properties(temp_inputs, input_requirements, dep_step)
                
                if matched:
                    # Normalize any nested inputs from the matching
                    normalized_deps = self._normalize_inputs(temp_inputs)
                    
                    # Add to our main inputs dictionary
                    inputs.update(normalized_deps)
                    logger.info(f"Found inputs from dependency: {getattr(dep_step, 'name', None)}")
                    
        # Get training inputs (TrainingInput objects)
        training_inputs = self._get_training_inputs(inputs)
        
        # Make sure we have the inputs we need
        if len(training_inputs) == 0:
            raise ValueError("No training inputs available. Provide input_path or ensure dependencies supply necessary outputs.")
        
        logger.info(f"Final training inputs: {list(training_inputs.keys())}")
        
        # Create estimator with output path if provided
        estimator = self._create_estimator(output_path)
        
        # Create the training step
        try:
            training_step = TrainingStep(
                name=step_name,
                estimator=estimator,
                inputs=training_inputs,
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching)
            )
            
            # Add model output properties
            model_output_key = None
            if hasattr(self.config, 'output_names'):
                # Find the output name mapped to model artifacts
                for key, value in self.config.output_names.items():
                    if "model" in key.lower():
                        model_output_key = key
                        break
            
            # Log successful creation
            logger.info(f"Created TrainingStep with name: {training_step.name}")
            
            return training_step
            
        except Exception as e:
            logger.error(f"Error creating XGBoost TrainingStep: {str(e)}")
            raise ValueError(f"Failed to create XGBoostTrainingStep: {str(e)}") from e
