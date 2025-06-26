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

from .config_training_step_xgboost import XGBoostTrainingConfig
from .builder_step_base import StepBuilderBase

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
        if "train" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'train'")
        
        if "val" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'val'")
            
        logger.info("XGBoostTrainingConfig validation succeeded.")

    def _create_estimator(self) -> XGBoost:
        """
        Creates and configures the XGBoost estimator for the SageMaker Training Job.
        This defines the execution environment for the training script, including the instance
        type, framework version, and environment variables.

        Returns:
            An instance of sagemaker.xgboost.XGBoost.
        """
        # Note: We don't pass hyperparameters directly here because they are passed
        # through the "config" input channel instead
        
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
            output_path=self.config.output_path,
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
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Define the output properties for the training step
        output_props = {
            "ModelArtifacts.S3ModelArtifacts": "S3 URI of the model artifacts"
        }
        return output_props

        
    def _get_training_inputs(self, inputs: Dict[str, Any]) -> Dict[str, TrainingInput]:
        """
        Constructs a dictionary of TrainingInput objects from the provided inputs dictionary.
        This defines the data channels for the training job, mapping S3 locations
        to input channels for the training container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'train', 'val')
                    to their S3 URIs or dynamic Step properties.
                    Can be either:
                    - Flat format: {"train": "s3://path/...", "val": "s3://path/..."}
                    - Nested format: {"inputs": {"train": "s3://path/...", "val": "s3://path/..."}}

        Returns:
            A dictionary of channel names to sagemaker.inputs.TrainingInput objects.
        """
        training_inputs = {}
        
        # Get the input names from config for standard channels
        train_key = self.config.input_names.get("train", "train")
        val_key = self.config.input_names.get("val", "val")
        test_key = self.config.input_names.get("test", "test") if "test" in self.config.input_names else "test"
        config_key = self.config.input_names.get("config", "config") if "config" in self.config.input_names else "config"
        
        # Handle different input structures
        if not isinstance(inputs, dict):
            logger.warning(f"Expected inputs to be a dictionary, got {type(inputs)}")
            return training_inputs
            
        # First, normalize the inputs structure by merging nested and flat formats
        normalized_inputs = {}
        
        # Copy items directly from the inputs dict (flat format)
        for key, value in inputs.items():
            if key != "inputs" and key != "hyperparameters_s3_uri":
                normalized_inputs[key] = value
                
        # Add items from the nested format if present
        if "inputs" in inputs and isinstance(inputs["inputs"], dict):
            for key, value in inputs["inputs"].items():
                normalized_inputs[key] = value
        
        # Handle hyperparameters_s3_uri specially
        if "hyperparameters_s3_uri" in inputs and config_key not in normalized_inputs:
            normalized_inputs[config_key] = inputs["hyperparameters_s3_uri"]
            
        # Process training data - required
        if train_key in normalized_inputs:
            s3_uri = normalized_inputs[train_key]
            if self._validate_s3_uri(s3_uri, "training data"):
                training_inputs[train_key] = TrainingInput(s3_data=s3_uri)
                logger.info(f"Adding training data channel: {train_key} from {s3_uri.expr if hasattr(s3_uri, 'expr') else s3_uri}")
        else:
            logger.warning(f"No training data found for channel: {train_key}")
                
        # Process validation data - required
        if val_key in normalized_inputs:
            s3_uri = normalized_inputs[val_key]
            if self._validate_s3_uri(s3_uri, "validation data"):
                training_inputs[val_key] = TrainingInput(s3_data=s3_uri)
                logger.info(f"Adding validation data channel: {val_key} from {s3_uri.expr if hasattr(s3_uri, 'expr') else s3_uri}")
        else:
            logger.warning(f"No validation data found for channel: {val_key}")
                
        # Process test data (optional)
        if test_key in normalized_inputs:
            s3_uri = normalized_inputs[test_key]
            if self._validate_s3_uri(s3_uri, "test data"):
                training_inputs[test_key] = TrainingInput(s3_data=s3_uri)
                logger.info(f"Adding test data channel: {test_key} from {s3_uri.expr if hasattr(s3_uri, 'expr') else s3_uri}")
                
        # Process hyperparameter config (optional)
        if config_key in normalized_inputs:
            s3_uri = normalized_inputs[config_key]
            if self._validate_s3_uri(s3_uri, "hyperparameter config"):
                training_inputs[config_key] = TrainingInput(s3_data=s3_uri)
                logger.info(f"Adding hyperparameter config channel: {config_key} from {s3_uri.expr if hasattr(s3_uri, 'expr') else s3_uri}")
                
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

            # Construct S3 URI
            prefix = self.config.hyperparameters_s3_uri if hasattr(self.config, 'hyperparameters_s3_uri') else None
            if not prefix:
                # Fallback path construction
                bucket = self.config.bucket if hasattr(self.config, 'bucket') else "sandboxdependency-abuse-secureaisandboxteamshare-1l77v9am252um"
                pipeline_name = self.config.pipeline_name if hasattr(self.config, 'pipeline_name') else "xgboost-model"
                current_date = getattr(self.config, 'current_date', "2025-06-02")
                prefix = f"s3://{bucket}/{pipeline_name}/training_config/{current_date}"
            
            target_s3_uri = f"{prefix}/hyperparameters.json"

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
        
        Args:
            uri: The URI to validate
            description: Description of what the URI is for (used in error messages)
            
        Returns:
            True if valid, False otherwise
        """
        import re
        
        # Handle PipelineVariable objects
        if hasattr(uri, 'expr'):
            # For PipelineVariables, we trust they'll resolve to valid URIs at execution time
            return True
            
        if not isinstance(uri, str):
            logger.warning(f"Invalid {description} URI: type {type(uri).__name__}")
            return False
            
        # Basic S3 URI validation
        s3_pattern = r'^s3://[a-zA-Z0-9.-]+(/[a-zA-Z0-9._-]+)*/?$'
        if not re.match(s3_pattern, uri):
            logger.warning(f"Invalid {description} URI format: {uri}")
            return False
            
        return True
        
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
        
        # Get the configured input channel keys from config
        train_key = self.config.input_names.get("train", "train") 
        val_key = self.config.input_names.get("val", "val")
        test_key = self.config.input_names.get("test", "test") if "test" in self.config.input_names else "test"
        
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
                
            # TabularPreprocessingStep stores outputs in subfolders (train, val, test)
            # according to tabular_preprocess.py script
            base_path = processed_data_output.destination
            base_path = base_path.rstrip("/")
            
            # Initialize inputs dict if needed
            if "inputs" not in inputs:
                inputs["inputs"] = {}
                
            # Add train data - explicitly point to the /train subdirectory
            if train_key not in inputs.get("inputs", {}):
                train_path = f"{base_path}/train"
                inputs["inputs"][train_key] = train_path
                matched_inputs.add("inputs")
                logger.info(f"Added training data path: {train_path}")
                
            # Add validation data - explicitly point to the /val subdirectory
            if val_key not in inputs.get("inputs", {}):
                val_path = f"{base_path}/val"
                inputs["inputs"][val_key] = val_path
                matched_inputs.add("inputs")
                logger.info(f"Added validation data path: {val_path}")
                
            # Add test data (optional) - explicitly point to the /test subdirectory
            if test_key not in inputs.get("inputs", {}) and "test" in self.config.input_names:
                test_path = f"{base_path}/test"
                inputs["inputs"][test_key] = test_path
                matched_inputs.add("inputs")
                logger.info(f"Added test data path: {test_path}")
                
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
        config_key = self.config.input_names.get("config", "config") if "config" in self.config.input_names else "config"
        
        # Check if hyperparameters are available in the step outputs
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "Outputs"):
            try:
                if "hyperparameters_s3_uri" in prev_step.properties.Outputs:
                    hyperparameters_s3_uri = prev_step.properties.Outputs["hyperparameters_s3_uri"]
                    
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                        
                    # Add to config channel directly
                    if config_key not in inputs.get("inputs", {}):
                        inputs["inputs"][config_key] = hyperparameters_s3_uri
                        matched_inputs.add("inputs")
                        logger.info(f"Connected hyperparameters to config input channel: {hyperparameters_s3_uri.expr if hasattr(hyperparameters_s3_uri, 'expr') else hyperparameters_s3_uri}")
            except (AttributeError, KeyError) as e:
                logger.warning(f"Error matching HyperparameterPrepStep outputs: {e}")
                
        # Check for direct hyperparameters property (older pattern)
        if hasattr(prev_step, "hyperparameters_s3_uri"):
            try:
                hyperparameters_s3_uri = prev_step.hyperparameters_s3_uri
                
                # Initialize inputs dict if needed
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                    
                # Add to config channel directly
                if config_key not in inputs.get("inputs", {}):
                    inputs["inputs"][config_key] = hyperparameters_s3_uri
                    matched_inputs.add("inputs")
                    logger.info(f"Connected hyperparameters to config input channel: {hyperparameters_s3_uri}")
            except AttributeError as e:
                logger.warning(f"Error accessing hyperparameters_s3_uri property: {e}")
                
        return matched_inputs
        
    def _match_generic_outputs(self, inputs: Dict[str, Any], prev_step: Step) -> Set[str]:
        """
        Generic matching for other step types.
        
        Args:
            inputs: Dictionary to add matched inputs to
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Get the configured input channel keys from config
        train_key = self.config.input_names.get("train", "train") 
        val_key = self.config.input_names.get("val", "val")
        test_key = self.config.input_names.get("test", "test") if "test" in self.config.input_names else "test"
        config_key = self.config.input_names.get("config", "config") if "config" in self.config.input_names else "config"
        
        # Look for outputs from a ProcessingStep
        if hasattr(prev_step, "outputs") and prev_step.outputs:
            # Try to detect if this might be a TabularPreprocessingStep based on output structure
            # even if not explicitly identified by name
            if len(prev_step.outputs) == 1 and hasattr(prev_step.outputs[0], "destination"):
                # Check if this single output might be a base path with train/val/test subdirectories
                base_path = prev_step.outputs[0].destination.rstrip("/")
                output_name = prev_step.outputs[0].output_name.lower() if hasattr(prev_step.outputs[0], "output_name") else ""
                
                # If this appears to be a processed data output, try inferring subdirectories
                if "processed" in output_name or "data" in output_name:
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                        
                    # Add potential train/val/test subdirectory paths if not already set
                    if train_key not in inputs.get("inputs", {}):
                        train_path = f"{base_path}/train"
                        inputs["inputs"][train_key] = train_path
                        matched_inputs.add("inputs")
                        logger.info(f"Generic match - inferred training data path: {train_path}")
                    
                    if val_key not in inputs.get("inputs", {}):
                        val_path = f"{base_path}/val"
                        inputs["inputs"][val_key] = val_path
                        matched_inputs.add("inputs")
                        logger.info(f"Generic match - inferred validation data path: {val_path}")
                    
                    # Only add test if configured
                    if "test" in self.config.input_names and test_key not in inputs.get("inputs", {}):
                        test_path = f"{base_path}/test"
                        inputs["inputs"][test_key] = test_path
                        matched_inputs.add("inputs")
                        logger.info(f"Generic match - inferred test data path: {test_path}")
                    
                    # If we've matched all required inputs, return early
                    if matched_inputs:
                        return matched_inputs
            
            # Fallback to standard matching by names if the above approach doesn't match
            # Match train data
            self._match_output_by_name(inputs, prev_step, matched_inputs, 
                                     train_key, ["train", "training"])
                
            # Match validation data
            self._match_output_by_name(inputs, prev_step, matched_inputs, 
                                     val_key, ["valid", "val", "validation"])
                
            # Match test data
            self._match_output_by_name(inputs, prev_step, matched_inputs, 
                                     test_key, ["test", "testing"])
        
        # Look for hyperparameters from a step with standard SageMaker Pipeline output
        # NO CHANGES TO THIS PART
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "Outputs"):
            try:
                # Try to get the hyperparameters_s3_uri from standard SageMaker Pipeline output
                if "hyperparameters_s3_uri" in prev_step.properties.Outputs:
                    hyperparameters_s3_uri = prev_step.properties.Outputs["hyperparameters_s3_uri"]
                    
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    # Add to config channel directly
                    if config_key not in inputs.get("inputs", {}):
                        inputs["inputs"][config_key] = hyperparameters_s3_uri
                        matched_inputs.add("inputs")
                        logger.info(f"Connected hyperparameters to config input channel: '{config_key}'")
            except (AttributeError, KeyError) as e:
                logger.warning(f"Could not extract hyperparameters from step outputs: {e}")
                
        return matched_inputs
        
    def _match_output_by_name(self, inputs: Dict[str, Any], step: Step, 
                             matched_inputs: Set[str], input_key: str, 
                             output_name_patterns: List[str]) -> None:
        """
        Match a specific output by name patterns.
        
        Args:
            inputs: Dictionary to add matched inputs to
            step: The dependency step
            matched_inputs: Set to add matched input names to
            input_key: Name of the input key to use
            output_name_patterns: List of patterns to look for in output names
        """
        try:
            # Look for an output with a name that matches any of the patterns
            for output in step.outputs:
                if hasattr(output, "output_name"):
                    matches_pattern = any(pattern in output.output_name.lower() 
                                        for pattern in output_name_patterns)
                    
                    # Ensure we don't match more specific patterns
                    if not matches_pattern:
                        continue
                        
                    # Skip if the output name contains more specific patterns that shouldn't match
                    # Example: "train_data" should match "train" but "train_validation_split" should NOT match "train"
                    if (input_key == self.config.input_names.get("train", "train") and 
                        any(term in output.output_name.lower() 
                            for term in ["validation", "val", "test"])):
                        continue
                    
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    # Add output destination to inputs
                    if input_key not in inputs.get("inputs", {}):
                        inputs["inputs"][input_key] = output.destination
                        matched_inputs.add("inputs")
                        logger.info(f"Found {input_key} data from output: {output.output_name}")
                        break
                        
        except Exception as e:
            logger.warning(f"Error matching output for {input_key}: {e}")
    
    def create_step(self, **kwargs) -> TrainingStep:
        """
        Creates the final, fully configured SageMaker TrainingStep for the pipeline.
        This method orchestrates the assembly of the estimator and its inputs
        into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - hyperparameters_s3_uri: Optional S3 URI to a JSON file containing hyperparameters.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.TrainingStep instance.
        """
        logger.info("Creating XGBoost TrainingStep...")

        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs', {})
        hyperparameters_s3_uri = self._extract_param(kwargs, 'hyperparameters_s3_uri')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Prepare hyperparameters - generate and upload hyperparameters.json
        if not hyperparameters_s3_uri:
            hyperparameters_s3_uri = self._prepare_hyperparameters_file()
            logger.info(f"Generated hyperparameters at: {hyperparameters_s3_uri}")
            
        # Add the hyperparameters URI to the config channel input
        config_key = self.config.input_names.get("config", "config") if "config" in self.config.input_names else "config"
        if "inputs" not in inputs:
            inputs["inputs"] = {}
        
        if config_key not in inputs.get("inputs", {}):
            inputs["inputs"][config_key] = hyperparameters_s3_uri
        
        
        # Auto-detect inputs from dependencies if needed
        if dependencies:
            input_requirements = self.get_input_requirements()
            
            # Initialize inputs dictionary if not provided
            if not inputs:
                inputs = {}
                
            # Extract both regular inputs and hyperparameters_s3_uri from dependencies
            for dep_step in dependencies:
                matched = self._match_custom_properties(inputs, input_requirements, dep_step)
                if matched:
                    logger.info(f"Found inputs from dependency: {getattr(dep_step, 'name', str(dep_step))}")
            
            # If hyperparameters_s3_uri was found in inputs during matching, add it to the config channel
            config_key = self.config.input_names.get("config", "config") if "config" in self.config.input_names else "config"
            
            if not hyperparameters_s3_uri and 'hyperparameters_s3_uri' in inputs:
                # Extract the URI
                hyperparameters_s3_uri = inputs.pop('hyperparameters_s3_uri')
                logger.info(f"Found hyperparameters_s3_uri in inputs: {hyperparameters_s3_uri.expr if hasattr(hyperparameters_s3_uri, 'expr') else hyperparameters_s3_uri}")
                
                # Create inputs structure if needed
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                    
                # Add to config channel directly
                if config_key not in inputs["inputs"]:
                    inputs["inputs"][config_key] = hyperparameters_s3_uri
                    logger.info(f"Added hyperparameters to {config_key} input channel")
        
        # Validate we have required inputs - more comprehensive check
        input_sources = []
        
        # Check both flat and nested input formats
        if "inputs" in inputs and isinstance(inputs["inputs"], dict):
            input_sources.extend(list(inputs["inputs"].keys()))
        
        for key in inputs.keys():
            if key != "inputs" and key != "hyperparameters_s3_uri" and key != "dependencies" and key != "enable_caching":
                input_sources.append(key)
        
        # Get the configured train and val keys
        train_key = self.config.input_names.get("train", "train")
        val_key = self.config.input_names.get("val", "val")
        
        # Check for required channels
        missing_channels = []
        if train_key not in input_sources:
            missing_channels.append(f"training data ('{train_key}')")
        if val_key not in input_sources:
            missing_channels.append(f"validation data ('{val_key}')")
            
        if missing_channels:
            raise ValueError(f"Missing required input channels: {', '.join(missing_channels)}. "
                            f"Could not extract them from dependencies or provided inputs.")

        # Create and configure the estimator
        estimator = self._create_estimator()

        # Note: We don't set estimator.hyperparameters_file_s3_uri because our script
        # reads hyperparameters directly from the "config" input channel 

        # Convert the inputs dict to properly formatted TrainingInput objects
        train_inputs = self._get_training_inputs(inputs)
        logger.info(f"Final training inputs: {list(train_inputs.keys())}")

        step_name = self._get_step_name('XGBoostTraining')
        
        training_step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=train_inputs,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created TrainingStep with name: {training_step.name}")
        return training_step
