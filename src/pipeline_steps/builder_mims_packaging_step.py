from typing import Dict, Optional, Any, List, Set
from pathlib import Path
import logging

from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor

from .config_mims_packaging_step import PackageStepConfig
from .builder_step_base import StepBuilderBase

# Register property path for MIMS packaging output - using the VALUE from output_names
# This follows the standard pattern:
# 1. In config classes: output_names = {"logical_name": "DescriptiveValue"}
# 2. In step builders: use the VALUE ("PackagedModel") as the property path name
StepBuilderBase.register_property_path(
    "PackagingStep", 
    "PackagedModel",                             # VALUE from output_names as property path name
    "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"  # Runtime access path
)

# Also register with name-based access path for better robustness
StepBuilderBase.register_property_path(
    "PackagingStep", 
    "PackagedModel",                             # VALUE from output_names as property path name
    "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"  # Name-based path
)

# Register for backward compatibility, though this isn't following the standard pattern
StepBuilderBase.register_property_path(
    "PackagingStep", 
    "packaged_model_output",                     # Logical name (KEY) in output_names
    "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"  # Runtime access path
)

logger = logging.getLogger(__name__)


class MIMSPackagingStepBuilder(StepBuilderBase):
    """
    Builder for a MIMS Packaging ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that packages a model for MIMS registration.
    """

    def __init__(
        self,
        config: PackageStepConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the MIMS packaging step.

        Args:
            config: A PackageStepConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, PackageStepConfig):
            raise ValueError(
                "MIMSPackagingStepBuilder requires a PackageStepConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: PackageStepConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating PackageStepConfig...")
        
        # Validate processing attributes from the base class
        required_attrs = [
            'processing_instance_count',
            'processing_volume_size',
            'processing_entry_point',
            'processing_source_dir',
            'processing_framework_version',
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"PackageStepConfig missing required attribute: {attr}")
        
        # Validate instance type settings
        if not hasattr(self.config, 'processing_instance_type_large'):
            raise ValueError("Missing required attribute: processing_instance_type_large")
        if not hasattr(self.config, 'processing_instance_type_small'):
            raise ValueError("Missing required attribute: processing_instance_type_small")
        if not hasattr(self.config, 'use_large_processing_instance'):
            raise ValueError("Missing required attribute: use_large_processing_instance")
        
        # Validate input and output names
        if "model_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'model_input'")
        
        if "inference_scripts_input" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'inference_scripts_input'")
        
        if "packaged_model_output" not in (self.config.output_names or {}):
            raise ValueError("output_names must contain key 'packaged_model_output'")
        
        logger.info("PackageStepConfig validation succeeded.")

    def _create_processor(self) -> SKLearnProcessor:
        """
        Creates and configures the SKLearnProcessor for the SageMaker Processing Job.
        This defines the execution environment for the script, including the instance
        type and framework version.

        Returns:
            An instance of sagemaker.sklearn.SKLearnProcessor.
        """
        # Get the appropriate instance type based on use_large_processing_instance
        instance_type = self.config.processing_instance_type_large if self.config.use_large_processing_instance else self.config.processing_instance_type_small
        
        return SKLearnProcessor(
            framework_version=self.config.processing_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('MIMSPackaging')}"
            ),
            sagemaker_session=self.session,
            env={},  # No environment variables needed for this script
        )

    def _get_processor_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects from the provided inputs dictionary.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names (e.g., 'model_input')
                    to their S3 URIs or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Check if inputs is empty
        if not inputs:
            raise ValueError("Inputs dictionary is empty. Must supply required inputs.")
        
        # Validate required inputs against KEYS in input_names (standard pattern)
        required_input_keys = ["model_input", "inference_scripts_input"]
        missing_inputs = [key for key in required_input_keys if key not in inputs]
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {', '.join(missing_inputs)}")

        # Define the input channels using script input names from VALUES in input_names
        processing_inputs = [
            ProcessingInput(
                input_name=self.config.input_names["model_input"],  # Script expects this name
                source=inputs["model_input"],  # Use LOGICAL name for lookup
                destination="/opt/ml/processing/input/model"
            ),
            ProcessingInput(
                input_name=self.config.input_names["inference_scripts_input"],  # Script name
                source=inputs["inference_scripts_input"],  # Use LOGICAL name for lookup
                destination="/opt/ml/processing/input/script"
            )
        ]
        
        return processing_inputs

    def _get_processor_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """
        Constructs the ProcessingOutput objects needed for this step.
        This defines the S3 location where the results of the processing job will be stored.

        Args:
            outputs: A dictionary mapping output descriptors (VALUES from output_names)
                     to their S3 destination URIs.

        Returns:
            A list containing sagemaker.processing.ProcessingOutput objects.
        """
        # Following standard pattern:
        # 1. Get logical_name first
        logical_name = "packaged_model_output"
        
        # 2. Get VALUE from output_names using logical_name
        # This follows the standard pattern: output_value = step.config.output_names["logical_name"]
        output_descriptor = self.config.output_names[logical_name]
        
        # 3. Validate that outputs contains the VALUE
        # For outputs: validate using VALUES
        if not outputs:
            raise ValueError("Outputs dictionary is empty. Must supply outputs.")
            
        # Log available keys in outputs for debugging
        logger.info(f"Available keys in outputs dictionary: {list(outputs.keys())}")
        
        # Check if our required output descriptor exists
        if output_descriptor not in outputs:
            # Extended error with more diagnostic info
            available = ', '.join(list(outputs.keys()))
            raise ValueError(f"Must supply an S3 URI for '{output_descriptor}' in 'outputs'. Available keys are: {available}")
        
        # 4. Define the output using the VALUE as the output_name and key in outputs dict
        logger.info(f"Creating ProcessingOutput with output_name='{output_descriptor}' (value from output_names)")
        processing_outputs = [
            ProcessingOutput(
                output_name=output_descriptor,  # Use VALUE from output_names
                source="/opt/ml/processing/output",
                destination=outputs[output_descriptor]  # Use VALUE as key in outputs dict
            )
        ]
        
        logger.info(f"Created ProcessingOutput with output_name='{output_descriptor}' and destination={outputs[output_descriptor]}")
        return processing_outputs

        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
            "outputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.output_names or {}).keys()])} S3 paths",
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
        return {k: v for k, v in (self.config.output_names or {}).items()}
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to MIMSPackaging step.
        Uses a generic approach to find model artifacts from standard SageMaker step types.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        step_name = getattr(prev_step, 'name', str(prev_step))

        # Register outputs using VALUES from output_names for future steps to find
        # This ensures our outputs can be discovered following standard pattern
        if hasattr(self.config, "output_names"):
            # Log the complete output_names mapping for clarity
            logger.info(f"Registering property paths from output_names: {self.config.output_names}")
            
            for logical_name, output_descriptor in self.config.output_names.items():
                # Register using the VALUE for lookups as per standard pattern
                StepBuilderBase.register_property_path(
                    type(self).__name__,
                    output_descriptor,  # Use VALUE as the property path name
                    f"properties.ProcessingOutputConfig.Outputs['{output_descriptor}'].S3Output.S3Uri"
                )
                logger.info(f"Registered property path for {output_descriptor} (logical name: {logical_name}) following standard pattern")
                
                # Also register using name-based path for better robustness
                StepBuilderBase.register_property_path(
                    type(self).__name__,
                    output_descriptor,  # Use VALUE as the property path name
                    f"properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
                )
                logger.info(f"Registered index-based property path for {output_descriptor}")
                
                # Register property paths under more general step types for better matching
                for step_type in ["Package", "PackagingStep", "ProcessingStep"]:
                    # Register with name-based access
                    StepBuilderBase.register_property_path(
                        step_type,
                        output_descriptor,  # Use VALUE as property path name
                        f"properties.ProcessingOutputConfig.Outputs['{output_descriptor}'].S3Output.S3Uri"
                    )
                    
                    # Register with index-based access
                    StepBuilderBase.register_property_path(
                        step_type,
                        output_descriptor,  # Use VALUE as property path name
                        f"properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"
                    )
                
                logger.info(f"Registered property paths for {output_descriptor} under multiple step types")
        
        # PART 1: Try TrainingStep standard paths
        found_model = False
        
        # Log available properties for debugging
        if hasattr(prev_step, "properties"):
            logger.info(f"Available properties for {step_name}: {dir(prev_step.properties)}")
            
            # Add additional debug logging for model artifacts
            if hasattr(prev_step.properties, "ModelArtifacts"):
                logger.info(f"ModelArtifacts properties: {dir(prev_step.properties.ModelArtifacts)}")
        
        # Most common path for trained model artifacts from TrainingStep
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ModelArtifacts"):
            try:
                model_artifacts = prev_step.properties.ModelArtifacts.S3ModelArtifacts
                model_input_key = "model_input"
                inputs[model_input_key] = model_artifacts
                matched_inputs.add(model_input_key)
                logger.info(f"Found model artifacts from TrainingStep path: {step_name}")
                found_model = True
            except AttributeError as e:
                logger.info(f"Error accessing ModelArtifacts.S3ModelArtifacts: {e}")
        
        # XGBoost-specific model output paths - check direct properties
        if not found_model and "xgboost" in step_name.lower():
            # Try direct attribute access for common model output properties
            model_attrs = ["model_data", "output_path", "ModelOutputPath", "ModelArtifacts"]
            for attr in model_attrs:
                if hasattr(prev_step, attr):
                    try:
                        model_uri = getattr(prev_step, attr)
                        model_input_key = "model_input"
                        inputs[model_input_key] = model_uri
                        matched_inputs.add(model_input_key)
                        self.log_info("Found model via direct attribute: %s", attr)
                        found_model = True
                        break
                    except Exception as e:
                        logger.info(f"Error accessing {attr}: {e}")
        
        # PART 2: Try ProcessingStep standard paths (useful for packaging steps)
        if not found_model and hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
            try:
                outputs = prev_step.properties.ProcessingOutputConfig.Outputs
                
                # Log the type of outputs object to help with debugging
                logger.info(f"Processing outputs of type: {outputs.__class__.__name__ if hasattr(outputs, '__class__') else type(outputs)}")
                
                # Special handling for PropertiesList type
                if hasattr(outputs, "__class__") and outputs.__class__.__name__ == "PropertiesList":
                    logger.info("Detected PropertiesList object - using direct attribute access")
                    
                    # Try common output names that might contain processed data
                    model_output_names = ["ProcessedTabularData", "ModelOutputPath", "ModelArtifacts"]
                    for name in model_output_names:
                        if hasattr(outputs, name):
                            try:
                                s3_uri = outputs[name].S3Output.S3Uri
                                model_input_key = "model_input"
                                inputs[model_input_key] = s3_uri
                                matched_inputs.add(model_input_key)
                                self.log_info("Found model from PropertiesList attribute %s: %s", name, s3_uri)
                                found_model = True
                                break
                            except (AttributeError, KeyError) as e:
                                logger.debug(f"Error accessing PropertiesList attribute {name}: {e}")
                                
                # For indexed outputs - use safe iteration
                if not found_model and hasattr(outputs, "__getitem__"):
                    try:
                        # Try getting first element safely with iteration
                        first_item = next(iter(outputs), None)
                        if first_item is not None:
                            if hasattr(first_item, "S3Output") and hasattr(first_item.S3Output, "S3Uri"):
                                s3_uri = first_item.S3Output.S3Uri
                                model_input_key = "model_input"
                                inputs[model_input_key] = s3_uri
                                matched_inputs.add(model_input_key)
                                logger.info(f"Found model from ProcessingStep first output: {step_name}")
                                found_model = True
                    except (TypeError, StopIteration) as e:
                        logger.debug(f"Error getting first item from outputs: {e}")
                
                # Try by name if available (for model-related outputs)
                if not found_model:
                    model_output_names = ["model", "model_output", "packaged_model", "artifact"]
                    for out_name in model_output_names:
                        try:
                            if out_name in outputs and hasattr(outputs[out_name], "S3Output"):
                                s3_uri = outputs[out_name].S3Output.S3Uri
                                model_input_key = "model_input"
                                inputs[model_input_key] = s3_uri
                                matched_inputs.add(model_input_key)
                                logger.info(f"Found model from ProcessingStep named output: {out_name}")
                                found_model = True
                                break
                        except (TypeError, KeyError) as e:
                            logger.debug(f"Error accessing output {out_name}: {e}")
            except Exception as e:
                logger.warning(f"Error accessing ProcessingStep outputs: {e}")
        
        # PART 3: Try direct attribute access for common model properties
        if not found_model:
            # Direct attribute names that might contain model artifacts
            model_attr_names = ["model_data", "model_uri", "model_path", "ModelOutputPath", "model_artifacts_path"]
            
            for attr_name in model_attr_names:
                if hasattr(prev_step, attr_name):
                    try:
                        model_uri = getattr(prev_step, attr_name)
                        if model_uri is not None:
                            model_input_key = "model_input"
                            inputs[model_input_key] = model_uri
                            matched_inputs.add(model_input_key)
                            self.log_info("Found model via direct attribute: %s", attr_name)
                            found_model = True
                            break
                    except Exception:
                        continue
        
        # PART 4: Try the outputs collection (used by some step types)
        if not found_model and hasattr(prev_step, "outputs"):
            try:
                # Look for model-related output names
                for output in prev_step.outputs:
                    if hasattr(output, "output_name") and hasattr(output, "destination"):
                        output_name = output.output_name.lower()
                        if any(term in output_name for term in ["model", "artifact", "packaged"]):
                            model_input_key = "model_input"
                            inputs[model_input_key] = output.destination
                            matched_inputs.add(model_input_key)
                            logger.info(f"Found model in outputs collection: {output.output_name}")
                            found_model = True
                            break
            except Exception as e:
                logger.warning(f"Error checking step outputs: {e}")
        
        # PART 5: Always add inference_scripts_input
        inference_scripts_key = "inference_scripts_input"
        if inference_scripts_key not in inputs:
            inference_scripts_path = self.config.source_dir
            if not inference_scripts_path:
                inference_scripts_path = str(self.notebook_root / "inference") if self.notebook_root else "inference"
            
            inputs[inference_scripts_key] = inference_scripts_path
            matched_inputs.add(inference_scripts_key)
            logger.info(f"Using inference scripts path: {inference_scripts_path}")
        
        return matched_inputs
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates the final, fully configured SageMaker ProcessingStep for the pipeline.
        This method orchestrates the assembly of the processor, inputs, outputs, and
        script arguments into a single, executable pipeline step.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                  Can be nested (e.g., {'inputs': {'model_input': uri}}) or flat (e.g., {'model_input': uri}).
                - model_input: Direct parameter for model input URI (alternative to nested inputs).
                - inference_scripts_input: Direct parameter for inference scripts path (alternative to nested inputs).
                - outputs: A dictionary mapping output channel names to their S3 destinations.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step
                                to speed up subsequent pipeline runs with the same inputs.

        Returns:
            A configured sagemaker.workflow.steps.ProcessingStep instance.
        """
        logger.info("Creating MIMSPackaging ProcessingStep...")

        # Extract parameters
        inputs_raw = self._extract_param(kwargs, 'inputs')
        outputs = self._extract_param(kwargs, 'outputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        # Normalize inputs - handles both nested and flat structures
        # This ensures inputs work regardless of whether they're passed as:
        # 1. inputs={'model_input': uri, 'inference_scripts_input': path}
        # 2. inputs={'inputs': {'model_input': uri, 'inference_scripts_input': path}}
        # 3. model_input=uri, inference_scripts_input=path (direct kwargs)
        inputs = self._normalize_inputs(inputs_raw)
        
        # Add direct parameters if provided (overriding any in inputs)
        for param in ['model_input', 'inference_scripts_input']:
            if param in kwargs:
                inputs[param] = kwargs[param]
                logger.debug(f"Added direct parameter: {param}")
        
        # Log the normalized inputs for debugging
        logger.debug(f"Normalized inputs: {list(inputs.keys())}")
        
        # Validate required parameters
        if not inputs:
            raise ValueError("inputs must be provided")
        if not outputs:
            raise ValueError("outputs must be provided")

        processor = self._create_processor()
        proc_inputs = self._get_processor_inputs(inputs)
        proc_outputs = self._get_processor_outputs(outputs)
        # Remove job_arguments completely as the script doesn't need any arguments

        step_name = self._get_step_name('Package')
        
        processing_step = ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=self.config.get_script_path(),
            # job_arguments parameter removed
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
        logger.info(f"Created ProcessingStep with name: {processing_step.name}")
        return processing_step
