from typing import Dict, Optional, Any, List, Set, Union
from pathlib import Path
import logging

from sagemaker.workflow.steps import Step
from sagemaker.processing import ProcessingInput
from sagemaker.workflow.properties import Properties

# Import the customized step
from secure_ai_sandbox_workflow_python_sdk.mims_model_registration.mims_model_registration_processing_step import (
    MimsModelRegistrationProcessingStep,
)

from .config_mims_registration_step import ModelRegistrationConfig
from .builder_step_base import StepBuilderBase

# Register property paths for inputs to ModelRegistrationStep
# Following the standard pattern:
# 1. In config classes: output_names = {"logical_name": "DescriptiveValue"}
# 2. In pipeline code: output_value = step.config.output_names["logical_name"]
# 3. Properties access: step.properties.ProcessingOutputConfig.Outputs[output_value].S3Output.S3Uri

# Primary registration using the PackagedModel VALUE
StepBuilderBase.register_property_path(
    "PackagingStep", 
    "PackagedModel",                      # Use VALUE from output_names as property name
    "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"  
)

# Register with index-based access for robustness
StepBuilderBase.register_property_path(
    "PackagingStep", 
    "PackagedModel",                      # Use VALUE from output_names 
    "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"  
)

# Additional registrations for variants of step name, still using the standard pattern
StepBuilderBase.register_property_path(
    "Package", 
    "PackagedModel",                      # Use VALUE from output_names
    "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"  
)

StepBuilderBase.register_property_path(
    "Package", 
    "PackagedModel",                      # Use VALUE from output_names
    "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"  
)

StepBuilderBase.register_property_path(
    "ProcessingStep", 
    "PackagedModel",                      # Use VALUE from output_names
    "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"  
)

StepBuilderBase.register_property_path(
    "ProcessingStep", 
    "PackagedModel",                      # Use VALUE from output_names
    "properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri"  
)

# NOTE: For backward compatibility only - these do NOT follow the standard pattern
# but are kept to ensure existing pipelines continue to work
StepBuilderBase.register_property_path(
    "PackagingStep", 
    "packaged_model_output",              # Using KEY instead of VALUE (non-standard)
    "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"  
)

StepBuilderBase.register_property_path(
    "Package", 
    "packaged_model_output",              # Using KEY instead of VALUE (non-standard)
    "properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri"  
)

# Standard Pattern for Input/Output Naming:
# 1. In config classes:
#    output_names = {"logical_name": "DescriptiveValue"}  # VALUE used as key in outputs dict
#    input_names = {"DescriptiveValue": "ScriptInputName"} # KEY matches output VALUE
# 2. In this file:
#    - "GeneratedPayloadSamples" KEY in input_names matches OUTPUT DESCRIPTOR from PayloadStep
#    - "PackagedModel" KEY matches OUTPUT DESCRIPTOR from PackagingStep

# Register property path for payload step output
StepBuilderBase.register_property_path(
    "PayloadTestStep", 
    "GeneratedPayloadSamples",  # Changed to match KEY in input_names                                                   
    "properties.ProcessingOutputConfig.Outputs['GeneratedPayloadSamples'].S3Output.S3Uri"  
)

logger = logging.getLogger(__name__)


class ModelRegistrationStepBuilder(StepBuilderBase):
    """
    Builder for a Model Registration ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that registers a model with MIMS.
    """

    def __init__(
        self,
        config: ModelRegistrationConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the model registration step.

        Args:
            config: A ModelRegistrationConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, ModelRegistrationConfig):
            raise ValueError(
                "ModelRegistrationStepBuilder requires a ModelRegistrationConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: ModelRegistrationConfig = config

    def validate_configuration(self) -> None:
        """
        Validates the provided configuration to ensure all required fields for this
        specific step are present and valid before attempting to build the step.

        Raises:
            ValueError: If any required configuration is missing or invalid.
        """
        logger.info("Validating ModelRegistrationConfig...")
        
        # Validate required attributes that are actually defined in the config
        required_attrs = [
            'region',
            'model_registration_domain',
            'model_registration_objective',
            'framework',
            'inference_instance_type',
            'inference_entry_point',
            'source_model_inference_content_types',
            'source_model_inference_response_types',
            'source_model_inference_input_variable_list',
            'source_model_inference_output_variable_list'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelRegistrationConfig missing required attribute: {attr}")
        
        # Validate input names
        # Following standard pattern: input KEYs should match output descriptors from upstream steps
        if "PackagedModel" not in (self.config.input_names or {}):
            raise ValueError("input_names must contain key 'PackagedModel'")
        
        if "GeneratedPayloadSamples" not in (self.config.input_names or {}):  # Changed from "payload_sample"
            raise ValueError("input_names must contain key 'GeneratedPayloadSamples'")
        
        # Registration step has no outputs, so no validation needed for output_names
        
        logger.info("ModelRegistrationConfig validation succeeded.")

    def _get_processing_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """
        Constructs a list of ProcessingInput objects using the standardized helper methods.
        This defines the data channels for the processing job, mapping S3 locations
        to local directories inside the container.

        Args:
            inputs: A dictionary mapping logical input channel names to their S3 URIs 
                  or dynamic Step properties.

        Returns:
            A list of sagemaker.processing.ProcessingInput objects.
        """
        # Use the standard helper method for input validation
        # We'll need to do a custom validation due to backward compatibility options
        if not inputs:
            raise ValueError("Inputs dictionary is empty")
        
        # Get logical input names for better readability
        model_package_key = "PackagedModel"  # Changed from "packaged_model_output"
        payload_sample_key = "GeneratedPayloadSamples"  # Updated to match output descriptor
        
        # For backward compatibility
        payload_key = "payload_s3_key"
        payload_uri_key = "payload_s3_uri"
        
        # Map logical names to script input names using config.input_names
        model_script_param = self.config.input_names.get(model_package_key, "ModelPackage")
        payload_script_param = self.config.input_names.get(payload_sample_key, "PayloadSamples")
        
        self.log_info("Using script parameter '%s' for logical input '%s'", model_script_param, model_package_key)
        self.log_info("Using script parameter '%s' for logical input '%s'", payload_script_param, payload_sample_key)
        
        # Enhanced debugging for input structure
        self.log_info("Input structure before normalization: %s", list(inputs.keys()))
        if "inputs" in inputs:
            self.log_info("Nested inputs keys: %s", list(inputs['inputs'].keys()))
        
        # Check if we have a nested structure and normalize it
        if "inputs" in inputs and isinstance(inputs["inputs"], dict):
            self.log_info("Detected nested inputs structure - normalizing")
            nested_inputs = inputs["inputs"]
            
            # CRITICAL FIX: Always copy relevant keys to top level, even if they already exist there
            for key in [model_package_key, payload_sample_key, payload_key, payload_uri_key]:
                if key in nested_inputs:
                    inputs[key] = nested_inputs[key]
                    self.log_info("Normalized nested input: '%s' to top level (overwriting if exists)", key)
            
            # CRITICAL FIX: Additional fix for GeneratedPayloadSamples edge case
            # Sometimes payload is stored in a different key than we're looking for
            if payload_sample_key not in inputs and payload_sample_key not in nested_inputs:
                # If we have any keys that might contain payload data, try using them
                payload_candidates = []
                for key, value in nested_inputs.items():
                    if any(payload_term in key.lower() for payload_term in ["payload", "sample", "generated"]):
                        payload_candidates.append((key, value))
                        
                if payload_candidates:
                    key, value = payload_candidates[0]
                    inputs[payload_sample_key] = value
                    self.log_info("Found payload from alternate key '%s', normalized to '%s'", key, payload_sample_key)
        
        # Post-normalization debugging
        self.log_info("Input structure after normalization: %s", list(inputs.keys()))
        # Add comprehensive logging to track payload location
        for key in [payload_sample_key, payload_key, payload_uri_key]:
            if key in inputs:
                self.log_info("SUCCESS: Found payload key at top level: %s = %s", key, inputs[key])
            elif "inputs" in inputs and key in inputs["inputs"]:
                self.log_info("FOUND NESTED ONLY: Payload key %s found in nested structure but not at top level", key)
        
        # Validate required model package input
        if model_package_key not in inputs:
            raise ValueError(f"Must supply an S3 URI for '{model_package_key}' in 'inputs'")
        
        # Validate we have at least one form of payload input
        if (payload_sample_key not in inputs and 
            payload_key not in inputs and payload_uri_key not in inputs):
            raise ValueError(f"Must supply an S3 URI for either '{payload_sample_key}', '{payload_key}', or '{payload_uri_key}' in 'inputs'")
        
        # Define the input channels - use standard helper for model input
        processing_inputs = [
            self._create_standard_processing_input(
                model_package_key,  # Logical name
                inputs, 
                "/opt/ml/processing/input/model",
                s3_data_distribution_type="FullyReplicated",
                s3_input_mode="File"
            )
        ]
        
        # Handle payload input with preference order - use standard helper when possible
        if payload_sample_key in inputs:
            # Use standard helper for preferred payload sample key
            processing_inputs.append(
                self._create_standard_processing_input(
                    payload_sample_key,
                    inputs,
                    "/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        # Fallback to old keys for backward compatibility
        # We can still use the helper by passing the payload key directly rather than using input_names mapping
        elif payload_key in inputs:
            processing_inputs.append(
                ProcessingInput(
                    input_name=payload_script_param,  # Use the script param name from config
                    source=inputs[payload_key],
                    destination="/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        elif payload_uri_key in inputs:
            processing_inputs.append(
                ProcessingInput(
                    input_name=payload_script_param,  # Use the script param name from config
                    source=inputs[payload_uri_key],
                    destination="/opt/ml/processing/mims_payload",
                    s3_data_distribution_type="FullyReplicated",
                    s3_input_mode="File"
                )
            )
        
        return processing_inputs
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get input requirements from config's input_names
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
            "dependencies": self.COMMON_PROPERTIES["dependencies"],
            "performance_metadata_location": "Optional S3 location of performance metadata file"
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Note: The MimsModelRegistrationProcessingStep does not produce any accessible outputs.
        The step registers the model in MIMS as a side effect but doesn't create any
        output properties that can be referenced by subsequent steps.
        
        Returns:
            Empty dictionary since this step doesn't produce any outputs
        """
        # Registration step has no outputs
        return {}

    def _try_fallback_s3_config(self, model_package_key: str, inputs: Dict[str, Any], 
                                matched_inputs: Set[str]) -> Set[str]:
        """
        Fallback to constructing a path using pipeline_s3_loc from config.
        
        Args:
            model_package_key: Key to use in inputs dictionary
            inputs: Dictionary to add matched inputs to
            matched_inputs: Set of input names that were successfully matched
            
        Returns:
            Updated set of matched input names
        """
        try:
            # Find base_s3_loc in configs
            base_s3_loc = None
            
            # First try from self.config if available
            if hasattr(self, 'config') and hasattr(self.config, 'pipeline_s3_loc'):
                base_s3_loc = self.config.pipeline_s3_loc
                self.log_info("Using base_s3_loc from self.config: %s", base_s3_loc)
            
            # Alternatively check other configs
            if not base_s3_loc and hasattr(self, "config_map"):
                for cfg_name, cfg in self.config_map.items():
                    if hasattr(cfg, 'pipeline_s3_loc') and cfg.pipeline_s3_loc:
                        base_s3_loc = cfg.pipeline_s3_loc
                        self.log_info("Using base_s3_loc from config_map: %s", base_s3_loc)
                        break
            
            # If we found a base S3 location and have a pipeline name
            if base_s3_loc and hasattr(self.config, 'pipeline_name'):
                # Construct S3 path using pattern from generated outputs
                s3_uri = f"{base_s3_loc}/package/packaged_model_output"
                inputs["inputs"][model_package_key] = s3_uri
                matched_inputs.add("inputs")
                self.log_info("Connected packaged model using fallback path from config: %s", s3_uri)
                return matched_inputs
        except Exception as e:
            logger.debug(f"Config-based fallback failed: {e}")
        
        return matched_inputs
        
    def _handle_properties_list(self, outputs, model_package_key: str, inputs: Dict[str, Any], 
                               matched_inputs: Set[str]) -> Set[str]:
        """
        Special handler for PropertiesList objects to safely extract S3Uri
        
        Args:
            outputs: The PropertiesList object to handle
            model_package_key: Key to use in inputs dictionary
            inputs: Dictionary to add matched inputs to
            matched_inputs: Set of input names that were successfully matched
            
        Returns:
            Updated set of matched input names
        """
        try:
            # DIRECT APPROACH: Always try index-based access first for PropertiesList
            first_output = outputs[0]
            logger.info(f"First output type: {type(first_output).__name__}")
            
            if hasattr(first_output, "S3Output") and hasattr(first_output.S3Output, "S3Uri"):
                s3_uri = first_output.S3Output.S3Uri
                inputs["inputs"][model_package_key] = s3_uri
                matched_inputs.add("inputs")
                logger.info("Connected packaged model using direct index access [0] for PropertiesList")
                return matched_inputs
        except Exception as e:
            logger.info(f"Error accessing PropertiesList by index: {e}")
        
        return matched_inputs
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to ModelRegistration step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # Get step type name and step_name for better logging
        step_type = prev_step.__class__.__name__ 
        step_name = getattr(prev_step, 'name', str(prev_step))
        logger.info(f"Matching properties for step: {step_name}, type: {step_type}")
        
        # FIRST ATTEMPT: Special handling for PackagingStep connections
        # This is the most critical connection and needs dedicated logic
        if hasattr(prev_step, "name") and ("packaging" in prev_step.name.lower() or "package" in step_type.lower()):
            try:
                logger.info(f"Attempting to connect from packaging step: {prev_step.name}")
                # Direct connection for packaging step - try both methods
                if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                    outputs = prev_step.properties.ProcessingOutputConfig.Outputs
                    model_package_key = "PackagedModel"  # Changed from "packaged_model_output"
                    
                    # Initialize inputs dict if needed
                    if "inputs" not in inputs:
                        inputs["inputs"] = {}
                    
                    # CRITICAL ENHANCEMENT: Inspect the actual structure of the outputs collection
                    logger.info(f"Output collection type: {type(outputs).__name__}")
                    if hasattr(outputs, "__dict__"):
                        logger.info(f"Output collection attributes: {dir(outputs)}")
                    
                    # Get the output name that should be used from the packaging step config
                    # Try to find the output descriptor (VALUE) from the packaging step config
                    from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
                    
                    # Look for the correct output descriptor from packaging step's output_names
                    packaging_output_name = "PackagedModel"  # Default if can't find
                    
                    # If we can find the step's config directly, get the output descriptor VALUE
                    packaging_cfg = None
                    if hasattr(self, "config_map"):
                        for step_name, cfg in self.config_map.items():
                            if isinstance(cfg, PackageStepConfig):
                                packaging_cfg = cfg
                                break
                    
                    if packaging_cfg and hasattr(packaging_cfg, 'output_names'):
                        # Get the VALUE from output_names for packaged_model_output
                        packaging_output_name = packaging_cfg.output_names.get("packaged_model_output", packaging_output_name)
                        self.log_info("Found packaging output descriptor: %s from config", packaging_output_name)
                    
                    # CRITICAL FIX: Special handling for PropertiesList objects
                    if hasattr(outputs, "__class__") and outputs.__class__.__name__ == "PropertiesList":
                        logger.info("Detected PropertiesList - using safe access methods")
                        matched_inputs = self._handle_properties_list(outputs, model_package_key, inputs, matched_inputs)
                        if matched_inputs:
                            return matched_inputs
                            
                    # Try by dict key access (for non-PropertiesList objects)
                    if hasattr(outputs, "keys") and callable(outputs.keys):
                        try:
                            available_keys = list(outputs.keys())
                            logger.info(f"Available named outputs: {available_keys}")
                            
                            # Try our preferred keys in order
                            key_tries = [packaging_output_name]  # Start with the actual output descriptor from config
                            # Then try fallbacks
                            if packaging_output_name not in key_tries:
                                key_tries.append("PackagedModel")
                            key_tries.extend(["Model", "model", "packaged_model", "packaged"])
                            
                            logger.info(f"Trying keys in order: {key_tries}")
                            for key in key_tries:
                                try:
                                    if key in available_keys:  # Pre-check with keys we know exist
                                        output = outputs[key]
                                        if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                            s3_uri = output.S3Output.S3Uri
                                            inputs["inputs"][model_package_key] = s3_uri
                                            matched_inputs.add("inputs")
                                            logger.info(f"Connected packaged model using safe key lookup: {key}")
                                            return matched_inputs
                                except Exception as e:
                                    logger.debug(f"Safe key access failed for {key}: {e}")
                        except Exception as e:
                            logger.debug(f"Error checking output keys: {e}")
                    
                    # Try index access as fallback for any collection with __getitem__
                    if model_package_key not in inputs.get("inputs", {}) and hasattr(outputs, "__getitem__"):
                        try:
                            s3_uri = outputs[0].S3Output.S3Uri
                            inputs["inputs"][model_package_key] = s3_uri
                            matched_inputs.add("inputs")
                            logger.info("Connected packaged model using index access")
                            return matched_inputs
                        except Exception as e:
                            logger.debug(f"Index access failed: {e}")
                            
                    # FALLBACK S3 LOC: Use config for S3 path if needed
                    if model_package_key not in inputs.get("inputs", {}):
                        matched_inputs = self._try_fallback_s3_config(model_package_key, inputs, matched_inputs)
                        if matched_inputs:
                            return matched_inputs
            except Exception as e:
                logger.warning(f"Failed to connect packaging step output: {e}")
        
        # Try to find payload from payload_test step
        if hasattr(prev_step, "name") and "payload" in prev_step.name.lower():
            try:
                logger.info(f"Attempting to connect from payload step: {prev_step.name}")
                payload_key = "GeneratedPayloadSamples"
                
                # Check if the step properties has Outputs
                if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                    outputs = prev_step.properties.ProcessingOutputConfig.Outputs
                    
                    # CRITICAL ENHANCEMENT: Inspect the actual structure of the outputs collection
                    logger.info(f"Payload step output collection type: {type(outputs).__name__}")
                    if hasattr(outputs, "__dict__"):
                        logger.info(f"Payload step output collection attributes: {dir(outputs)}")
                        
                    # Get the output descriptor from config if possible
                    from src.pipeline_steps.config_mims_payload_step import PayloadConfig
                    
                    # Look for the output descriptor from payload step's output_names
                    payload_output_name = "GeneratedPayloadSamples"  # Default
                    
                    # If we can find the step's config directly
                    payload_cfg = None
                    if hasattr(self, "config_map"):
                        for step_name, cfg in self.config_map.items():
                            if isinstance(cfg, PayloadConfig):
                                payload_cfg = cfg
                                break
                    
                    if payload_cfg and hasattr(payload_cfg, 'output_names'):
                        # Get the VALUE from output_names
                        payload_output_name = payload_cfg.output_names.get("payload_sample", payload_output_name)
                        self.log_info("Found payload output descriptor: %s from config", payload_output_name)
                    
                    # CRITICAL FIX: Special handling for PropertiesList objects
                    if hasattr(outputs, "__class__") and outputs.__class__.__name__ == "PropertiesList":
                        logger.info("Detected PropertiesList for payload outputs - using safe access methods")
                        try:
                            # Try all possible output descriptors
                            key_tries = ["GeneratedPayloadSamples", "payload_sample", "PayloadSamples", payload_output_name]
                            
                            # First try direct property access by various keys
                            for key in key_tries:
                                try:
                                    if hasattr(outputs, key):
                                        output = getattr(outputs, key)
                                        if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                            s3_uri = output.S3Output.S3Uri
                                            if "inputs" not in inputs:
                                                inputs["inputs"] = {}
                                            inputs["inputs"][payload_key] = s3_uri
                                            matched_inputs.add("inputs")
                                            logger.info(f"Connected payload sample using direct attribute access with key: {key}")
                                            return matched_inputs
                                except Exception as e:
                                    logger.debug(f"Direct attribute access failed for key {key}: {e}")
                            
                            # Try index-based access for PropertiesList
                            try:
                                for i in range(min(len(outputs), 3)):  # Try first few outputs
                                    output = outputs[i]
                                    if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                        s3_uri = output.S3Output.S3Uri
                                        if "inputs" not in inputs:
                                            inputs["inputs"] = {}
                                        inputs["inputs"][payload_key] = s3_uri
                                        matched_inputs.add("inputs")
                                        self.log_info("Connected payload sample using index access: [%d]", i)
                                        return matched_inputs
                            except Exception as e:
                                logger.debug(f"Index-based access failed: {e}")
                                
                        except Exception as e:
                            logger.debug(f"All PropertiesList access methods failed: {e}")
                    
                    # Try by dict key access (for non-PropertiesList objects)
                    if hasattr(outputs, "keys") and callable(outputs.keys):
                        try:
                            available_keys = list(outputs.keys())
                            logger.info(f"Available payload output keys: {available_keys}")
                            
                            # Try all possible output keys
                            key_tries = [payload_output_name, "GeneratedPayloadSamples", "payload_sample"]
                            
                            for key in key_tries:
                                try:
                                    if key in available_keys:
                                        output = outputs[key]
                                        if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                            s3_uri = output.S3Output.S3Uri
                                            if "inputs" not in inputs:
                                                inputs["inputs"] = {}
                                            inputs["inputs"][payload_key] = s3_uri
                                            matched_inputs.add("inputs")
                                            logger.info(f"Connected payload sample using key lookup: {key}")
                                            return matched_inputs
                                except Exception as e:
                                    logger.debug(f"Key lookup failed for {key}: {e}")
                        except Exception as e:
                            logger.debug(f"Error checking output keys: {e}")
                                
                    # Try any access method as fallback
                    try:
                        # Try to get payload_sample from any available output
                        first_output = outputs[0]
                        if hasattr(first_output, "S3Output") and hasattr(first_output.S3Output, "S3Uri"):
                            s3_uri = first_output.S3Output.S3Uri
                            if "inputs" not in inputs:
                                inputs["inputs"] = {}
                                
                            # Store in nested structure
                            inputs["inputs"][payload_key] = s3_uri
                            
                            # CRITICAL FIX: Also store at top level
                            inputs[payload_key] = s3_uri
                            
                            matched_inputs.add("inputs")
                            logger.info(f"Connected payload sample using first output fallback")
                            self.log_info("CRITICAL: Storing payload at both nested and top level: %s=%s", payload_key, s3_uri)
                            return matched_inputs
                    except Exception as e:
                        logger.debug(f"Error accessing payload output: {e}")
            except Exception as e:
                logger.warning(f"Failed to connect payload output: {e}")
        
        # Use registered property paths as a fallback
        step_property_paths = {}
        for registered_type in [step_type, step_name.replace("-", ""), "Package", "PackagingStep", "ProcessingStep"]:
            if registered_type in self._PROPERTY_PATH_REGISTRY:
                # Add all property paths for this step type to our search list
                step_property_paths.update(self._PROPERTY_PATH_REGISTRY[registered_type])
                
        # Try to extract values using registered property paths
        for logical_name, property_path in step_property_paths.items():
            try:
                # Skip if not a model registration input or if already matched
                if logical_name not in self.config.input_names:
                    continue
                    
                if "inputs" in inputs and logical_name in inputs["inputs"]:
                    continue
                
                # Dynamically evaluate the property path
                path_parts = property_path.split('.')
                value = prev_step
                for part in path_parts:
                    # Handle dictionary access with quotes: ['key']
                    if part.startswith("[") and part.endswith("]"):
                        key = part[2:-2]  # Remove ["..."]
                        value = value[key]
                    # Handle regular attribute access
                    else:
                        value = getattr(value, part)
                
                # Add extracted value to inputs
                if "inputs" not in inputs:
                    inputs["inputs"] = {}
                
                inputs["inputs"][logical_name] = value
                matched_inputs.add("inputs")
                logger.info(f"Found {logical_name} using registered property path")
            except Exception as e:
                logger.debug(f"Could not extract {logical_name} using property path: {e}")
                
        return matched_inputs
    
    def create_step(self, **kwargs) -> Step:
        """
        Creates a specialized MimsModelRegistrationProcessingStep for the pipeline.
        This method orchestrates the assembly of the inputs and configuration
        into a single, executable pipeline step.

        Note: The MimsModelRegistrationProcessingStep does not define property files (outputs)
        that can be referenced by subsequent steps in the pipeline. It registers the model in MIMS
        as a side effect but doesn't produce output artifacts that can be accessed through properties.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: A dictionary mapping input channel names to their sources (S3 URIs or Step properties).
                - OR individual parameters:
                  - packaged_model_output: S3 URI of the packaged model
                  - payload_s3_key: S3 key for the payload
                  - payload_s3_uri: S3 URI for the payload (alternative to payload_s3_key)
                - dependencies: Optional list of steps that this step depends on.
                - performance_metadata_location: Optional S3 location of performance metadata file.
                  If not provided, no performance metadata will be used.
                - regions: Optional list of regions to register the model in.

        Returns:
            A configured MimsModelRegistrationProcessingStep instance that registers the model in MIMS.
        """
        logger.info("Creating MimsModelRegistrationProcessingStep...")

        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        performance_metadata_location = self._extract_param(kwargs, 'performance_metadata_location')
        
        # Check if individual input parameters were provided instead of 'inputs' dictionary
        packaged_model_output = self._extract_param(kwargs, 'packaged_model_output')
        payload_s3_key = self._extract_param(kwargs, 'payload_s3_key')
        payload_s3_uri = self._extract_param(kwargs, 'payload_s3_uri')
        
        # Extract parameters using standardized keys
        packaged_model = self._extract_param(kwargs, 'PackagedModel')
        generated_payload_samples = self._extract_param(kwargs, 'GeneratedPayloadSamples')  # Updated key
        payload_sample = self._extract_param(kwargs, 'payload_sample')  # Keep for backward compatibility
        
        # If individual parameters were provided, build the inputs dictionary
        if not inputs and (packaged_model or packaged_model_output or generated_payload_samples or payload_sample or payload_s3_key or payload_s3_uri):
            inputs = {}
            # Prefer the new key name if provided
            if packaged_model:
                inputs["PackagedModel"] = packaged_model
            # Fall back to old key name for backward compatibility
            elif packaged_model_output:
                inputs["PackagedModel"] = packaged_model_output  # Use new key for internal consistency
                
            # Prefer the new standardized key name for payload
            if generated_payload_samples:
                inputs["GeneratedPayloadSamples"] = generated_payload_samples
            # Fall back to old key name for backward compatibility
            elif payload_sample:
                inputs["payload_sample"] = payload_sample
                
            # Keep support for older payload formats
            if payload_s3_key:
                inputs["payload_s3_key"] = payload_s3_key
            if payload_s3_uri:
                inputs["payload_s3_uri"] = payload_s3_uri
        
        # Validate required parameters
        if not inputs:
            raise ValueError("Either 'inputs' dictionary or individual 'packaged_model_output' and 'payload_s3_key'/'payload_s3_uri' must be provided")

        # Get processing inputs
        processing_inputs = self._get_processing_inputs(inputs)

        # Create step name
        step_name = f"{self._get_step_name('Registration')}-{self.config.region}"
        
        # Create the specialized step
        try:
            registration_step = MimsModelRegistrationProcessingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session,
                processing_input=processing_inputs,  # This parameter name matches the expected signature
                performance_metadata_location=performance_metadata_location,
                depends_on=dependencies or []
            )
            
            logger.info(f"Created MimsModelRegistrationProcessingStep with name: {registration_step.name}")
            return registration_step
            
        except Exception as e:
            logger.error(f"Error creating MimsModelRegistrationProcessingStep: {e}")
            raise ValueError(f"Failed to create MimsModelRegistrationProcessingStep: {e}") from e
