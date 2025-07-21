#!/usr/bin/env python
"""Builder for ModelCalibration processing step.

This module defines the ModelCalibrationStepBuilder class that builds a SageMaker
ProcessingStep for model calibration, connecting the configuration, specification, 
and script contract.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.entities import PipelineVariable

from ..pipeline_deps.base_specifications import StepSpecification
from ..pipeline_script_contracts.base_script_contract import ScriptContract
from .builder_step_base import StepBuilderBase
from .config_model_calibration_step import ModelCalibrationConfig
from ..pipeline_step_specs.model_calibration_spec import MODEL_CALIBRATION_SPEC

logger = logging.getLogger(__name__)

class ModelCalibrationStepBuilder(StepBuilderBase):
    """Builder for ModelCalibration processing step.
    
    This class builds a SageMaker ProcessingStep that calibrates model prediction
    scores to accurate probabilities. Calibration is essential for ensuring that
    prediction scores reflect true probabilities, which is crucial for reliable
    decision-making based on model outputs.
    """
    
    def __init__(
        self, 
        config, 
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        """Initialize the ModelCalibrationStepBuilder.
        
        Args:
            config: Configuration object for this step
            sagemaker_session: SageMaker session
            role: IAM role for SageMaker execution
            notebook_root: Root directory for notebooks
            registry_manager: Registry manager for steps
            dependency_resolver: Resolver for step dependencies
            
        Raises:
            ValueError: If config is not a ModelCalibrationConfig instance
        """
        if not isinstance(config, ModelCalibrationConfig):
            raise ValueError("ModelCalibrationStepBuilder requires a ModelCalibrationConfig instance.")
            
        super().__init__(
            config=config,
            spec=MODEL_CALIBRATION_SPEC,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: ModelCalibrationConfig = config
        
    def validate_configuration(self) -> None:
        """Validate the provided configuration.
        
        This method performs comprehensive validation of all configuration parameters,
        ensuring they meet the requirements for the calibration step.
        
        Raises:
            ValueError: If any configuration validation fails
        """
        self.log_info("Validating ModelCalibrationConfig...")
        
        # Validate required attributes
        required_attrs = [
            'processing_entry_point',
            'processing_source_dir',
            'processing_instance_count',
            'processing_volume_size',
            'calibration_method',
            'label_field',
            'score_field'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"ModelCalibrationConfig missing required attribute: {attr}")
        
        # Validate calibration method
        valid_methods = ['gam', 'isotonic', 'platt']
        if self.config.calibration_method.lower() not in valid_methods:
            raise ValueError(f"Invalid calibration method: {self.config.calibration_method}. "
                            f"Must be one of: {valid_methods}")
        
        # Validate numeric parameters
        if self.config.gam_splines <= 0:
            raise ValueError(f"gam_splines must be > 0, got {self.config.gam_splines}")
            
        if not 0 <= self.config.error_threshold <= 1:
            raise ValueError(f"error_threshold must be between 0 and 1, got {self.config.error_threshold}")
            
        # Check if script exists if notebook_root is provided
        if self.notebook_root:
            script_dir = Path(self.notebook_root) / self.config.processing_source_dir
            script_path = script_dir / self.config.processing_entry_point
            if not script_path.exists() and not self._is_pipeline_variable(script_path):
                self.log_warning(f"Script not found at {script_path}. It should exist before pipeline execution.")
            
        self.log_info("ModelCalibrationConfig validation succeeded.")
    
    def _is_pipeline_variable(self, value: Any) -> bool:
        """Check if a value is a PipelineVariable.
        
        Args:
            value: Value to check
            
        Returns:
            bool: True if the value is a PipelineVariable, False otherwise
        """
        return isinstance(value, PipelineVariable) or (hasattr(value, "expr") and callable(getattr(value, "expr", None)))
    
    def _normalize_s3_uri(self, uri: Union[str, PipelineVariable]) -> Union[str, PipelineVariable]:
        """Normalize S3 URI, handling PipelineVariable objects.
        
        This method handles both regular strings and PipelineVariable objects,
        providing a consistent interface for S3 URI handling throughout the builder.
        
        Args:
            uri: The S3 URI or PipelineVariable to normalize
            
        Returns:
            Union[str, PipelineVariable]: The normalized URI
            
        Raises:
            TypeError: If uri is not a string or PipelineVariable
        """
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            return uri
            
        # Handle PipelineVariable objects
        if self._is_pipeline_variable(uri):
            return uri
            
        if not isinstance(uri, str):
            raise TypeError(f"Expected string or PipelineVariable, got {type(uri)}")
            
        # Normalize string URI
        return uri
    
    def _get_s3_directory_path(self, s3_uri: Union[str, PipelineVariable, dict]) -> Union[str, PipelineVariable, dict]:
        """Ensure S3 URI is a directory path (ends with '/').
        
        This method is important for ensuring consistent directory path handling
        when working with S3 URIs in the pipeline.
        
        Args:
            s3_uri: The S3 URI to process
            
        Returns:
            Union[str, PipelineVariable, dict]: The normalized URI with trailing slash
        """
        # Handle Pipeline step references with Get key
        if isinstance(s3_uri, dict) and 'Get' in s3_uri:
            return s3_uri
            
        # Handle PipelineVariable objects
        if self._is_pipeline_variable(s3_uri):
            # We can't modify PipelineVariable directly, so return as is
            # The processing logic should handle directory paths appropriately
            return s3_uri
            
        # Normalize string URI
        normalized_uri = str(s3_uri)
        if not normalized_uri.endswith('/'):
            normalized_uri += '/'
            
        return normalized_uri
    
    def _validate_s3_uri(self, uri: Union[str, PipelineVariable, dict]) -> Union[str, PipelineVariable, dict]:
        """Validate that a given URI is a valid S3 URI.
        
        This method ensures that S3 URIs are properly formatted to prevent errors
        during pipeline execution.
        
        Args:
            uri: The URI to validate
            
        Returns:
            Union[str, PipelineVariable, dict]: The validated URI
            
        Raises:
            ValueError: If URI doesn't start with s3:// and is not a PipelineVariable or Get reference
        """
        # Handle Pipeline step references with Get key
        if isinstance(uri, dict) and 'Get' in uri:
            return uri
            
        # Handle PipelineVariable objects
        if self._is_pipeline_variable(uri):
            return uri
            
        # Validate string URI
        normalized_uri = str(uri)
        if not normalized_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")
            
        return uri
    
    def _detect_circular_references(self, var: Any, visited: Optional[Set] = None) -> bool:
        """Detect circular references in PipelineVariable objects.
        
        This method checks for circular references that could cause infinite recursion
        or other issues during pipeline execution.
        
        Args:
            var: The variable to check
            visited: Set of already visited objects (used for recursion)
            
        Returns:
            bool: True if a circular reference is detected, False otherwise
        """
        if visited is None:
            visited = set()
            
        if id(var) in visited:
            return True
        
        if self._is_pipeline_variable(var):
            visited.add(id(var))
            # Check for circular references in any dependent variables
            for dep in getattr(var, "_dependencies", []):
                if self._detect_circular_references(dep, visited):
                    return True
                    
        # For dictionaries, check values for circular references
        elif isinstance(var, dict):
            for key, value in var.items():
                if key == 'Get':  # Skip Get references
                    continue
                if self._detect_circular_references(value, visited.copy()):
                    return True
        
        return False
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor.
        
        This method creates a dictionary of environment variables needed by the
        calibration script, combining base variables with calibration-specific ones.
        
        Returns:
            Dict[str, str]: Environment variables dictionary
        """
        env_vars = super()._get_environment_variables()
        
        # Add calibration-specific environment variables
        env_vars.update({
            "CALIBRATION_METHOD": self.config.calibration_method.lower(),
            "LABEL_FIELD": self.config.label_field,
            "SCORE_FIELD": self.config.score_field,
            "MONOTONIC_CONSTRAINT": str(self.config.monotonic_constraint).lower(),
            "GAM_SPLINES": str(self.config.gam_splines),
            "ERROR_THRESHOLD": str(self.config.error_threshold)
        })
        
        return env_vars
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract.
        
        This method maps logical input names from the step specification to
        SageMaker ProcessingInput objects required by the processing script.
        
        Args:
            inputs: Dictionary of input values
            
        Returns:
            List[ProcessingInput]: List of configured ProcessingInput objects
            
        Raises:
            ValueError: If spec or contract is missing
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for input mapping")
        
        # Check for circular references in PipelineVariable inputs
        for input_name, input_value in inputs.items():
            if self._detect_circular_references(input_value):
                raise ValueError(f"Circular reference detected in input '{input_name}'")
            
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract.
        
        This method maps logical output names from the step specification to
        SageMaker ProcessingOutput objects that will be produced by the processing script.
        
        Args:
            outputs: Dictionary of output values
            
        Returns:
            List[ProcessingOutput]: List of configured ProcessingOutput objects
            
        Raises:
            ValueError: If spec or contract is missing
        """
        if not self.spec:
            raise ValueError("Step specification is required")
            
        if not self.contract:
            raise ValueError("Script contract is required for output mapping")
            
        return self._get_spec_driven_processor_outputs(outputs)
    
    def _get_processor(self) -> ScriptProcessor:
        """Create and configure the processor for this step.
        
        Returns:
            ScriptProcessor: The configured processor for running the step
        """
        return ScriptProcessor(
            image_uri=self.config.processing_image_uri,
            command=["python3"],
            instance_type=self.config.processing_instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            role=self.role,
            sagemaker_session=self.session,
            base_job_name=self._sanitize_name_for_sagemaker(
                f"{self._get_step_name('ModelCalibration')}"
            )
        )
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the model calibration processing step.
        
        This is the primary method for building the SageMaker ProcessingStep that will
        execute the calibration logic. It configures all necessary inputs, outputs,
        environment variables, and resources based on the step specification and configuration.
        
        Args:
            **kwargs: Additional keyword arguments for step creation.
                     Should include 'dependencies' list if step has dependencies.
                     
        Returns:
            ProcessingStep: The configured model calibration processing step
            
        Raises:
            ValueError: If any issues occur during step creation
        """
        try:
            self.log_info("Creating ModelCalibration ProcessingStep...")
            
            # Extract parameters
            inputs_raw = kwargs.get('inputs', {})
            outputs = kwargs.get('outputs', {})
            dependencies = kwargs.get('dependencies', [])
            enable_caching = kwargs.get('enable_caching', True)
            
            # Handle inputs
            inputs = {}
            
            # If dependencies are provided, extract inputs from them
            if dependencies:
                try:
                    extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
                    inputs.update(extracted_inputs)
                except Exception as e:
                    self.log_warning(f"Failed to extract inputs from dependencies: {str(e)}")
                    
            # Add explicitly provided inputs
            inputs.update(inputs_raw)
            
            # Get processor inputs and outputs
            processor_inputs = self._get_inputs(inputs)
            processor_outputs = self._get_outputs(outputs)
            
            # Create processor
            processor = self._get_processor()
            
            # Get environment variables
            env_vars = self._get_environment_variables()
            
            # Get step name and script path
            step_name = self._get_step_name()
            script_path = self.config.get_script_path()
            
            # Create step
            step = ProcessingStep(
                name=step_name,
                processor=processor,
                inputs=processor_inputs,
                outputs=processor_outputs,
                code=self.config.processing_source_dir,
                job_name=self._generate_job_name('ModelCalibration'),
                container_entrypoint=["python3", script_path],
                container_arguments=[],
                depends_on=dependencies,
                cache_config=self._get_cache_config(enable_caching),
                environment=env_vars
            )
            
            # Attach specification to the step
            setattr(step, '_spec', self.spec)
                
            self.log_info(f"Created ProcessingStep with name: {step.name}")
            return step
            
        except Exception as e:
            self.log_error(f"Error creating ModelCalibration step: {str(e)}")
            import traceback
            self.log_error(traceback.format_exc())
            raise ValueError(f"Failed to create ModelCalibration step: {str(e)}") from e
