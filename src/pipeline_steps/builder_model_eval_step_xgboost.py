from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.xgboost import XGBoostProcessor
from sagemaker.workflow.steps import ProcessingStep, Step
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import CacheConfig

from .config_model_eval_step_xgboost import XGBoostModelEvalConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class XGBoostModelEvalStepBuilder(StepBuilderBase):
    """
    Builder for XGBoost model evaluation ProcessingStep.
    """

    def __init__(
        self,
        config: XGBoostModelEvalConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        if not isinstance(config, XGBoostModelEvalConfig):
            raise ValueError("XGBoostModelEvalStepBuilder requires a XGBoostModelEvalConfig instance.")
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
        self.config: XGBoostModelEvalConfig = config

    def validate_configuration(self) -> None:
        logger.info(f"Running {self.__class__.__name__} specific configuration validation.")
        
        required_attrs = [
            'processing_entry_point',
            'processing_source_dir',
            'processing_instance_count', 
            'processing_volume_size',
            'pipeline_name',
            'job_type',
            'hyperparameters',
            'xgboost_framework_version'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) in [None, ""]:
                raise ValueError(f"XGBoostModelEvalConfig missing required attribute: {attr}")

        required_inputs = {"model_input", "eval_data_input"}
        required_outputs = {"eval_output", "metrics_output"}
        
        if not all(name in (self.config.input_names or {}) for name in required_inputs):
            raise ValueError(f"Required input names {required_inputs} must be defined")
        
        if not all(name in (self.config.output_names or {}) for name in required_outputs):
            raise ValueError(f"Required output names {required_outputs} must be defined")

        logger.info(f"{self.__class__.__name__} configuration validation passed.")
        
    def _get_environment_variables(self) -> Dict[str, str]:
        env_vars = {
            "ID_FIELD": str(self.config.hyperparameters.id_name),
            "LABEL_FIELD": str(self.config.hyperparameters.label_name),
        }
        logger.info(f"Evaluation environment variables: {env_vars}")
        return env_vars

    def _create_processor(self) -> XGBoostProcessor:
        """Create XGBoost processor for model evaluation."""
        instance_type = self.config.get_instance_type(
            'large' if self.config.use_large_processing_instance else 'small'
        )
        logger.info(f"Using processing instance type for evaluation: {instance_type}")

        base_job_name_prefix = self._sanitize_name_for_sagemaker(self.config.pipeline_name, 30)
        
        # Create a command that will run the entry point from within the package
        # Construct the command to run the entry point script from the source directory

        return XGBoostProcessor(
            framework_version=self.config.xgboost_framework_version,
            role=self.role,
            instance_type=instance_type,
            instance_count=self.config.processing_instance_count,
            volume_size_in_gb=self.config.processing_volume_size,
            sagemaker_session=self.session,
            base_job_name=f"{base_job_name_prefix}-xgb-eval",
            env=self._get_environment_variables(),
        )

    def _get_processing_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        required_inputs = (self.config.input_names or {}).keys()
        
        if not inputs or not all(k in inputs for k in required_inputs):
            raise ValueError(f"Must supply S3 URIs for all required inputs: {required_inputs}")

        input_destinations = {
            "model_input": "/opt/ml/processing/input/model",
            "eval_data_input": "/opt/ml/processing/input/eval_data",
        }

        return [
            ProcessingInput(
                input_name=channel_name,
                source=inputs[channel_name],
                destination=input_destinations[channel_name]
            )
            for channel_name in required_inputs
        ]

    def _get_processing_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        required_outputs = (self.config.output_names or {}).keys()
        
        if not outputs or not all(k in outputs for k in required_outputs):
            raise ValueError(f"Must supply S3 URIs for all required outputs: {required_outputs}")

        output_sources = {
            "eval_output": "/opt/ml/processing/output/eval",
            "metrics_output": "/opt/ml/processing/output/metrics"
        }

        return [
            ProcessingOutput(
                output_name=channel_name,
                source=output_sources[channel_name],
                destination=outputs[channel_name]
            )
            for channel_name in required_outputs
        ]

    def _get_job_arguments(self) -> List[str]:
        return ["--job_type", self.config.job_type]
    
    def _get_cache_config(self, enable_caching: bool = True) -> Optional[CacheConfig]:
        if not enable_caching:
            return None
        return CacheConfig(
            enable_caching=enable_caching,
            expire_after="30d"
        )
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # Get base input requirements and add additional ones
        input_reqs = {
            "inputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.input_names or {}).keys()])} S3 paths",
            "outputs": f"Dictionary containing {', '.join([f'{k}' for k in (self.config.output_names or {}).keys()])} S3 paths",
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
        # Get output properties from config's output_names
        return {k: v for k, v in (self.config.output_names or {}).items()}
        
    def extract_inputs_from_dependencies(self, dependency_steps: List[Step]) -> Dict[str, Any]:
        """
        Extract inputs from dependency steps.
        
        This method extracts the inputs required by the XGBoostModelEvalStep from the dependency steps.
        Specifically, it looks for:
        1. model_input from a ModelStep
        2. eval_data_input from a TabularPreprocessingStep
        
        Args:
            dependency_steps: List of dependency steps
            
        Returns:
            Dictionary of inputs extracted from dependency steps
        """
        inputs = {}
        outputs = {}
        
        # Look for model_input from a ModelStep
        for prev_step in dependency_steps:
            # Check for model step output
            if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ModelArtifacts"):
                try:
                    if "model_input" in self.config.input_names:
                        input_key = self.config.input_names["model_input"]
                        if "inputs" not in inputs:
                            inputs = {"inputs": {}}
                        inputs["inputs"][input_key] = prev_step.properties.ModelArtifacts.S3ModelArtifacts
                        logger.info(f"Found model_input from ModelStep: {prev_step.name}")
                except AttributeError as e:
                    logger.warning(f"Could not extract model artifacts from step: {e}")
            
            # Look for eval_data_input from a TabularPreprocessingStep
            elif hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
                try:
                    # Try to get the processed data output
                    if hasattr(prev_step.properties.ProcessingOutputConfig.Outputs, "__getitem__"):
                        # Try string keys (dict-like)
                        for key in prev_step.properties.ProcessingOutputConfig.Outputs:
                            output = prev_step.properties.ProcessingOutputConfig.Outputs[key]
                            if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                # If this is a processed data output, use it as eval_data_input
                                if key == "ProcessedTabularData":
                                    if "eval_data_input" in self.config.input_names:
                                        input_key = self.config.input_names["eval_data_input"]
                                        if "inputs" not in inputs:
                                            inputs = {"inputs": {}}
                                        inputs["inputs"][input_key] = output.S3Output.S3Uri
                                        logger.info(f"Found eval_data_input from step: {prev_step.name}")
                except (AttributeError, IndexError) as e:
                    logger.warning(f"Could not extract processed data output from step: {e}")
        
        # Set up outputs if not already set
        if not outputs and hasattr(self.config, "pipeline_s3_loc") and hasattr(self.config, "job_type"):
            base_output_path = f"{self.config.pipeline_s3_loc}/xgboost_model_eval/{self.config.job_type}"
            if "eval_output" in self.config.output_names and "metrics_output" in self.config.output_names:
                eval_key = self.config.output_names["eval_output"]
                metrics_key = self.config.output_names["metrics_output"]
                outputs = {
                    "outputs": {
                        eval_key: f"{base_output_path}/eval",
                        metrics_key: f"{base_output_path}/metrics"
                    }
                }
                logger.info(f"Set up output paths: {outputs}")
        
        result = {}
        if "inputs" in inputs:
            result["inputs"] = inputs["inputs"]
        if "outputs" in outputs:
            result["outputs"] = outputs["outputs"]
        
        # Add enable_caching
        result["enable_caching"] = True
        
        return result

    def create_step(self, **kwargs) -> ProcessingStep:
        """
        Creates a ProcessingStep for XGBoost model evaluation.
        
        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - inputs: Dictionary containing model_input and eval_data_input S3 paths
                - outputs: Dictionary containing eval_output and metrics_output S3 paths
                - dependencies: Optional list of step dependencies
                - enable_caching: Whether to enable caching for this step (default: True)
            
        Returns:
            ProcessingStep object
        """
        # Extract parameters
        inputs = self._extract_param(kwargs, 'inputs')
        outputs = self._extract_param(kwargs, 'outputs')
        dependencies = self._extract_param(kwargs, 'dependencies')
        enable_caching = self._extract_param(kwargs, 'enable_caching', True)
        
        logger.info("Creating XGBoost Model Evaluation ProcessingStep...")

        processor = self._create_processor()
        proc_inputs = self._get_processing_inputs(inputs)
        proc_outputs = self._get_processing_outputs(outputs)
        job_args = self._get_job_arguments()

        # FIX: The processor's `.run()` method is called to correctly package the code,
        # entrypoint, and dependencies. The output of this call (`step_args`) is then passed
        # to the ProcessingStep.
        step_args = processor.run(
            code=self.config.processing_entry_point, #self.config.get_script_path(),
            source_dir=self.config.processing_source_dir, # This is the crucial part
            inputs=proc_inputs,
            outputs=proc_outputs,
            arguments=job_args,
        )
        
        step_name = f"{self._get_step_name('XGBoostModelEval')}-{self.config.job_type.capitalize()}"
        logger.info(f"Created ProcessingStep with name: {step_name}")

        return ProcessingStep(
            name=step_name,
            step_args=step_args,
            depends_on=dependencies or [],
            cache_config=self._get_cache_config(enable_caching)
        )
