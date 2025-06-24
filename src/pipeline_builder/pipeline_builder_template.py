from typing import Dict, List, Any, Optional, Type, Set, Tuple
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import Step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from pathlib import Path
import logging
from collections import defaultdict

from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.builder_step_base import StepBuilderBase

from src.pipeline_builder.pipeline_dag import PipelineDAG


logger = logging.getLogger(__name__)



class PipelineBuilderTemplate:
    """
    Generic pipeline builder using a DAG and step builders.
    """
    def __init__(
        self,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_builder_map: Dict[str, Type[StepBuilderBase]],
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List[ParameterString]] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        dag: PipelineDAG instance
        config_map: Mapping from step name to config instance
        step_builder_map: Mapping from step type to StepBuilderBase subclass
        """
        self.dag = dag
        self.config_map = config_map
        self.step_builder_map = step_builder_map
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        self.pipeline_parameters = pipeline_parameters or []

        self.step_instances: Dict[str, Step] = {}
        self.step_builders: Dict[str, StepBuilderBase] = {}
        
        # Message passing data structures
        self.step_input_requirements: Dict[str, Dict[str, str]] = {}
        self.step_output_properties: Dict[str, Dict[str, str]] = {}
        self.step_messages: Dict[str, Dict[str, Any]] = defaultdict(dict)

    def _collect_step_io_requirements(self) -> None:
        """
        Collect input and output requirements from all steps.
        
        This method initializes the step builders for all steps in the DAG
        and collects their input requirements and output properties.
        """
        logger.info("Collecting input and output requirements for all steps")
        
        for step_name in self.dag.nodes:
            config = self.config_map[step_name]
            step_type = BasePipelineConfig.get_step_name(type(config).__name__)
            builder_cls = self.step_builder_map[step_type]
            
            # Initialize the builder
            builder = builder_cls(
                config=config,
                sagemaker_session=self.sagemaker_session,
                role=self.role,
                notebook_root=self.notebook_root,
            )
            self.step_builders[step_name] = builder
            
            # Collect input requirements
            try:
                input_requirements = builder.get_input_requirements()
                self.step_input_requirements[step_name] = input_requirements
                logger.info(f"Step {step_name} requires inputs: {list(input_requirements.keys())}")
            except Exception as e:
                logger.warning(f"Error getting input requirements for {step_name}: {e}")
                self.step_input_requirements[step_name] = {}
            
            # Collect output properties
            try:
                output_properties = builder.get_output_properties()
                self.step_output_properties[step_name] = output_properties
                logger.info(f"Step {step_name} provides outputs: {list(output_properties.keys())}")
            except Exception as e:
                logger.warning(f"Error getting output properties for {step_name}: {e}")
                self.step_output_properties[step_name] = {}
    
    def _propagate_messages(self) -> None:
        """
        Propagate messages between steps based on the DAG topology.
        
        This method implements a message passing algorithm where each step:
        1. Collects messages from its dependencies (previous steps)
        2. Verifies that its input requirements can be satisfied
        3. Prepares its own output messages for downstream steps
        """
        logger.info("Starting message propagation between steps")
        
        # Process steps in topological order
        for step_name in self.dag.topological_sort():
            # Get dependencies
            dependencies = self.dag.get_dependencies(step_name)
            
            # Skip if no dependencies
            if not dependencies:
                logger.info(f"Step {step_name} has no dependencies, skipping message verification")
                continue
            
            # Get input requirements for this step
            input_requirements = self.step_input_requirements[step_name]
            if not input_requirements:
                logger.info(f"Step {step_name} has no declared input requirements")
                continue
            
            # Check each input requirement
            for input_name in input_requirements:
                # Try to find a matching output from dependencies
                found_match = False
                
                for dep_name in dependencies:
                    # Check if dependency provides this output
                    dep_outputs = self.step_output_properties[dep_name]
                    
                    # Direct match by name
                    if input_name in dep_outputs:
                        self.step_messages[step_name][input_name] = {
                            "source_step": dep_name,
                            "source_output": input_name
                        }
                        found_match = True
                        logger.info(f"Matched input {input_name} for {step_name} to output from {dep_name}")
                        break
                
                # If no direct match, try pattern matching
                if not found_match:
                    # Common patterns for matching inputs to outputs
                    common_patterns = {
                        "model": ["model", "model_data", "model_artifacts", "model_path"],
                        "data": ["data", "dataset", "input_data", "training_data"],
                        "output": ["output", "result", "artifacts", "s3_uri"]
                    }
                    
                    # Try to find a pattern match
                    for dep_name in dependencies:
                        dep_outputs = self.step_output_properties[dep_name]
                        
                        for pattern_type, keywords in common_patterns.items():
                            if any(kw in input_name.lower() for kw in keywords):
                                # Find outputs that match the same pattern
                                for output_name in dep_outputs:
                                    if any(kw in output_name.lower() for kw in keywords):
                                        self.step_messages[step_name][input_name] = {
                                            "source_step": dep_name,
                                            "source_output": output_name,
                                            "pattern_match": True
                                        }
                                        found_match = True
                                        logger.info(f"Pattern-matched input {input_name} for {step_name} to output {output_name} from {dep_name}")
                                        break
                            
                            if found_match:
                                break
                        
                        if found_match:
                            break
                
                if not found_match:
                    logger.warning(f"Could not find a matching output for input {input_name} required by {step_name}")
    
    def _instantiate_step(self, step_name: str) -> Step:
        """
        Instantiate a pipeline step with appropriate inputs from dependencies.
        
        This method uses the step builder's build method to create a step with the
        appropriate inputs from dependency steps.
        
        Args:
            step_name: Name of the step to instantiate
            
        Returns:
            Instantiated SageMaker Pipeline Step
        """
        config = self.config_map[step_name]
        builder = self.step_builders[step_name]

        # Gather dependencies
        dependency_steps = [self.step_instances[parent] for parent in self.dag.get_dependencies(step_name)]
        
        # Add any configuration-provided inputs to the builder's config
        # This allows steps to get inputs directly from their config
        self._add_config_inputs_to_builder(builder, config)
        
        try:
            # Use the builder's build method to create the step
            step = builder.build(dependency_steps)
            logger.info(f"Created step {step_name} using builder's build method")
            return step
        except Exception as e:
            logger.warning(f"Error using builder's build method: {e}")
            
            # Fallback to the old approach if build method fails
            logger.info(f"Falling back to legacy step instantiation for {step_name}")
            
            # Start with basic dependencies
            kwargs = {"dependencies": dependency_steps}
            
            # Add any configuration-provided inputs
            self._add_config_inputs(kwargs, config)
            
            # Extract inputs from dependency steps
            if dependency_steps:
                try:
                    extracted_inputs = builder.extract_inputs_from_dependencies(dependency_steps)
                    kwargs.update(extracted_inputs)
                    logger.info(f"Extracted inputs for {step_name} using step builder's extract_inputs_from_dependencies method")
                except Exception as extract_error:
                    logger.warning(f"Error extracting inputs using step builder's method: {extract_error}")
                    # Fallback to message passing results
                    self._extract_inputs_from_message_passing(kwargs, step_name, dependency_steps)
            
            # Create the step with extracted inputs
            try:
                step = builder.create_step(**kwargs)
            except TypeError as type_error:
                logger.warning(f"Error creating step with extracted inputs: {type_error}")
                # Fallback for builders that don't accept our kwargs
                step = builder.create_step()
                # If possible, add depends_on after creation
                if hasattr(step, "add_depends_on"):
                    step.add_depends_on(dependency_steps)
            
            return step
        
    def _add_config_inputs(self, kwargs: dict, config: BasePipelineConfig) -> None:
        """
        Add inputs from the step's configuration to kwargs.
        
        Args:
            kwargs: Dictionary to add inputs to
            config: Step configuration
        """
        # Common input parameters that might be in configs
        common_inputs = [
            "model_data", 
            "data_uri", 
            "input_data", 
            "model_uri",
            "training_data"
        ]
        
        for input_param in common_inputs:
            if hasattr(config, input_param) and getattr(config, input_param) is not None:
                kwargs[input_param] = getattr(config, input_param)
    
    def _add_config_inputs_to_builder(self, builder: StepBuilderBase, config: BasePipelineConfig) -> None:
        """
        Add inputs from the step's configuration to the builder's config.
        
        Args:
            builder: Step builder to add inputs to
            config: Step configuration
        """
        # Common input parameters that might be in configs
        common_inputs = [
            "model_data", 
            "data_uri", 
            "input_data", 
            "model_uri",
            "training_data"
        ]
        
        # Add inputs to the builder's config if they're not already set
        for input_param in common_inputs:
            if hasattr(config, input_param) and getattr(config, input_param) is not None:
                # Only set the attribute if it doesn't already exist or is None
                if not hasattr(builder.config, input_param) or getattr(builder.config, input_param) is None:
                    setattr(builder.config, input_param, getattr(config, input_param))
    
    def _extract_inputs_from_message_passing(self, kwargs: dict, step_name: str, dependency_steps: List[Step]) -> None:
        """
        Extract inputs from dependency steps based on message passing results.
        
        This method uses the messages collected during the message propagation phase
        to determine what outputs from previous steps should be passed as inputs to
        the current step.
        
        Args:
            kwargs: Dictionary to add extracted inputs to
            step_name: Name of the current step
            dependency_steps: List of dependency steps
        """
        # Get messages for this step
        step_messages = self.step_messages.get(step_name, {})
        
        # Get the step type from the config
        config = self.config_map[step_name]
        step_type = BasePipelineConfig.get_step_name(type(config).__name__)
        
        # If we have messages, use them to extract inputs
        if step_messages:
            logger.info(f"Using message passing results for step {step_name}")
            
            # Process each message
            for input_name, message in step_messages.items():
                source_step_name = message["source_step"]
                source_output = message["source_output"]
                
                # Find the corresponding dependency step instance
                for dep_step in dependency_steps:
                    dep_step_name = getattr(dep_step, 'name', 'unknown')
                    
                    # Check if this is the source step
                    if source_step_name in dep_step_name:
                        # Try to get the output property from the step
                        if hasattr(dep_step, source_output):
                            kwargs[input_name] = getattr(dep_step, source_output)
                            logger.info(f"Set input {input_name} from {dep_step_name}.{source_output}")
                        else:
                            # Fallback to common output patterns
                            logger.info(f"Output {source_output} not found on step {dep_step_name}, trying common patterns")
                            self._extract_common_outputs(kwargs, dep_step, step_name, step_type)
        else:
            # Fallback to extracting common outputs
            logger.info(f"No message passing results for step {step_name}, using fallback extraction")
            
            # Check for common output patterns in each dependency step
            for prev_step in dependency_steps:
                prev_step_name = getattr(prev_step, 'name', 'unknown')
                logger.info(f"Processing dependency: {prev_step_name}")
                
                # Extract common outputs
                self._extract_common_outputs(kwargs, prev_step, step_name, step_type)
    
    def _extract_common_outputs(self, kwargs: dict, prev_step: Step, step_name: str, step_type: str) -> None:
        """
        Extract common outputs from a step based on known patterns.
        
        Args:
            kwargs: Dictionary to add extracted inputs to
            prev_step: The dependency step to extract outputs from
            step_name: Name of the current step
            step_type: Type of the current step
        """
        # Check for model artifacts path (common in model steps)
        if hasattr(prev_step, "model_artifacts_path"):
            # Different step types might use different parameter names for the same concept
            if step_type == "Package":
                kwargs["model_artifacts_input_source"] = prev_step.model_artifacts_path
            else:
                kwargs["model_data"] = prev_step.model_artifacts_path
        
        # Check for training output path (common in training steps)
        if hasattr(prev_step, "training_output_path"):
            kwargs["model_data"] = prev_step.training_output_path
        
        # Check for processing output (common in processing steps)
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "ProcessingOutputConfig"):
            # Extract the S3 URI from the processing step's output
            try:
                # Try to get the first output
                if hasattr(prev_step.properties.ProcessingOutputConfig.Outputs, "__getitem__"):
                    # If it's a list or dictionary-like object
                    try:
                        # Try numeric index first (list-like)
                        s3_uri = prev_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
                        
                        # Different step types might use different parameter names
                        if step_type == "Registration":
                            kwargs["packaging_step_output"] = s3_uri
                        elif step_type == "TabularPreprocessing":
                            # For tabular preprocessing, we need to check the input_names
                            config = self.config_map[step_name]
                            if hasattr(config, "input_names") and "data_input" in config.input_names:
                                kwargs["inputs"] = {config.input_names["data_input"]: s3_uri}
                        else:
                            # Use a generic name if no specific mapping exists
                            kwargs["processing_output"] = s3_uri
                    except (IndexError, TypeError):
                        # Try string keys (dict-like)
                        for key in prev_step.properties.ProcessingOutputConfig.Outputs:
                            output = prev_step.properties.ProcessingOutputConfig.Outputs[key]
                            if hasattr(output, "S3Output") and hasattr(output.S3Output, "S3Uri"):
                                s3_uri = output.S3Output.S3Uri
                                
                                # For tabular preprocessing, we need to check if this is the data output
                                if key == "ProcessedTabularData" and (step_type == "PytorchTraining" or step_type == "XGBoostTraining"):
                                    kwargs["input_path"] = s3_uri
                                elif step_type == "Registration" and key == "packaged_model_output":
                                    kwargs["packaging_step_output"] = s3_uri
                                else:
                                    # Use the output key as the input key
                                    kwargs[key.lower()] = s3_uri
            except (AttributeError, IndexError) as e:
                logger.warning(f"Could not extract processing output from step: {e}")
        
        # Check for transform output (common in transform steps)
        if hasattr(prev_step, "properties") and hasattr(prev_step.properties, "TransformOutput"):
            try:
                s3_uri = prev_step.properties.TransformOutput.S3OutputPath
                kwargs["transform_output"] = s3_uri
            except AttributeError as e:
                logger.warning(f"Could not extract transform output from step: {e}")
    
    
    def _match_inputs_to_outputs(
        self, 
        kwargs: dict, 
        input_requirements: Dict[str, str], 
        output_properties: Dict[str, str], 
        prev_step: Step
    ) -> None:
        """
        Match input requirements with output properties.
        
        Args:
            kwargs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            output_properties: Dictionary of output properties
            prev_step: The dependency step
        """
        # Common patterns for matching inputs to outputs
        common_patterns = {
            "model": ["model", "model_data", "model_artifacts", "model_path"],
            "data": ["data", "dataset", "input_data", "training_data"],
            "output": ["output", "result", "artifacts", "s3_uri"]
        }
        
        # Try to match inputs to outputs based on name similarity
        for input_name, input_desc in input_requirements.items():
            for output_name, output_desc in output_properties.items():
                # Direct match
                if input_name == output_name and hasattr(prev_step, output_name):
                    kwargs[input_name] = getattr(prev_step, output_name)
                    logger.info(f"Matched input {input_name} to output {output_name}")
                    break
                
                # Pattern-based matching
                for pattern_type, keywords in common_patterns.items():
                    if any(kw in input_name.lower() for kw in keywords) and any(kw in output_name.lower() for kw in keywords):
                        if hasattr(prev_step, output_name):
                            kwargs[input_name] = getattr(prev_step, output_name)
                            logger.info(f"Pattern-matched input {input_name} to output {output_name}")
                            break

    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """
        Build and return a SageMaker Pipeline object.
        """
        logger.info(f"Generating pipeline: {pipeline_name}")
        
        # First, collect input/output requirements from all steps
        self._collect_step_io_requirements()
        
        # Then, propagate messages between steps
        self._propagate_messages()
        
        # Topological sort to determine build order
        build_order = self.dag.topological_sort()
        
        # Instantiate steps in topological order
        for step_name in build_order:
            step = self._instantiate_step(step_name)
            self.step_instances[step_name] = step

        steps = [self.step_instances[name] for name in build_order]
        return Pipeline(
            name=pipeline_name,
            parameters=self.pipeline_parameters,
            steps=steps,
            sagemaker_session=self.sagemaker_session,
        )
