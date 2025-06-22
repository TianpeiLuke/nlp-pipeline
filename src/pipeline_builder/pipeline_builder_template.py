from typing import Dict, List, Any, Optional, Type
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import Step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from pathlib import Path
import logging

from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

class PipelineDAG:
    """
    Represents a pipeline topology as a directed acyclic graph (DAG).
    Each node is a step name; edges define dependencies.
    """
    def __init__(self, nodes: List[str], edges: List[tuple]):
        """
        nodes: List of step names (str)
        edges: List of (from_step, to_step) tuples
        """
        self.nodes = nodes
        self.edges = edges
        self.adj_list = {n: [] for n in nodes}
        for src, dst in edges:
            self.adj_list[src].append(dst)
        self.reverse_adj = {n: [] for n in nodes}
        for src, dst in edges:
            self.reverse_adj[dst].append(src)

    def get_dependencies(self, node: str) -> List[str]:
        """Return immediate dependencies (parents) of a node."""
        return self.reverse_adj.get(node, [])

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        from collections import deque

        in_degree = {n: 0 for n in self.nodes}
        for src, dst in self.edges:
            in_degree[dst] += 1

        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            raise ValueError("DAG has cycles or disconnected nodes")
        return order

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

    def _instantiate_step(self, step_name: str) -> Step:
        """
        Instantiate a pipeline step with appropriate inputs from dependencies.
        
        This method dynamically determines the inputs required by each step based on:
        1. The step's configuration
        2. The outputs available from dependency steps
        3. The step type and its expected input parameters
        
        Args:
            step_name: Name of the step to instantiate
            
        Returns:
            Instantiated SageMaker Pipeline Step
        """
        config = self.config_map[step_name]
        step_type = BasePipelineConfig.get_step_name(type(config).__name__)
        builder_cls = self.step_builder_map[step_type]
        builder = builder_cls(
            config=config,
            sagemaker_session=self.sagemaker_session,
            role=self.role,
            notebook_root=self.notebook_root,
        )
        self.step_builders[step_name] = builder

        # Gather dependencies
        dependency_steps = [self.step_instances[parent] for parent in self.dag.get_dependencies(step_name)]
        
        # Start with basic dependencies
        kwargs = {"dependencies": dependency_steps}
        
        # Add any configuration-provided inputs
        # This allows steps to get inputs directly from their config
        self._add_config_inputs(kwargs, config)
        
        # Extract inputs from dependency steps if there are any
        if dependency_steps:
            self._extract_inputs_from_dependencies(kwargs, step_type, dependency_steps)
        
        # Create the step with extracted inputs
        try:
            step = builder.create_step(**kwargs)
        except TypeError as e:
            logger.warning(f"Error creating step with extracted inputs: {e}")
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
    
    def _extract_inputs_from_dependencies(self, kwargs: dict, step_type: str, dependency_steps: List[Step]) -> None:
        """
        Extract inputs from dependency steps based on step type and available outputs.
        
        This method dynamically determines what outputs from previous steps should be
        passed as inputs to the current step.
        
        Args:
            kwargs: Dictionary to add extracted inputs to
            step_type: Type of the current step
            dependency_steps: List of dependency steps
        """
        # Get the current step builder to check its input requirements
        current_builder = self.step_builders.get(step_type)
        input_requirements = {}
        
        # If we have the current builder, get its input requirements
        if current_builder and hasattr(current_builder, 'get_input_requirements'):
            try:
                input_requirements = current_builder.get_input_requirements()
                logger.info(f"Step {step_type} requires inputs: {list(input_requirements.keys())}")
            except Exception as e:
                logger.warning(f"Error getting input requirements for {step_type}: {e}")
        
        # Process each dependency step
        for prev_step in dependency_steps:
            prev_step_name = getattr(prev_step, 'name', 'unknown')
            logger.info(f"Processing dependency: {prev_step_name}")
            
            # Try to get the builder for this dependency
            prev_builder = None
            for builder_name, builder in self.step_builders.items():
                if builder_name in prev_step_name:
                    prev_builder = builder
                    break
            
            # Get output properties if available
            output_properties = {}
            if prev_builder and hasattr(prev_builder, 'get_output_properties'):
                try:
                    output_properties = prev_builder.get_output_properties()
                    logger.info(f"Step {prev_step_name} provides outputs: {list(output_properties.keys())}")
                except Exception as e:
                    logger.warning(f"Error getting output properties for {prev_step_name}: {e}")
            
            # Check for common output patterns in the previous step
            self._extract_common_outputs(kwargs, prev_step, step_type)
            
            # Try to match input requirements with output properties
            if input_requirements and output_properties:
                self._match_inputs_to_outputs(kwargs, input_requirements, output_properties, prev_step)
    
    def _extract_common_outputs(self, kwargs: dict, prev_step: Step, step_type: str) -> None:
        """
        Extract common outputs from a step based on known patterns.
        
        Args:
            kwargs: Dictionary to add extracted inputs to
            prev_step: The dependency step to extract outputs from
            step_type: Type of the current step
        """
        # Check for model artifacts path (common in model steps)
        if hasattr(prev_step, "model_artifacts_path"):
            # Different step types might use different parameter names for the same concept
            if step_type == "PackagingStep":
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
                s3_uri = prev_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri
                
                # Different step types might use different parameter names
                if step_type == "RegistrationStep":
                    kwargs["packaging_step_output"] = s3_uri
                else:
                    # Use a generic name if no specific mapping exists
                    kwargs["processing_output"] = s3_uri
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
        # Topological sort to determine build order
        build_order = self.dag.topological_sort()
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
