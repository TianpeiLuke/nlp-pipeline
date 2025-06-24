from typing import Dict, List, Any, Optional, Type, Set, Tuple
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import Step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from pathlib import Path
import logging
import time
from collections import defaultdict

from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.builder_step_base import StepBuilderBase

from src.pipeline_builder.pipeline_dag import PipelineDAG


logger = logging.getLogger(__name__)


class PipelineBuilderTemplate:
    """
    Generic pipeline builder using a DAG and step builders.
    
    This class implements a template-based approach to building SageMaker Pipelines.
    It uses a directed acyclic graph (DAG) to define the pipeline structure and
    step builders to create the individual steps.
    
    The template follows these steps to build a pipeline:
    1. Initialize step builders for all steps in the DAG
    2. Determine the build order using topological sort
    3. Instantiate steps in topological order, extracting inputs from dependency steps
    4. Create the pipeline with the instantiated steps
    
    This approach allows for a flexible and modular pipeline definition, where
    each step is responsible for its own configuration and input/output handling.
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
        Initialize the pipeline builder template.
        
        Args:
            dag: PipelineDAG instance defining the pipeline structure
            config_map: Mapping from step name to config instance
            step_builder_map: Mapping from step type to StepBuilderBase subclass
            sagemaker_session: SageMaker session to use for creating the pipeline
            role: IAM role to use for the pipeline
            pipeline_parameters: List of pipeline parameters
            notebook_root: Root directory of the notebook environment
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
        
        # Validate inputs
        self._validate_inputs()
        
        # Initialize step builders
        self._initialize_step_builders()

    def _validate_inputs(self) -> None:
        """
        Validate that the inputs to the template are consistent.
        
        This method checks that:
        1. All nodes in the DAG have a corresponding config in config_map
        2. All configs in config_map have a corresponding step builder in step_builder_map
        3. All edges in the DAG connect nodes that exist in the DAG
        
        Raises:
            ValueError: If any of the validation checks fail
        """
        # Check that all nodes in the DAG have a corresponding config
        missing_configs = [node for node in self.dag.nodes if node not in self.config_map]
        if missing_configs:
            raise ValueError(f"Missing configs for nodes: {missing_configs}")
        
        # Check that all configs have a corresponding step builder
        for step_name, config in self.config_map.items():
            step_type = BasePipelineConfig.get_step_name(type(config).__name__)
            if step_type not in self.step_builder_map:
                raise ValueError(f"Missing step builder for step type: {step_type}")
        
        # Check that all edges in the DAG connect nodes that exist in the DAG
        for src, dst in self.dag.edges:
            if src not in self.dag.nodes:
                raise ValueError(f"Edge source node not in DAG: {src}")
            if dst not in self.dag.nodes:
                raise ValueError(f"Edge destination node not in DAG: {dst}")
        
        logger.info("Input validation successful")

    def _initialize_step_builders(self) -> None:
        """
        Initialize step builders for all steps in the DAG.
        
        This method creates a step builder instance for each step in the DAG,
        using the corresponding config from config_map and the appropriate
        builder class from step_builder_map.
        """
        logger.info("Initializing step builders")
        start_time = time.time()
        
        for step_name in self.dag.nodes:
            try:
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
                logger.info(f"Initialized builder for step {step_name} of type {step_type}")
            except Exception as e:
                logger.error(f"Error initializing builder for step {step_name}: {e}")
                raise ValueError(f"Failed to initialize step builder for {step_name}: {e}") from e
        
        elapsed_time = time.time() - start_time
        logger.info(f"Initialized {len(self.step_builders)} step builders in {elapsed_time:.2f} seconds")

    def _instantiate_step(self, step_name: str) -> Step:
        """
        Instantiate a pipeline step with appropriate inputs from dependencies.
        
        This method creates a step using the step builder's build method,
        which automatically extracts inputs from dependency steps and creates the step.
        
        Args:
            step_name: Name of the step to instantiate
            
        Returns:
            Instantiated SageMaker Pipeline Step
        """
        builder = self.step_builders[step_name]
        
        # Get dependency steps
        dependencies = self.dag.get_dependencies(step_name)
        if dependencies:
            logger.info(f"Step {step_name} depends on: {dependencies}")
        
        # Ensure all dependencies have been instantiated
        missing_deps = [dep for dep in dependencies if dep not in self.step_instances]
        if missing_deps:
            raise ValueError(f"Dependencies not instantiated for step {step_name}: {missing_deps}")
        
        dependency_steps = [self.step_instances[parent] for parent in dependencies]
        
        # Build the step with dependency steps
        start_time = time.time()
        try:
            # Use the build method which combines extract_inputs_from_dependencies and create_step
            step = builder.build(dependency_steps)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Built step {step_name} in {elapsed_time:.2f} seconds")
            
            # Get the input requirements for logging purposes
            input_requirements = builder.get_input_requirements()
            if input_requirements:
                logger.info(f"Step {step_name} input requirements: {list(input_requirements.keys())}")
            
            return step
        except Exception as e:
            logger.error(f"Error building step {step_name}: {e}")
            raise ValueError(f"Failed to build step {step_name}: {e}") from e
    
    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """
        Build and return a SageMaker Pipeline object.
        
        This method builds the pipeline by:
        1. Determining the build order using topological sort
        2. Instantiating steps in topological order
        3. Creating the pipeline with the instantiated steps
        
        Args:
            pipeline_name: Name of the pipeline
            
        Returns:
            SageMaker Pipeline object
        """
        logger.info(f"Generating pipeline: {pipeline_name}")
        start_time = time.time()
        
        # Reset step instances if we're regenerating the pipeline
        if self.step_instances:
            logger.info("Clearing existing step instances for pipeline regeneration")
            self.step_instances = {}
        
        # Topological sort to determine build order
        try:
            build_order = self.dag.topological_sort()
            logger.info(f"Build order: {build_order}")
        except ValueError as e:
            logger.error(f"Error in topological sort: {e}")
            raise ValueError(f"Failed to determine build order: {e}") from e
        
        # Instantiate steps in topological order
        for step_name in build_order:
            try:
                step = self._instantiate_step(step_name)
                self.step_instances[step_name] = step
            except Exception as e:
                logger.error(f"Error instantiating step {step_name}: {e}")
                raise ValueError(f"Failed to instantiate step {step_name}: {e}") from e

        # Create the pipeline
        steps = [self.step_instances[name] for name in build_order]
        pipeline = Pipeline(
            name=pipeline_name,
            parameters=self.pipeline_parameters,
            steps=steps,
            sagemaker_session=self.sagemaker_session,
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated pipeline {pipeline_name} with {len(steps)} steps in {elapsed_time:.2f} seconds")
        
        return pipeline
