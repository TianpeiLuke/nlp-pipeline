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
        
        # Add attributes expected by tests
        self.step_input_requirements: Dict[str, Dict[str, str]] = {}
        self.step_output_properties: Dict[str, Dict[str, str]] = {}
        self.step_messages = defaultdict(dict)
        
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

    def _collect_step_io_requirements(self) -> None:
        """
        Collect input requirements and output properties from all step builders.
        
        This method queries each step builder for its input requirements and output properties,
        and stores them in the step_input_requirements and step_output_properties dictionaries.
        """
        logger.info("Collecting step I/O requirements")
        
        for step_name, builder in self.step_builders.items():
            # Get input requirements
            input_requirements = builder.get_input_requirements()
            self.step_input_requirements[step_name] = input_requirements
            
            # Get output properties
            output_properties = builder.get_output_properties()
            self.step_output_properties[step_name] = output_properties
            
            logger.debug(f"Step {step_name} input requirements: {list(input_requirements.keys())}")
            logger.debug(f"Step {step_name} output properties: {list(output_properties.keys())}")

    def _propagate_messages(self) -> None:
        """
        Propagate messages between steps based on input requirements and output properties.
        
        This method analyzes the input requirements and output properties of each step,
        and creates messages that describe how inputs should be connected to outputs.
        
        It follows the standard pattern for input/output naming:
        1. Match logical input name (KEY from input_names) to logical output name (KEY from output_names)
        2. Use output descriptor VALUE (VALUE from output_names) when referencing the output
        3. Handle uppercase constants specially (DATA, METADATA, SIGNATURE)
        4. Fall back to pattern matching if other matching methods fail
        """
        logger.info("Propagating messages between steps")
        
        # Add enhanced debug logging for input/output connections
        logger.info("Analyzing step input/output connections:")
        for step_name in self.step_input_requirements:
            logger.info(f"  Step {step_name} input requirements: {list(self.step_input_requirements[step_name].keys())}")
            if step_name in self.config_map and hasattr(self.config_map[step_name], 'input_names'):
                logger.info(f"  Step {step_name} input_names: {self.config_map[step_name].input_names}")
        
        # Define common patterns for matching inputs to outputs
        input_patterns = {
            "model": ["model", "model_data", "model_artifacts", "model_path"],
            "data": ["data", "dataset", "input_data", "training_data"],
            "output": ["output", "result", "artifacts", "s3_uri"]
        }
        
        # Common uppercase constants used as input/output keys
        uppercase_constants = ["DATA", "METADATA", "SIGNATURE"]
        
        # Get the build order from the DAG
        build_order = self.dag.topological_sort()
        
        # Process steps in topological order
        for i, step_name in enumerate(build_order):
            # Skip the first step (no dependencies)
            if i == 0:
                continue
                
            # Get the input requirements for this step
            input_requirements = self.step_input_requirements.get(step_name, {})
            if not input_requirements:
                logger.debug(f"Step {step_name} has no input requirements")
                continue
            
            # Get the current step's config to access input_names
            current_config = self.config_map.get(step_name)
            
            # Get the dependencies for this step
            dependencies = self.dag.get_dependencies(step_name)
            if not dependencies:
                logger.debug(f"Step {step_name} has no dependencies")
                continue
                
                # Process each input requirement
            for input_name, input_desc in input_requirements.items():
                # Skip if this input already has a message
                if step_name in self.step_messages and input_name in self.step_messages[step_name]:
                    continue
                    
                # Try to find a matching output in the dependencies
                for dep_name in dependencies:
                    # Get the source step's config to access output_names
                    source_config = self.config_map.get(dep_name)
                    if not source_config or not hasattr(source_config, 'output_names') or not source_config.output_names:
                        continue
                    
                    # Track if we found a match
                    matched = False
                    
                    # Try logical name match first (KEY from input_names to KEY from output_names)
                    for out_logical_name, out_descriptor in source_config.output_names.items():
                        if input_name == out_logical_name:
                            # Found a match between logical names
                            self.step_messages[step_name][input_name] = {
                                'source_step': dep_name,
                                'source_output': out_descriptor,  # Use VALUE from output_names
                                'match_type': 'logical_name'
                            }
                            logger.info(f"Logical name match: {dep_name}.{out_logical_name} (descriptor: {out_descriptor}) -> {step_name}.{input_name}")
                            matched = True
                            break
                    
                    if matched:
                        break
                    
                    # Check for uppercase constant names (DATA, METADATA, SIGNATURE)
                    if input_name.isupper() and input_name in uppercase_constants:
                        # First check if this input name is also a key in source_config.output_names
                        if input_name in source_config.output_names:
                            self.step_messages[step_name][input_name] = {
                                'source_step': dep_name,
                                'source_output': source_config.output_names[input_name],  # Use VALUE from output_names
                                'match_type': 'uppercase_constant_key'
                            }
                            logger.info(f"Uppercase constant key match: {dep_name}.{input_name} -> {step_name}.{input_name}")
                            matched = True
                            break
                        
                        # Also check if this exact name exists as a value in output_names
                        for out_logical_name, out_descriptor in source_config.output_names.items():
                            if input_name == out_descriptor:
                                self.step_messages[step_name][input_name] = {
                                    'source_step': dep_name,
                                    'source_output': input_name,  # Use the uppercase constant directly
                                    'match_type': 'uppercase_constant_value'
                                }
                                logger.info(f"Uppercase constant value match: {dep_name}.{out_logical_name} -> {step_name}.{input_name}")
                                matched = True
                                break
                        
                        # As a last resort, check if the source step has this exact attribute
                        # This handles direct uppercase constants from CradleDataLoadingStep
                        source_step_instance = self.step_instances.get(dep_name)
                        if source_step_instance and hasattr(source_step_instance, input_name):
                            self.step_messages[step_name][input_name] = {
                                'source_step': dep_name,
                                'source_output': input_name,  # Use the same name for source output
                                'match_type': 'direct_attribute'
                            }
                            logger.info(f"Direct attribute match: {dep_name}.{input_name} -> {step_name}.{input_name}")
                            matched = True
                            break
                    
                    if matched:
                        break
                    
                    # Check if input name directly matches an output VALUE
                    for out_logical_name, out_descriptor in source_config.output_names.items():
                        if input_name == out_descriptor:
                            self.step_messages[step_name][input_name] = {
                                'source_step': dep_name,
                                'source_output': input_name,
                                'match_type': 'direct_output_value'
                            }
                            logger.info(f"Direct output value match: {dep_name}.{out_descriptor} -> {step_name}.{input_name}")
                            matched = True
                            break
                    
                    if matched:
                        break
                    
                    # If no previous matches found, try pattern match as fallback
                    # Get the output properties for this dependency
                    output_properties = self.step_output_properties.get(dep_name, {})
                    if not output_properties:
                        continue
                        
                        # Try pattern match
                    for pattern_type, keywords in input_patterns.items():
                        if not any(kw in input_name.lower() for kw in keywords):
                            continue
                            
                        # Find outputs by looking for matches in output_names logical names
                        matching_outputs = []
                        for out_logical_name, out_descriptor in source_config.output_names.items():
                            if any(kw in out_logical_name.lower() for kw in keywords):
                                matching_outputs.append((out_logical_name, out_descriptor))
                        
                        if matching_outputs:
                            # Use the first matching output
                            out_logical_name, out_descriptor = matching_outputs[0]
                            self.step_messages[step_name][input_name] = {
                                'source_step': dep_name,
                                'source_output': out_descriptor,  # Use VALUE from output_names
                                'match_type': 'pattern_match',
                                'pattern_type': pattern_type
                            }
                            logger.info(f"Pattern match ({pattern_type}): {dep_name}.{out_logical_name} (descriptor: {out_descriptor}) -> {step_name}.{input_name}")
                            matched = True
                            break
                            
                    if matched:
                        break
                
                # Log warning if no match found
                if step_name not in self.step_messages or input_name not in self.step_messages[step_name]:
                    logger.warning(f"No output match found for input: {step_name}.{input_name}")

    def _generate_outputs(self, step_name: str) -> Dict[str, Any]:
        """
        Generate default outputs for a step using the VALUES from output_names as keys.
        
        Creates paths in the format:
        {base_s3_loc}/{step_type}/{job_type (if present)}/{output_key}
        
        Args:
            step_name: Name of the step to generate outputs for
            
        Returns:
            Dictionary mapping output values to S3 URIs
        """
        outputs = {}
        config = self.config_map[step_name]
        
        # Find base_s3_loc in configs - typically in the base config
        base_s3_loc = None
        for cfg in self.config_map.values():
            if hasattr(cfg, 'pipeline_s3_loc') and getattr(cfg, 'pipeline_s3_loc'):
                base_s3_loc = getattr(cfg, 'pipeline_s3_loc')
                break
        
        if not base_s3_loc:
            base_s3_loc = 's3://default-bucket/pipeline'
            logger.info(f"No base_s3_loc found, using default: {base_s3_loc}")
        
        # Use output_names from config to generate outputs
        if hasattr(config, 'output_names') and config.output_names:
            # Get step type from config class name
            step_type = BasePipelineConfig.get_step_name(type(config).__name__)
            
            # Get job_type if available (used by steps like TabularPreprocessing)
            job_type = getattr(config, 'job_type', '')
            
            # Construct base path components
            path_parts = [base_s3_loc, step_type.lower()]
            if job_type:
                path_parts.append(job_type)
                
            # Join with slashes and ensure no duplicate slashes
            base_path = "/".join([p for p in path_parts if p])
            
            # Generate paths for all output_names values
            for logical_name, output_descriptor in config.output_names.items():
                # Use VALUE as the dictionary key (instead of logical name)
                outputs[output_descriptor] = f"{base_path}/{logical_name}"
                
                # Special handling for constants used by CradleDataLoadingStep
                if logical_name.upper() in ["DATA", "METADATA", "SIGNATURE"]:
                    outputs[logical_name.upper()] = outputs[output_descriptor]
            
            # Double check we have all required outputs by ensuring VALUES are in outputs
            for logical_name, output_descriptor in config.output_names.items():
                if output_descriptor not in outputs:
                    logger.info(f"Adding missing required output '{output_descriptor}' for {step_name}")
                    outputs[output_descriptor] = f"{base_path}/{logical_name}"
        
        logger.info(f"Generated {len(outputs)} outputs for {step_name}")
        
        # Add detailed info logging for all outputs
        logger.info(f"Generated output details for {step_name}:")
        for key, path in outputs.items():
            logger.info(f"  Output: '{key}' => {path}")
            
        return outputs
    
    def _instantiate_step(self, step_name: str) -> Step:
        """
        Instantiate a pipeline step with appropriate inputs from dependencies.
        
        This method creates a step using the step builder's create_step method,
        extracting inputs from dependency steps based on the messages.
        
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
        
        # Extract inputs from messages
        kwargs = {'dependencies': dependency_steps}
        
        # Add inputs from messages
        if step_name in self.step_messages:
            for input_name, message in self.step_messages[step_name].items():
                source_step = message['source_step']
                source_output = message['source_output']
                
                # Get the source step instance
                if source_step not in self.step_instances:
                    logger.warning(f"Source step {source_step} not instantiated for input {input_name}")
                    continue
                    
                source_step_instance = self.step_instances[source_step]
                
                # Get the output value from the source step
                if hasattr(source_step_instance, source_output):
                    output_value = getattr(source_step_instance, source_output)
                    kwargs[input_name] = output_value
                    logger.debug(f"Added input {input_name} from {source_step}.{source_output}")
                else:
                    logger.warning(f"Source step {source_step} has no output {source_output}")
        
        # Generate outputs if not already provided
        if 'outputs' not in kwargs:
            outputs = self._generate_outputs(step_name)
            if outputs:
                kwargs['outputs'] = outputs
                logger.info(f"Adding generated outputs for step {step_name}")
        
        # Build the step with extracted inputs and outputs
        start_time = time.time()
        try:
            step = builder.create_step(**kwargs)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Built step {step_name} in {elapsed_time:.2f} seconds")
            
            return step
        except Exception as e:
            logger.error(f"Error building step {step_name}: {e}")
            raise ValueError(f"Failed to build step {step_name}: {e}") from e

    def _add_config_inputs(self, kwargs: Dict[str, Any], config: BasePipelineConfig) -> None:
        """
        Add inputs from a config object to the kwargs dictionary.
        
        This method extracts non-None attributes from the config object and adds them to the kwargs dictionary.
        
        Args:
            kwargs: Dictionary to add inputs to
            config: Config object to extract inputs from
        """
        # Get all attributes of the config object
        for attr_name in dir(config):
            # Skip private attributes and methods
            if attr_name.startswith('_') or callable(getattr(config, attr_name)):
                continue
                
            # Get the attribute value
            attr_value = getattr(config, attr_name)
            
            # Skip None values
            if attr_value is None:
                continue
                
            # Add the attribute to kwargs
            kwargs[attr_name] = attr_value
            logger.debug(f"Added config input {attr_name}")

    def _extract_common_outputs(self, kwargs: Dict[str, Any], prev_step: Step, step_name: str, step_type: str) -> None:
        """
        Extract common outputs from a previous step and add them to the kwargs dictionary.
        
        This method extracts common outputs like model artifacts and processing outputs from a previous step,
        and adds them to the kwargs dictionary with appropriate names based on the step type.
        
        Args:
            kwargs: Dictionary to add outputs to
            prev_step: Previous step to extract outputs from
            step_name: Name of the current step
            step_type: Type of the current step
        """
        # Extract model artifacts
        if hasattr(prev_step, 'model_artifacts_path'):
            model_path = prev_step.model_artifacts_path
            
            # Add model_data for normal steps
            if step_type != "PackagingStep":
                kwargs['model_data'] = model_path
                logger.debug(f"Added model_data from {prev_step.name}.model_artifacts_path")
            else:
                # Add model_artifacts_input_source for packaging steps
                kwargs['model_artifacts_input_source'] = model_path
                logger.debug(f"Added model_artifacts_input_source from {prev_step.name}.model_artifacts_path")
        
        # Extract processing output
        if hasattr(prev_step, 'properties') and hasattr(prev_step.properties, 'ProcessingOutputConfig'):
            try:
                outputs = prev_step.properties.ProcessingOutputConfig.Outputs
                if hasattr(outputs, '__getitem__') and hasattr(outputs[0], 'S3Output'):
                    s3_uri = outputs[0].S3Output.S3Uri
                    
                    # Add processing_output for normal steps
                    if step_type != "RegistrationStep":
                        kwargs['processing_output'] = s3_uri
                        logger.debug(f"Added processing_output from {prev_step.name}.ProcessingOutputConfig")
                    else:
                        # Add packaging_step_output for registration steps
                        kwargs['packaging_step_output'] = s3_uri
                        logger.debug(f"Added packaging_step_output from {prev_step.name}.ProcessingOutputConfig")
            except (AttributeError, IndexError) as e:
                logger.warning(f"Error extracting processing output: {e}")

    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """
        Build and return a SageMaker Pipeline object.
        
        This method builds the pipeline by:
        1. Collecting step I/O requirements
        2. Propagating messages between steps
        3. Instantiating steps in topological order
        4. Creating the pipeline with the instantiated steps
        
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
        
        # Collect step I/O requirements
        self._collect_step_io_requirements()
        
        # Propagate messages between steps
        self._propagate_messages()
        
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
