# Pipeline Assembler

The `PipelineAssembler` serves as a low-level pipeline assembler that translates a declarative pipeline structure into a SageMaker Pipeline. It takes a directed acyclic graph (DAG), configurations, and step builder classes as inputs and handles the complex task of instantiating steps, managing dependencies, and connecting components.

## Purpose

The pipeline assembler addresses several challenges in pipeline construction:

1. **Complexity Reduction**: Simplifies pipeline creation with a component-based approach
2. **Separation of Concerns**: Cleanly separates pipeline structure (DAG) from implementation details
3. **Dependency Management**: Handles step dependencies using specification-based dependency resolution
4. **Consistent Assembly**: Ensures consistent pipeline assembly across different pipeline types

## Class Structure

```python
class PipelineAssembler:
    def __init__(
        self,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_builder_map: Dict[str, Type[StepBuilderBase]],
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List[ParameterString]] = None,
        notebook_root: Optional[Path] = None,
        registry_manager: Optional[RegistryManager] = None,
        dependency_resolver: Optional[UnifiedDependencyResolver] = None
    ):
        """Initialize the pipeline assembler."""
        pass
        
    def _initialize_step_builders(self) -> None:
        """Initialize step builders for all steps in the DAG."""
        pass
        
    def _propagate_messages(self) -> None:
        """Initialize step connections using the dependency resolver."""
        pass
        
    def _generate_outputs(self, step_name: str) -> Dict[str, Any]:
        """Generate outputs dictionary using step builder's specification."""
        pass
        
    def _instantiate_step(self, step_name: str) -> Step:
        """Instantiate a pipeline step with appropriate inputs from dependencies."""
        pass
        
    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """Build and return a SageMaker Pipeline object."""
        pass
        
    @classmethod
    def create_with_components(cls, 
                             dag: PipelineDAG,
                             config_map: Dict[str, BasePipelineConfig],
                             step_builder_map: Dict[str, Type[StepBuilderBase]],
                             context_name: Optional[str] = None,
                             **kwargs) -> "PipelineAssembler":
        """Create pipeline assembler with managed components."""
        pass
```

## Key Components

### 1. Constructor Parameters

The assembler requires several key parameters:

- **dag**: The pipeline structure as a directed acyclic graph
- **config_map**: Maps step names to their configurations
- **step_builder_map**: Maps step types to builder classes
- **registry_manager**: Component for managing specification registries
- **dependency_resolver**: Component for resolving dependencies

These parameters provide everything needed to instantiate and connect pipeline steps.

### 2. Step Builder Initialization

The assembler initializes step builders for all steps in the DAG:

```python
def _initialize_step_builders(self) -> None:
    for step_name in self.dag.nodes:
        config = self.config_map[step_name]
        step_type = BasePipelineConfig.get_step_name(type(config).__name__)
        builder_cls = self.step_builder_map[step_type]
        
        # Initialize the builder with dependency components
        builder = builder_cls(
            config=config,
            sagemaker_session=self.sagemaker_session,
            role=self.role,
            notebook_root=self.notebook_root,
            registry_manager=self._registry_manager,
            dependency_resolver=self._dependency_resolver,
        )
        self.step_builders[step_name] = builder
```

This creates a step builder for each node in the DAG, passing the appropriate configuration and dependency components.

### 3. Dependency Propagation

The assembler uses the dependency resolver to connect steps:

```python
def _propagate_messages(self) -> None:
    resolver = self._get_dependency_resolver()
    
    # Process each edge in the DAG
    for src_step, dst_step in self.dag.edges:
        src_builder = self.step_builders[src_step]
        dst_builder = self.step_builders[dst_step]
        
        # Use resolver to match outputs to inputs
        for dep_name, dep_spec in dst_builder.spec.dependencies.items():
            # Check if source step can provide this dependency
            for out_name, out_spec in src_builder.spec.outputs.items():
                compatibility = resolver._calculate_compatibility(dep_spec, out_spec, src_builder.spec)
                if compatibility > 0.5:
                    self.step_messages[dst_step][dep_name] = {
                        'source_step': src_step,
                        'source_output': out_name,
                        'match_type': 'specification_match',
                        'compatibility': compatibility
                    }
```

This uses the specification-based dependency resolver to intelligently match step outputs to inputs.

### 4. Output Generation

The assembler generates standard output paths based on specifications:

```python
def _generate_outputs(self, step_name: str) -> Dict[str, Any]:
    builder = self.step_builders[step_name]
    config = self.config_map[step_name]
    
    base_s3_loc = getattr(config, 'pipeline_s3_loc', 's3://default-bucket/pipeline')
    outputs = {}
    step_type = builder.spec.step_type.lower()
    
    # Use each output specification to generate standard output path
    for logical_name, output_spec in builder.spec.outputs.items():
        outputs[logical_name] = f"{base_s3_loc}/{step_type}/{logical_name}"
```

This creates a dictionary of output paths based on the step's specification, ensuring consistent output locations.

### 5. Step Instantiation

The assembler instantiates steps with appropriate inputs:

```python
def _instantiate_step(self, step_name: str) -> Step:
    builder = self.step_builders[step_name]
    
    # Get dependency steps
    dependencies = []
    for dep_name in self.dag.get_dependencies(step_name):
        if dep_name in self.step_instances:
            dependencies.append(self.step_instances[dep_name])
    
    # Extract parameters from message dictionaries
    inputs = {}
    if step_name in self.step_messages:
        for input_name, message in self.step_messages[step_name].items():
            src_step = message['source_step']
            if src_step in self.step_instances:
                inputs[input_name] = {
                    "Get": f"Steps.{src_step}.{message['source_output']}"
                }
    
    # Generate outputs
    outputs = self._generate_outputs(step_name)
    
    # Create step with extracted inputs and outputs
    step = builder.create_step(
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies,
        enable_caching=True
    )
    
    return step
```

This extracts inputs from dependencies, generates outputs, and uses the step builder to create a SageMaker Pipeline step.

### 6. Pipeline Generation

The assembler builds the pipeline by following these steps:

```python
def generate_pipeline(self, pipeline_name: str) -> Pipeline:
    # Propagate messages between steps
    self._propagate_messages()
    
    # Topological sort to determine build order
    build_order = self.dag.topological_sort()
    
    # Instantiate steps in topological order
    for step_name in build_order:
        step = self._instantiate_step(step_name)
        self.step_instances[step_name] = step

    # Create the pipeline
    steps = [self.step_instances[name] for name in build_order]
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=self.pipeline_parameters,
        steps=steps,
        sagemaker_session=self.sagemaker_session,
    )
    
    return pipeline
```

This follows a clear sequence of steps to build the pipeline, ensuring that steps are instantiated in the correct order.

## Relationship to PipelineTemplateBase

The `PipelineAssembler` operates at a lower level than `PipelineTemplateBase`:

1. `PipelineAssembler` is responsible for:
   - Assembling the pipeline from the DAG, config_map, and step_builder_map
   - Handling the low-level details of step instantiation
   - Connecting steps according to the DAG
   - Using specifications for dependency resolution

2. `PipelineTemplateBase` is responsible for:
   - Loading configurations from file
   - Creating the DAG, config_map, and step_builder_map
   - Managing dependency components
   - Providing a standard structure for templates

The relationship can be visualized as:

```
PipelineTemplateBase (High-level template)
├── Loads configurations from file
├── Creates DAG, config_map, step_builder_map
├── Manages dependency components
└── Uses PipelineAssembler for pipeline assembly
    ├── PipelineAssembler (Low-level assembler)
    ├── Instantiates steps based on the DAG
    ├── Connects steps according to the DAG
    └── Generates the final pipeline
```

## Implementation Pattern

To use `PipelineAssembler` directly:

1. Create a PipelineDAG
2. Create a config_map
3. Create a step_builder_map
4. Create a PipelineAssembler
5. Call generate_pipeline

Example:

```python
# Create DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("data_load", "preprocessing")
dag.add_edge("preprocessing", "training")

# Create config map
config_map = {
    "data_load": data_load_config,
    "preprocessing": preprocessing_config,
    "training": training_config,
}

# Create step builder map
step_builder_map = {
    "DataLoad": DataLoadingStepBuilder,
    "Preprocessing": PreprocessingStepBuilder,
    "Training": TrainingStepBuilder,
}

# Create pipeline assembler
assembler = PipelineAssembler(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=session,
    role=role,
)

# Generate pipeline
pipeline = assembler.generate_pipeline("my-pipeline")
```

However, in most cases, you would use `PipelineTemplateBase` which provides a higher-level interface that creates the DAG, config_map, and step_builder_map for you.

## Best Practices

1. **DAG Construction**:
   - Ensure the DAG is a valid directed acyclic graph
   - Use descriptive node names that match config keys

2. **Config Mapping**:
   - Ensure all nodes in the DAG have a corresponding config
   - Use config classes that inherit from BasePipelineConfig

3. **Step Builder Mapping**:
   - Ensure all step types have a corresponding builder class
   - Use builder classes that inherit from StepBuilderBase

4. **Component Management**:
   - Use the provided factory methods for component creation
   - Pass registry_manager and dependency_resolver explicitly

5. **Error Handling**:
   - Validate inputs before creating the assembler
   - Handle errors gracefully during pipeline generation

By following these best practices, you can create reliable, maintainable, and efficient pipelines using the `PipelineAssembler`.

## Factory Methods

The assembler includes a factory method for component creation:

```python
@classmethod
def create_with_components(cls, 
                         dag: PipelineDAG,
                         config_map: Dict[str, BasePipelineConfig],
                         step_builder_map: Dict[str, Type[StepBuilderBase]],
                         context_name: Optional[str] = None,
                         **kwargs) -> "PipelineAssembler":
    components = create_pipeline_components(context_name)
    return cls(
        dag=dag,
        config_map=config_map,
        step_builder_map=step_builder_map,
        registry_manager=components["registry_manager"],
        dependency_resolver=components["resolver"],
        **kwargs
    )
```

This method creates an assembler with properly configured components, making it easier to create templates with isolated dependency components.
