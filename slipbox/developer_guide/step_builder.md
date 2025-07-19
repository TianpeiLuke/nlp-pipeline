# Step Builder Implementation

Step builders bridge the gap between declarative specifications and executable SageMaker steps. They transform configuration and specification information into concrete SageMaker pipeline steps that can be executed.

## Purpose of Step Builders

Step builders serve several important purposes:

1. **SageMaker Integration**: They create the actual SageMaker step objects
2. **Resource Configuration**: They apply configuration settings for SageMaker resources
3. **Input/Output Mapping**: They connect inputs and outputs based on specifications
4. **Environment Setup**: They configure environment variables for scripts
5. **Dependency Resolution**: They extract dependencies from connected steps

## Builder Structure

A step builder is implemented as a class that extends `StepBuilderBase` and includes the following key components:

```python
from typing import Dict, List, Any, Optional
from pathlib import Path

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

from ..pipeline_deps.base_specifications import StepSpecification
from ..pipeline_script_contracts.base_script_contract import ScriptContract
from .builder_step_base import StepBuilderBase
from .config_your_step import YourStepConfig
from ..pipeline_step_specs.your_step_spec import YOUR_STEP_SPEC

class YourStepBuilder(StepBuilderBase):
    """Builder for YourStep processing step."""
    
    def __init__(
        self, 
        config, 
        sagemaker_session=None, 
        role=None, 
        notebook_root=None,
        registry_manager=None,
        dependency_resolver=None
    ):
        # Get job type if available
        job_type = getattr(config, 'job_type', None)
        
        # Get the appropriate specification based on job type
        if job_type and hasattr(self, '_get_spec_for_job_type'):
            spec = self._get_spec_for_job_type(job_type)
        else:
            spec = YOUR_STEP_SPEC
        
        super().__init__(
            config=config,
            spec=spec,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root,
            registry_manager=registry_manager,
            dependency_resolver=dependency_resolver
        )
        self.config: YourStepConfig = config
    
    def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
        """Get inputs for the processor using the specification and contract."""
        return self._get_spec_driven_processor_inputs(inputs)
    
    def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
        """Get outputs for the processor using the specification and contract."""
        return self._get_spec_driven_processor_outputs(outputs)
    
    def _get_environment_variables(self) -> Dict[str, str]:
        """Get environment variables for the processor."""
        # Use the standardized method from the base class
        env_vars = super()._get_environment_variables()
        
        # Add any additional environment variables needed by your script
        env_vars.update({
            "ADDITIONAL_PARAM": self.config.additional_param
        })
        
        return env_vars
    
    def _get_processor(self):
        """Create and return a SageMaker processor."""
        # Create the processor using SageMaker SDK
        from sagemaker.processing import Processor
        
        processor = Processor(
            role=self.role,
            image_uri=self.config.get_image_uri(),
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            volume_size_in_gb=self.config.volume_size_gb,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        return processor
    
    def create_step(self, **kwargs) -> ProcessingStep:
        """Create the processing step.
        
        Args:
            **kwargs: Additional keyword arguments for step creation.
                     Should include 'dependencies' list if step has dependencies.
        """
        # Extract inputs from dependencies using the resolver
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Get processor inputs and outputs
        inputs = self._get_inputs(extracted_inputs)
        outputs = self._get_outputs({})
        
        # Create processor
        processor = self._get_processor()
        
        # Set environment variables
        env_vars = self._get_environment_variables()
        
        # Create and return the step
        step_name = kwargs.get('step_name', 'YourStep')
        step = processor.run(
            inputs=inputs,
            outputs=outputs,
            container_arguments=[],
            container_entrypoint=["python", self.config.get_script_path()],
            job_name=self._generate_job_name(),
            wait=False,
            environment=env_vars
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

## How to Develop a Step Builder

### 1. Choose the Right Base Class

Select the appropriate base builder for your step type:

- **StepBuilderBase**: Base class for all step builders
- **ProcessingStepBuilder**: For processing steps
- **TrainingStepBuilder**: For training steps
- **ModelStepBuilder**: For model steps
- **TransformStepBuilder**: For transform steps

### 2. Handle Job Type Variants

If your step supports multiple job types (training, calibration, etc.), implement the logic to select the appropriate specification:

```python
def _get_spec_for_job_type(self, job_type: str) -> StepSpecification:
    """Get the appropriate specification based on job type."""
    job_type = job_type.lower()
    
    if job_type == "calibration":
        return CALIBRATION_SPEC
    elif job_type == "validation":
        return VALIDATION_SPEC
    else:
        return TRAINING_SPEC  # Default
```

### 3. Implement Input/Output Handling

Use the specification-driven approach for inputs and outputs:

```python
def _get_inputs(self, inputs: Dict[str, Any]) -> List[ProcessingInput]:
    """Get inputs for the processor using the specification and contract."""
    # Use the built-in method that leverages the specification and contract
    return self._get_spec_driven_processor_inputs(inputs)

def _get_outputs(self, outputs: Dict[str, Any]) -> List[ProcessingOutput]:
    """Get outputs for the processor using the specification and contract."""
    # Use the built-in method that leverages the specification and contract
    return self._get_spec_driven_processor_outputs(outputs)
```

### 4. Set Up Environment Variables

Configure environment variables required by your script:

```python
def _get_processor_env_vars(self) -> Dict[str, str]:
    """Get environment variables for the processor."""
    # Base environment variables
    env_vars = {
        # Map configuration parameters to environment variables
        "MODEL_TYPE": self.config.model_type,
        "NUM_EPOCHS": str(self.config.num_epochs),
        "LEARNING_RATE": str(self.config.learning_rate),
        "DEBUG_MODE": str(self.config.debug_mode).lower()
    }
    
    # Add job type specific variables if needed
    job_type = getattr(self.config, 'job_type', 'training')
    if job_type.lower() == "training":
        env_vars["TRAINING_MODE"] = "True"
    
    return env_vars
```

### 5. Create the Processor

Implement the processor creation logic:

```python
def _get_processor(self):
    """Create and return a SageMaker processor."""
    # For processing steps
    from sagemaker.processing import Processor
    
    processor = Processor(
        role=self.role,
        image_uri=self.config.get_image_uri(),
        instance_count=self.config.instance_count,
        instance_type=self.config.instance_type,
        volume_size_in_gb=self.config.volume_size_gb,
        max_runtime_in_seconds=self.config.max_runtime_seconds,
        sagemaker_session=self.sagemaker_session
    )
    
    return processor
    
    # For training steps
    # from sagemaker.estimator import Estimator
    # return Estimator(...)
```

### 6. Implement Step Creation

Override the `create_step` method to create the actual SageMaker step:

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create the processing step."""
    # Extract inputs from dependencies using the resolver
    dependencies = kwargs.get('dependencies', [])
    extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
    
    # Get processor inputs and outputs
    inputs = self._get_inputs(extracted_inputs)
    outputs = self._get_outputs({})
    
    # Create processor
    processor = self._get_processor()
    
    # Set environment variables
    env_vars = self._get_processor_env_vars()
    
    # Create and return the step
    # Use automatic step naming - no parameter needed for _get_step_name()
    step = processor.run(
        inputs=inputs,
        outputs=outputs,
        container_arguments=[],
        container_entrypoint=["python", self.config.get_script_path()],
        job_name=self._generate_job_name(),  # Automatic step type detection - no parameter needed
        wait=False,
        environment=env_vars
    )
    
    # Store specification in step for future reference
    setattr(step, '_spec', self.spec)
    
    return step
```

## Dependency Resolution

### Extracting Inputs from Dependencies

The `extract_inputs_from_dependencies` method is crucial for connecting steps:

```python
# In StepBuilderBase
def extract_inputs_from_dependencies(self, dependencies: List) -> Dict[str, str]:
    """Extract inputs from the given dependencies using the spec and resolver."""
    if not dependencies or not self.dependency_resolver:
        return {}
    
    # Use the dependency resolver to extract inputs based on the specification
    return self.dependency_resolver.resolve_dependencies(self.spec, dependencies)
```

This method:
1. Takes a list of dependency steps
2. Uses the dependency resolver to match outputs from those steps with dependencies in the specification
3. Returns a dictionary mapping logical input names to S3 URIs

### Understanding the Dependency Resolution Process

The dependency resolution process:

1. **Input Identification**: For each dependency in the specification, identify required inputs
2. **Output Matching**: For each dependency step, find outputs that match the required inputs
3. **Semantic Matching**: Use logical names, types, and semantic keywords to find the best matches
4. **URI Resolution**: Extract S3 URIs from the matched outputs
5. **Input Mapping**: Create a dictionary mapping logical input names to S3 URIs

## Processor Creation

Different step types require different SageMaker components:

### Processing Step

```python
from sagemaker.processing import Processor

processor = Processor(
    role=self.role,
    image_uri=self.config.get_image_uri(),
    instance_count=self.config.instance_count,
    instance_type=self.config.instance_type,
    volume_size_in_gb=self.config.volume_size_gb,
    max_runtime_in_seconds=self.config.max_runtime_seconds,
    sagemaker_session=self.sagemaker_session
)
```

### Training Step

```python
from sagemaker.estimator import Estimator

estimator = Estimator(
    role=self.role,
    image_uri=self.config.get_image_uri(),
    instance_count=self.config.instance_count,
    instance_type=self.config.instance_type,
    volume_size=self.config.volume_size_gb,
    max_run=self.config.max_runtime_seconds,
    sagemaker_session=self.sagemaker_session,
    hyperparameters=self.config.get_hyperparameters()
)
```

### Model Step

```python
from sagemaker.model import Model

model = Model(
    image_uri=self.config.get_image_uri(),
    model_data=self.config.model_data,
    role=self.role,
    sagemaker_session=self.sagemaker_session
)
```

## Environment Variable Handling

Environment variables connect configuration parameters to script requirements. The `StepBuilderBase` class now includes a standardized `_get_environment_variables()` method that automatically extracts environment variables from the script contract:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """
    Create environment variables for the processing job based on the script contract.
    
    This base implementation:
    1. Uses required_env_vars from the script contract
    2. Gets values from the config object
    3. Adds optional variables with defaults from the contract
    4. Can be overridden by child classes to add custom logic
    
    Returns:
        Dict[str, str]: Environment variables for the processing job
    """
    env_vars = {}
    
    if not hasattr(self, 'contract') or self.contract is None:
        self.log_warning("No script contract available for environment variable definition")
        return env_vars
    
    # Process required environment variables
    for env_var in self.contract.required_env_vars:
        # Convert from ENV_VAR_NAME format to config attribute style (env_var_name)
        config_attr = env_var.lower()
        
        # Try to get from config (direct attribute)
        if hasattr(self.config, config_attr):
            env_vars[env_var] = str(getattr(self.config, config_attr))
        # Try to get from config.hyperparameters
        elif hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, config_attr):
            env_vars[env_var] = str(getattr(self.config.hyperparameters, config_attr))
        else:
            self.log_warning(f"Required environment variable '{env_var}' not found in config")
    
    # Add optional environment variables with defaults
    for env_var, default_value in self.contract.optional_env_vars.items():
        # Convert from ENV_VAR_NAME format to config attribute style (env_var_name)
        config_attr = env_var.lower()
        
        # Try to get from config, fall back to default
        if hasattr(self.config, config_attr):
            env_vars[env_var] = str(getattr(self.config, config_attr))
        # Try to get from config.hyperparameters
        elif hasattr(self.config, 'hyperparameters') and hasattr(self.config.hyperparameters, config_attr):
            env_vars[env_var] = str(getattr(self.config.hyperparameters, config_attr))
        else:
            env_vars[env_var] = default_value
            self.log_debug(f"Using default value for optional environment variable '{env_var}': {default_value}")
    
    return env_vars
```

### How to Use the Standardized Method

In your step builder, you can use this method directly:

```python
def _get_processor(self):
    """Create and return a SageMaker processor."""
    from sagemaker.processing import ScriptProcessor
    
    processor = ScriptProcessor(
        role=self.role,
        image_uri=self.config.get_image_uri(),
        command=["python"],
        instance_count=self.config.instance_count,
        instance_type=self.config.instance_type,
        volume_size_in_gb=self.config.volume_size_gb,
        max_runtime_in_seconds=self.config.max_runtime_seconds,
        sagemaker_session=self.sagemaker_session,
        env=self._get_environment_variables()  # Use the standardized method
    )
    
    return processor
```

### Extending the Base Method

If you need additional environment variables beyond what's in the script contract:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Get environment variables for the processor."""
    # Get standard environment variables from contract
    env_vars = super()._get_environment_variables()
    
    # Add step-specific environment variables
    env_vars.update({
        "ADDITIONAL_PARAM": self.config.additional_param,
        "DEBUG_MODE": str(self.config.debug_mode).lower()
    })
    
    return env_vars
```

### Completely Overriding the Method

For cases where the standard approach doesn't fit:

```python
def _get_environment_variables(self) -> Dict[str, str]:
    """Get custom environment variables for this specific step."""
    return {
        "CUSTOM_PARAM_1": self.config.custom_param1,
        "CUSTOM_PARAM_2": str(self.config.custom_param2),
        "JOB_TYPE": self.config.job_type
    }
```

Best practices:
1. Use the base implementation unless you have specific requirements
2. When extending, call `super()._get_environment_variables()` first
3. Handle type conversion (all environment variables are strings)
4. Log warnings for missing required variables
5. Provide sensible defaults for optional variables

## Standardized Job Name Generation

The `StepBuilderBase` class now includes a standardized `_generate_job_name()` method to create consistent job names with automatic step type detection and uniqueness:

```python
def _generate_job_name(self) -> str:
    """
    Generate a standardized job name for SageMaker processing/training jobs.
    
    This method automatically determines the step type from the class name
    using the centralized step name registry. It also adds a timestamp to 
    ensure uniqueness across executions.
        
    Returns:
        Sanitized job name suitable for SageMaker
    """
    import time
    
    # Determine step type from the class name
    class_name = self.__class__.__name__
    determined_step_type = None
    
    # Try to find a matching entry in the STEP_NAMES registry
    for canonical_name, info in self.STEP_NAMES.items():
        if info["builder_step_name"] == class_name or class_name.startswith(info["builder_step_name"]):
            determined_step_type = canonical_name
            break
    
    # If no match found, fall back to class name with "StepBuilder" removed
    if determined_step_type is None:
        if class_name.endswith("StepBuilder"):
            determined_step_type = class_name[:-11]  # Remove "StepBuilder"
        else:
            determined_step_type = class_name
            
    # Generate a timestamp for uniqueness (unix timestamp in seconds)
    timestamp = int(time.time())
    
    # Build the job name
    if hasattr(self.config, 'job_type') and self.config.job_type:
        job_name = f"{determined_step_type}-{self.config.job_type.capitalize()}-{timestamp}"
    else:
        job_name = f"{determined_step_type}-{timestamp}"
        
    # Sanitize and return
    return self._sanitize_name_for_sagemaker(job_name)
```

The updated method provides several key improvements:

1. **Automatic Step Type Detection**: The method can now automatically determine the step type from the class name using the centralized step name registry, eliminating the need to pass it as a parameter
2. **Guaranteed Uniqueness**: A timestamp is added to ensure unique job names across multiple executions
3. **Registry-Based Naming**: Uses the centralized `STEP_NAMES` registry to maintain consistent naming conventions
4. **Backward Compatibility**: Still supports explicitly passing a step type for legacy code

### How to Use the Standardized Method

In your step builder's `create_step()` method, you can now call the method without any parameters:

```python
def create_step(self, **kwargs) -> ProcessingStep:
    """Create the processing step."""
    # [...]
    
    # Create and return the step
    step = processor.run(
        inputs=processing_inputs,
        outputs=processing_outputs,
        arguments=script_args,
        job_name=self._generate_job_name(),  # No parameter needed!
        wait=False,
        cache_config=cache_config
    )
    
    return step
```

This approach ensures that all jobs created by your step builders will have consistent and unique naming, making them easier to identify in the SageMaker console and avoiding job name conflicts.

## Builder Examples

### Processing Step Builder

```python
class TabularPreprocessingStepBuilder(StepBuilderBase):
    """Builder for TabularPreprocessing step."""
    
    def __init__(self, config, **kwargs):
        job_type = getattr(config, 'job_type', 'training')
        
        # Select appropriate spec based on job type
        if job_type.lower() == "calibration":
            spec = PREPROCESSING_CALIBRATION_SPEC
        else:
            spec = PREPROCESSING_TRAINING_SPEC
            
        super().__init__(config=config, spec=spec, **kwargs)
    
    def _get_processor(self):
        """Create and return a SageMaker processor."""
        from sagemaker.processing import ScriptProcessor
        
        processor = ScriptProcessor(
            role=self.role,
            image_uri=self.config.get_image_uri(),
            command=["python"],
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            volume_size_in_gb=self.config.volume_size_gb,
            max_runtime_in_seconds=self.config.max_runtime_seconds,
            sagemaker_session=self.sagemaker_session
        )
        
        return processor
    
    def _get_environment_variables(self):
        """Get environment variables for the processor."""
        # Get base environment variables from script contract
        env_vars = super()._get_environment_variables()
        
        # Add or override with specific variables for this step
        env_vars.update({
            "FEATURE_COLUMNS": ",".join(self.config.feature_columns),
            "DEBUG_MODE": str(self.config.debug_mode).lower()
        })
        
        return env_vars
```

### Training Step Builder

```python
class XGBoostTrainingStepBuilder(StepBuilderBase):
    """Builder for XGBoost training step."""
    
    def __init__(self, config, **kwargs):
        super().__init__(config=config, spec=XGBOOST_TRAINING_SPEC, **kwargs)
    
    def _get_hyperparameters(self):
        """Get hyperparameters for the estimator."""
        return {
            "max_depth": str(self.config.max_depth),
            "eta": str(self.config.learning_rate),
            "gamma": str(self.config.gamma),
            "min_child_weight": str(self.config.min_child_weight),
            "subsample": str(self.config.subsample),
            "silent": "0",
            "objective": self.config.objective,
            "num_round": str(self.config.num_round)
        }
    
    def create_step(self, **kwargs):
        """Create the training step."""
        from sagemaker.xgboost.estimator import XGBoost
        from sagemaker.inputs import TrainingInput
        from sagemaker.workflow.steps import TrainingStep
        
        # Extract inputs from dependencies
        dependencies = kwargs.get('dependencies', [])
        extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Create training inputs
        inputs = {}
        for channel_name, s3_uri in extracted_inputs.items():
            inputs[channel_name] = TrainingInput(
                s3_data=s3_uri,
                content_type="csv"
            )
        
        # Create estimator
        estimator = XGBoost(
            entry_point=self.config.get_script_path(),
            hyperparameters=self._get_hyperparameters(),
            role=self.role,
            instance_count=self.config.instance_count,
            instance_type=self.config.instance_type,
            volume_size=self.config.volume_size_gb,
            max_run=self.config.max_runtime_seconds,
            sagemaker_session=self.sagemaker_session,
            framework_version=self.config.framework_version
        )
        
        # Create and return the step
        step_name = kwargs.get('step_name', 'XGBoostTraining')
        step = TrainingStep(
            name=step_name,
            estimator=estimator,
            inputs=inputs
        )
        
        # Store specification in step for future reference
        setattr(step, '_spec', self.spec)
        
        return step
```

## Best Practices

1. **Use Specification-Driven Methods**: Leverage built-in methods for input/output handling
2. **Handle Job Type Variants**: Implement proper selection of specifications for different job types
3. **Validate Configuration**: Add validation to ensure configuration is complete and valid
4. **Provide Meaningful Errors**: Include helpful error messages when validation fails
5. **Log Key Information**: Log important information for debugging
6. **Follow SageMaker Best Practices**: Adhere to SageMaker's conventions for resource creation
7. **Use Strong Typing**: Add type hints to improve code quality
8. **Test Edge Cases**: Write unit tests for various configuration scenarios

By following these guidelines, your step builders will provide a robust implementation that connects specifications to SageMaker steps while maintaining separation of concerns.
