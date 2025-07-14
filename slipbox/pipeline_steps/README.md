# Pipeline Steps Documentation

This directory contains documentation for each step in the MODS_BSM pipeline. Each markdown file provides a detailed description of a specific pipeline step, including its purpose, inputs, outputs, configuration parameters, and usage examples.

> **New Feature**: These pipeline steps can now be used with the [Pipeline Builder Template](../pipeline_builder/README.md) system, which provides a declarative approach to defining pipeline structure and automatically handles the connections between steps.

## Available Pipeline Steps

### Data Loading and Preprocessing
- [Cradle Data Load Step](data_load_step_cradle.md): Loads data from various sources (MDS, EDX, or ANDES) using the Cradle service
- [Tabular Preprocessing Step](tabular_preprocessing_step.md): Prepares tabular data for model training
- [Risk Table Mapping Step](risk_table_map_step.md): Processes raw data and applies risk table mappings
- [Currency Conversion Step](currency_conversion_step.md): Performs currency normalization on monetary values

### Model Configuration
- [Hyperparameter Preparation Step](hyperparameter_prep_step.md): Serializes model hyperparameters to JSON and uploads them to S3

### Model Training
- [PyTorch Training Step](training_step_pytorch.md): Configures and executes a PyTorch model training job
- [XGBoost Training Step](training_step_xgboost.md): Configures and executes an XGBoost model training job

### Model Evaluation and Transformation
- [XGBoost Model Evaluation Step](model_eval_step_xgboost.md): Evaluates a trained XGBoost model on a specified dataset
- [Batch Transform Step](batch_transform_step.md): Generates predictions using a trained model

### Model Packaging and Registration
- [XGBoost Model Step](model_step_xgboost.md): Creates a SageMaker model artifact from a trained XGBoost model
- [MIMS Packaging Step](mims_packaging_step.md): Prepares a trained model for deployment in MIMS
- [MIMS Payload Step](mims_payload_step.md): Generates and uploads test payloads for model testing
- [MIMS Registration Step](mims_registration_step.md): Registers a packaged model with MIMS

## Pipeline Architecture

Each step in the pipeline follows a consistent pattern:
- **Config Class (`config_xxx_step.py`)**: Defines the configuration parameters for the step using Pydantic models
- **Builder Class (`builder_xxx_step.py`)**: Implements the logic to create a SageMaker Pipeline step using the configuration

## Common Base Classes
- **BasePipelineConfig**: Base configuration class with common parameters
- **ProcessingStepConfigBase**: Base configuration for processing steps
- **StepBuilderBase**: Base builder class with common functionality
- **ModelHyperparameters**: Base class for model hyperparameters
- **XGBoostModelHyperparameters**: XGBoost-specific hyperparameters

## Usage Patterns

### Traditional Pattern

The traditional usage pattern for these steps is:

1. Create a configuration object with the required parameters
2. Create a builder object with the configuration
3. Use the builder to create a step with appropriate inputs and dependencies
4. Add the step to the pipeline

Example:
```python
# Create configuration
config = StepConfig(param1="value1", param2="value2")

# Create builder
builder = StepBuilder(config=config)

# Create step
step = builder.create_step(
    input_data=previous_step.properties.OutputPath,
    dependencies=[previous_step]
)

# Add to pipeline
pipeline.add_step(step)
```

### Template-Based Pattern

With the new [Pipeline Builder Template](../pipeline_builder/README.md) system, you can use a more declarative approach:

1. Define the pipeline structure as a DAG
2. Create a config map that maps step names to configuration instances
3. Create a step builder map that maps step types to step builder classes
4. Use the template to generate the pipeline

Example:
```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_edge("data_load", "preprocess")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
}

# Create the step builder map
step_builder_map = {
    "DataLoadStep": DataLoadStepBuilder,
    "PreprocessStep": PreprocessStepBuilder,
}

# Create the template
template = PipelineBuilderTemplate(
    dag=dag,
    config_map=config_map,
    step_builder_map=step_builder_map,
    sagemaker_session=sagemaker_session,
    role=role,
)

# Generate the pipeline
pipeline = template.generate_pipeline("my-pipeline")
```

This template-based approach automatically handles the connections between steps, eliminating the need for manual wiring of inputs and outputs. It's particularly valuable for handling placeholder variables like `dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`.

See the [Pipeline Builder](../pipeline_builder/README.md) documentation for more details.
