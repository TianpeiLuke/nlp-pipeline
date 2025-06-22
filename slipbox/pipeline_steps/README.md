# Pipeline Steps Documentation

This directory contains documentation for each step in the MODS_BSM pipeline. Each markdown file provides a detailed description of a specific pipeline step, including its purpose, inputs, outputs, configuration parameters, and usage examples.

## Available Pipeline Steps

### Data Loading and Preprocessing
- [Cradle Data Load Step](data_load_step_cradle.md): Loads data from various sources (MDS, EDX, or ANDES) using the Cradle service
- [Tabular Preprocessing Step](tabular_preprocessing_step.md): Prepares tabular data for model training
- [Risk Table Mapping Step](risk_table_map_step.md): Processes raw data and applies risk table mappings
- [Currency Conversion Step](currency_conversion_step.md): Performs currency normalization on monetary values

### Model Training
- [PyTorch Training Step](training_step_pytorch.md): Configures and executes a PyTorch model training job
- [XGBoost Training Step](training_step_xgboost.md): Configures and executes an XGBoost model training job

### Model Evaluation and Transformation
- [XGBoost Model Evaluation Step](model_eval_step_xgboost.md): Evaluates a trained XGBoost model on a specified dataset
- [Batch Transform Step](batch_transform_step.md): Generates predictions using a trained model

### Model Packaging and Registration
- [XGBoost Model Step](model_step_xgboost.md): Creates a SageMaker model artifact from a trained XGBoost model
- [MIMS Packaging Step](mims_packaging_step.md): Prepares a trained model for deployment in MIMS
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

## Usage Pattern

The typical usage pattern for these steps is:

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
