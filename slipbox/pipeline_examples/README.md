# MODS_BSM Pipeline Examples

This directory contains documentation for the various pipeline examples in the MODS_BSM system. Each markdown file provides a detailed description of a specific pipeline, including its steps, connections between steps, and input/output relationships.

## Available Pipelines

### XGBoost Pipelines

- [XGBoost End-to-End Pipeline](mods_pipeline_xgboost_end_to_end.md): A complete machine learning workflow including data loading, preprocessing, model training, model creation, packaging, and registration.

- [XGBoost End-to-End Simple Pipeline](mods_pipeline_xgboost_end_to_end_simple.md): A streamlined version of the end-to-end pipeline that skips the explicit model creation step.

### PyTorch Pipelines

- [PyTorch BSM Pipeline](mods_pipeline_bsm_pytorch.md): A model deployment pipeline that takes a pre-trained PyTorch model and prepares it for deployment.

## Pipeline Architecture

All pipelines follow a consistent design pattern:

1. **Configuration-Driven**: Pipelines extract their settings from JSON configuration files
2. **Builder Pattern**: Each pipeline uses a builder class to construct the pipeline
3. **Step-Based**: Pipelines are composed of individual steps that are connected together
4. **Execution Document Support**: Pipelines provide methods to fill execution documents with step configurations

## Common Pipeline Components

Most pipelines include some combination of these components:

- **Data Loading**: Using the [Cradle Data Load Step](../pipelines/data_load_step_cradle.md)
- **Data Preprocessing**: Using the [Tabular Preprocessing Step](../pipelines/tabular_preprocessing_step.md)
- **Model Training**: Using either [XGBoost Training Step](../pipelines/training_step_xgboost.md) or [PyTorch Training Step](../pipelines/training_step_pytorch.md)
- **Model Creation**: Using either [XGBoost Model Step](../pipelines/model_step_xgboost.md) or [PyTorch Model Step](../pipelines/model_step_pytorch.md)
- **Model Packaging**: Using the [MIMS Packaging Step](../pipelines/mims_packaging_step.md)
- **Model Registration**: Using the [MIMS Registration Step](../pipelines/mims_registration_step.md)

## Pipeline Flow Patterns

The pipelines demonstrate several common flow patterns:

1. **Training Flow**: Data Loading → Preprocessing → Training → Model Creation → Packaging → Registration
2. **Calibration Flow**: Data Loading → Preprocessing
3. **Deployment Flow**: Model Creation → Packaging → Registration

## Usage Pattern

The typical usage pattern for these pipelines is:

1. Create a pipeline builder with a configuration file
2. Generate the pipeline
3. Execute the pipeline
4. Fill the execution document with step configurations

Example:
```python
# Create builder
builder = MDSXGBoostPipelineBuilder(config_path="path/to/config.json")

# Generate pipeline
pipeline = builder.generate_pipeline()

# Execute pipeline
execution = pipeline.start()

# Fill execution document
execution_doc = builder.fill_execution_document(execution.describe())
```
