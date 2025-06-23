# Tabular Preprocessing Step

## Task Summary
The Tabular Preprocessing Step prepares tabular data for model training by performing various preprocessing operations. This step:

1. Takes raw tabular data as input
2. Performs data cleaning, transformation, and feature engineering
3. Optionally splits the data into training, validation, and testing sets
4. Outputs the processed data to S3 for use in subsequent pipeline steps

## Input and Output Format

### Input
- **Raw Data**: Raw tabular data from a previous step or S3 location
- **Optional Metadata**: Metadata about the dataset (optional)
- **Optional Signature**: Data signatures for verification (optional)

### Output
- **Processed Data**: Preprocessed tabular data ready for model training
- **Full Data**: Complete dataset before splitting (if applicable)
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| processing_entry_point | Relative path to preprocessing script | tabular_preprocess.py |
| processing_source_dir | Directory containing processing scripts | Required |
| hyperparameters | Model hyperparameters (only label_name is used) | ModelHyperparameters() |
| job_type | Dataset type ('training', 'validation', 'testing', 'calibration') | training |
| train_ratio | Fraction of data for training set | 0.7 |
| test_val_ratio | Fraction of holdout for test vs validation | 0.5 |
| input_names | Dictionary mapping input names | {"data_input": "RawData", "metadata_input": "Metadata", "signature_input": "Signature"} |
| output_names | Dictionary mapping output names | {"processed_data": "ProcessedTabularData", "full_data": "FullTabularData"} |

## Validation Rules
- processing_entry_point must be a non-empty relative path
- job_type must be one of: 'training', 'validation', 'testing', 'calibration'
- train_ratio and test_val_ratio must be strictly between 0 and 1
- hyperparameters.label_name must be provided and non-empty
- input_names must contain key 'data_input'
- output_names must contain keys 'processed_data' and 'full_data'
- Input channel names must be one of: 'data_input', 'metadata_input', 'signature_input'

## Environment Variables
The preprocessing step sets the following environment variables for the processing job:
- **LABEL_FIELD**: The name of the label field from hyperparameters
- **TRAIN_RATIO**: The fraction of data to allocate to the training set
- **TEST_VAL_RATIO**: The fraction of the holdout to allocate to the test set vs. validation

## Usage Example
```python
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

# Create hyperparameters
hyperparams = ModelHyperparameters(
    label_name="target",
    feature_names=["feature1", "feature2", "feature3"]
)

# Create configuration
config = TabularPreprocessingConfig(
    processing_entry_point="tabular_preprocess.py",
    processing_source_dir="s3://my-bucket/scripts/",
    hyperparameters=hyperparams,
    job_type="training",
    train_ratio=0.8,
    test_val_ratio=0.5
)

# Create builder
builder = TabularPreprocessingStepBuilder(config=config)

# Define input and output locations
inputs = {
    "data_input": "s3://my-bucket/raw-data/"
}

outputs = {
    "processed_data": "s3://my-bucket/processed-data/",
    "full_data": "s3://my-bucket/full-data/"
}

# Create step
preprocessing_step = builder.create_step(
    inputs=inputs,
    outputs=outputs,
    enable_caching=True
)

# Add to pipeline
pipeline.add_step(preprocessing_step)
```

## Processing Inputs and Outputs

### Processing Inputs
- **data_input**: Raw data input (destination: /opt/ml/processing/input/data)
- **metadata_input**: Optional metadata input
- **signature_input**: Optional signature input

### Processing Outputs
- **processed_data**: Processed data output (source: /opt/ml/processing/output)
- **full_data**: Full data output before splitting

## Integration with Pipeline Builder Template

### Input Arguments

The `TabularPreprocessingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| data_input | Raw data input location | Yes | Previous step's data output |
| metadata_input | Metadata input location | No | Previous step's metadata output |
| signature_input | Signature input location | No | Previous step's signature output |

### Output Properties

The `TabularPreprocessingStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| processed_data | Processed data location | `step.properties.ProcessingOutputConfig.Outputs["processed_data"].S3Output.S3Uri` |
| full_data | Full data location | `step.properties.ProcessingOutputConfig.Outputs["full_data"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

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
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
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

For more details on how the Pipeline Builder Template handles connections between steps, see the [Pipeline Builder documentation](../pipeline_builder/README.md).
