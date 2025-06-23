# PyTorch Training Step

## Task Summary
The PyTorch Training Step configures and executes a PyTorch model training job in SageMaker. This step:

1. Creates a PyTorch estimator with the specified configuration and hyperparameters
2. Configures input channels for training, validation, and test data
3. Sets up checkpointing for model state saving during training
4. Configures profiling and metric monitoring
5. Executes the training job with the specified instance type and count
6. Outputs the trained model artifacts to S3

## Input and Output Format

### Input
- **Train Data**: Training dataset from S3 (train/train.parquet)
- **Validation Data**: Validation dataset from S3 (val/val.parquet)
- **Test Data**: Test dataset from S3 (test/test.parquet)
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Model Artifacts**: Trained PyTorch model artifacts stored in S3
- **Checkpoints**: Model checkpoints saved during training
- **TrainingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| input_path | S3 path for input data | Generated from bucket and current date |
| output_path | S3 path for output data | Generated from bucket and current date |
| checkpoint_path | Optional S3 path for model checkpoints | Generated from bucket and current date |
| training_instance_type | Instance type for training job | ml.g5.12xlarge |
| training_instance_count | Number of instances for training job | 1 |
| training_volume_size | Volume size (GB) for training instances | 30 |
| training_entry_point | Entry point script for training | train.py |
| source_dir | Directory containing training scripts | Inherited from base |
| framework_version | PyTorch framework version | Inherited from base |
| py_version | Python version | Inherited from base |
| hyperparameters | Model hyperparameters | Required |

## Validation Rules
- input_path, output_path, and checkpoint_path must be valid S3 URIs
- All paths (input, output, checkpoint) must be different
- Paths must have at least 2 levels of hierarchy
- training_instance_type must be one of the valid instances
- hyperparameters must be provided
- All fields in tab_field_list and cat_field_list must be in full_field_list
- label_name and id_name must be in full_field_list

## Metrics Monitoring
The training step monitors the following metrics during training:
- **Train Loss**: Training loss value
- **Validation Loss**: Validation loss value
- **Validation F1 Score**: F1 score on validation data
- **Validation AUC ROC**: Area under ROC curve on validation data

## Environment Variables
The training step sets the following environment variables for the training job:
- **CA_REPOSITORY_ARN**: ARN for the secure PyPI repository

## Usage Example
```python
from src.pipeline_steps.config_training_step_pytorch import PytorchTrainingConfig
from src.pipeline_steps.builder_training_step_pytorch import PyTorchTrainingStepBuilder
from src.pipeline_steps.hyperparameters_base import ModelHyperparameters

# Create hyperparameters
hyperparams = ModelHyperparameters(
    id_name="customer_id",
    label_name="target",
    full_field_list=["customer_id", "target", "feature1", "feature2", "feature3"],
    tab_field_list=["feature1", "feature2"],
    cat_field_list=["feature3"],
    learning_rate=0.001,
    batch_size=64,
    epochs=10
)

# Create configuration
config = PytorchTrainingConfig(
    input_path="s3://my-bucket/preprocessed-data/",
    output_path="s3://my-bucket/model-artifacts/",
    checkpoint_path="s3://my-bucket/checkpoints/",
    training_instance_type="ml.g5.12xlarge",
    training_instance_count=1,
    hyperparameters=hyperparams
)

# Create builder and step
builder = PyTorchTrainingStepBuilder(config=config)
training_step = builder.create_step(
    dependencies=[preprocessing_step]
)

# Add to pipeline
pipeline.add_step(training_step)
```

## Input Channels
The training step configures the following input channels for the PyTorch estimator:
- **train**: Training data from {input_path}/train/train.parquet
- **val**: Validation data from {input_path}/val/val.parquet
- **test**: Test data from {input_path}/test/test.parquet

## Integration with Pipeline Builder Template

### Input Arguments

The `PyTorchTrainingStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| train_data | Training data location | Yes | Previous step's processed_data output |
| validation_data | Validation data location | No | Previous step's processed_data output |
| test_data | Test data location | No | Previous step's processed_data output |

### Output Properties

The `PyTorchTrainingStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| model_artifacts | Trained model artifacts | `step.properties.ModelArtifacts.S3ModelArtifacts` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_node("model")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "model")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "model": model_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "PyTorchTrainingStep": PyTorchTrainingStepBuilder,
    "PytorchModelStep": PytorchModelStepBuilder,
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
