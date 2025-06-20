# XGBoost Training Step

## Task Summary
The XGBoost Training Step configures and executes an XGBoost model training job in SageMaker. This step:

1. Uploads hyperparameters as a JSON file to S3
2. Creates an XGBoost estimator with the specified configuration
3. Configures input channels for training, validation, and test data
4. Executes the training job with the specified instance type and count
5. Outputs the trained model artifacts to S3

## Input and Output Format

### Input
- **Train Data**: Training dataset from S3 (train/ subfolder)
- **Validation Data**: Validation dataset from S3 (val/ subfolder)
- **Test Data**: Test dataset from S3 (test/ subfolder)
- **Config**: Hyperparameters JSON file uploaded to S3
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Model Artifacts**: Trained XGBoost model artifacts stored in S3
- **TrainingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| input_path | S3 path for input training data | Generated from bucket and pipeline name |
| output_path | S3 path for output model artifacts | Generated from bucket and pipeline name |
| checkpoint_path | Optional S3 path for model checkpoints | Generated from bucket and pipeline name |
| training_instance_type | Instance type for XGBoost training job | ml.m5.xlarge |
| training_instance_count | Number of instances for XGBoost training job | 1 |
| training_volume_size | Volume size (GB) for training instances | 30 |
| training_entry_point | Entry point script for XGBoost training | train_xgb.py |
| framework_version | SageMaker XGBoost framework version | 1.7-1 |
| py_version | Python version for the SageMaker XGBoost container | py3 |
| hyperparameters | XGBoost model hyperparameters | Required |
| hyperparameters_s3_uri | S3 URI prefix for hyperparameters.json | Generated from bucket and pipeline name |

## Validation Rules
- input_path, output_path, and checkpoint_path must be valid S3 URIs
- All defined paths (input, output, checkpoint) must be unique
- Paths must have at least 2 levels of hierarchy (bucket + prefix)
- training_instance_type must be a valid SageMaker instance type for XGBoost
- All fields in tab_field_list and cat_field_list must be in full_field_list
- label_name and id_name must be in full_field_list

## Environment Variables
The training step sets the following environment variables for the training job:
- **CA_REPOSITORY_ARN**: ARN for the secure PyPI repository

## Usage Example
```python
from src.pipelines.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipelines.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipelines.hyperparameters_xgboost import XGBoostModelHyperparameters

# Create hyperparameters
hyperparams = XGBoostModelHyperparameters(
    id_name="customer_id",
    label_name="target",
    full_field_list=["customer_id", "target", "feature1", "feature2", "feature3"],
    tab_field_list=["feature1", "feature2"],
    cat_field_list=["feature3"],
    xgb_params={
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.3,
        "num_round": 100
    }
)

# Create configuration
config = XGBoostTrainingConfig(
    input_path="s3://my-bucket/preprocessed-data/",
    output_path="s3://my-bucket/model-artifacts/",
    training_instance_type="ml.m5.2xlarge",
    training_instance_count=1,
    hyperparameters=hyperparams
)

# Create builder and step
builder = XGBoostTrainingStepBuilder(config=config)
training_step = builder.create_step(
    dependencies=[preprocessing_step]
)

# Add to pipeline
pipeline.add_step(training_step)
```

## Input Channels
The training step configures the following input channels for the XGBoost estimator:
- **train**: Training data from {input_path}/train/
- **val**: Validation data from {input_path}/val/
- **test**: Test data from {input_path}/test/
- **config**: Hyperparameters JSON file from {hyperparameters_s3_uri}/hyperparameters.json
