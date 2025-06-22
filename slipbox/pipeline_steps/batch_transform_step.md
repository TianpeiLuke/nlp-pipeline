# Batch Transform Step

## Task Summary
The Batch Transform Step creates a SageMaker Batch Transform job to generate predictions using a trained model. This step:

1. Configures a SageMaker Transformer object with specified instance type, count, and output path
2. Sets up the transform input with data location and content negotiation parameters
3. Creates a TransformStep in the SageMaker pipeline with appropriate dependencies
4. Validates the configuration parameters before execution

## Input and Output Format

### Input
- **Model Name**: Name or Properties reference to a trained SageMaker model
- **Input Data**: CSV or other formatted data in S3 that needs predictions
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Predictions**: Model predictions stored in the specified S3 output location
- **TransformStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| job_type | Dataset slice to transform ('training', 'testing', 'validation', 'calibration') | Required |
| batch_input_location | S3 URI for input data | Required |
| batch_output_location | S3 URI for transform outputs | Required |
| transform_instance_type | Instance type for the transform job | ml.m5.large |
| transform_instance_count | Number of instances for the transform job | 1 |
| content_type | MIME type of input data | text/csv |
| accept | Response MIME type | text/csv |
| split_type | How to split the input file | Line |
| assemble_with | How to reassemble input+output | Line |
| input_filter | JMESPath filter on each input record | $[1:] |
| output_filter | JMESPath filter on each joined record | $[-1] |
| join_source | Whether to join on 'Input' or 'Output' stream | Input |

## Validation Rules
- job_type must be one of: 'training', 'testing', 'validation', 'calibration'
- batch_input_location and batch_output_location must be valid S3 URIs
- transform_instance_type must start with 'ml.'
- When join_source='Input', assemble_with must equal split_type

## Usage Example
```python
from src.pipeline_steps.config_batch_transform_step import BatchTransformStepConfig
from src.pipeline_steps.builder_batch_transform_step import BatchTransformStepBuilder

# Create configuration
config = BatchTransformStepConfig(
    job_type="testing",
    batch_input_location="s3://my-bucket/test-data/",
    batch_output_location="s3://my-bucket/test-predictions/",
    transform_instance_type="ml.c5.xlarge",
    transform_instance_count=1
)

# Create builder and step
builder = BatchTransformStepBuilder(config=config)
transform_step = builder.create_step(
    model_name=model_step.properties.ModelName,
    dependencies=[preprocessing_step]
)

# Add to pipeline
pipeline.add_step(transform_step)
```
