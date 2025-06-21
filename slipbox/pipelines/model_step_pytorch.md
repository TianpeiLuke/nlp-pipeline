# PyTorch Model Step

## Task Summary
The PyTorch Model Step creates a SageMaker model artifact from a trained PyTorch model. This step:

1. Takes trained model artifacts from a previous training step
2. Creates a SageMaker model with the specified inference configuration
3. Configures the model with appropriate environment variables and resource limits
4. Prepares the model for deployment or batch transformation

## Input and Output Format

### Input
- **Model Data**: S3 path to trained PyTorch model artifacts
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Model**: SageMaker model artifact that can be deployed to an endpoint or used for batch transformation
- **ModelStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| inference_instance_type | Instance type for inference endpoint/transform job | ml.m5.large |
| inference_entry_point | Entry point script for inference | inference.py |
| source_dir | Directory containing inference scripts | Required |
| framework_version | PyTorch framework version | Inherited from base |
| py_version | Python version | Inherited from base |
| initial_instance_count | Initial instance count for endpoint | 1 |
| container_startup_health_check_timeout | Container startup health check timeout (seconds) | 300 |
| container_memory_limit | Container memory limit (MB) | 6144 |
| data_download_timeout | Model data download timeout (seconds) | 900 |
| inference_memory_limit | Inference memory limit (MB) | 6144 |
| max_concurrent_invocations | Max concurrent invocations per instance | 10 |
| max_payload_size | Max payload size (MB) for inference | 6 |

## Validation Rules
- inference_entry_point must be provided
- inference_instance_type must start with 'ml.'
- inference_memory_limit cannot exceed container_memory_limit
- container_startup_health_check_timeout should not exceed data_download_timeout
- If source_dir is provided and not an S3 URI, the inference entry point script must exist

## Environment Variables
The model step sets the following environment variables for the model container:

| Environment Variable | Description | Value |
|---------------------|-------------|-------|
| MMS_DEFAULT_RESPONSE_TIMEOUT | Container startup health check timeout | From config |
| SAGEMAKER_CONTAINER_LOG_LEVEL | Log level | 20 |
| SAGEMAKER_PROGRAM | Entry point script | From config |
| SAGEMAKER_SUBMIT_DIRECTORY | Code directory | /opt/ml/model/code |
| SAGEMAKER_CONTAINER_MEMORY_LIMIT | Container memory limit | From config |
| SAGEMAKER_MODEL_DATA_DOWNLOAD_TIMEOUT | Data download timeout | From config |
| SAGEMAKER_INFERENCE_MEMORY_LIMIT | Inference memory limit | From config |
| SAGEMAKER_MAX_CONCURRENT_INVOCATIONS | Max concurrent invocations | From config |
| SAGEMAKER_MAX_PAYLOAD_IN_MB | Max payload size | From config |
| AWS_REGION | AWS region | From session |

## Usage Example
```python
from src.pipelines.config_model_step_pytorch import PytorchModelCreationConfig
from src.pipelines.builder_model_step_pytorch import PytorchModelStepBuilder

# Create configuration
config = PytorchModelCreationConfig(
    inference_instance_type="ml.m5.large",
    inference_entry_point="inference.py",
    source_dir="s3://my-bucket/inference-scripts/",
    framework_version="1.13.1",
    py_version="py39",
    container_memory_limit=8192,
    inference_memory_limit=6144,
    max_concurrent_invocations=5,
    max_payload_size=10
)

# Create builder and step
builder = PytorchModelStepBuilder(config=config)
model_step = builder.create_step(
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    dependencies=[training_step]
)

# Add to pipeline
pipeline.add_step(model_step)
```
