# MIMS Registration Step

## Task Summary
The MIMS Registration Step registers a packaged model with the Model Inventory Management System (MIMS). This step:

1. Takes a packaged model artifact from a previous packaging step
2. Registers the model with MIMS, providing metadata about the model's inputs, outputs, and usage
3. Optionally registers the model in multiple AWS regions
4. Creates a model package group in SageMaker Model Registry

This step is the final component in the MIMS registration workflow:
- It takes the packaged model from the [MIMS Packaging Step](mims_packaging_step.md) as a required input
- It takes the test payloads from the [MIMS Payload Step](mims_payload_step.md) as a required input
- It combines these inputs to register the model with MIMS and make it available for deployment

In the pipeline templates (e.g., `template_pipeline_pytorch_model_registration.py`), this step is positioned at the end of the pipeline, after both the packaging and payload steps, as seen in the DAG configuration:
```python
# Define the DAG structure
nodes = ["CreatePytorchModelStep", "PackagingStep", "PayloadStep", "RegistrationStep"]
edges = [
    ("CreatePytorchModelStep", "PackagingStep"),
    ("PackagingStep", "PayloadStep"),
    ("PayloadStep", "RegistrationStep")
]
```

## Input and Output Format

### Input
- **Packaged Model**: Model packaged according to MIMS requirements from a previous packaging step
- **Optional Payload**: Custom MIMS payload configuration
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Registered Model**: Model registered in MIMS and SageMaker Model Registry
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| framework | ML framework used for the model | xgboost |
| inference_instance_type | Instance type for inference endpoint/transform job | ml.m5.large |
| inference_entry_point | Entry point script for inference | inference.py |
| model_owner | Team ID of model owner | team id |
| model_registration_domain | Domain for model registration | BuyerSellerMessaging |
| model_registration_objective | Objective of model registration | Required |
| source_model_inference_content_types | Content type for model inference input | ["text/csv"] |
| source_model_inference_response_types | Response type for model inference output | ["application/json"] |
| source_model_inference_output_variable_list | Dictionary of output variables and their types | {"legacy-score": "NUMERIC"} |
| source_model_inference_input_variable_list | Input variables and their types | {} |

## Validation Rules
- model_registration_objective must be provided
- inference_instance_type must start with 'ml.'
- framework must be one of: 'xgboost', 'sklearn', 'pytorch', 'tensorflow'
- source_model_inference_content_types must be exactly ["text/csv"] or ["application/json"]
- source_model_inference_response_types must be exactly ["text/csv"] or ["application/json"]
- At least one output variable must be defined
- Variable types must be either 'NUMERIC' or 'TEXT'
- If source_dir is provided and not an S3 URI, the inference entry point script must exist

## Usage Example
```python
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig, VariableType
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder

# Create configuration
config = ModelRegistrationConfig(
    framework="xgboost",
    inference_instance_type="ml.m5.large",
    model_owner="my-team-id",
    model_registration_domain="BuyerSellerMessaging",
    model_registration_objective="FraudDetection",
    source_model_inference_content_types=["text/csv"],
    source_model_inference_response_types=["application/json"],
    source_model_inference_output_variable_list={
        "fraud_score": VariableType.NUMERIC,
        "risk_category": VariableType.TEXT
    },
    source_model_inference_input_variable_list={
        "account_age": VariableType.NUMERIC,
        "transaction_count": VariableType.NUMERIC,
        "country_code": VariableType.TEXT
    }
)

# Create builder and step
builder = ModelRegistrationStepBuilder(config=config)

# Basic usage with just the packaging step
registration_step = builder.create_step(
    packaging_step_output=packaging_step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"].S3Output.S3Uri,
    dependencies=[packaging_step]
)

# Complete usage with both packaging and payload steps
registration_step_with_payload = builder.create_step(
    packaging_step_output=packaging_step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"].S3Output.S3Uri,
    payload_s3_uri=payload_step.properties.payload_s3_uri,
    payload_s3_key=payload_step.properties.payload_s3_key,
    dependencies=[packaging_step, payload_step]
)

# Register in multiple regions
registration_steps = builder.create_step(
    packaging_step_output=packaging_step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"].S3Output.S3Uri,
    payload_s3_uri=payload_step.properties.payload_s3_uri,
    dependencies=[packaging_step, payload_step],
    regions=["us-east-1", "eu-west-1"]
)

# Add to pipeline
pipeline.add_step(registration_step)
```

## Variable Schema
The registration step generates a variable schema for MIMS registration with the following structure:

```json
{
  "input": {
    "variables": [
      {"name": "variable_name", "type": "NUMERIC|TEXT"}
    ]
  },
  "output": {
    "variables": [
      {"name": "variable_name", "type": "NUMERIC|TEXT"}
    ]
  }
}
```

## Integration with Pipeline Builder Template

### Input Arguments

The `ModelRegistrationStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| packaging_step_output | Packaged model location | Yes | Previous packaging step's packaged_model_output |
| payload_s3_uri | Test payload location | No | Previous payload step's payload_s3_uri |
| payload_s3_key | Test payload S3 key | No | Previous payload step's payload_s3_key |

### Output Properties

The `ModelRegistrationStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| registered_model | Registered model information | `step.properties.ProcessingOutputConfig.Outputs["registered_model"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("train")
dag.add_node("package")
dag.add_node("register")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "train")
dag.add_edge("train", "package")
dag.add_edge("package", "register")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "train": train_config,
    "package": package_config,
    "register": register_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
    "MIMSPackagingStep": MIMSPackagingStepBuilder,
    "ModelRegistrationStep": ModelRegistrationStepBuilder,
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

## Related Steps
- [MIMS Packaging Step](mims_packaging_step.md): Prepares a trained model for deployment in MIMS
- [MIMS Payload Step](mims_payload_step.md): Generates test payloads for the model
