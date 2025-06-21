# MIMS Registration Step

## Task Summary
The MIMS Registration Step registers a packaged model with the Model Inventory Management System (MIMS). This step:

1. Takes a packaged model artifact from a previous packaging step
2. Registers the model with MIMS, providing metadata about the model's inputs, outputs, and usage
3. Optionally registers the model in multiple AWS regions
4. Creates a model package group in SageMaker Model Registry

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
from src.pipelines.config_mims_registration_step import ModelRegistrationConfig, VariableType
from src.pipelines.builder_mims_registration_step import ModelRegistrationStepBuilder

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
registration_step = builder.create_step(
    packaging_step_output=packaging_step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"].S3Output.S3Uri,
    dependencies=[packaging_step]
)

# Register in multiple regions
registration_steps = builder.create_step(
    packaging_step_output=packaging_step.properties.ProcessingOutputConfig.Outputs["packaged_model_output"].S3Output.S3Uri,
    dependencies=[packaging_step],
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
