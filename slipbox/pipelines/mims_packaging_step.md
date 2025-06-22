# MIMS Packaging Step

## Task Summary
The MIMS Packaging Step prepares a trained model for deployment in the Model Inventory Management System (MIMS). This step:

1. Takes a trained model artifact and inference scripts as inputs
2. Packages them together in a format compatible with MIMS requirements
3. Outputs the packaged model to an S3 location
4. Prepares the model for subsequent registration with MIMS

## Input and Output Format

### Input
- **Model Artifacts**: Trained model artifacts from a previous training or model step
- **Inference Scripts**: Scripts needed for model inference (inference.py, etc.)
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Packaged Model**: Model packaged according to MIMS requirements, stored in S3
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| processing_entry_point | Entry point script for packaging | mims_package.py |
| processing_source_dir | Directory containing processing scripts | Required |
| source_dir | Directory containing inference scripts | Required |
| processing_framework_version | SKLearn framework version | Inherited from base |
| processing_instance_type_small | Instance type for small processing | Inherited from base |
| processing_instance_type_large | Instance type for large processing | Inherited from base |
| processing_instance_count | Number of instances for processing | Inherited from base |
| processing_volume_size | EBS volume size for processing | Inherited from base |
| use_large_processing_instance | Whether to use large instance type | Inherited from base |
| input_names | Dictionary mapping input names to descriptions | Default dictionary |
| output_names | Dictionary mapping output names to descriptions | Default dictionary |
| enable_caching_package_step | Whether to enable caching for the step | True |

## Validation Rules
- Either processing_source_dir or source_dir must be set
- Required input names 'model_input' and 'inference_scripts_input' must be defined
- Required output name 'packaged_model_output' must be defined
- processing_entry_point must be provided

## Usage Example
```python
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder

# Create configuration
config = PackageStepConfig(
    processing_entry_point="mims_package.py",
    source_dir="s3://my-bucket/inference-scripts/",
    processing_source_dir="s3://my-bucket/processing-scripts/",
    pipeline_name="MyModelPipeline",
    pipeline_s3_loc="s3://my-bucket/pipeline-outputs/"
)

# Create builder and step
builder = MIMSPackagingStepBuilder(config=config)
packaging_step = builder.create_step(
    model_artifacts_input_source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    dependencies=[training_step]
)

# Add to pipeline
pipeline.add_step(packaging_step)
```

## Default Input/Output Names

### Default Input Names
- **model_input**: Input name for model artifacts
- **inference_scripts_input**: Input name for inference scripts

### Default Output Names
- **packaged_model_output**: Output name for the packaged model
