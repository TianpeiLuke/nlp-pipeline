# Currency Conversion Step

## Task Summary
The Currency Conversion Step performs currency normalization on monetary values in the dataset. This step:

1. Takes input data from a previous processing step (typically TabularPreprocessingStep)
2. Converts monetary values from various currencies to a standard currency using provided conversion rates
3. Optionally splits the data into training/validation/testing sets after conversion
4. Outputs the converted data to S3 for use in subsequent pipeline steps

## Input and Output Format

### Input
- **Data Input**: Processed tabular data from a previous step (typically TabularPreprocessingStep)
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Converted Data**: Dataset with monetary values converted to a standard currency
- **ProcessingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| job_type | Dataset type ('training', 'validation', 'testing', 'calibration') | validation |
| mode | Processing mode ('per_split', 'split_after_conversion') | per_split |
| train_ratio | Train fraction when split_after_conversion | 0.7 |
| test_val_ratio | Test vs val split within holdout | 0.5 |
| label_field | Label column name for stratified splitting | Required |
| processing_entry_point | Entry point script for currency conversion | currency_conversion.py |
| use_large_processing_instance | Whether to use large instance type | False |
| marketplace_id_col | Column with marketplace IDs | Required |
| currency_col | Optional column with currency codes | None (infer from marketplace_info) |
| currency_conversion_var_list | Which numeric columns to convert | [] |
| currency_conversion_dict | Map currency code → conversion rate | Required |
| marketplace_info | Map marketplace ID → {'currency_code':...} | Required |
| enable_currency_conversion | Turn off conversion if False | True |
| default_currency | Fallback currency code | USD |
| skip_invalid_currencies | If True, fill invalid codes with default_currency | False |
| input_names | Input channel names | {"data_input": "ProcessedTabularData"} |
| output_names | Output channel names | {"converted_data": "ConvertedCurrencyData"} |

## Validation Rules
- job_type must be one of: 'training', 'validation', 'testing', 'calibration'
- mode must be one of: 'per_split', 'split_after_conversion'
- currency_conversion_dict cannot be empty and must include a rate of 1.0
- All conversion rates must be positive
- When currency conversion is enabled:
  - marketplace_id_col is required
  - currency_conversion_var_list cannot be empty
  - marketplace_info must be provided
  - For split_after_conversion mode, label_field is required for stratification

## Usage Example
```python
from src.pipeline_steps.config_currency_conversion_step import CurrencyConversionConfig
from src.pipeline_steps.builder_currency_conversion_step import CurrencyConversionStepBuilder

# Create configuration
config = CurrencyConversionConfig(
    job_type="training",
    label_field="target_column",
    marketplace_id_col="marketplace_id",
    currency_conversion_var_list=["price", "cost", "revenue"],
    currency_conversion_dict={
        "USD": 1.0,
        "EUR": 1.1,
        "GBP": 1.3,
        "JPY": 0.0091
    },
    marketplace_info={
        "US": {"currency_code": "USD"},
        "UK": {"currency_code": "GBP"},
        "DE": {"currency_code": "EUR"},
        "JP": {"currency_code": "JPY"}
    }
)

# Create builder and step
builder = CurrencyConversionStepBuilder(config=config)
conversion_step = builder.create_step(
    data_input=preprocessing_step.properties.ProcessingOutputConfig.Outputs["ProcessedTabularData"].S3Output.S3Uri,
    dependencies=[preprocessing_step]
)

# Add to pipeline
pipeline.add_step(conversion_step)
```

## Integration with Pipeline Builder Template

### Input Arguments

The `CurrencyConversionStepBuilder` defines the following input arguments that can be automatically connected by the Pipeline Builder Template:

| Argument | Description | Required | Source |
|----------|-------------|----------|--------|
| data_input | Processed data input location | Yes | Previous step's processed_data output |

### Output Properties

The `CurrencyConversionStepBuilder` provides the following output properties that can be used by subsequent steps:

| Property | Description | Access Pattern |
|----------|-------------|---------------|
| converted_data | Currency-converted data location | `step.properties.ProcessingOutputConfig.Outputs["converted_data"].S3Output.S3Uri` |

### Usage with Pipeline Builder Template

When using the Pipeline Builder Template, the inputs and outputs are automatically connected based on the DAG structure:

```python
# Create the DAG
dag = PipelineDAG()
dag.add_node("data_load")
dag.add_node("preprocess")
dag.add_node("currency_conversion")
dag.add_node("train")
dag.add_edge("data_load", "preprocess")
dag.add_edge("preprocess", "currency_conversion")
dag.add_edge("currency_conversion", "train")

# Create the config map
config_map = {
    "data_load": data_load_config,
    "preprocess": preprocess_config,
    "currency_conversion": currency_conversion_config,
    "train": train_config,
}

# Create the step builder map
step_builder_map = {
    "CradleDataLoadStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "CurrencyConversionStep": CurrencyConversionStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
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
