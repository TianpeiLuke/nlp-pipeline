# Essential Inputs User Guide

This document serves as a user guide for the simplified XGBoost evaluation pipeline configuration approach. It explains how to use the streamlined interface to configure a pipeline with minimal input while maintaining full control over the resulting pipeline.

## Introduction

The Essential Inputs approach dramatically simplifies the pipeline configuration process by focusing on only three key input areas:

1. **Data Configuration** - Where your data comes from and what fields to use
2. **Model Configuration** - Core model settings and hyperparameters
3. **Registration Configuration** - How the model will be deployed

All other configuration details are automatically derived from these essential inputs using smart defaults and best practices.

## Getting Started

### Prerequisites

Before using the simplified configuration notebook, ensure you have:

1. Access to SAIS environment
2. Proper permissions for MDS/EDX data sources
3. Necessary Python dependencies installed

### Opening the Notebook

1. Navigate to the notebook directory
2. Open `template_config_xgb_eval_simplified.ipynb`
3. Run the setup cells to initialize the environment

## Step-by-Step Configuration Guide

### Step 1: Data Configuration

#### Region Selection

Select your region from the dropdown:

```
Region: [North America (NA) ▼]
```

Available options:
- North America (NA)
- Europe (EU)
- Far East (FE)

#### Date Range Configuration

Set the date ranges for training and calibration data:

```
Training Period:
    Start Date: [2025-01-01] End Date: [2025-04-17]
    
Calibration Period:
    Start Date: [2025-04-17] End Date: [2025-04-28]
```

**Important**: Ensure there's no overlap between training and calibration periods to prevent data leakage.

#### Feature Group Selection

Select which feature groups to include in your model:

```
[ ✓ ] Buyer Profile Metrics      - Account age, history metrics
[ ✓ ] Order Behavior Metrics     - Order frequency, value stats
[ ✓ ] Refund and Claim Metrics   - Return history, claim patterns
[ ✓ ] Shipping Status Metrics    - Delivery status patterns
[ ✓ ] Message Metrics            - Buyer-seller communication stats
[ ✓ ] Abuse Pattern Indicators   - Previous abuse flags, warnings
```

To view what fields are included in each feature group, click the "View Fields" button next to the group name.

#### Custom Fields

If you need additional fields not covered by the feature groups, you can add them to the custom fields section:

```
Custom Fields:
[baseCurrency                              ] [Add]
[Abuse.currency_exchange_rate_inline.exchangeRate] [Add]
```

### Step 2: Model Configuration

#### Model Type

Select the model type:

```
Model Type: [Binary Classification ▼]
```

Available options:
- Binary Classification
- Multi-class Classification

#### Target and ID Fields

Specify the label field and ID field:

```
Label Field: [is_abuse]
ID Field: [order_id]
Marketplace ID Field: [marketplace_id]
```

#### XGBoost Hyperparameters

Configure the core XGBoost hyperparameters:

```
Number of Rounds: [300]
Maximum Depth: [10]
Min Child Weight: [1]
```

#### Evaluation Metrics

Select evaluation metrics:

```
[ ✓ ] F1 Score
[ ✓ ] AUC-ROC
[ ] Precision
[ ] Recall
[ ] Accuracy
```

### Step 3: Registration Configuration

#### Model Ownership

Specify the model owner and domain:

```
Model Owner: [amzn1.abacus.team.djmdvixm5abr3p75c5ca]
Model Domain: [AtoZ]
```

#### Performance Requirements

Set the expected performance characteristics:

```
Expected TPS: [2]
Max Latency (ms): [800]
Max Error Rate: [0.2]
```

## Reviewing and Generating Configuration

After completing the essential inputs, you can generate the full configuration:

1. Click the "Generate Configuration" button
2. The system will display a preview of the derived configuration
3. Review the automatically generated settings

### Configuration Preview

The configuration preview shows a summary of both your inputs and the derived values:

```
Configuration Preview:
------------------------
Region: NA
Pipeline Name: username-AtoZ-XGBoostModel-NA

Data Configuration:
  - Training Period: 2025-01-01 to 2025-04-17
  - Calibration Period: 2025-04-17 to 2025-04-28
  - Feature Groups: 6 enabled (67 total fields)
  - Data Sources: MDS (AtoZ), EDX (trms-abuse-analytics/qingyuye-notr-exp/atoz-tag)

Model Configuration:
  - Type: Binary Classification
  - Fields: 55 tabular, 4 categorical
  - Hyperparameters: num_round=300, max_depth=10, min_child_weight=1
  - Objective: binary:logistic

Registration Configuration:
  - Model Objective: AtoZ_Claims_SM_Model_NA
  - Expected TPS: 2
  - Max Latency: 800ms
```

### Advanced Configuration Options

If you need to customize any of the derived values:

1. Click "Show Advanced Options"
2. This will expand sections for each derived configuration area
3. Modify any values as needed
4. Click "Update Configuration" to regenerate with your customizations

## Executing the Pipeline

After finalizing your configuration:

1. Click "Save Configuration" to export the configuration to JSON
2. The notebook will ask if you want to proceed with pipeline execution
3. If you click "Yes", the notebook will execute the pipeline using your configuration

## Troubleshooting

### Common Issues and Solutions

#### "Invalid feature group selection"

**Problem**: The system indicates that your feature group selection is invalid.

**Solution**: Ensure you have selected at least one feature group. If the error persists, try selecting the "Buyer Profile" and "Order Behavior" groups as a minimum.

#### "Date range error"

**Problem**: Error indicating issues with the date ranges.

**Solution**: Ensure that:
- The training end date is not after the calibration start date
- The date format is correct (YYYY-MM-DDThh:mm:ss)
- The date ranges are reasonable (not too short or too long)

#### "Missing required fields"

**Problem**: The system indicates that required fields are missing.

**Solution**: Ensure that all essential fields are populated:
- Region must be selected
- Date ranges must be specified
- Label field and ID field must be provided

## Tips and Best Practices

1. **Feature Group Selection**:
   - Start with all feature groups enabled, then selectively disable if needed
   - The "Buyer Profile" and "Order Behavior" groups provide core features

2. **Date Ranges**:
   - For initial model development, use 3-4 months of training data
   - For calibration, use 1-2 weeks of data

3. **Hyperparameters**:
   - The default values work well in most cases
   - For large datasets, consider increasing `num_round` to 500
   - For complex relationships, consider increasing `max_depth` to 15

4. **Configuration Review**:
   - Always review the generated configuration before execution
   - Pay special attention to the field lists and SQL transformations

## Advanced Topics

### Custom Templates

You can save and load configuration templates:

1. Configure the essential inputs as desired
2. Click "Save as Template"
3. Provide a name for your template
4. To load a template, select it from the "Load Template" dropdown

### Configuration Export Options

You can export the configuration in different formats:

- **Full JSON**: Complete configuration with all derived values
- **Essential JSON**: Only your essential inputs (can be shared/reused)
- **Python Dict**: Python dictionary representation

### Custom SQL Transformations

If you need to customize the SQL transformations:

1. Click "Show Advanced Options"
2. Expand the "Data Loading" section
3. Modify the transform SQL as needed
4. Click "Update Configuration"

## Conclusion

The Essential Inputs approach provides a streamlined way to configure XGBoost evaluation pipelines with minimal effort. By focusing on just the essential inputs and automating everything else, you can create pipelines faster and with fewer errors.

Remember that you always have the option to customize any part of the configuration through the advanced options if needed.

For more detailed information, refer to the following resources:
- Essential Inputs Notebook Design document
- Essential Inputs Implementation Strategy document
- XGBoost Pipeline Technical Documentation
