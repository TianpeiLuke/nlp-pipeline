# Pipeline Step Specifications

This directory contains declarative specifications for all pipeline step types, defining their input dependencies and output properties for automatic dependency resolution.

## Overview

Each step specification defines:
- **Step Type**: The type identifier for the step
- **Dependencies**: Required and optional inputs with semantic matching criteria
- **Outputs**: Available outputs with property paths for runtime access

## Available Specifications

### 1. Data Loading Specification (`data_loading_spec.py`)
- **Step Type**: `CradleDataLoading`
- **Dependencies**: None (source step)
- **Outputs**:
  - `DATA`: Main data output from Cradle data loading
  - `METADATA`: Metadata output from Cradle data loading  
  - `SIGNATURE`: Signature output from Cradle data loading

### 2. Preprocessing Specification (`preprocessing_spec.py`)
- **Step Type**: `TabularPreprocessing`
- **Dependencies**:
  - `input_data` (required): Training data from data loading or other processing steps
- **Outputs**:
  - `processed_data`: Preprocessed tabular data ready for training
  - `ProcessedTabularData`: Alias for processed_data

### 3. XGBoost Training Specification (`xgboost_training_spec.py`)
- **Step Type**: `XGBoostTraining`
- **Dependencies**:
  - `training_data` (required): Preprocessed training data
  - `hyperparameters` (optional): Hyperparameter configuration
- **Outputs**:
  - `model_artifacts`: Trained XGBoost model artifacts
  - `ModelArtifacts`: Alias for model_artifacts

### 4. Packaging Specification (`packaging_spec.py`)
- **Step Type**: `Package`
- **Dependencies**:
  - `model_input` (required): Trained model artifacts to be packaged
  - `inference_scripts_input` (required): Inference scripts and code for deployment
- **Outputs**:
  - `packaged_model_output`: Packaged model ready for deployment
  - `PackagedModel`: Alias for packaged_model_output

### 5. Payload Specification (`payload_spec.py`)
- **Step Type**: `Payload`
- **Dependencies**:
  - `model_input` (required): Trained model artifacts for payload generation
- **Outputs**:
  - `payload_sample`: Generated payload samples for model testing
  - `GeneratedPayloadSamples`: Alias for payload_sample
  - `payload_metadata`: Metadata about the generated payload samples
  - `PayloadMetadata`: Alias for payload_metadata

### 6. Registration Specification (`registration_spec.py`)
- **Step Type**: `Registration`
- **Dependencies**:
  - `model_input` (required): Packaged model artifacts for registration
  - `payload_input` (optional): Payload samples for model testing
- **Outputs**:
  - `registered_model`: Information about the registered model
  - `RegisteredModel`: Alias for registered_model

## Dependency Types

The specifications use the following dependency types:

- `MODEL_ARTIFACTS`: Trained model files and artifacts
- `TRAINING_DATA`: Data used for model training
- `PROCESSING_OUTPUT`: General processing step outputs
- `HYPERPARAMETERS`: Model hyperparameter configurations
- `PAYLOAD_SAMPLES`: Sample payloads for model testing
- `CUSTOM_PROPERTY`: Custom or specialized properties

## Property Paths

Each output specification includes property paths that define how to access the output at runtime:
- **Name-based access**: `properties.ProcessingOutputConfig.Outputs['OutputName'].S3Output.S3Uri`
- **Index-based access**: `properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`
- **Direct attribute access**: For special cases like `step.ModelArtifacts.S3ModelArtifacts`

## Semantic Matching

Dependencies include semantic keywords for intelligent matching:
- **Model-related**: `["model", "artifacts", "trained", "output"]`
- **Data-related**: `["data", "processed", "training", "input"]`
- **Processing-related**: `["processed", "tabular", "preprocessed"]`
- **Deployment-related**: `["packaged", "inference", "scripts", "payload"]`

## Usage

These specifications are used by the `UnifiedDependencyResolver` to automatically:
1. Match step dependencies with compatible outputs from other steps
2. Calculate compatibility scores based on type, semantics, and keywords
3. Resolve property paths for runtime access
4. Generate dependency resolution reports

## Example Pipeline Flow

```
CradleDataLoading → TabularPreprocessing → XGBoostTraining → Package → Payload → Registration
     ↓                      ↓                    ↓             ↓         ↓          ↓
   DATA              processed_data      model_artifacts  PackagedModel  payload_sample  registered_model
```

Each arrow represents an automatically resolved dependency based on these specifications.
