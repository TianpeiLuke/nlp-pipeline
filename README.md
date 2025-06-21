# Pipeline for MLOps and LLMOps

A modular pipeline for processing emails, messages, and tabular data, supporting end-to-end ML workflows from raw data to model evaluation and registration. This package is designed for robust, production-grade NLP and tabular ML pipelines, with a focus on extensibility, reproducibility, and integration with AWS SageMaker.

---

## Table of Contents

- [Overview](#overview)
- [Processors](#processors)
- [Pipeline Steps](#pipeline-steps)
- [Pipeline Examples](#pipeline-examples)
- [Docker Images](#docker-images)
- [Benefits](#benefits)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository provides a comprehensive framework for building and deploying machine learning pipelines for both NLP and tabular data. It includes:

- Modular processors for text and tabular data
- Configurable pipeline steps for data loading, preprocessing, training, evaluation, packaging, and registration
- End-to-end pipeline examples for XGBoost and PyTorch
- Docker images for reproducible processing and training environments

---

## Processors

Processors are modular components for data cleaning, feature engineering, and transformation. They can be composed and reused across different pipeline steps.

- **Text Processing:** Normalization, tokenization, dialogue splitting, emoji removal, HTML cleaning, etc.
- **Categorical Data Processing:** Label encoding, risk table mapping, multi-class encoding
- **Numerical Data Processing:** Imputation, binning, scaling

See [slipbox/processing/README.md](slipbox/processing/README.md) for a full list and usage examples.

---

## Pipeline Steps

Each pipeline step encapsulates a specific stage of the ML workflow, with configuration-driven design and strong validation.

- **Data Loading:** [Cradle Data Load Step](slipbox/pipelines/data_load_step_cradle.md)
- **Tabular Preprocessing:** [Tabular Preprocessing Step](slipbox/pipelines/tabular_preprocessing_step.md)
- **Risk Table Mapping:** [Risk Table Mapping Step](slipbox/pipelines/risk_table_map_step.md)
- **Model Training:** [XGBoost Training Step](slipbox/pipelines/training_step_xgboost.md), [PyTorch Training Step](slipbox/pipelines/training_step_pytorch.md)
- **Model Creation:** [XGBoost Model Step](slipbox/pipelines/model_step_xgboost.md), [PyTorch Model Step](slipbox/pipelines/model_step_pytorch.md)
- **Model Evaluation:** [XGBoost Model Evaluation Step](slipbox/pipelines/model_eval_step_xgboost.md)
- **Packaging:** [MIMS Packaging Step](slipbox/pipelines/mims_packaging_step.md)
- **Registration:** [MIMS Registration Step](slipbox/pipelines/mims_registration_step.md)
- **Batch Transform:** [Batch Transform Step](slipbox/pipelines/batch_transform_step.md)

See [slipbox/pipelines/README.md](slipbox/pipelines/README.md) for a complete overview and usage pattern.

---

## Pipeline Examples

Ready-to-use pipeline blueprints for common ML workflows:

- **XGBoost End-to-End:** [mods_pipeline_xgboost_end_to_end.md](slipbox/pipeline_examples/mods_pipeline_xgboost_end_to_end.md)
- **XGBoost End-to-End (Simple):** [mods_pipeline_xgboost_end_to_end_simple.md](slipbox/pipeline_examples/mods_pipeline_xgboost_end_to_end_simple.md)
- **PyTorch BSM Pipeline:** [mods_pipeline_bsm_pytorch.md](slipbox/pipeline_examples/mods_pipeline_bsm_pytorch.md)

Each example demonstrates step connections, configuration, and S3 input/output conventions.

---

## Docker Images

Dockerfiles and build scripts are provided for:

- **Processing Containers:** For data preprocessing, risk mapping, and feature engineering
- **Training Containers:** For XGBoost and PyTorch model training
- **Inference Containers:** For model serving and batch inference

These images ensure reproducibility and compatibility with SageMaker Processing and Training jobs.

---

## Benefits

- **Modularity:** Reusable processors and pipeline steps for rapid development
- **Configurability:** All steps are driven by Pydantic configs for validation and clarity
- **Production-Ready:** Designed for SageMaker Pipelines and large-scale ML workflows
- **Extensibility:** Easy to add new processors, steps, or pipeline patterns
- **Documentation:** Extensive [slipbox/](slipbox/) markdown docs for every component and step

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. See [CONTRIBUTING.md](CONTRIBUTING.md) if available.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.