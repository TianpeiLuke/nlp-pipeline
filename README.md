# Pipeline for MLOps and LLMOps

A modular pipeline for processing emails, messages, and tabular data, supporting end-to-end ML workflows from raw data to model evaluation and registration. This package is designed for robust, production-grade NLP and tabular ML pipelines, with a focus on extensibility, reproducibility, and integration with AWS SageMaker.

---

## Table of Contents

- [Overview](#overview)
- [Processors](#processors)
- [Pipeline Steps](#pipeline-steps)
- [Pipeline Examples](#pipeline-examples)
- [Pipeline Builder Template](#pipeline-builder-template)
- [PyTorch Models](#pytorch-models)
- [Docker Images](#docker-images)
- [Slipbox Documentation](#slipbox-documentation)
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

- **Data Loading:** [Cradle Data Load Step](slipbox/pipeline_steps/data_load_step_cradle.md)
- **Tabular Preprocessing:** [Tabular Preprocessing Step](slipbox/pipeline_steps/tabular_preprocessing_step.md)
- **Risk Table Mapping:** [Risk Table Mapping Step](slipbox/pipeline_steps/risk_table_map_step.md)
- **Model Training:** [XGBoost Training Step](slipbox/pipeline_steps/training_step_xgboost.md), [PyTorch Training Step](slipbox/pipeline_steps/training_step_pytorch.md)
- **Model Creation:** [XGBoost Model Step](slipbox/pipeline_steps/model_step_xgboost.md), [PyTorch Model Step](slipbox/pipeline_steps/model_step_pytorch.md)
- **Model Evaluation:** [XGBoost Model Evaluation Step](slipbox/pipeline_steps/model_eval_step_xgboost.md)
- **Packaging:** [MIMS Packaging Step](slipbox/pipeline_steps/mims_packaging_step.md)
- **Registration:** [MIMS Registration Step](slipbox/pipeline_steps/mims_registration_step.md)
- **Batch Transform:** [Batch Transform Step](slipbox/pipeline_steps/batch_transform_step.md)

See [slipbox/pipeline_steps/README.md](slipbox/pipeline_steps/README.md) for a complete overview and usage pattern.

---

## Pipeline Examples

Ready-to-use pipeline blueprints for common ML workflows:

- **XGBoost End-to-End:** [mods_pipeline_xgboost_end_to_end.md](slipbox/pipeline_examples/mods_pipeline_xgboost_end_to_end.md)
- **XGBoost End-to-End (Simple):** [mods_pipeline_xgboost_end_to_end_simple.md](slipbox/pipeline_examples/mods_pipeline_xgboost_end_to_end_simple.md)
- **PyTorch BSM Pipeline:** [mods_pipeline_bsm_pytorch.md](slipbox/pipeline_examples/mods_pipeline_bsm_pytorch.md)

Each example demonstrates step connections, configuration, and S3 input/output conventions.

---

## Pipeline Builder Template

A new template-based approach for building SageMaker pipelines that simplifies the creation of complex pipelines and automatically handles the connections between steps.

### Key Features

- **Declarative Pipeline Definition:** Define pipeline structure as a Directed Acyclic Graph (DAG)
- **Automatic Connection:** Steps are automatically connected based on the DAG structure
- **Message Passing Algorithm:** Propagates information between steps to eliminate manual wiring
- **Placeholder Handling:** Automatically handles placeholder variables like `dependency_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri`

### Components

- **Pipeline DAG:** Represents the structure of a pipeline as a directed acyclic graph
- **Pipeline Builder Template:** Core component that uses the DAG to generate a SageMaker pipeline
- **Template Implementation:** Explains how the template handles placeholder variables

See [slipbox/pipeline_builder/README.md](slipbox/pipeline_builder/README.md) for a complete overview of the template-based approach.

---

## PyTorch Models

A collection of modular PyTorch Lightning models for NLP and multimodal tasks, including BERT-based, LSTM, CNN, and advanced fusion architectures. These models are designed for easy integration into the pipeline and support flexible configuration for various tasks.

- See the [slipbox/lightning_models/README.md](slipbox/lightning_models/README.md) for an overview of available models and usage instructions.

### Available PyTorch Model Architectures

- [pl_bert.md](slipbox/lightning_models/pl_bert.md): BERT-based classification model
- [pl_bert_classification.md](slipbox/lightning_models/pl_bert_classification.md): BERT for multi-class classification
- [pl_lstm.md](slipbox/lightning_models/pl_lstm.md): LSTM-based sequence model
- [pl_text_cnn.md](slipbox/lightning_models/pl_text_cnn.md): Text CNN for sentence classification
- [pl_multimodal_bert.md](slipbox/lightning_models/pl_multimodal_bert.md): Multimodal BERT model
- [pl_multimodal_cnn.md](slipbox/lightning_models/pl_multimodal_cnn.md): Multimodal CNN model
- [pl_multimodal_cross_attn.md](slipbox/lightning_models/pl_multimodal_cross_attn.md): Multimodal model with cross-attention
- [pl_multimodal_gate_fusion.md](slipbox/lightning_models/pl_multimodal_gate_fusion.md): Multimodal model with gated fusion
- [pl_multimodal_moe.md](slipbox/lightning_models/pl_multimodal_moe.md): Mixture-of-Experts multimodal model
- [pl_tab_ae.md](slipbox/lightning_models/pl_tab_ae.md): Tabular autoencoder for feature learning
- [pl_model_plots.md](slipbox/lightning_models/pl_model_plots.md): Model visualization and plotting utilities
- [pl_train.md](slipbox/lightning_models/pl_train.md): Training utilities and scripts for PyTorch Lightning models
- [dist_utils.md](slipbox/lightning_models/dist_utils.md): Distributed training utilities

Refer to each markdown file above for architecture details, configuration options, and usage examples.

---

## Docker Images

Dockerfiles and build scripts are provided for:

- **Processing Containers:** For data preprocessing, risk mapping, and feature engineering
- **Training Containers:** For XGBoost and PyTorch model training
- **Inference Containers:** For model serving and batch inference

These images ensure reproducibility and compatibility with SageMaker Processing and Training jobs.

---

## Slipbox Documentation

The [slipbox/](slipbox/) directory contains comprehensive documentation, guides, examples, and design specifications for the entire pipeline framework. This knowledge base is organized into specialized sections covering all aspects of development, implementation, and usage.

### Core Documentation Sections

- **[Developer Demo](slipbox/developer_demo/)** - Complete demonstrations and examples of pipeline implementation, including dummy training validation examples, implementation plans, and risk table mapping validation reports.

- **[Developer Guide](slipbox/developer_guide/)** - Essential guides for developers working with the pipeline framework, covering best practices, design principles, component guides, validation checklists, and step-by-step creation processes.

- **[Developer Prompts](slipbox/developer_prompts/)** - AI-assisted development prompts and templates for planning, validation, programming, and revision workflows to accelerate development cycles.

- **[Lightning Models](slipbox/lightning_models/)** - Detailed documentation for PyTorch Lightning model architectures, including BERT, LSTM, CNN, multimodal models, and distributed training utilities.

- **[Notebook Digest Prompts](slipbox/notebook_digest_prompts/)** - Templates and analyzers for converting Jupyter notebooks to SageMaker pipelines, with usage guides and example outputs.

- **[Notebook Digests](slipbox/notebook_digests/)** - Processed notebook analyses and summaries for model training and pipeline development.

- **[Notebooks](slipbox/notebooks/)** - Documentation for notebook-based workflows and configurations.

- **[Pipeline Builder](slipbox/pipeline_builder/)** - Comprehensive documentation for the template-based pipeline building system, including assemblers, examples, and implementation guides.

- **[Pipeline DAG](slipbox/pipeline_dag/)** - Documentation for the Directed Acyclic Graph system used in pipeline construction, covering base DAG structures, edge types, and enhanced DAG features.

- **[Pipeline Dependencies](slipbox/pipeline_deps/)** - Dependency management system documentation, including resolvers, registries, semantic matching, and factory patterns.

- **[Pipeline Design](slipbox/pipeline_design/)** - Core design documentation covering configuration management, dependency resolution, design principles, and architectural evolution.

- **[Pipeline Examples](slipbox/pipeline_examples/)** - Real-world pipeline examples and blueprints for various ML workflows.

- **[Pipeline Script Contracts](slipbox/pipeline_script_contracts/)** - Contract specifications and interfaces for pipeline scripts.

- **[Pipeline Step Specs](slipbox/pipeline_step_specs/)** - Detailed specifications for individual pipeline steps and their configurations.

- **[Pipeline Steps](slipbox/pipeline_steps/)** - Documentation for all available pipeline steps, from data loading to model registration.

- **[Processing](slipbox/processing/)** - Documentation for data processing components and utilities.

- **[Project Planning](slipbox/project_planning/)** - Project planning documents and roadmaps.

- **[Review](slipbox/review/)** - Code review guidelines and documentation review processes.

### Getting Started with Slipbox

1. **New Developers**: Start with [slipbox/developer_guide/README.md](slipbox/developer_guide/README.md) for an overview of development practices and principles.

2. **Pipeline Creation**: Refer to [slipbox/pipeline_builder/README.md](slipbox/pipeline_builder/README.md) for template-based pipeline construction.

3. **Model Development**: Check [slipbox/lightning_models/README.md](slipbox/lightning_models/README.md) for PyTorch model architectures and training utilities.

4. **Design Understanding**: Review [slipbox/pipeline_design/README.md](slipbox/pipeline_design/README.md) for core architectural concepts and design evolution.

The slipbox serves as both a reference manual and a living knowledge base that evolves with the project, ensuring comprehensive documentation coverage for all aspects of the pipeline framework.

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
