"""
Tabular Preprocessing Training Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for training data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Tabular Preprocessing Training Step Specification
PREPROCESSING_TRAINING_SPEC = StepSpecification(
    step_type="TabularPreprocessing_Training",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading_Training"],
            semantic_keywords=["training", "train", "data", "input", "raw", "dataset", "source", "tabular", "model_training"],
            data_type="S3Uri",
            description="Raw training data for preprocessing"
        ),
        DependencySpec(
            logical_name="METADATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Training"],
            semantic_keywords=["training", "train", "metadata", "schema", "info", "description", "model_training"],
            data_type="S3Uri",
            description="Optional training metadata about the dataset"
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Training"],
            semantic_keywords=["training", "train", "signature", "validation", "checksum", "model_training"],
            data_type="S3Uri",
            description="Optional training data signature for validation"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed training data with train/val/test splits",
            semantic_keywords=["training", "train", "processed", "data", "tabular", "splits", "model_training", "preprocessed"]
        ),
        OutputSpec(
            logical_name="ProcessedTabularData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed training data (alias for processed_data)",
            semantic_keywords=["training", "train", "processed", "data", "tabular", "splits", "model_training", "preprocessed"]
        ),
        OutputSpec(
            logical_name="full_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed training dataset without splits (optional)",
            semantic_keywords=["training", "train", "full", "data", "complete", "dataset", "model_training"]
        ),
        OutputSpec(
            logical_name="FullData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed training dataset (alias for full_data)",
            semantic_keywords=["training", "train", "full", "data", "complete", "dataset", "model_training"]
        )
    ]
)
