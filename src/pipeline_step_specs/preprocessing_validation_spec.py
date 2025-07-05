"""
Tabular Preprocessing Validation Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for validation data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Tabular Preprocessing Validation Step Specification
PREPROCESSING_VALIDATION_SPEC = StepSpecification(
    step_type="TabularPreprocessing_Validation",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading_Validation"],
            semantic_keywords=["validation", "val", "data", "input", "raw", "dataset", "source", "tabular", "model_validation", "holdout"],
            data_type="S3Uri",
            description="Raw validation data for preprocessing"
        ),
        DependencySpec(
            logical_name="METADATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Validation"],
            semantic_keywords=["validation", "val", "metadata", "schema", "info", "description", "model_validation", "holdout"],
            data_type="S3Uri",
            description="Optional validation metadata about the dataset"
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Validation"],
            semantic_keywords=["validation", "val", "signature", "validation", "checksum", "model_validation", "holdout"],
            data_type="S3Uri",
            description="Optional validation data signature for validation"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed validation data",
            semantic_keywords=["validation", "val", "processed", "data", "tabular", "model_validation", "holdout", "preprocessed"]
        ),
        OutputSpec(
            logical_name="ProcessedTabularData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed validation data (alias for processed_data)",
            semantic_keywords=["validation", "val", "processed", "data", "tabular", "model_validation", "holdout", "preprocessed"]
        ),
        OutputSpec(
            logical_name="full_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['full_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed validation dataset (optional)",
            semantic_keywords=["validation", "val", "full", "data", "complete", "dataset", "model_validation", "holdout"]
        ),
        OutputSpec(
            logical_name="FullData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed validation dataset (alias for full_data)",
            semantic_keywords=["validation", "val", "full", "data", "complete", "dataset", "model_validation", "holdout"]
        )
    ]
)
