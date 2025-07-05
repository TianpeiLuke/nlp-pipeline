"""
Tabular Preprocessing Testing Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for testing data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Tabular Preprocessing Testing Step Specification
PREPROCESSING_TESTING_SPEC = StepSpecification(
    step_type="TabularPreprocessing_Testing",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading_Testing"],
            semantic_keywords=["testing", "test", "data", "input", "raw", "dataset", "source", "tabular", "model_testing", "holdout"],
            data_type="S3Uri",
            description="Raw testing data for preprocessing"
        ),
        DependencySpec(
            logical_name="METADATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Testing"],
            semantic_keywords=["testing", "test", "metadata", "schema", "info", "description", "model_testing", "holdout"],
            data_type="S3Uri",
            description="Optional testing metadata about the dataset"
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Testing"],
            semantic_keywords=["testing", "test", "signature", "validation", "checksum", "model_testing", "holdout"],
            data_type="S3Uri",
            description="Optional testing data signature for validation"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed testing data",
            semantic_keywords=["testing", "test", "processed", "data", "tabular", "model_testing", "holdout", "preprocessed"]
        ),
        OutputSpec(
            logical_name="ProcessedTabularData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed testing data (alias for processed_data)",
            semantic_keywords=["testing", "test", "processed", "data", "tabular", "model_testing", "holdout", "preprocessed"]
        ),
        OutputSpec(
            logical_name="full_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed testing dataset (optional)",
            semantic_keywords=["testing", "test", "full", "data", "complete", "dataset", "model_testing", "holdout"]
        ),
        OutputSpec(
            logical_name="FullData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed testing dataset (alias for full_data)",
            semantic_keywords=["testing", "test", "full", "data", "complete", "dataset", "model_testing", "holdout"]
        )
    ]
)
