"""
Tabular Preprocessing Step Specification.

This module defines the declarative specification for tabular preprocessing steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Tabular Preprocessing Step Specification
PREPROCESSING_SPEC = StepSpecification(
    step_type="TabularPreprocessing",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=["data", "input", "raw", "dataset", "source", "tabular"],
            data_type="S3Uri",
            description="Raw tabular data for preprocessing"
        ),
        DependencySpec(
            logical_name="METADATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=["metadata", "schema", "info", "description"],
            data_type="S3Uri",
            description="Optional metadata about the dataset"
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading", "DataLoad", "ProcessingStep"],
            semantic_keywords=["signature", "validation", "checksum"],
            data_type="S3Uri",
            description="Optional data signature for validation"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed tabular data with train/val/test splits"
        ),
        OutputSpec(
            logical_name="ProcessedTabularData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed tabular data (alias for processed_data)"
        ),
        OutputSpec(
            logical_name="full_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['full_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed dataset without splits (optional)"
        ),
        OutputSpec(
            logical_name="calibration_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data for model calibration (optional)"
        ),
        OutputSpec(
            logical_name="FullData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed dataset (alias for full_data)"
        ),
        OutputSpec(
            logical_name="CalibrationData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['CalibrationData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data (alias for calibration_data)"
        )
    ]
)
