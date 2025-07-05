"""
Tabular Preprocessing Calibration Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for calibration data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Tabular Preprocessing Calibration Step Specification
PREPROCESSING_CALIBRATION_SPEC = StepSpecification(
    step_type="TabularPreprocessing_Calibration",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading_Calibration"],
            semantic_keywords=["calibration", "calib", "eval", "data", "input", "raw", "dataset", "source", "tabular", "evaluation", "model_eval"],
            data_type="S3Uri",
            description="Raw calibration data for preprocessing"
        ),
        DependencySpec(
            logical_name="METADATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Calibration"],
            semantic_keywords=["calibration", "calib", "eval", "metadata", "schema", "info", "description", "evaluation", "model_eval"],
            data_type="S3Uri",
            description="Optional calibration metadata about the dataset"
        ),
        DependencySpec(
            logical_name="SIGNATURE",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=False,
            compatible_sources=["CradleDataLoading_Calibration"],
            semantic_keywords=["calibration", "calib", "eval", "signature", "validation", "checksum", "evaluation", "model_eval"],
            data_type="S3Uri",
            description="Optional calibration data signature for validation"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed calibration data for model evaluation",
            semantic_keywords=["calibration", "calib", "eval", "processed", "data", "tabular", "evaluation", "model_eval", "preprocessed"]
        ),
        OutputSpec(
            logical_name="ProcessedTabularData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['ProcessedTabularData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed calibration data (alias for processed_data)",
            semantic_keywords=["calibration", "calib", "eval", "processed", "data", "tabular", "evaluation", "model_eval", "preprocessed"]
        ),
        OutputSpec(
            logical_name="full_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['full_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed calibration dataset (optional)",
            semantic_keywords=["calibration", "calib", "eval", "full", "data", "complete", "dataset", "evaluation", "model_eval"]
        ),
        OutputSpec(
            logical_name="calibration_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['calibration_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data for model calibration (optional)",
            semantic_keywords=["calibration", "calib", "eval", "data", "model_calibration", "evaluation", "model_eval"]
        ),
        OutputSpec(
            logical_name="FullData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['FullData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Full processed calibration dataset (alias for full_data)",
            semantic_keywords=["calibration", "calib", "eval", "full", "data", "complete", "dataset", "evaluation", "model_eval"]
        ),
        OutputSpec(
            logical_name="CalibrationData",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['CalibrationData'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data (alias for calibration_data)",
            semantic_keywords=["calibration", "calib", "eval", "data", "model_calibration", "evaluation", "model_eval"]
        )
    ]
)
