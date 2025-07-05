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
        )
    ]
)
