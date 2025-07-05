"""
Cradle Data Loading Calibration Step Specification.

This module defines the declarative specification for Cradle data loading steps
specifically for calibration data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Cradle Data Loading Calibration Step Specification
DATA_LOADING_CALIBRATION_SPEC = StepSpecification(
    step_type="CradleDataLoading_Calibration",
    node_type=NodeType.SOURCE,
    dependencies=[
        # Note: CradleDataLoading is typically the first step in a pipeline
        # and doesn't depend on other pipeline steps - it loads data from external sources
    ],
    outputs=[
        OutputSpec(
            logical_name="DATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['DATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration data output from Cradle data loading",
            semantic_keywords=["calibration", "calib", "eval", "data", "input", "raw", "dataset", "evaluation", "model_eval", "source"]
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration metadata output from Cradle data loading",
            semantic_keywords=["calibration", "calib", "eval", "metadata", "schema", "info", "description", "evaluation", "model_eval"]
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Calibration signature output from Cradle data loading",
            semantic_keywords=["calibration", "calib", "eval", "signature", "validation", "checksum", "evaluation", "model_eval"]
        )
    ]
)
