"""
Cradle Data Loading Validation Step Specification.

This module defines the declarative specification for Cradle data loading steps
specifically for validation data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Cradle Data Loading Validation Step Specification
DATA_LOADING_VALIDATION_SPEC = StepSpecification(
    step_type="CradleDataLoading_Validation",
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
            description="Validation data output from Cradle data loading",
            semantic_keywords=["validation", "val", "data", "input", "raw", "dataset", "model_validation", "holdout", "source"]
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Validation metadata output from Cradle data loading",
            semantic_keywords=["validation", "val", "metadata", "schema", "info", "description", "model_validation", "holdout"]
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Validation signature output from Cradle data loading",
            semantic_keywords=["validation", "val", "signature", "validation", "checksum", "model_validation", "holdout"]
        )
    ]
)
