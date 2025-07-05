"""
Cradle Data Loading Testing Step Specification.

This module defines the declarative specification for Cradle data loading steps
specifically for testing data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Cradle Data Loading Testing Step Specification
DATA_LOADING_TESTING_SPEC = StepSpecification(
    step_type="CradleDataLoading_Testing",
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
            description="Testing data output from Cradle data loading",
            semantic_keywords=["testing", "test", "data", "input", "raw", "dataset", "model_testing", "holdout", "source"]
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Testing metadata output from Cradle data loading",
            semantic_keywords=["testing", "test", "metadata", "schema", "info", "description", "model_testing", "holdout"]
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Testing signature output from Cradle data loading",
            semantic_keywords=["testing", "test", "signature", "validation", "checksum", "model_testing", "holdout"]
        )
    ]
)
