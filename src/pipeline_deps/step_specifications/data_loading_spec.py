"""
Cradle Data Loading Step Specification.

This module defines the declarative specification for Cradle data loading steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType

# Cradle Data Loading Step Specification
DATA_LOADING_SPEC = StepSpecification(
    step_type="CradleDataLoading",
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
            description="Main data output from Cradle data loading"
        ),
        OutputSpec(
            logical_name="METADATA",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['METADATA'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata output from Cradle data loading"
        ),
        OutputSpec(
            logical_name="SIGNATURE",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['SIGNATURE'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Signature output from Cradle data loading"
        )
    ]
)
