"""
Cradle Data Loading Step Specification for Validation Job Type.

This module defines the declarative specification for Cradle data loading steps
specifically for validation jobs, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..pipeline_registry.step_names import get_spec_step_type

# Cradle Data Loading Step Specification for Validation
DATA_LOADING_VALIDATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("CradleDataLoading") + "_Validation",
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
