"""
MIMS Payload Step Specification.

This module defines the declarative specification for MIMS payload generation steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType

# MIMS Payload Step Specification
PAYLOAD_SPEC = StepSpecification(
    step_type="Payload",
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=["model", "artifacts", "trained", "output", "ModelArtifacts"],
            data_type="S3Uri",
            description="Trained model artifacts for payload generation"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="payload_sample",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['GeneratedPayloadSamples'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated payload samples for model testing"
        ),
        OutputSpec(
            logical_name="GeneratedPayloadSamples",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['GeneratedPayloadSamples'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated payload samples (alias for payload_sample)"
        ),
        OutputSpec(
            logical_name="payload_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['PayloadMetadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata about the generated payload samples"
        ),
        OutputSpec(
            logical_name="PayloadMetadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['PayloadMetadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Payload metadata (alias for payload_metadata)"
        )
    ]
)
