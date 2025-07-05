"""
MIMS Payload Step Specification.

This module defines the declarative specification for MIMS payload generation steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Import the contract at runtime to avoid circular imports
def _get_mims_payload_contract():
    from ..pipeline_script_contracts.mims_payload_contract import MIMS_PAYLOAD_CONTRACT
    return MIMS_PAYLOAD_CONTRACT

# MIMS Payload Step Specification
PAYLOAD_SPEC = StepSpecification(
    step_type="Payload",
    node_type=NodeType.INTERNAL,
    script_contract=_get_mims_payload_contract(),
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
            property_path="properties.ProcessingOutputConfig.Outputs['payload_sample'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Generated payload samples for model testing"
        ),
        OutputSpec(
            logical_name="payload_metadata",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['payload_metadata'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Metadata about the generated payload samples"
        )
    ]
)
