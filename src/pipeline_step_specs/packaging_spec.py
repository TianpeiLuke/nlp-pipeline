"""
MIMS Packaging Step Specification.

This module defines the declarative specification for MIMS model packaging steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# MIMS Packaging Step Specification
PACKAGING_SPEC = StepSpecification(
    step_type="Package",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=["model", "artifacts", "trained", "output", "ModelArtifacts"],
            data_type="S3Uri",
            description="Trained model artifacts to be packaged"
        ),
        DependencySpec(
            logical_name="inference_scripts_input",
            dependency_type=DependencyType.CUSTOM_PROPERTY,
            required=True,
            compatible_sources=["ProcessingStep", "ScriptStep"],
            semantic_keywords=["inference", "scripts", "code", "InferenceScripts"],
            data_type="S3Uri",
            description="Inference scripts and code for model deployment"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="packaged_model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Packaged model ready for deployment"
        ),
        OutputSpec(
            logical_name="PackagedModel",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['PackagedModel'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Packaged model (alias for packaged_model_output)"
        )
    ]
)
