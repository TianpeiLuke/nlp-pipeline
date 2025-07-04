"""
PyTorch Model Step Specification.

This module defines the declarative specification for PyTorch model steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# PyTorch Model Step Specification
PYTORCH_MODEL_SPEC = StepSpecification(
    step_type="PyTorchModel",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="model_data",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["PyTorchTraining", "ProcessingStep", "ModelArtifactsStep"],
            semantic_keywords=["model", "artifacts", "pytorch", "training", "output", "model_data"],
            data_type="S3Uri",
            description="PyTorch model artifacts from training or processing"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.ModelName",
            data_type="String",
            description="SageMaker model name"
        ),
        OutputSpec(
            logical_name="model_artifacts_path",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="S3 path to model artifacts"
        ),
        OutputSpec(
            logical_name="ModelName",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.ModelName",
            data_type="String",
            description="SageMaker model name (alias for model)"
        ),
        OutputSpec(
            logical_name="ModelArtifactsPath",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="S3 path to model artifacts (alias for model_artifacts_path)"
        ),
        OutputSpec(
            logical_name="model_input",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Model input reference (alias for model_artifacts_path)"
        )
    ]
)
