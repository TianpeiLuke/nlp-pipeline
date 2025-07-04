"""
PyTorch Training Step Specification.

This module defines the declarative specification for PyTorch training steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# PyTorch Training Step Specification
PYTORCH_TRAINING_SPEC = StepSpecification(
    step_type="PyTorchTraining",
    node_type=NodeType.INTERNAL,
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "input", "training", "dataset", "processed", "train", "pytorch"],
            data_type="S3Uri",
            description="Training dataset S3 location"
        ),
        DependencySpec(
            logical_name="checkpoint_path",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=False,
            compatible_sources=["PyTorchTraining", "ProcessingStep"],
            semantic_keywords=["checkpoint", "model", "weights", "pretrained", "resume"],
            data_type="S3Uri",
            description="Optional checkpoint to resume training from"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained PyTorch model artifacts"
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingMetrics",
            data_type="String",
            description="Training metrics from the job"
        ),
        OutputSpec(
            logical_name="training_job_name",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingJobName",
            data_type="String",
            description="SageMaker training job name"
        ),
        OutputSpec(
            logical_name="ModelArtifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Model artifacts (alias for model_output)"
        ),
        OutputSpec(
            logical_name="TrainingJobName",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingJobName",
            data_type="String",
            description="Training job name (alias for training_job_name)"
        ),
        OutputSpec(
            logical_name="TrainingMetrics",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingMetrics",
            data_type="String",
            description="Training metrics (alias for metrics_output)"
        ),
        OutputSpec(
            logical_name="model_data",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Model data (alias for model_output)"
        ),
        OutputSpec(
            logical_name="output_path",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Output path (alias for model_output)"
        ),
        OutputSpec(
            logical_name="model_input",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Model input reference (alias for model_output)"
        )
    ]
)
