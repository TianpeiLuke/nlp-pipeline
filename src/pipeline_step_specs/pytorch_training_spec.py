"""
PyTorch Training Step Specification.

This module defines the declarative specification for PyTorch training steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..pipeline_registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_pytorch_train_contract():
    from ..pipeline_script_contracts.pytorch_train_contract import PYTORCH_TRAIN_CONTRACT
    return PYTORCH_TRAIN_CONTRACT

# PyTorch Training Step Specification
PYTORCH_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("PytorchTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_pytorch_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "input", "training", "dataset", "processed", "train", "pytorch"],
            data_type="S3Uri",
            description="Training dataset S3 location with train/val/test subdirectories"
        ),
        DependencySpec(
            logical_name="config",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=True,
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=["config", "params", "hyperparameters", "settings"],
            data_type="S3Uri",
            description="Hyperparameters configuration file"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained PyTorch model artifacts",
            aliases=["ModelArtifacts", "model_data", "output_path", "model_input"]
        ),
        OutputSpec(
            logical_name="data_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.TrainingJobDefinition.OutputDataConfig.S3OutputPath",
            data_type="S3Uri",
            description="Training evaluation results and predictions"
        ),
        OutputSpec(
            logical_name="checkpoints",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.CheckpointConfig.S3Uri",
            data_type="S3Uri",
            description="Training checkpoints for resuming"
        ),
        OutputSpec(
            logical_name="training_job_name",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingJobName",
            data_type="String",
            description="SageMaker training job name",
            aliases=["TrainingJobName"]
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingMetrics",
            data_type="String",
            description="Training metrics from the job",
            aliases=["TrainingMetrics"]
        )
    ]
)
