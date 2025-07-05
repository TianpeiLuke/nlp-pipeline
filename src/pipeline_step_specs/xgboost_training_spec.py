"""
XGBoost Training Step Specification.

This module defines the declarative specification for XGBoost training steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Import the contract at runtime to avoid circular imports
def _get_xgboost_train_contract():
    from ..pipeline_script_contracts.xgboost_train_contract import XGBOOST_TRAIN_CONTRACT
    return XGBOOST_TRAIN_CONTRACT

# XGBoost Training Step Specification
XGBOOST_TRAINING_SPEC = StepSpecification(
    step_type="XGBoostTraining",
    node_type=NodeType.INTERNAL,
    script_contract=_get_xgboost_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "input", "training", "dataset", "processed", "train", "tabular"],
            data_type="S3Uri",
            description="Training dataset S3 location with train/val/test subdirectories"
        ),
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,  # Can be generated internally
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=["config", "params", "hyperparameters", "settings", "hyperparams"],
            data_type="S3Uri",
            description="Hyperparameters configuration file (optional, can be generated internally)"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained XGBoost model artifacts"
        ),
        OutputSpec(
            logical_name="ModelOutputPath",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Model output path (alias for model_output)"
        ),
        OutputSpec(
            logical_name="ModelArtifacts",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Model artifacts (alias for model_output)"
        ),
        OutputSpec(
            logical_name="training_job_name",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingJobName",
            data_type="String",
            description="SageMaker training job name"
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.CUSTOM_PROPERTY,
            property_path="properties.TrainingMetrics",
            data_type="String",
            description="Training metrics from the job"
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
        )
    ]
)
