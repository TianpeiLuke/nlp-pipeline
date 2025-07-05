"""
XGBoost Model Evaluation Step Specification.

This module defines the declarative specification for XGBoost model evaluation steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..pipeline_script_contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT

# Import the contract at runtime to avoid circular imports
def _get_model_evaluation_contract():
    from ..pipeline_script_contracts.model_evaluation_contract import MODEL_EVALUATION_CONTRACT
    return MODEL_EVALUATION_CONTRACT

# XGBoost Model Evaluation Step Specification
MODEL_EVAL_SPEC = StepSpecification(
    step_type="XGBoostModelEvaluation",
    node_type=NodeType.INTERNAL,
    script_contract=_get_model_evaluation_contract(),
    dependencies=[
        DependencySpec(
            logical_name="model_input",
            dependency_type=DependencyType.MODEL_ARTIFACTS,
            required=True,
            compatible_sources=["XGBoostTraining", "TrainingStep", "ModelStep"],
            semantic_keywords=["model", "artifacts", "trained", "output", "ModelArtifacts"],
            data_type="S3Uri",
            description="Trained model artifacts to be evaluated (includes hyperparameters.json)"
        ),
        DependencySpec(
            logical_name="eval_data_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep", "DataLoad"],
            semantic_keywords=["data", "evaluation", "calibration", "validation", "test", "processed"],
            data_type="S3Uri",
            description="Evaluation dataset for model assessment"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="eval_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['EvaluationResults'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Model evaluation results including predictions"
        ),
        OutputSpec(
            logical_name="metrics_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['EvaluationMetrics'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Model evaluation metrics (AUC, precision, recall, etc.)"
        ),
        OutputSpec(
            logical_name="EvaluationResults",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['EvaluationResults'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Evaluation results (alias for eval_output)"
        ),
        OutputSpec(
            logical_name="EvaluationMetrics",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['EvaluationMetrics'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Evaluation metrics (alias for metrics_output)"
        )
    ]
)
