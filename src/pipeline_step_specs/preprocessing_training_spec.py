"""
Tabular Preprocessing Training Step Specification.

This module defines the declarative specification for tabular preprocessing steps
specifically for training data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType

# Import the contract at runtime to avoid circular imports
def _get_tabular_preprocess_contract():
    from ..pipeline_script_contracts.tabular_preprocess_contract import TABULAR_PREPROCESS_CONTRACT
    return TABULAR_PREPROCESS_CONTRACT

# Tabular Preprocessing Training Step Specification
PREPROCESSING_TRAINING_SPEC = StepSpecification(
    step_type="TabularPreprocessing_Training",
    node_type=NodeType.INTERNAL,
    script_contract=_get_tabular_preprocess_contract(),
    dependencies=[
        DependencySpec(
            logical_name="DATA",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["CradleDataLoading_Training"],
            semantic_keywords=["training", "train", "data", "input", "raw", "dataset", "source", "tabular", "model_training"],
            data_type="S3Uri",
            description="Raw training data for preprocessing"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed training data with train/val/test splits",
            semantic_keywords=["training", "train", "processed", "data", "tabular", "splits", "model_training", "preprocessed"]
        )
    ]
)
