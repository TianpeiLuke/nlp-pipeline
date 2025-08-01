"""
Risk Table Mapping Validation Step Specification.

This module defines the declarative specification for risk table mapping steps
specifically for validation data, including their dependencies and outputs.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..pipeline_registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_risk_table_mapping_contract():
    from ..pipeline_script_contracts.risk_table_mapping_contract import RISK_TABLE_MAPPING_CONTRACT
    return RISK_TABLE_MAPPING_CONTRACT

# Risk Table Mapping Validation Step Specification
RISK_TABLE_MAPPING_VALIDATION_SPEC = StepSpecification(
    step_type=get_spec_step_type("RiskTableMapping") + "_Validation",
    node_type=NodeType.INTERNAL,
    script_contract=_get_risk_table_mapping_contract(),
    dependencies=[
        DependencySpec(
            logical_name="data_input",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["TabularPreprocessing", "ProcessingStep"],
            semantic_keywords=["validation", "val", "data", "input", "preprocessed", "tabular"],
            data_type="S3Uri",
            description="Preprocessed validation data from tabular preprocessing step"
        ),
        # Hyperparameters are optional as they can be generated internally
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,
            compatible_sources=[
                "HyperparameterPrep", "ProcessingStep", "ConfigurationStep",
                "DataPrep", "ModelTraining", "FeatureEngineering", "DataQuality"
            ],
            semantic_keywords=["config", "params", "hyperparameters", "settings", "hyperparams"],
            data_type="S3Uri",
            description="Optional external hyperparameters configuration file (will be overridden by internal generation)"
        ),
        DependencySpec(
            logical_name="risk_tables",
            dependency_type=DependencyType.PROCESSING_OUTPUT,
            required=True,
            compatible_sources=["RiskTableMapping_Training"],
            semantic_keywords=["risk_tables", "bin_mapping", "categorical_mappings", "model_artifacts"],
            data_type="S3Uri",
            description="Risk tables and imputation models from training step"
        )
    ],
    outputs=[
        OutputSpec(
            logical_name="processed_data",
            aliases=["validation_data", "model_validation_data"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['processed_data'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Processed validation data with risk table mappings applied"
        ),
        OutputSpec(
            logical_name="risk_tables",
            aliases=["bin_mapping", "risk_table_artifacts", "categorical_mappings"],
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.ProcessingOutputConfig.Outputs['risk_tables'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Risk tables and imputation models (passthrough from training)"
        )
    ]
)
