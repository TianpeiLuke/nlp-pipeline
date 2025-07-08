"""
MIMS Packaging Step Specification.

This module defines the declarative specification for MIMS model packaging steps,
including their dependencies and outputs based on the actual implementation.
"""

from ..pipeline_deps.base_specifications import StepSpecification, DependencySpec, OutputSpec, DependencyType, NodeType
from ..pipeline_registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_mims_package_contract():
    from ..pipeline_script_contracts.mims_package_contract import MIMS_PACKAGE_CONTRACT
    return MIMS_PACKAGE_CONTRACT

# MIMS Packaging Step Specification
PACKAGING_SPEC = StepSpecification(
    step_type=get_spec_step_type("Package"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_mims_package_contract(),
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
            logical_name="packaged_model",
            aliases=["PackagedModel"],  # Add alias to match registration dependency
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ProcessingOutputConfig.Outputs['packaged_model'].S3Output.S3Uri",
            data_type="S3Uri",
            description="Packaged model ready for deployment"
        )
    ]
)
