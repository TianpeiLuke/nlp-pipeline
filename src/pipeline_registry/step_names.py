"""
Central registry for all pipeline step names.
Single source of truth for step naming across config, builders, and specifications.
"""

from typing import Dict, List

# Core step name registry - canonical names used throughout the system
STEP_NAMES = {
    # Base Steps
    "Base": {
        "config_class": "BasePipelineConfig",
        "builder_step_name": "BaseStep",
        "spec_type": "Base",
        "description": "Base pipeline configuration"
    },
    
    # Processing Steps
    "Processing": {
        "config_class": "ProcessingStepConfigBase", 
        "builder_step_name": "ProcessingStep",
        "spec_type": "Processing",
        "description": "Base processing step"
    },
    "TabularPreprocessing": {
        "config_class": "TabularPreprocessingConfig",
        "builder_step_name": "TabularPreprocessingStep",
        "spec_type": "TabularPreprocessing",
        "description": "Tabular data preprocessing step"
    },
    "RiskTableMapping": {
        "config_class": "RiskTableMappingConfig",
        "builder_step_name": "RiskTableMappingStepBuilder",
        "spec_type": "RiskTableMapping",
        "description": "Risk table mapping step for categorical features"
    },
    "CurrencyConversion": {
        "config_class": "CurrencyConversionConfig",
        "builder_step_name": "CurrencyConversionStep",
        "spec_type": "CurrencyConversion",
        "description": "Currency conversion processing step"
    },
    
    # Data Loading Steps
    "CradleDataLoading": {
        "config_class": "CradleDataLoadConfig",
        "builder_step_name": "CradleDataLoadingStep",
        "spec_type": "CradleDataLoading",
        "description": "Cradle data loading step"
    },
    
    # Training Steps
    "PytorchTraining": {  # Canonical: PytorchTraining (not PyTorchTraining)
        "config_class": "PytorchTrainingConfig",
        "builder_step_name": "PytorchTrainingStep",
        "spec_type": "PytorchTraining",
        "description": "PyTorch model training step"
    },
    "XGBoostTraining": {
        "config_class": "XGBoostTrainingConfig", 
        "builder_step_name": "XGBoostTrainingStep",
        "spec_type": "XGBoostTraining",
        "description": "XGBoost model training step"
    },
    "DummyTraining": {
        "config_class": "DummyTrainingConfig",
        "builder_step_name": "DummyTrainingStepBuilder",
        "spec_type": "DummyTraining",
        "description": "Training step that uses a pretrained model"
    },
    
    # Model Creation Steps
    "PytorchModel": {  # Canonical: PytorchModel (not PyTorchModel)
        "config_class": "PytorchModelCreationConfig",
        "builder_step_name": "CreatePytorchModelStep",
        "spec_type": "PytorchModel",
        "description": "PyTorch model creation step"
    },
    "XGBoostModel": {
        "config_class": "XGBoostModelCreationConfig",
        "builder_step_name": "CreateXGBoostModelStep",
        "spec_type": "XGBoostModel",
        "description": "XGBoost model creation step"
    },
    
    # Evaluation Steps
    "XGBoostModelEval": {
        "config_class": "XGBoostModelEvalConfig",
        "builder_step_name": "XGBoostModelEvaluationStep",
        "spec_type": "XGBoostModelEval",
        "description": "XGBoost model evaluation step"
    },
    "PytorchModelEval": {
        "config_class": "PytorchModelEvalConfig", 
        "builder_step_name": "PytorchModelEvaluationStep",
        "spec_type": "PytorchModelEval",
        "description": "PyTorch model evaluation step"
    },
    
    # Deployment Steps
    "Package": {
        "config_class": "PackageStepConfig",
        "builder_step_name": "PackagingStep",
        "spec_type": "Package",
        "description": "Model packaging step"
    },
    "Registration": {
        "config_class": "ModelRegistrationConfig",
        "builder_step_name": "RegistrationStep",
        "spec_type": "Registration",
        "description": "Model registration step"
    },
    "Payload": {
        "config_class": "PayloadConfig",
        "builder_step_name": "PayloadTestStep",
        "spec_type": "Payload",
        "description": "Payload testing step"
    },
    
    # Transform Steps
    "BatchTransform": {
        "config_class": "BatchTransformStepConfig",
        "builder_step_name": "BatchTransformStep",
        "spec_type": "BatchTransform",
        "description": "Batch transform step"
    },
    
    # Utility Steps
    "HyperparameterPrep": {
        "config_class": "HyperparameterPrepConfig",
        "builder_step_name": "HyperparameterPrepStep",
        "spec_type": "HyperparameterPrep",
        "description": "Hyperparameter preparation step"
    }
}

# Generate the mappings that existing code expects
CONFIG_STEP_REGISTRY = {
    info["config_class"]: step_name 
    for step_name, info in STEP_NAMES.items()
}

BUILDER_STEP_NAMES = {
    step_name: info["builder_step_name"]
    for step_name, info in STEP_NAMES.items()
}

# Generate step specification types
SPEC_STEP_TYPES = {
    step_name: info["spec_type"]
    for step_name, info in STEP_NAMES.items()
}

# Helper functions
def get_config_class_name(step_name: str) -> str:
    """Get config class name for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["config_class"]

def get_builder_step_name(step_name: str) -> str:
    """Get builder step class name for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["builder_step_name"]

def get_spec_step_type(step_name: str) -> str:
    """Get step_type value for StepSpecification."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["spec_type"]

def get_spec_step_type_with_job_type(step_name: str, job_type: str = None) -> str:
    """Get step_type with optional job_type suffix."""
    base_type = get_spec_step_type(step_name)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type

def get_step_name_from_spec_type(spec_type: str) -> str:
    """Get canonical step name from spec_type."""
    # Handle job type variants (e.g., "TabularPreprocessing_Training" -> "TabularPreprocessing")
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    
    reverse_mapping = {info["spec_type"]: step_name 
                      for step_name, info in STEP_NAMES.items()}
    return reverse_mapping.get(base_spec_type, spec_type)

def get_all_step_names() -> List[str]:
    """Get all canonical step names."""
    return list(STEP_NAMES.keys())

# Validation functions
def validate_step_name(step_name: str) -> bool:
    """Validate that a step name exists in the registry."""
    return step_name in STEP_NAMES

def validate_spec_type(spec_type: str) -> bool:
    """Validate that a spec_type exists in the registry."""
    # Handle job type variants
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    return base_spec_type in [info["spec_type"] for info in STEP_NAMES.values()]

def get_step_description(step_name: str) -> str:
    """Get description for a step."""
    if step_name not in STEP_NAMES:
        raise ValueError(f"Unknown step name: {step_name}")
    return STEP_NAMES[step_name]["description"]

def list_all_step_info() -> Dict[str, Dict[str, str]]:
    """Get complete step information for all registered steps."""
    return STEP_NAMES.copy()
