"""
Example of using the PipelineBuilderTemplate to reconstruct the XGBoost end-to-end pipeline.
"""
import logging
from pathlib import Path
from typing import Dict, Type, List, Any

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterString

from src.pipeline_steps.utils import load_configs
from src.pipeline_builder.pipeline_dag import PipelineDAG
from src.pipeline_builder.pipeline_builder_template import PipelineBuilderTemplate

# Config classes
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.config_model_step_xgboost import XGBoostModelCreationConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from src.pipeline_steps.builder_model_step_xgboost import XGBoostModelStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common parameters
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir", default_value="/tmp")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMSEncryptionKey", default_value="")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId", default_value="")
VPC_SUBNET = ParameterString(name="VPCEndpointSubnet", default_value="")

# Map config classes to step builder classes
BUILDER_MAP = {
    "CradleDataLoadingStep": CradleDataLoadingStepBuilder,
    "TabularPreprocessingStep": TabularPreprocessingStepBuilder,
    "XGBoostTrainingStep": XGBoostTrainingStepBuilder,
    "CreateXGBoostModelStep": XGBoostModelStepBuilder,
    "PackagingStep": MIMSPackagingStepBuilder,
    "RegistrationStep": ModelRegistrationStepBuilder,
}

def create_pipeline_from_template(
    config_path: str,
    sagemaker_session: PipelineSession = None,
    role: str = None,
    notebook_root: Path = None
) -> Pipeline:
    """
    Create a pipeline using the PipelineBuilderTemplate.
    
    Args:
        config_path: Path to the configuration file
        sagemaker_session: SageMaker session
        role: IAM role
        notebook_root: Root directory of notebook
        
    Returns:
        SageMaker pipeline
    """
    # Load configurations
    logger.info(f"Loading configs from: {config_path}")
    config_classes = {
        'BasePipelineConfig': BasePipelineConfig,
        'CradleDataLoadConfig': CradleDataLoadConfig,
        'TabularPreprocessingConfig': TabularPreprocessingConfig,
        'XGBoostTrainingConfig': XGBoostTrainingConfig,
        'XGBoostModelCreationConfig': XGBoostModelCreationConfig,
        'PackageStepConfig': PackageStepConfig,
        'ModelRegistrationConfig': ModelRegistrationConfig,
        'PayloadConfig': PayloadConfig,
    }
    configs = load_configs(config_path, config_classes)
    
    # Extract base config
    base_config = configs['Base']
    
    # Find configs by type and job type
    cradle_train_key = _find_config_key(configs, 'CradleDataLoadConfig', job_type='training')
    cradle_test_key = _find_config_key(configs, 'CradleDataLoadConfig', job_type='calibration')
    tp_train_key = _find_config_key(configs, 'TabularPreprocessingConfig', job_type='training')
    tp_test_key = _find_config_key(configs, 'TabularPreprocessingConfig', job_type='calibration')
    
    # Find XGBoost training config
    xgb_train_config = _find_config_by_type(configs, XGBoostTrainingConfig)
    
    # Find XGBoost model creation config
    xgb_model_config = _find_config_by_type(configs, XGBoostModelCreationConfig)
    
    # Find packaging config
    package_config = _find_config_by_type(configs, PackageStepConfig)
    
    # Find registration config
    registration_config = _find_config_by_type(configs, ModelRegistrationConfig)
    
    # Create config map
    config_map = {
        "CradleDataLoadingStep_Training": configs[cradle_train_key],
        "TabularPreprocessingStep_Training": configs[tp_train_key],
        "XGBoostTrainingStep": xgb_train_config,
        "CreateXGBoostModelStep": xgb_model_config,
        "PackagingStep": package_config,
        "RegistrationStep": registration_config,
        "CradleDataLoadingStep_Calibration": configs[cradle_test_key],
        "TabularPreprocessingStep_Calibration": configs[tp_test_key],
    }
    
    # Define DAG nodes and edges
    nodes = [
        "CradleDataLoadingStep_Training",
        "TabularPreprocessingStep_Training",
        "XGBoostTrainingStep",
        "CreateXGBoostModelStep",
        "PackagingStep",
        "RegistrationStep",
        "CradleDataLoadingStep_Calibration",
        "TabularPreprocessingStep_Calibration",
    ]
    
    edges = [
        ("CradleDataLoadingStep_Training", "TabularPreprocessingStep_Training"),
        ("TabularPreprocessingStep_Training", "XGBoostTrainingStep"),
        ("XGBoostTrainingStep", "CreateXGBoostModelStep"),
        ("CreateXGBoostModelStep", "PackagingStep"),
        ("PackagingStep", "RegistrationStep"),
        ("CradleDataLoadingStep_Calibration", "TabularPreprocessingStep_Calibration"),
    ]
    
    # Create DAG
    dag = PipelineDAG(nodes=nodes, edges=edges)
    
    # Create pipeline parameters
    pipeline_parameters = [
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    ]
    
    # Create template
    template = PipelineBuilderTemplate(
        dag=dag,
        config_map=config_map,
        step_builder_map=BUILDER_MAP,
        sagemaker_session=sagemaker_session,
        role=role,
        pipeline_parameters=pipeline_parameters,
        notebook_root=notebook_root,
    )
    
    # Generate pipeline
    pipeline_name = f"{base_config.pipeline_name}-xgb-e2e"
    return template.generate_pipeline(pipeline_name)

def _find_config_key(configs: Dict[str, BasePipelineConfig], class_name: str, **attrs) -> str:
    """
    Find the unique step_name for configs of type `class_name`
    that have all of the given attribute=value pairs in their suffix.
    """
    base = BasePipelineConfig.get_step_name(class_name)
    candidates = []
    for step_name in configs:
        if not step_name.startswith(base + "_"):
            continue
        # extract suffix parts
        parts = step_name[len(base) + 1:].split("_")
        if all(str(val) in parts for val in attrs.values()):
            candidates.append(step_name)

    if not candidates:
        raise ValueError(f"No config found for {class_name} with {attrs}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple configs found for {class_name} with {attrs}: {candidates}")
    return candidates[0]

def _find_config_by_type(configs: Dict[str, BasePipelineConfig], config_type: Type) -> BasePipelineConfig:
    """
    Find a configuration of a specific type.
    
    Args:
        configs: Dictionary of configurations
        config_type: Type of configuration to find
        
    Returns:
        Configuration of the specified type
        
    Raises:
        ValueError: If no configuration of the specified type is found
    """
    for config in configs.values():
        if isinstance(config, config_type):
            return config
    raise ValueError(f"No configuration of type {config_type.__name__} found")

if __name__ == "__main__":
    # Example usage
    pipeline = create_pipeline_from_template(
        config_path="path/to/config.json",
        sagemaker_session=PipelineSession(),
        role="arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole",
    )
    
    # Print pipeline definition
    print(pipeline.definition())
