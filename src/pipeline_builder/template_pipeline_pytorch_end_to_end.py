"""
Example of using the PipelineBuilderTemplate to reconstruct the PyTorch end-to-end pipeline.

This template creates a complete PyTorch training and deployment pipeline with the following steps:
1. Data loading from Cradle (training data)
2. Tabular preprocessing (training data)
3. PyTorch model training
4. PyTorch model creation
5. Model packaging
6. Payload testing
7. Model registration
8. Data loading from Cradle (calibration data)
9. Tabular preprocessing (calibration data)

The pipeline is defined using a DAG structure and step builders, which are orchestrated
by the PipelineBuilderTemplate.
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
from src.pipeline_steps.config_training_step_pytorch import PytorchTrainingConfig
from src.pipeline_steps.config_model_step_pytorch import PytorchModelCreationConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

# Step builders
from src.pipeline_steps.builder_step_base import StepBuilderBase
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from src.pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from src.pipeline_steps.builder_training_step_pytorch import PyTorchTrainingStepBuilder
from src.pipeline_steps.builder_model_step_pytorch import PytorchModelStepBuilder
from src.pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from src.pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
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
    "CradleDataLoading": CradleDataLoadingStepBuilder,
    "TabularPreprocessing": TabularPreprocessingStepBuilder,
    "PytorchTraining": PyTorchTrainingStepBuilder,
    "PytorchModel": PytorchModelStepBuilder,
    "Package": MIMSPackagingStepBuilder,
    "Payload": MIMSPayloadStepBuilder,
    "Registration": ModelRegistrationStepBuilder,
}

def create_pipeline_from_template(
    config_path: str,
    sagemaker_session: PipelineSession = None,
    role: str = None,
    notebook_root: Path = None
) -> Pipeline:
    """
    Create a PyTorch end-to-end pipeline using the PipelineBuilderTemplate.
    
    This function:
    1. Loads configurations from the specified config file
    2. Identifies the required configurations for each step
    3. Creates a DAG defining the pipeline structure
    4. Uses the PipelineBuilderTemplate to generate the pipeline
    
    Args:
        config_path: Path to the configuration file
        sagemaker_session: SageMaker session
        role: IAM role
        notebook_root: Root directory of notebook
        
    Returns:
        SageMaker pipeline
        
    Raises:
        ValueError: If required configurations are missing or if there are duplicate configurations
    """
    import time
    start_time = time.time()
    
    # Load configurations
    logger.info(f"Loading configs from: {config_path}")
    config_classes = {
        'BasePipelineConfig': BasePipelineConfig,
        'CradleDataLoadConfig': CradleDataLoadConfig,
        'TabularPreprocessingConfig': TabularPreprocessingConfig,
        'PytorchTrainingConfig': PytorchTrainingConfig,
        'PytorchModelCreationConfig': PytorchModelCreationConfig,
        'PackageStepConfig': PackageStepConfig,
        'ModelRegistrationConfig': ModelRegistrationConfig,
        'PayloadConfig': PayloadConfig,
    }
    
    try:
        configs = load_configs(config_path, config_classes)
    except Exception as e:
        logger.error(f"Error loading configurations: {e}")
        raise ValueError(f"Failed to load configurations from {config_path}: {e}") from e
    
    logger.info(f"Loaded {len(configs)} configurations")
    
    # Extract base config
    if 'Base' not in configs:
        raise ValueError("Base configuration not found in config file")
    base_config = configs['Base']
    
    # Find configs by type and job type
    try:
        cradle_train_key = _find_config_key(configs, 'CradleDataLoadConfig', job_type='training')
        cradle_test_key = _find_config_key(configs, 'CradleDataLoadConfig', job_type='calibration')
        tp_train_key = _find_config_key(configs, 'TabularPreprocessingConfig', job_type='training')
        tp_test_key = _find_config_key(configs, 'TabularPreprocessingConfig', job_type='calibration')
        
        # Find PyTorch training config
        pytorch_train_config = _find_config_by_type(configs, PytorchTrainingConfig)
        
        # Find PyTorch model creation config
        pytorch_model_config = _find_config_by_type(configs, PytorchModelCreationConfig)
        
        # Find packaging config
        package_config = _find_config_by_type(configs, PackageStepConfig)
        
        # Find registration config
        registration_config = _find_config_by_type(configs, ModelRegistrationConfig)
        
        # Find payload config
        payload_config = _find_config_by_type(configs, PayloadConfig)
    except ValueError as e:
        logger.error(f"Error finding required configurations: {e}")
        raise
    
    # Create config map
    config_map = {
        "CradleDataLoading_Training": configs[cradle_train_key],
        "TabularPreprocessing_Training": configs[tp_train_key],
        "PytorchTraining": pytorch_train_config,
        "PytorchModel": pytorch_model_config,
        "Package": package_config,
        "Payload": payload_config,
        "Registration": registration_config,
        "CradleDataLoading_Calibration": configs[cradle_test_key],
        "TabularPreprocessing_Calibration": configs[tp_test_key],
    }
    
    logger.info(f"Created config map with {len(config_map)} entries")
    
    # Define DAG nodes and edges
    nodes = [
        "CradleDataLoading_Training",
        "TabularPreprocessing_Training",
        "PytorchTraining",
        "PytorchModel",
        "Package",
        "Payload",
        "Registration",
        "CradleDataLoading_Calibration",
        "TabularPreprocessing_Calibration",
    ]
    
    edges = [
        ("CradleDataLoading_Training", "TabularPreprocessing_Training"),
        ("TabularPreprocessing_Training", "PytorchTraining"),
        ("PytorchTraining", "PytorchModel"),
        ("PytorchModel", "Package"),
        ("Package", "Payload"),
        ("Payload", "Registration"),
        ("CradleDataLoading_Calibration", "TabularPreprocessing_Calibration"),
    ]
    
    # Create DAG
    dag = PipelineDAG(nodes=nodes, edges=edges)
    logger.info(f"Created DAG with {len(nodes)} nodes and {len(edges)} edges")
    
    # Create pipeline parameters
    pipeline_parameters = [
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    ]
    
    # Create template
    try:
        template = PipelineBuilderTemplate(
            dag=dag,
            config_map=config_map,
            step_builder_map=BUILDER_MAP,
            sagemaker_session=sagemaker_session,
            role=role,
            pipeline_parameters=pipeline_parameters,
            notebook_root=notebook_root,
        )
    except Exception as e:
        logger.error(f"Error creating pipeline template: {e}")
        raise ValueError(f"Failed to create pipeline template: {e}") from e
    
    # Generate pipeline
    pipeline_name = f"{base_config.pipeline_name}-pytorch-e2e"
    try:
        pipeline = template.generate_pipeline(pipeline_name)
        elapsed_time = time.time() - start_time
        logger.info(f"Generated pipeline {pipeline_name} in {elapsed_time:.2f} seconds")
        return pipeline
    except Exception as e:
        logger.error(f"Error generating pipeline: {e}")
        raise ValueError(f"Failed to generate pipeline {pipeline_name}: {e}") from e

def _find_config_key(configs: Dict[str, BasePipelineConfig], class_name: str, **attrs) -> str:
    """
    Find the unique step_name for configs of type `class_name`
    that have all of the given attribute=value pairs in their suffix.
    
    Args:
        configs: Dictionary of configurations
        class_name: Name of the configuration class to find
        **attrs: Attribute-value pairs to match in the configuration suffix
        
    Returns:
        Step name of the matching configuration
        
    Raises:
        ValueError: If no matching configuration is found or if multiple matching configurations are found
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
