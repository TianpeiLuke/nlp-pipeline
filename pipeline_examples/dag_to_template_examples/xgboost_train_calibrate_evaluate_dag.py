#!/usr/bin/env python3
"""
Example of converting a DAG to a SageMaker pipeline using dag_to_pipeline_template.

This script demonstrates how to convert a PipelineDAG to a SageMaker pipeline
without using the template-based approach. It extracts the DAG from
template_pipeline_xgboost_train_calibrate_evaluate_e2e.py and uses the
dag_to_pipeline_template function from the pipeline_api module.

The resulting pipeline performs:
1) Data Loading (for training set)
2) Tabular Preprocessing (for training set)
3) XGBoost Model Training
4) Model Calibration
5) Packaging
6) MIMS Registration
7) Data Loading (for calibration set)
8) Tabular Preprocessing (for calibration set)
9) Model Evaluation (on calibration set)
"""

from typing import Dict, List, Any, Optional
import argparse
import logging
import json
import os
from pathlib import Path

from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.network import NetworkConfig
from sagemaker.workflow.parameters import ParameterString
from sagemaker.image_uris import retrieve

# Import DAG-related components
from src.pipeline_dag.base_dag import PipelineDAG
from src.pipeline_api.dag_converter import dag_to_pipeline_template, PipelineDAGConverter
from src.pipeline_api.validation import ConversionReport
from src.pipeline_steps.utils import load_configs

# Import configuration classes for type checking
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig 
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig
from src.pipeline_steps.config_model_calibration_step import ModelCalibrationConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import constants from core library (these are the parameters that will be wrapped)
try:
    from mods_workflow_core.utils.constants import (
        PIPELINE_EXECUTION_TEMP_DIR,
        KMS_ENCRYPTION_KEY_PARAM,
        PROCESSING_JOB_SHARED_NETWORK_CONFIG,
        SECURITY_GROUP_ID,
        VPC_SUBNET,
    )
    logger.info("Successfully imported constants from mods_workflow_core")
except ImportError:
    logger.warning("Could not import constants from mods_workflow_core, using local definitions")
    # Define pipeline parameters locally if import fails - match exact definitions from original module
    PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="EXECUTION_S3_PREFIX")
    KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KMS_ENCRYPTION_KEY_PARAM")
    SECURITY_GROUP_ID = ParameterString(name="SECURITY_GROUP_ID")
    VPC_SUBNET = ParameterString(name="VPC_SUBNET")
    # Also create the network config as defined in the original module
    PROCESSING_JOB_SHARED_NETWORK_CONFIG = NetworkConfig(
        enable_network_isolation=False,
        security_group_ids=[SECURITY_GROUP_ID],
        subnets=[VPC_SUBNET],
        encrypt_inter_container_traffic=True,
    )


def create_xgboost_pipeline_dag() -> PipelineDAG:
    """
    Create the DAG structure for the XGBoost train-calibrate-evaluate pipeline.
    
    This is extracted from the _create_pipeline_dag method in the
    XGBoostTrainCalibrateEvaluateE2ETemplate class.
    
    Returns:
        PipelineDAG: The directed acyclic graph representing the pipeline
    """
    dag = PipelineDAG()
    
    # Add all nodes - renamed to match configuration names exactly
    dag.add_node("CradleDataLoading_training")    # Data load for training
    dag.add_node("TabularPreprocessing_training") # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")              # XGBoost training step
    dag.add_node("ModelCalibration")             # Model calibration step
    dag.add_node("Package")                      # Package step
    dag.add_node("Registration")                 # MIMS registration step
    dag.add_node("Payload")                      # Payload step
    dag.add_node("CradleDataLoading_calibration") # Data load for calibration
    dag.add_node("TabularPreprocessing_calibration") # Tabular preprocessing for calibration
    dag.add_node("XGBoostModelEval_calibration")     # Model evaluation step
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")
    dag.add_edge("XGBoostTraining", "ModelCalibration")
    
    # Output flow
    dag.add_edge("ModelCalibration", "Package")
    dag.add_edge("XGBoostTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("XGBoostTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")
    
    # Calibration flow
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
    
    # Evaluation flow
    dag.add_edge("XGBoostTraining", "XGBoostModelEval_calibration")
    dag.add_edge("TabularPreprocessing_calibration", "XGBoostModelEval_calibration")
    
    logger.info(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag


def validate_config_structure(configs: Dict[str, BasePipelineConfig]) -> bool:
    """
    Perform lightweight validation of configuration structure.
    
    This validates the presence of required configurations and basic structural
    requirements, similar to _validate_configuration in the template.
    
    Args:
        configs: Dictionary mapping config names to config objects
        
    Returns:
        bool: True if configuration is valid, False otherwise
        
    Raises:
        ValueError: If configuration structure is invalid
    """
    # Find preprocessing configs
    tp_configs = [cfg for name, cfg in configs.items() 
                 if isinstance(cfg, TabularPreprocessingConfig)]
    
    if len(tp_configs) < 2:
        raise ValueError("Expected at least two TabularPreprocessingConfig instances")
        
    # Check for presence of training and calibration configs
    training_config = next((cfg for cfg in tp_configs if getattr(cfg, 'job_type', None) == 'training'), None)
    if not training_config:
        raise ValueError("No TabularPreprocessingConfig found with job_type='training'")
        
    calibration_config = next((cfg for cfg in tp_configs if getattr(cfg, 'job_type', None) == 'calibration'), None)
    if not calibration_config:
        raise ValueError("No TabularPreprocessingConfig found with job_type='calibration'")
    
    # Check for required single-instance configs
    for config_type, name in [
        (XGBoostTrainingConfig, "XGBoost training"),
        (ModelCalibrationConfig, "model calibration"),
        (PackageStepConfig, "model packaging"),
        (PayloadConfig, "payload testing"),
        (ModelRegistrationConfig, "model registration"),
        (XGBoostModelEvalConfig, "model evaluation")
    ]:
        instances = [cfg for _, cfg in configs.items() if type(cfg) is config_type]
        if not instances:
            raise ValueError(f"No {name} configuration found")
        if len(instances) > 1:
            raise ValueError(f"Multiple {name} configurations found, expected exactly one")
            
    logger.info("Basic configuration structure validation passed")
    return True


def create_execution_doc_config(configs: Dict[str, BasePipelineConfig]) -> Dict[str, Any]:
    """
    Create execution document configuration for registration steps.
    
    This is similar to the _create_execution_doc_config method in the template.
    
    Args:
        configs: Dictionary mapping config names to config objects
        
    Returns:
        Dict: Configuration for execution document
    """
    # Find needed configs
    registration_cfg = next((cfg for _, cfg in configs.items() 
                           if isinstance(cfg, ModelRegistrationConfig) and not isinstance(cfg, PayloadConfig)), None)
    payload_cfg = next((cfg for _, cfg in configs.items() 
                       if isinstance(cfg, PayloadConfig)), None)
    package_cfg = next((cfg for _, cfg in configs.items() 
                       if isinstance(cfg, PackageStepConfig)), None)
    
    if not registration_cfg or not payload_cfg or not package_cfg:
        raise ValueError("Missing required configs for execution document")
    
    # Get image URI
    try:
        image_uri = retrieve(
            framework=registration_cfg.framework,
            region=registration_cfg.aws_region,
            version=registration_cfg.framework_version,
            py_version=registration_cfg.py_version,
            instance_type=registration_cfg.inference_instance_type,
            image_scope="inference"
        )
        logger.info(f"Retrieved image URI: {image_uri}")
    except Exception as e:
        logger.warning(f"Could not retrieve image URI: {e}")
        image_uri = "image-uri-placeholder"  # Use placeholder for template
    
    return {
        "model_domain": registration_cfg.model_registration_domain,
        "model_objective": registration_cfg.model_registration_objective,
        "source_model_inference_content_types": registration_cfg.source_model_inference_content_types,
        "source_model_inference_response_types": registration_cfg.source_model_inference_response_types,
        "source_model_inference_input_variable_list": registration_cfg.source_model_inference_input_variable_list,
        "source_model_inference_output_variable_list": registration_cfg.source_model_inference_output_variable_list,
        "model_registration_region": registration_cfg.region,
        "source_model_inference_image_arn": image_uri,
        "source_model_region": registration_cfg.aws_region,
        "model_owner": registration_cfg.model_owner,
        "source_model_environment_variable_map": {
            "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
            "SAGEMAKER_PROGRAM": registration_cfg.inference_entry_point,
            "SAGEMAKER_REGION": registration_cfg.aws_region,
            "SAGEMAKER_SUBMIT_DIRECTORY": '/opt/ml/model/code',
        },
        'load_testing_info_map': {
            "sample_payload_s3_bucket": registration_cfg.bucket,
            "sample_payload_s3_key": payload_cfg.sample_payload_s3_key,
            "expected_tps": payload_cfg.expected_tps,
            "max_latency_in_millisecond": payload_cfg.max_latency_in_millisecond,
            "instance_type_list": [package_cfg.get_instance_type() if hasattr(package_cfg, 'get_instance_type') else package_cfg.processing_instance_type_small],
            "max_acceptable_error_rate": payload_cfg.max_acceptable_error_rate,
        },
    }


def fill_execution_document(execution_document: Dict[str, Any], 
                           configs: Dict[str, BasePipelineConfig],
                           pipeline_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fill in the execution document with pipeline metadata.
    
    This is similar to the fill_execution_document method in the template.
    
    Args:
        execution_document: Execution document to fill
        configs: Dictionary of configurations
        pipeline_metadata: Pipeline metadata from conversion
        
    Returns:
        Dict: Updated execution document
    """
    if "PIPELINE_STEP_CONFIGS" not in execution_document:
        raise KeyError("Execution document missing 'PIPELINE_STEP_CONFIGS' key")

    pipeline_configs = execution_document["PIPELINE_STEP_CONFIGS"]

    # Fill Cradle configurations
    cradle_requests = pipeline_metadata.get('cradle_loading_requests', {})
    for step_name, request_dict in cradle_requests.items():
        if step_name not in pipeline_configs:
            logger.warning(f"Cradle step '{step_name}' not found in execution document")
            continue
        pipeline_configs[step_name]["STEP_CONFIG"] = request_dict
        logger.info(f"Updated execution config for Cradle step: {step_name}")

    # Find registration config
    registration_cfg = next(
        (cfg for _, cfg in configs.items() 
         if isinstance(cfg, ModelRegistrationConfig)), 
        None
    )
    
    # Create execution document config
    if registration_cfg:
        try:
            # Create execution document config
            exec_config = create_execution_doc_config(configs)
            
            # Check multiple naming patterns for the registration step
            registration_step_found = False
            for registration_step_name in [
                f"ModelRegistration-{registration_cfg.region}",  # Format from error log
                f"Registration_{registration_cfg.region}",       # Format from template code
                "model_registration"                             # Generic fallback
            ]:
                if registration_step_name in pipeline_configs:
                    pipeline_configs[registration_step_name]["STEP_CONFIG"] = exec_config
                    logger.info(f"Updated execution config for registration step: {registration_step_name}")
                    registration_step_found = True
                    break
                
            if not registration_step_found:
                logger.warning(f"Registration step not found in execution document with any known naming pattern")
                
        except Exception as e:
            logger.warning(f"Failed to create execution document config: {e}")

    return execution_document


def main(config_path: str,
         role: str,
         pipeline_name: Optional[str] = None,
         output_path: Optional[str] = None,
         preview_only: bool = False) -> None:
    """
    Main function to build the pipeline from a DAG.
    
    Args:
        config_path: Path to the configuration file
        role: IAM role ARN for pipeline execution
        pipeline_name: Optional pipeline name override
        output_path: Optional path to save pipeline definition
        preview_only: If True, only preview the resolution without creating the pipeline
    """
    # Create SageMaker session
    sagemaker_session = PipelineSession()
    
    # Load configurations
    logger.info(f"Loading configurations from {config_path}")
    configs = load_configs(config_path)
    
    # Validate configuration structure
    try:
        validate_config_structure(configs)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    # Extract pipeline name from base config if not provided
    if not pipeline_name:
        base_config = next((cfg for cfg in configs.values() 
                          if isinstance(cfg, BasePipelineConfig)), None)
        if base_config and hasattr(base_config, 'pipeline_name'):
            pipeline_name = f"{base_config.pipeline_name}-xgb-train-calibrate-eval"
        else:
            pipeline_name = "xgboost-train-calibrate-eval"
    
    # Create DAG
    dag = create_xgboost_pipeline_dag()
    
    # Create converter
    converter = PipelineDAGConverter(
        config_path=config_path,
        sagemaker_session=sagemaker_session,
        role=role
    )
    
    # Preview resolution if requested
    if preview_only:
        preview = converter.preview_resolution(dag)
        logger.info("DAG node resolution preview:")
        for node, config_type in preview.node_config_map.items():
            confidence = preview.resolution_confidence.get(node, 0.0)
            logger.info(f"  {node} → {config_type} (confidence: {confidence:.2f})")
        
        if preview.recommendations:
            logger.info("Recommendations:")
            for recommendation in preview.recommendations:
                logger.info(f"  - {recommendation}")
        
        validation = converter.validate_dag_compatibility(dag)
        logger.info(f"DAG validation: {'VALID' if validation.is_valid else 'INVALID'}")
        if not validation.is_valid:
            if validation.missing_configs:
                logger.warning(f"Missing configs: {validation.missing_configs}")
            if validation.unresolvable_builders:
                logger.warning(f"Unresolvable builders: {validation.unresolvable_builders}")
            if validation.config_errors:
                logger.warning(f"Config errors: {validation.config_errors}")
        
        return
    
    # Convert DAG to pipeline and get report
    try:
        logger.info(f"Converting DAG to pipeline '{pipeline_name}'")
        pipeline, report = converter.convert_with_report(
            dag=dag,
            pipeline_name=pipeline_name
        )
        
        # Log report summary
        logger.info(f"Conversion complete: {report.summary()}")
        for node, details in report.resolution_details.items():
            logger.info(f"  {node} → {details['config_type']} ({details['builder_type']})")
        
        # Save pipeline definition if path provided
        if output_path:
            definition = json.loads(pipeline.definition())
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(definition, f, indent=2)
            logger.info(f"Pipeline definition saved to {output_path}")
        
        # Log pipeline creation details
        logger.info(f"Pipeline '{pipeline.name}' created successfully")
        logger.info(f"Pipeline ARN: {pipeline.arn if hasattr(pipeline, 'arn') else 'Not available until upserted'}")
        logger.info("To upsert the pipeline, call pipeline.upsert()")
        
        return pipeline, report
        
    except Exception as e:
        logger.error(f"Failed to convert DAG to pipeline: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert XGBoost Train-Calibrate-Evaluate DAG to SageMaker pipeline")
    parser.add_argument("--config-path", type=str, required=True, 
                        help="Path to configuration file")
    parser.add_argument("--role", type=str, required=True,
                        help="IAM role ARN for pipeline execution")
    parser.add_argument("--pipeline-name", type=str, default=None,
                        help="Optional pipeline name override")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Path to save pipeline definition JSON")
    parser.add_argument("--preview-only", action="store_true",
                        help="Only preview resolution without creating pipeline")
    
    args = parser.parse_args()
    
    main(
        config_path=args.config_path,
        role=args.role,
        pipeline_name=args.pipeline_name,
        output_path=args.output_path,
        preview_only=args.preview_only
    )
