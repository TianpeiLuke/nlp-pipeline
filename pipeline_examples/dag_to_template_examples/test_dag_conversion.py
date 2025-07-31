#!/usr/bin/env python3
"""
Simple test script to demonstrate the DAG to pipeline conversion.

This script shows how to use the DAG-to-pipeline approach with a simplified
example that just shows the pipeline structure without actually creating it.
"""

import logging
import argparse
from pathlib import Path
import sys

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import components from the example script
from pipeline_examples.dag_to_template_examples.xgboost_train_calibrate_evaluate_dag import (
    create_xgboost_pipeline_dag,
    validate_config_structure,
    load_configs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def visualize_dag_structure():
    """
    Create and visualize the DAG structure without loading configs.
    
    This provides a simple way to see the pipeline structure.
    """
    # Create the DAG
    dag = create_xgboost_pipeline_dag()
    
    # Print basic information about the DAG
    logger.info(f"DAG Structure:")
    logger.info(f"  Nodes ({len(dag.nodes)}): {', '.join(sorted(dag.nodes))}")
    logger.info(f"  Edges ({len(dag.edges)}):")
    
    # Print edges in a readable format
    for src, targets in sorted(dag.edges.items()):
        for target in sorted(targets):
            logger.info(f"    {src} → {target}")
    
    # Print topological sort (execution order)
    try:
        topo_sort = dag.topological_sort()
        logger.info(f"Execution order: {' → '.join(topo_sort)}")
    except Exception as e:
        logger.error(f"Error in topological sort: {e}")
    
    # Check for cycles
    try:
        cycles = dag.find_cycles()
        if cycles:
            logger.warning(f"DAG contains cycles: {cycles}")
        else:
            logger.info("DAG is acyclic (no cycles)")
    except Exception as e:
        logger.error(f"Error checking for cycles: {e}")
    
    # Print potential parallelization
    parallelizable_nodes = dag.get_independent_nodes()
    if parallelizable_nodes:
        logger.info(f"Parallelizable nodes: {', '.join(parallelizable_nodes)}")


def preview_dag_config_mapping(config_path):
    """
    Preview how DAG nodes will map to configuration objects.
    
    Args:
        config_path: Path to configuration file
    """
    try:
        # Load configurations
        logger.info(f"Loading configurations from {config_path}")
        configs = load_configs(config_path)
        
        # Try to validate configuration structure
        try:
            validate_config_structure(configs)
            logger.info("Configuration validation passed")
        except ValueError as e:
            logger.error(f"Configuration validation failed: {e}")
            return
        
        # Create the DAG
        dag = create_xgboost_pipeline_dag()
        
        # Print node to config mapping based on naming patterns
        logger.info(f"Expected node to configuration mapping:")
        
        # Dictionary to store the node-to-config mapping
        node_config_map = {}
        
        # Find configs by type and job_type attribute for preprocessing steps
        from src.pipeline_steps.config_data_load_step_cradle import CradleDataLoadConfig
        from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
        
        cradle_configs = {
            getattr(cfg, 'job_type', 'unknown'): cfg 
            for _, cfg in configs.items() 
            if isinstance(cfg, CradleDataLoadConfig)
        }
        
        tp_configs = {
            getattr(cfg, 'job_type', 'unknown'): cfg 
            for _, cfg in configs.items() 
            if isinstance(cfg, TabularPreprocessingConfig)
        }
        
        # Map training and calibration data loading and preprocessing steps
        node_config_map["CradleDataLoading_training"] = cradle_configs.get('training')
        node_config_map["TabularPreprocessing_training"] = tp_configs.get('training')
        node_config_map["CradleDataLoading_calibration"] = cradle_configs.get('calibration')
        node_config_map["TabularPreprocessing_calibration"] = tp_configs.get('calibration')
        
        # Import types for single-instance config steps
        from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
        from src.pipeline_steps.config_model_calibration_step import ModelCalibrationConfig
        from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
        from src.pipeline_steps.config_mims_payload_step import PayloadConfig
        from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
        from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
        
        # Map for node name to config type
        node_config_type_map = {
            "XGBoostTraining": XGBoostTrainingConfig,
            "ModelCalibration": ModelCalibrationConfig,
            "Package": PackageStepConfig,
            "Payload": PayloadConfig,
            "Registration": ModelRegistrationConfig,
            "XGBoostModelEval_calibration": XGBoostModelEvalConfig
        }
        
        # Find and map single instance configs
        for node_name, config_type in node_config_type_map.items():
            instances = [cfg for _, cfg in configs.items() if type(cfg) is config_type]
            if instances:
                node_config_map[node_name] = instances[0]
            else:
                logger.warning(f"No configuration found for node '{node_name}' of type {config_type.__name__}")
        
        # Print the node-to-config mapping
        for node in sorted(dag.nodes):
            config = node_config_map.get(node)
            if config:
                config_type = type(config).__name__
                # Add job_type if available
                job_type = getattr(config, 'job_type', None)
                job_type_str = f" (job_type='{job_type}')" if job_type else ""
                logger.info(f"  {node} → {config_type}{job_type_str}")
            else:
                logger.warning(f"  {node} → NO MATCHING CONFIG FOUND")
                
    except Exception as e:
        logger.error(f"Error previewing DAG-config mapping: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DAG to Pipeline Conversion")
    parser.add_argument("--config-path", type=str, 
                        help="Path to config file for mapping preview")
    
    args = parser.parse_args()
    
    # Always visualize the DAG structure
    visualize_dag_structure()
    
    # Preview config mapping if config path provided
    if args.config_path:
        preview_dag_config_mapping(args.config_path)
    else:
        logger.info("\nRun with --config-path to preview node-to-config mapping")
