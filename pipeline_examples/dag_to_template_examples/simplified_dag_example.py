#!/usr/bin/env python3
"""
Simplified example showing the DAG structure of an XGBoost pipeline.

This script demonstrates the concept of using a DAG (Directed Acyclic Graph)
to define a pipeline structure without executing the full conversion process.
It avoids circular import issues while showing how the DAG approach works.
"""

import logging
import argparse
from pathlib import Path
import json
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import just the DAG class directly
from src.pipeline_dag.base_dag import PipelineDAG


def create_xgboost_pipeline_dag() -> PipelineDAG:
    """
    Create the DAG structure for the XGBoost train-calibrate-evaluate pipeline.
    
    This is extracted from the template-based implementation in
    template_pipeline_xgboost_train_calibrate_evaluate_e2e.py.
    
    Returns:
        PipelineDAG: The directed acyclic graph representing the pipeline
    """
    dag = PipelineDAG()
    
    # Add all nodes - named to match configuration names
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


def visualize_dag_structure(dag, output_file=None):
    """
    Visualize the DAG structure and optionally save to a file.
    
    Args:
        dag: The PipelineDAG to visualize
        output_file: Optional file path to save the visualization
    """
    # Print basic information about the DAG
    logger.info(f"DAG Structure:")
    logger.info(f"  Nodes ({len(dag.nodes)}): {', '.join(sorted(dag.nodes))}")
    logger.info(f"  Edges ({len(dag.edges)}):")
    
    # Create visualization data
    node_connections = []
    
    # Print edges in a readable format
    # Adapting to the actual structure of the DAG.edges
    # If edges is a list of tuples (src, dst)
    if isinstance(dag.edges, list):
        # Sort edges by source, then by target
        sorted_edges = sorted(dag.edges, key=lambda x: (x[0], x[1]))
        for src, target in sorted_edges:
            logger.info(f"    {src} → {target}")
            node_connections.append({"from": src, "to": target})
    # If edges is a dict mapping src -> list of targets
    elif isinstance(dag.edges, dict):
        for src, targets in sorted(dag.edges.items()):
            sorted_targets = sorted(targets) if isinstance(targets, list) else sorted([targets])
            for target in sorted_targets:
                logger.info(f"    {src} → {target}")
                node_connections.append({"from": src, "to": target})
    else:
        logger.error(f"Unexpected type for dag.edges: {type(dag.edges)}")
    
    # Print topological sort (execution order)
    try:
        topo_sort = dag.topological_sort()
        logger.info(f"Execution order: {' → '.join(topo_sort)}")
    except Exception as e:
        logger.error(f"Error in topological sort: {e}")
    
    # Check for cycles - manually implementing since find_cycles might not be available
    try:
        # A simple cycle detection using DFS
        def has_cycle(graph, node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True
            
            # For all neighbors of this node
            for neighbor in graph.get(node, []):
                if not visited.get(neighbor, False):
                    if has_cycle(graph, neighbor, visited, rec_stack):
                        return True
                elif rec_stack.get(neighbor, False):
                    return True
            
            rec_stack[node] = False
            return False
        
        # Create a dict representation of the graph
        graph = {}
        if isinstance(dag.edges, list):
            for src, dst in dag.edges:
                if src not in graph:
                    graph[src] = []
                graph[src].append(dst)
        elif isinstance(dag.edges, dict):
            graph = dag.edges
        
        # Check for cycles
        visited = {}
        rec_stack = {}
        has_cycles = False
        
        for node in dag.nodes:
            if not visited.get(node, False):
                if has_cycle(graph, node, visited, rec_stack):
                    has_cycles = True
                    break
        
        if has_cycles:
            logger.warning("DAG contains cycles")
        else:
            logger.info("DAG is acyclic (no cycles)")
    except Exception as e:
        logger.error(f"Error checking for cycles: {e}")
    
    # Print potential parallelization - find nodes with no incoming edges
    try:
        # Determine nodes with no incoming edges
        incoming_edges = {node: 0 for node in dag.nodes}
        
        if isinstance(dag.edges, list):
            for _, dst in dag.edges:
                incoming_edges[dst] += 1
        elif isinstance(dag.edges, dict):
            for src, targets in dag.edges.items():
                if isinstance(targets, list):
                    for dst in targets:
                        incoming_edges[dst] += 1
                else:
                    incoming_edges[targets] += 1
        
        # Nodes with no incoming edges can be executed in parallel at the start
        parallelizable_nodes = [node for node, count in incoming_edges.items() if count == 0]
        if parallelizable_nodes:
            logger.info(f"Parallelizable nodes: {', '.join(parallelizable_nodes)}")
    except Exception as e:
        logger.error(f"Error finding parallelizable nodes: {e}")
    
    # Save visualization data to file if requested
    if output_file:
        visualization_data = {
            "nodes": [{"id": node, "label": node.replace('_', ' ').title()} for node in dag.nodes],
            "edges": node_connections
        }
        
        with open(output_file, 'w') as f:
            json.dump(visualization_data, f, indent=2)
        logger.info(f"Saved visualization data to {output_file}")


def explain_dag_to_pipeline_concept():
    """
    Explain the concept of DAG-to-pipeline conversion.
    """
    logger.info("\nDAG to Pipeline Conversion Concept:")
    logger.info("---------------------------------")
    logger.info("In a complete implementation, the DAG would be converted to a SageMaker pipeline as follows:")
    logger.info("1. Load configuration from a JSON file")
    logger.info("2. Create the DAG structure as shown above")
    logger.info("3. Create a PipelineDAGConverter with the configuration")
    logger.info("4. Use the converter to map DAG nodes to configurations")
    logger.info("5. Convert configurations to SageMaker pipeline steps")
    logger.info("6. Build and return the final pipeline")
    logger.info("\nThe actual implementation is more complex due to:")
    logger.info("- Configuration resolution: Matching DAG nodes to config objects")
    logger.info("- Step builder registration: Finding builders for each step type")
    logger.info("- Dependency resolution: Managing inputs and outputs between steps")
    logger.info("- Validation: Ensuring all nodes have valid configurations and builders")
    logger.info("\nThe DAG approach provides several benefits:")
    logger.info("- More explicit definition of pipeline structure")
    logger.info("- Better separation of concerns (DAG, configs, builders)")
    logger.info("- Easier to understand for new users")
    logger.info("- Enhanced validation and debugging capabilities")


def main():
    parser = argparse.ArgumentParser(description="Demonstrate DAG structure for XGBoost pipeline")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for visualization data (JSON format)")
    
    args = parser.parse_args()
    
    # Create and visualize the DAG
    dag = create_xgboost_pipeline_dag()
    visualize_dag_structure(dag, args.output)
    
    # Explain the concept
    explain_dag_to_pipeline_concept()


if __name__ == "__main__":
    main()
