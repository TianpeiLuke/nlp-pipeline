#!/usr/bin/env python3
"""
Script to generate a visualization of the DAG structure.

This script creates a Graphviz DOT file from the DAG structure
which can be converted to a PNG or SVG for visualization.

Requirements:
- graphviz Python package: pip install graphviz
- Graphviz binaries: https://graphviz.org/download/

Usage:
python visualize_dag.py --output dag_visualization.png
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import components from the example script
from pipeline_examples.dag_to_template_examples.xgboost_train_calibrate_evaluate_dag import (
    create_xgboost_pipeline_dag
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import graphviz
except ImportError:
    logger.error("graphviz package not installed. Install with: pip install graphviz")
    logger.error("You also need to install Graphviz binaries: https://graphviz.org/download/")
    sys.exit(1)

# Define node colors by type
NODE_COLORS = {
    # Data loading nodes (green)
    'CradleDataLoading_training': 'lightgreen',
    'CradleDataLoading_calibration': 'lightgreen',
    
    # Preprocessing nodes (blue)
    'TabularPreprocessing_training': 'lightblue',
    'TabularPreprocessing_calibration': 'lightblue',
    
    # Training node (orange)
    'XGBoostTraining': 'orange',
    
    # Calibration node (yellow)
    'ModelCalibration': 'yellow',
    
    # Evaluation node (pink)
    'XGBoostModelEval_calibration': 'pink',
    
    # Deployment-related nodes (purple)
    'Package': 'lightpurple',
    'Registration': 'purple',
    'Payload': 'lavender'
}

# Define node shapes by category
NODE_SHAPES = {
    # Data loading nodes
    'CradleDataLoading_training': 'cylinder',
    'CradleDataLoading_calibration': 'cylinder',
    
    # Processing nodes
    'TabularPreprocessing_training': 'box',
    'TabularPreprocessing_calibration': 'box',
    
    # Training/modeling nodes
    'XGBoostTraining': 'ellipse',
    'ModelCalibration': 'ellipse',
    'XGBoostModelEval_calibration': 'ellipse',
    
    # Deployment nodes
    'Package': 'diamond',
    'Registration': 'diamond',
    'Payload': 'diamond'
}

def visualize_dag(output_file, format='png'):
    """
    Create a visualization of the DAG using Graphviz.
    
    Args:
        output_file: Path to save the visualization
        format: Output format (png, svg, pdf, etc.)
    """
    try:
        # Create the DAG
        dag = create_xgboost_pipeline_dag()
        
        # Create a new Digraph
        dot = graphviz.Digraph(comment='XGBoost Pipeline DAG')
        dot.attr(rankdir='TB', size='8,11', dpi='300')
        
        # Add title
        dot.attr(label='XGBoost Train-Calibrate-Evaluate Pipeline', fontsize='20', fontcolor='black')
        
        # Add nodes
        for node in dag.nodes:
            color = NODE_COLORS.get(node, 'lightgray')
            shape = NODE_SHAPES.get(node, 'box')
            
            # Clean up node name for display
            node_label = node.replace('_', ' ').title()
            
            dot.node(node, node_label, style='filled', fillcolor=color, shape=shape)
        
        # Add edges
        for src, targets in dag.edges.items():
            for target in targets:
                dot.edge(src, target)
        
        # Add legend
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Legend', style='filled', fillcolor='white', fontsize='14')
            
            # Add legend nodes
            legend.node('data_legend', 'Data Loading', style='filled', fillcolor='lightgreen', shape='cylinder')
            legend.node('preprocess_legend', 'Preprocessing', style='filled', fillcolor='lightblue', shape='box')
            legend.node('train_legend', 'Training/Modeling', style='filled', fillcolor='orange', shape='ellipse')
            legend.node('deploy_legend', 'Deployment', style='filled', fillcolor='purple', shape='diamond')
            
            # Arrange legend nodes
            legend.edge('data_legend', 'preprocess_legend', style='invis')
            legend.edge('preprocess_legend', 'train_legend', style='invis')
            legend.edge('train_legend', 'deploy_legend', style='invis')
            
            # Set legend to rank same to keep nodes on same level
            legend.attr(rank='same')
        
        # Render the graph
        dot.render(output_file, format=format, cleanup=True)
        logger.info(f"DAG visualization saved to {output_file}.{format}")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DAG Structure")
    parser.add_argument("--output", type=str, default="xgboost_pipeline_dag",
                        help="Output file path (without extension)")
    parser.add_argument("--format", type=str, default="png",
                        choices=["png", "svg", "pdf"],
                        help="Output format")
    
    args = parser.parse_args()
    
    visualize_dag(args.output, args.format)
