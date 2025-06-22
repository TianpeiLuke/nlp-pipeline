from typing import Dict, List, Any, Optional, Type
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import Step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline_context import PipelineSession
from pathlib import Path
import logging

from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

class PipelineDAG:
    """
    Represents a pipeline topology as a directed acyclic graph (DAG).
    Each node is a step name; edges define dependencies.
    """
    def __init__(self, nodes: List[str], edges: List[tuple]):
        """
        nodes: List of step names (str)
        edges: List of (from_step, to_step) tuples
        """
        self.nodes = nodes
        self.edges = edges
        self.adj_list = {n: [] for n in nodes}
        for src, dst in edges:
            self.adj_list[src].append(dst)
        self.reverse_adj = {n: [] for n in nodes}
        for src, dst in edges:
            self.reverse_adj[dst].append(src)

    def get_dependencies(self, node: str) -> List[str]:
        """Return immediate dependencies (parents) of a node."""
        return self.reverse_adj.get(node, [])

    def topological_sort(self) -> List[str]:
        """Return nodes in topological order."""
        from collections import deque

        in_degree = {n: 0 for n in self.nodes}
        for src, dst in self.edges:
            in_degree[dst] += 1

        queue = deque([n for n in self.nodes if in_degree[n] == 0])
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in self.adj_list[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.nodes):
            raise ValueError("DAG has cycles or disconnected nodes")
        return order

class PipelineBuilderTemplate:
    """
    Generic pipeline builder using a DAG and step builders.
    """
    def __init__(
        self,
        dag: PipelineDAG,
        config_map: Dict[str, BasePipelineConfig],
        step_builder_map: Dict[str, Type[StepBuilderBase]],
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        pipeline_parameters: Optional[List[ParameterString]] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        dag: PipelineDAG instance
        config_map: Mapping from step name to config instance
        step_builder_map: Mapping from step type to StepBuilderBase subclass
        """
        self.dag = dag
        self.config_map = config_map
        self.step_builder_map = step_builder_map
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.notebook_root = notebook_root or Path.cwd()
        self.pipeline_parameters = pipeline_parameters or []

        self.step_instances: Dict[str, Step] = {}
        self.step_builders: Dict[str, StepBuilderBase] = {}

    def _instantiate_step(self, step_name: str) -> Step:
        config = self.config_map[step_name]
        step_type = BasePipelineConfig.get_step_name(type(config).__name__)
        builder_cls = self.step_builder_map[step_type]
        builder = builder_cls(
            config=config,
            sagemaker_session=self.sagemaker_session,
            role=self.role,
            notebook_root=self.notebook_root,
        )
        self.step_builders[step_name] = builder

        # Gather dependencies
        dependencies = [self.step_instances[parent] for parent in self.dag.get_dependencies(step_name)]
        # Create the step, passing dependencies if supported
        try:
            step = builder.create_step(dependencies=dependencies)
        except TypeError:
            # Fallback for builders that don't accept dependencies
            step = builder.create_step()
            # If possible, add dependencies after creation
            if hasattr(step, "add_depends_on"):
                step.add_depends_on(dependencies)
        return step

    def generate_pipeline(self, pipeline_name: str) -> Pipeline:
        """
        Build and return a SageMaker Pipeline object.
        """
        logger.info(f"Generating pipeline: {pipeline_name}")
        # Topological sort to determine build order
        build_order = self.dag.topological_sort()
        for step_name in build_order:
            step = self._instantiate_step(step_name)
            self.step_instances[step_name] = step

        steps = [self.step_instances[name] for name in build_order]
        return Pipeline(
            name=pipeline_name,
            parameters=self.pipeline_parameters,
            steps=steps,
            sagemaker_session=self.sagemaker_session,
        )