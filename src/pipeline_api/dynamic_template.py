"""
Dynamic Pipeline Template for the Pipeline API.

This module provides a dynamic implementation of PipelineTemplateBase that can work
with any PipelineDAG structure without requiring custom template classes.
"""

from typing import Dict, Type, Any, Optional
import logging

from ..pipeline_dag.base_dag import PipelineDAG
from ..pipeline_builder.pipeline_template_base import PipelineTemplateBase
from ..pipeline_steps.builder_step_base import StepBuilderBase
from ..pipeline_steps.config_base import BasePipelineConfig
from ..pipeline_steps.utils import build_complete_config_classes

from .config_resolver import StepConfigResolver
from .builder_registry import StepBuilderRegistry
from .validation import ValidationEngine
from .exceptions import ConfigurationError, RegistryError, ValidationError

logger = logging.getLogger(__name__)


class DynamicPipelineTemplate(PipelineTemplateBase):
    """
    Dynamic pipeline template that works with any PipelineDAG.
    
    This template automatically implements the abstract methods of
    PipelineTemplateBase by using intelligent resolution mechanisms
    to map DAG nodes to configurations and step builders.
    """
    
    def __init__(
        self,
        dag: PipelineDAG,
        config_path: str,
        config_resolver: Optional[StepConfigResolver] = None,
        builder_registry: Optional[StepBuilderRegistry] = None,
        **kwargs
    ):
        """
        Initialize dynamic template.
        
        Args:
            dag: PipelineDAG instance defining pipeline structure
            config_path: Path to configuration file
            config_resolver: Custom config resolver (optional)
            builder_registry: Custom builder registry (optional)
            **kwargs: Additional arguments for base template
        """
        self._dag = dag
        self._config_resolver = config_resolver or StepConfigResolver()
        self._builder_registry = builder_registry or StepBuilderRegistry()
        self._validation_engine = ValidationEngine()
        
        # Auto-detect required config classes based on DAG nodes
        self.CONFIG_CLASSES = self._detect_config_classes()
        
        # Store resolved mappings for later use
        self._resolved_config_map = None
        self._resolved_builder_map = None
        
        self.logger = logging.getLogger(__name__)
        
        super().__init__(config_path, **kwargs)
    
    def _detect_config_classes(self) -> Dict[str, Type[BasePipelineConfig]]:
        """
        Automatically detect required config classes from DAG nodes.
        
        This method analyzes the DAG structure and determines which
        configuration classes are needed based on:
        1. Node naming patterns
        2. Available configurations in the config file
        3. Step builder registry mappings
        
        Returns:
            Dictionary mapping config class names to config classes
        """
        # Get all available config classes
        all_config_classes = build_complete_config_classes()
        
        # For dynamic templates, we include all available config classes
        # since we don't know in advance which ones will be needed
        self.logger.debug(f"Detected {len(all_config_classes)} available config classes")
        
        return all_config_classes
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        """
        Return the provided DAG.
        
        Returns:
            The PipelineDAG instance provided during initialization
        """
        return self._dag
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        """
        Auto-map DAG nodes to configurations.
        
        Uses StepConfigResolver to intelligently match DAG node names
        to configuration instances from the loaded config file.
        
        Returns:
            Dictionary mapping DAG node names to configuration instances
            
        Raises:
            ConfigurationError: If nodes cannot be resolved to configurations
        """
        if self._resolved_config_map is not None:
            return self._resolved_config_map
        
        try:
            dag_nodes = list(self._dag.nodes)
            self.logger.info(f"Resolving {len(dag_nodes)} DAG nodes to configurations")
            
            # Use the config resolver to map nodes to configs
            self._resolved_config_map = self._config_resolver.resolve_config_map(
                dag_nodes=dag_nodes,
                available_configs=self.configs
            )
            
            self.logger.info(f"Successfully resolved all {len(self._resolved_config_map)} nodes")
            
            # Log resolution details
            for node, config in self._resolved_config_map.items():
                config_type = type(config).__name__
                job_type = getattr(config, 'job_type', 'N/A')
                self.logger.debug(f"  {node} → {config_type} (job_type: {job_type})")
            
            return self._resolved_config_map
            
        except Exception as e:
            self.logger.error(f"Failed to resolve DAG nodes to configurations: {e}")
            raise ConfigurationError(f"Configuration resolution failed: {e}")
    
    def _create_step_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Auto-map step types to builders using registry.
        
        Uses StepBuilderRegistry to map configuration types to their
        corresponding step builder classes.
        
        Returns:
            Dictionary mapping step types to step builder classes
            
        Raises:
            RegistryError: If step builders cannot be found for config types
        """
        if self._resolved_builder_map is not None:
            return self._resolved_builder_map
        
        try:
            # Get the complete builder registry
            self._resolved_builder_map = self._builder_registry.get_builder_map()
            
            self.logger.info(f"Using {len(self._resolved_builder_map)} registered step builders")
            
            # Validate that all required builders are available
            config_map = self._create_config_map()
            missing_builders = []
            
            for node, config in config_map.items():
                try:
                    builder_class = self._builder_registry.get_builder_for_config(config)
                    step_type = self._builder_registry._config_class_to_step_type(type(config).__name__)
                    self.logger.debug(f"  {step_type} → {builder_class.__name__}")
                except RegistryError as e:
                    missing_builders.append(f"{node} ({type(config).__name__})")
            
            if missing_builders:
                available_builders = list(self._resolved_builder_map.keys())
                raise RegistryError(
                    f"Missing step builders for {len(missing_builders)} configurations",
                    unresolvable_types=missing_builders,
                    available_builders=available_builders
                )
            
            return self._resolved_builder_map
            
        except Exception as e:
            self.logger.error(f"Failed to create step builder map: {e}")
            raise RegistryError(f"Step builder mapping failed: {e}")
    
    def _validate_configuration(self) -> None:
        """
        Validate that all DAG nodes have corresponding configs.
        
        Performs comprehensive validation including:
        1. All DAG nodes have matching configurations
        2. All configurations have corresponding step builders
        3. Configuration-specific validation passes
        4. Dependency resolution is possible
        
        Raises:
            ValidationError: If validation fails
        """
        try:
            self.logger.info("Validating dynamic pipeline configuration")
            
            # Get resolved mappings
            dag_nodes = list(self._dag.nodes)
            config_map = self._create_config_map()
            builder_map = self._create_step_builder_map()
            
            # Run comprehensive validation
            validation_result = self._validation_engine.validate_dag_compatibility(
                dag_nodes=dag_nodes,
                available_configs=self.configs,
                config_map=config_map,
                builder_registry=builder_map
            )
            
            if not validation_result.is_valid:
                self.logger.error("Configuration validation failed")
                self.logger.error(validation_result.detailed_report())
                raise ValidationError(
                    "Dynamic pipeline configuration validation failed",
                    validation_errors={
                        'missing_configs': validation_result.missing_configs,
                        'unresolvable_builders': validation_result.unresolvable_builders,
                        'config_errors': validation_result.config_errors,
                        'dependency_issues': validation_result.dependency_issues
                    }
                )
            
            # Log warnings if any
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    self.logger.warning(warning)
            
            self.logger.info("Configuration validation passed successfully")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValidationError(f"Validation failed: {e}")
    
    def get_resolution_preview(self) -> Dict[str, Any]:
        """
        Get a preview of how DAG nodes will be resolved.
        
        Returns:
            Dictionary with resolution preview information
        """
        try:
            dag_nodes = list(self._dag.nodes)
            preview_data = self._config_resolver.preview_resolution(
                dag_nodes=dag_nodes,
                available_configs=self.configs
            )
            
            # Convert to display format
            preview = {
                'nodes': len(dag_nodes),
                'resolutions': {}
            }
            
            for node, candidates in preview_data.items():
                if candidates:
                    best_candidate = candidates[0]
                    preview['resolutions'][node] = {
                        'config_type': best_candidate['config_type'],
                        'confidence': best_candidate['confidence'],
                        'method': best_candidate['method'],
                        'job_type': best_candidate['job_type'],
                        'alternatives': len(candidates) - 1
                    }
                else:
                    preview['resolutions'][node] = {
                        'config_type': 'UNRESOLVED',
                        'confidence': 0.0,
                        'method': 'none',
                        'job_type': 'N/A',
                        'alternatives': 0
                    }
            
            return preview
            
        except Exception as e:
            self.logger.error(f"Failed to generate resolution preview: {e}")
            return {'error': str(e)}
    
    def get_builder_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the builder registry.
        
        Returns:
            Dictionary with registry statistics
        """
        return self._builder_registry.get_registry_stats()
    
    def validate_before_build(self) -> bool:
        """
        Validate the configuration before building the pipeline.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            self._validate_configuration()
            return True
        except ValidationError:
            return False
    
    def get_step_dependencies(self) -> Dict[str, list]:
        """
        Get the dependencies for each step based on the DAG.
        
        Returns:
            Dictionary mapping step names to their dependencies
        """
        dependencies = {}
        for node in self._dag.nodes:
            dependencies[node] = list(self._dag.get_dependencies(node))
        return dependencies
    
    def get_execution_order(self) -> list:
        """
        Get the topological execution order of steps.
        
        Returns:
            List of step names in execution order
        """
        try:
            return self._dag.topological_sort()
        except Exception as e:
            self.logger.error(f"Failed to get execution order: {e}")
            return list(self._dag.nodes)
