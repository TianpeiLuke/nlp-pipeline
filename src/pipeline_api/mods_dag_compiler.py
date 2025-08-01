"""
MODS DAG Compiler for the Pipeline API.

This module provides an extension of the PipelineDAGCompiler that creates
templates decorated with the MODSTemplate decorator for MODS integration.
"""

from typing import Optional, Dict, Any
import logging
from pathlib import Path

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession

from mods.mods_template import MODSTemplate

from ..pipeline_dag.base_dag import PipelineDAG
from .dag_compiler import PipelineDAGCompiler
from .dynamic_template import DynamicPipelineTemplate
from .config_resolver import StepConfigResolver
from ..pipeline_registry.builder_registry import StepBuilderRegistry
from .exceptions import PipelineAPIError
from ..pipeline_steps.config_base import BasePipelineConfig
from ..pipeline_steps.utils import load_configs, build_complete_config_classes

logger = logging.getLogger(__name__)


class MODSPipelineDAGCompiler(PipelineDAGCompiler):
    """
    Advanced API for DAG-to-template compilation with MODS integration.
    
    This class extends the PipelineDAGCompiler to provide templates
    decorated with the MODSTemplate decorator for MODS integration.
    """
    
    def __init__(
        self,
        config_path: str,
        sagemaker_session: Optional[PipelineSession] = None,
        role: Optional[str] = None,
        config_resolver: Optional[StepConfigResolver] = None,
        builder_registry: Optional[StepBuilderRegistry] = None,
        **kwargs
    ):
        """
        Initialize compiler with configuration and session.
        
        Args:
            config_path: Path to configuration file
            sagemaker_session: SageMaker session for pipeline execution
            role: IAM role for pipeline execution
            config_resolver: Custom config resolver (optional)
            builder_registry: Custom builder registry (optional)
            **kwargs: Additional arguments for template constructor
        """
        super().__init__(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            role=role,
            config_resolver=config_resolver,
            builder_registry=builder_registry,
            **kwargs
        )
        
        # Load base configuration to extract MODS metadata
        self.base_config = self._load_base_config(config_path)
        
    def _load_base_config(self, config_path: str) -> BasePipelineConfig:
        """
        Load the base configuration from the config file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Base configuration
            
        Raises:
            ValueError: If base configuration not found
        """
        # Build complete config classes dictionary
        complete_classes = build_complete_config_classes()
        
        # Load all configs
        configs = load_configs(config_path, complete_classes)
        
        # Extract base config
        base_config = configs.get('Base')
        if not base_config:
            raise ValueError("Base configuration not found in config file")
            
        return base_config
        
    def create_template(self, dag: PipelineDAG, **kwargs) -> Any:
        """
        Create a MODS-decorated pipeline template from the DAG.
        
        This method decorates the DynamicPipelineTemplate class with the
        MODSTemplate decorator before instantiating it.
        
        Args:
            dag: PipelineDAG instance to create a template for
            **kwargs: Additional arguments for template
            
        Returns:
            MODS-decorated DynamicPipelineTemplate instance
            
        Raises:
            PipelineAPIError: If template creation fails
        """
        try:
            self.logger.info(f"Creating MODS-decorated template for DAG with {len(dag.nodes)} nodes")
            
            # Merge kwargs with default values
            template_kwargs = {**self.template_kwargs}
            
            # Set default skip_validation if not provided
            if 'skip_validation' not in kwargs:
                template_kwargs['skip_validation'] = False  # Enable validation by default
            
            # Update with any other kwargs provided
            template_kwargs.update(kwargs)
            
            # Extract MODS metadata from base config
            author = getattr(self.base_config, 'author', 'Unknown')
            version = getattr(self.base_config, 'pipeline_version', '1.0')
            description = getattr(self.base_config, 'pipeline_description', 'MODS Pipeline')
            
            self.logger.info(f"Using MODS metadata: author={author}, version={version}")
            
            # Decorate the DynamicPipelineTemplate class with MODSTemplate
            MODSDecoratedTemplate = MODSTemplate(
                author=author,
                version=version,
                description=description
            )(DynamicPipelineTemplate)
            
            # Create dynamic template from the decorated class
            template = MODSDecoratedTemplate(
                dag=dag,
                config_path=self.config_path,
                config_resolver=self.config_resolver,
                builder_registry=self.builder_registry,
                sagemaker_session=self.sagemaker_session,
                role=self.role,
                **template_kwargs
            )
            
            self.logger.info(f"Successfully created MODS-decorated template")
            return template
            
        except Exception as e:
            self.logger.error(f"Failed to create MODS-decorated template: {e}")
            raise PipelineAPIError(f"MODS template creation failed: {e}") from e


def compile_mods_dag_to_pipeline(
    dag: PipelineDAG,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    pipeline_name: Optional[str] = None,
    **kwargs
) -> Pipeline:
    """
    Compile a PipelineDAG into a complete SageMaker Pipeline with MODS integration.
    
    This is a convenience function that creates a MODSPipelineDAGCompiler and
    compiles the provided DAG into a pipeline.
    
    Args:
        dag: PipelineDAG instance defining the pipeline structure
        config_path: Path to configuration file containing step configs
        sagemaker_session: SageMaker session for pipeline execution
        role: IAM role for pipeline execution
        pipeline_name: Optional pipeline name override
        **kwargs: Additional arguments passed to template constructor
        
    Returns:
        Generated SageMaker Pipeline ready for execution with MODS integration
        
    Raises:
        ValueError: If DAG nodes don't have corresponding configurations
        ConfigurationError: If configuration validation fails
        RegistryError: If step builders not found for config types
        
    Example:
        >>> dag = PipelineDAG()
        >>> dag.add_node("data_load")
        >>> dag.add_node("preprocess")
        >>> dag.add_edge("data_load", "preprocess")
        >>> 
        >>> pipeline = compile_mods_dag_to_pipeline(
        ...     dag=dag,
        ...     config_path="configs/my_pipeline.json",
        ...     sagemaker_session=session,
        ...     role="arn:aws:iam::123456789012:role/SageMakerRole"
        ... )
        >>> pipeline.upsert()
    """
    try:
        logger.info(f"Compiling DAG with {len(dag.nodes)} nodes to MODS pipeline")
        
        # Create MODS compiler
        compiler = MODSPipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=sagemaker_session,
            role=role,
            **kwargs
        )
        
        # Compile pipeline
        pipeline = compiler.compile(dag, pipeline_name=pipeline_name)
        
        logger.info(f"Successfully compiled DAG to MODS pipeline: {pipeline.name}")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to compile DAG to MODS pipeline: {e}")
        raise PipelineAPIError(f"MODS DAG compilation failed: {e}") from e
