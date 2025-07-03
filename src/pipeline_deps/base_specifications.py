"""
Base specifications for declarative dependency management.

This module provides the core classes for defining step dependencies and outputs
in a declarative, type-safe manner.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Types of dependencies in the pipeline."""
    MODEL_ARTIFACTS = "model_artifacts"
    PROCESSING_OUTPUT = "processing_output"
    TRAINING_DATA = "training_data"
    HYPERPARAMETERS = "hyperparameters"
    PAYLOAD_SAMPLES = "payload_samples"
    CUSTOM_PROPERTY = "custom_property"


class NodeType(Enum):
    """Types of nodes in the pipeline based on their dependency/output characteristics."""
    SOURCE = "source"      # No dependencies, has outputs (e.g., data loading)
    INTERNAL = "internal"  # Has both dependencies and outputs (e.g., processing, training)
    SINK = "sink"         # Has dependencies, no outputs (e.g., model registration)
    SINGULAR = "singular" # No dependencies, no outputs (e.g., standalone operations)


@dataclass
class DependencySpec:
    """Declarative specification for a step's dependency requirement."""
    logical_name: str                           # How this dependency is referenced
    dependency_type: DependencyType             # Type of dependency
    required: bool = True                       # Whether this dependency is required
    compatible_sources: List[str] = field(default_factory=list)  # Compatible step types
    semantic_keywords: List[str] = field(default_factory=list)   # Keywords for semantic matching
    data_type: str = "S3Uri"                   # Expected data type
    description: str = ""                       # Human-readable description
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.logical_name:
            raise ValueError("logical_name cannot be empty")
        if not isinstance(self.dependency_type, DependencyType):
            raise ValueError("dependency_type must be a DependencyType enum")


@dataclass
class OutputSpec:
    """Declarative specification for a step's output."""
    logical_name: str                           # How this output is referenced
    output_type: DependencyType                 # Type of output
    property_path: str                          # Runtime SageMaker property path
    data_type: str = "S3Uri"                   # Output data type
    description: str = ""                       # Human-readable description
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        if not self.logical_name:
            raise ValueError("logical_name cannot be empty")
        if not self.property_path:
            raise ValueError("property_path cannot be empty")
        if not isinstance(self.output_type, DependencyType):
            raise ValueError("output_type must be a DependencyType enum")


@dataclass
class PropertyReference:
    """Lazy evaluation reference bridging definition-time and runtime."""
    step_name: str
    output_spec: OutputSpec
    
    def to_sagemaker_property(self):
        """Convert to SageMaker Properties object at runtime."""
        return {"Get": f"Steps.{self.step_name}.{self.output_spec.property_path}"}
    
    def __str__(self):
        return f"{self.step_name}.{self.output_spec.logical_name}"
    
    def __repr__(self):
        return f"PropertyReference(step='{self.step_name}', output='{self.output_spec.logical_name}')"


class StepSpecification:
    """Complete specification for a step's dependencies and outputs."""
    
    def __init__(self, step_type: str, dependencies: List[DependencySpec], 
                 outputs: List[OutputSpec], node_type: NodeType):
        """
        Initialize step specification.
        
        Args:
            step_type: Type identifier for this step
            dependencies: List of dependency specifications
            outputs: List of output specifications
            node_type: Node type classification for validation
        """
        if not step_type:
            raise ValueError("step_type cannot be empty")
        if not isinstance(node_type, NodeType):
            raise ValueError("node_type must be a NodeType enum")
        
        self.step_type = step_type
        self.node_type = node_type
        self.dependencies = {dep.logical_name: dep for dep in dependencies}
        self.outputs = {out.logical_name: out for out in outputs}
        
        # Validate no duplicate names
        if len(self.dependencies) != len(dependencies):
            raise ValueError("Duplicate dependency logical names found")
        if len(self.outputs) != len(outputs):
            raise ValueError("Duplicate output logical names found")
        
        # Validate node type constraints
        self._validate_node_type_constraints()
    
    def _validate_node_type_constraints(self):
        """Validate that dependencies and outputs match the node type."""
        has_deps = len(self.dependencies) > 0
        has_outputs = len(self.outputs) > 0
        
        if self.node_type == NodeType.SOURCE:
            if has_deps:
                raise ValueError(f"SOURCE node '{self.step_type}' cannot have dependencies")
            if not has_outputs:
                raise ValueError(f"SOURCE node '{self.step_type}' must have outputs")
        elif self.node_type == NodeType.INTERNAL:
            if not has_deps:
                raise ValueError(f"INTERNAL node '{self.step_type}' must have dependencies")
            if not has_outputs:
                raise ValueError(f"INTERNAL node '{self.step_type}' must have outputs")
        elif self.node_type == NodeType.SINK:
            if not has_deps:
                raise ValueError(f"SINK node '{self.step_type}' must have dependencies")
            if has_outputs:
                raise ValueError(f"SINK node '{self.step_type}' cannot have outputs")
        elif self.node_type == NodeType.SINGULAR:
            if has_deps:
                raise ValueError(f"SINGULAR node '{self.step_type}' cannot have dependencies")
            if has_outputs:
                raise ValueError(f"SINGULAR node '{self.step_type}' cannot have outputs")
    
    def get_dependency(self, logical_name: str) -> Optional[DependencySpec]:
        """Get dependency specification by logical name."""
        return self.dependencies.get(logical_name)
    
    def get_output(self, logical_name: str) -> Optional[OutputSpec]:
        """Get output specification by logical name."""
        return self.outputs.get(logical_name)
    
    def list_required_dependencies(self) -> List[DependencySpec]:
        """Get list of required dependencies."""
        return [dep for dep in self.dependencies.values() if dep.required]
    
    def list_optional_dependencies(self) -> List[DependencySpec]:
        """Get list of optional dependencies."""
        return [dep for dep in self.dependencies.values() if not dep.required]
    
    def list_dependencies_by_type(self, dependency_type: DependencyType) -> List[DependencySpec]:
        """Get list of dependencies of a specific type."""
        return [dep for dep in self.dependencies.values() if dep.dependency_type == dependency_type]
    
    def list_outputs_by_type(self, output_type: DependencyType) -> List[OutputSpec]:
        """Get list of outputs of a specific type."""
        return [out for out in self.outputs.values() if out.output_type == output_type]
    
    def validate(self) -> List[str]:
        """Validate the specification for consistency."""
        errors = []
        
        # Check for empty collections (except for SINGULAR nodes which should have neither)
        if not self.dependencies and not self.outputs and self.node_type != NodeType.SINGULAR:
            errors.append(f"Step '{self.step_type}' has no dependencies or outputs")
        
        # Validate dependency specifications
        for dep_name, dep_spec in self.dependencies.items():
            try:
                # This will trigger __post_init__ validation
                DependencySpec(
                    logical_name=dep_spec.logical_name,
                    dependency_type=dep_spec.dependency_type,
                    required=dep_spec.required,
                    compatible_sources=dep_spec.compatible_sources,
                    semantic_keywords=dep_spec.semantic_keywords,
                    data_type=dep_spec.data_type,
                    description=dep_spec.description
                )
            except ValueError as e:
                errors.append(f"Invalid dependency '{dep_name}': {e}")
        
        # Validate output specifications
        for out_name, out_spec in self.outputs.items():
            try:
                # This will trigger __post_init__ validation
                OutputSpec(
                    logical_name=out_spec.logical_name,
                    output_type=out_spec.output_type,
                    property_path=out_spec.property_path,
                    data_type=out_spec.data_type,
                    description=out_spec.description
                )
            except ValueError as e:
                errors.append(f"Invalid output '{out_name}': {e}")
        
        return errors
    
    def __repr__(self):
        return (f"StepSpecification(type='{self.step_type}', "
                f"dependencies={len(self.dependencies)}, "
                f"outputs={len(self.outputs)})")


class SpecificationRegistry:
    """Registry for managing step specifications."""
    
    def __init__(self):
        self._specifications: Dict[str, StepSpecification] = {}
        self._step_type_to_names: Dict[str, List[str]] = {}
    
    def register(self, step_name: str, specification: StepSpecification):
        """Register a step specification."""
        if not isinstance(specification, StepSpecification):
            raise ValueError("specification must be a StepSpecification instance")
        
        # Validate the specification
        errors = specification.validate()
        if errors:
            raise ValueError(f"Invalid specification for '{step_name}': {errors}")
        
        self._specifications[step_name] = specification
        
        # Track step type mappings
        step_type = specification.step_type
        if step_type not in self._step_type_to_names:
            self._step_type_to_names[step_type] = []
        self._step_type_to_names[step_type].append(step_name)
        
        logger.info(f"Registered specification for step '{step_name}' of type '{step_type}'")
    
    def get_specification(self, step_name: str) -> Optional[StepSpecification]:
        """Get specification by step name."""
        return self._specifications.get(step_name)
    
    def get_specifications_by_type(self, step_type: str) -> List[StepSpecification]:
        """Get all specifications of a given step type."""
        step_names = self._step_type_to_names.get(step_type, [])
        return [self._specifications[name] for name in step_names]
    
    def list_step_names(self) -> List[str]:
        """Get list of all registered step names."""
        return list(self._specifications.keys())
    
    def list_step_types(self) -> List[str]:
        """Get list of all registered step types."""
        return list(self._step_type_to_names.keys())
    
    def find_compatible_outputs(self, dependency_spec: DependencySpec) -> List[tuple]:
        """Find outputs compatible with a dependency specification."""
        compatible = []
        
        for step_name, spec in self._specifications.items():
            for output_name, output_spec in spec.outputs.items():
                if self._are_compatible(dependency_spec, output_spec):
                    score = self._calculate_compatibility_score(dependency_spec, output_spec, spec.step_type)
                    compatible.append((step_name, output_name, output_spec, score))
        
        return sorted(compatible, key=lambda x: x[3], reverse=True)
    
    def _are_compatible(self, dep_spec: DependencySpec, out_spec: OutputSpec) -> bool:
        """Check basic compatibility between dependency and output."""
        # Type compatibility
        if dep_spec.dependency_type != out_spec.output_type:
            return False
        
        # Data type compatibility
        if dep_spec.data_type != out_spec.data_type:
            return False
        
        return True
    
    def _calculate_compatibility_score(self, dep_spec: DependencySpec, 
                                     out_spec: OutputSpec, step_type: str) -> float:
        """Calculate compatibility score between dependency and output."""
        score = 0.5  # Base compatibility score
        
        # Compatible source bonus
        if dep_spec.compatible_sources and step_type in dep_spec.compatible_sources:
            score += 0.3
        
        # Semantic keyword matching
        if dep_spec.semantic_keywords:
            keyword_matches = sum(
                1 for keyword in dep_spec.semantic_keywords
                if keyword.lower() in out_spec.logical_name.lower()
            )
            score += (keyword_matches / len(dep_spec.semantic_keywords)) * 0.2
        
        return min(score, 1.0)  # Cap at 1.0


# Global registry instance
global_registry = SpecificationRegistry()
