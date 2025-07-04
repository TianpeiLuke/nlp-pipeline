"""
Base specifications for declarative dependency management.

This module provides the core classes for defining step dependencies and outputs
in a declarative, type-safe manner using Pydantic V2 BaseModel.
"""

from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
from abc import ABC, abstractmethod
import logging

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

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


class DependencySpec(BaseModel):
    """Declarative specification for a step's dependency requirement."""
    
    model_config = ConfigDict(
        # Enable enum validation by value
        use_enum_values=True,
        # Validate assignment after object creation
        validate_assignment=True,
        # Allow arbitrary types for complex objects
        arbitrary_types_allowed=True,
        # Generate JSON schema
        json_schema_extra={
            "examples": [
                {
                    "logical_name": "training_data",
                    "dependency_type": "processing_output",
                    "required": True,
                    "compatible_sources": ["DataLoadingStep", "PreprocessingStep"],
                    "semantic_keywords": ["data", "dataset", "input"],
                    "data_type": "S3Uri",
                    "description": "Training dataset for model training"
                }
            ]
        }
    )
    
    logical_name: str = Field(
        ...,
        description="How this dependency is referenced",
        min_length=1,
        examples=["training_data", "model_input", "config_data"]
    )
    dependency_type: DependencyType = Field(
        ...,
        description="Type of dependency"
    )
    required: bool = Field(
        default=True,
        description="Whether this dependency is required"
    )
    compatible_sources: List[str] = Field(
        default_factory=list,
        description="Compatible step types that can provide this dependency"
    )
    semantic_keywords: List[str] = Field(
        default_factory=list,
        description="Keywords for semantic matching during dependency resolution"
    )
    data_type: str = Field(
        default="S3Uri",
        description="Expected data type of the dependency"
    )
    description: str = Field(
        default="",
        description="Human-readable description of the dependency"
    )
    
    @field_validator('logical_name')
    @classmethod
    def validate_logical_name(cls, v: str) -> str:
        """Validate logical name is not empty and follows naming conventions."""
        if not v or not v.strip():
            raise ValueError("logical_name cannot be empty or whitespace")
        
        # Optional: Add naming convention validation
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("logical_name should contain only alphanumeric characters, underscores, and hyphens")
        
        return v.strip()
    
    @field_validator('dependency_type')
    @classmethod
    def validate_dependency_type(cls, v) -> DependencyType:
        """Validate dependency type is a valid enum value."""
        if isinstance(v, str):
            try:
                return DependencyType(v)
            except ValueError:
                valid_values = [e.value for e in DependencyType]
                raise ValueError(f"dependency_type must be one of: {valid_values}")
        elif isinstance(v, DependencyType):
            return v
        else:
            raise ValueError("dependency_type must be a DependencyType enum or valid string value")
    
    @field_validator('compatible_sources')
    @classmethod
    def validate_compatible_sources(cls, v: List[str]) -> List[str]:
        """Validate compatible sources list."""
        if not isinstance(v, list):
            raise ValueError("compatible_sources must be a list")
        
        # Remove empty strings and duplicates
        cleaned = list(set(source.strip() for source in v if source and source.strip()))
        return cleaned
    
    @field_validator('semantic_keywords')
    @classmethod
    def validate_semantic_keywords(cls, v: List[str]) -> List[str]:
        """Validate semantic keywords list."""
        if not isinstance(v, list):
            raise ValueError("semantic_keywords must be a list")
        
        # Remove empty strings, convert to lowercase, and remove duplicates
        cleaned = list(set(keyword.strip().lower() for keyword in v if keyword and keyword.strip()))
        return cleaned


class OutputSpec(BaseModel):
    """Declarative specification for a step's output."""
    
    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "logical_name": "processed_data",
                    "output_type": "processing_output",
                    "property_path": "properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
                    "data_type": "S3Uri",
                    "description": "Processed training data output"
                }
            ]
        }
    )
    
    logical_name: str = Field(
        ...,
        description="How this output is referenced",
        min_length=1,
        examples=["processed_data", "model_artifacts", "evaluation_results"]
    )
    output_type: DependencyType = Field(
        ...,
        description="Type of output"
    )
    property_path: str = Field(
        ...,
        description="Runtime SageMaker property path to access this output",
        min_length=1
    )
    data_type: str = Field(
        default="S3Uri",
        description="Output data type"
    )
    description: str = Field(
        default="",
        description="Human-readable description of the output"
    )
    
    @field_validator('logical_name')
    @classmethod
    def validate_logical_name(cls, v: str) -> str:
        """Validate logical name is not empty and follows naming conventions."""
        if not v or not v.strip():
            raise ValueError("logical_name cannot be empty or whitespace")
        
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("logical_name should contain only alphanumeric characters, underscores, and hyphens")
        
        return v.strip()
    
    @field_validator('output_type')
    @classmethod
    def validate_output_type(cls, v) -> DependencyType:
        """Validate output type is a valid enum value."""
        if isinstance(v, str):
            try:
                return DependencyType(v)
            except ValueError:
                valid_values = [e.value for e in DependencyType]
                raise ValueError(f"output_type must be one of: {valid_values}")
        elif isinstance(v, DependencyType):
            return v
        else:
            raise ValueError("output_type must be a DependencyType enum or valid string value")
    
    @field_validator('property_path')
    @classmethod
    def validate_property_path(cls, v: str) -> str:
        """Validate property path is not empty and has basic structure."""
        if not v or not v.strip():
            raise ValueError("property_path cannot be empty or whitespace")
        
        # Basic validation for SageMaker property path structure
        v = v.strip()
        if not v.startswith('properties.'):
            raise ValueError("property_path should start with 'properties.'")
        
        return v


class PropertyReference(BaseModel):
    """Lazy evaluation reference bridging definition-time and runtime."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True
    )
    
    step_name: str = Field(
        ...,
        description="Name of the step that produces this output",
        min_length=1
    )
    output_spec: OutputSpec = Field(
        ...,
        description="Output specification for the referenced output"
    )
    
    @field_validator('step_name')
    @classmethod
    def validate_step_name(cls, v: str) -> str:
        """Validate step name is not empty."""
        if not v or not v.strip():
            raise ValueError("step_name cannot be empty or whitespace")
        return v.strip()
    
    def to_sagemaker_property(self) -> Dict[str, str]:
        """Convert to SageMaker Properties object at runtime."""
        return {"Get": f"Steps.{self.step_name}.{self.output_spec.property_path}"}
    
    def __str__(self) -> str:
        return f"{self.step_name}.{self.output_spec.logical_name}"
    
    def __repr__(self) -> str:
        return f"PropertyReference(step='{self.step_name}', output='{self.output_spec.logical_name}')"


class StepSpecification(BaseModel):
    """Complete specification for a step's dependencies and outputs."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "step_type": "DataProcessingStep",
                    "node_type": "internal",
                    "dependencies": {
                        "input_data": {
                            "logical_name": "input_data",
                            "dependency_type": "processing_output",
                            "required": True,
                            "compatible_sources": ["DataLoadingStep"],
                            "data_type": "S3Uri"
                        }
                    },
                    "outputs": {
                        "processed_data": {
                            "logical_name": "processed_data",
                            "output_type": "processing_output",
                            "property_path": "properties.ProcessingOutputConfig.Outputs['ProcessedData'].S3Output.S3Uri",
                            "data_type": "S3Uri"
                        }
                    }
                }
            ]
        }
    )
    
    step_type: str = Field(
        ...,
        description="Type identifier for this step",
        min_length=1
    )
    node_type: NodeType = Field(
        ...,
        description="Node type classification for validation"
    )
    dependencies: Dict[str, DependencySpec] = Field(
        default_factory=dict,
        description="Dictionary of dependency specifications keyed by logical name"
    )
    outputs: Dict[str, OutputSpec] = Field(
        default_factory=dict,
        description="Dictionary of output specifications keyed by logical name"
    )
    
    def __init__(self, step_type: str = None, dependencies: List[DependencySpec] = None, 
                 outputs: List[OutputSpec] = None, node_type: NodeType = None, **data):
        """
        Initialize step specification with backward compatibility.
        
        Args:
            step_type: Type identifier for this step
            dependencies: List of dependency specifications
            outputs: List of output specifications
            node_type: Node type classification for validation
        """
        # Handle direct model_validate calls
        if step_type is None and 'step_type' in data:
            step_type = data.pop('step_type')
        if node_type is None and 'node_type' in data:
            node_type = data.pop('node_type')
        if dependencies is None and 'dependencies' in data:
            deps_data = data.pop('dependencies')
            if isinstance(deps_data, dict):
                # Already in dict format from model_validate
                dependencies = list(deps_data.values())
            else:
                dependencies = deps_data
        if outputs is None and 'outputs' in data:
            outputs_data = data.pop('outputs')
            if isinstance(outputs_data, dict):
                # Already in dict format from model_validate
                outputs = list(outputs_data.values())
            else:
                outputs = outputs_data
        
        # Set defaults
        if dependencies is None:
            dependencies = []
        if outputs is None:
            outputs = []
        
        # Convert lists to dictionaries for internal storage
        if isinstance(dependencies, list):
            deps_dict = {dep.logical_name: dep for dep in dependencies}
        else:
            deps_dict = dependencies
            
        if isinstance(outputs, list):
            outputs_dict = {out.logical_name: out for out in outputs}
        else:
            outputs_dict = outputs
        
        # Check for duplicates
        if isinstance(dependencies, list) and len(deps_dict) != len(dependencies):
            raise ValueError("Duplicate dependency logical names found")
        if isinstance(outputs, list) and len(outputs_dict) != len(outputs):
            raise ValueError("Duplicate output logical names found")
        
        super().__init__(
            step_type=step_type,
            node_type=node_type,
            dependencies=deps_dict,
            outputs=outputs_dict,
            **data
        )
    
    @field_validator('step_type')
    @classmethod
    def validate_step_type(cls, v: str) -> str:
        """Validate step type is not empty."""
        if not v or not v.strip():
            raise ValueError("step_type cannot be empty or whitespace")
        return v.strip()
    
    @field_validator('node_type')
    @classmethod
    def validate_node_type(cls, v) -> NodeType:
        """Validate node type is a valid enum value."""
        if isinstance(v, str):
            try:
                return NodeType(v)
            except ValueError:
                valid_values = [e.value for e in NodeType]
                raise ValueError(f"node_type must be one of: {valid_values}")
        elif isinstance(v, NodeType):
            return v
        else:
            raise ValueError("node_type must be a NodeType enum or valid string value")
    
    @model_validator(mode='after')
    def validate_node_type_constraints(self) -> 'StepSpecification':
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
        
        return self
    
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
        """Validate the specification for consistency (legacy method for backward compatibility)."""
        # This method is kept for backward compatibility
        # Pydantic V2 handles most validation automatically
        errors = []
        
        # Additional custom validation can be added here
        if not self.dependencies and not self.outputs and self.node_type != NodeType.SINGULAR:
            errors.append(f"Step '{self.step_type}' has no dependencies or outputs")
        
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
        
        # Validate the specification (Pydantic handles most validation automatically)
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
