"""
Validation and preview classes for the Pipeline API.

This module provides classes for validating DAG-config compatibility
and previewing resolution results before pipeline generation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
import re

# Import registry components needed for step type resolution
from ..pipeline_registry.step_names import CONFIG_STEP_REGISTRY
from ..pipeline_registry.builder_registry import StepBuilderRegistry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of DAG-config compatibility validation."""
    is_valid: bool
    missing_configs: List[str]
    unresolvable_builders: List[str]
    config_errors: Dict[str, List[str]]
    dependency_issues: List[str]
    warnings: List[str]
    
    def summary(self) -> str:
        """Human-readable validation summary."""
        if self.is_valid:
            summary = "✅ Validation passed"
            if self.warnings:
                summary += f" with {len(self.warnings)} warnings"
        else:
            issues = []
            if self.missing_configs:
                issues.append(f"{len(self.missing_configs)} missing configs")
            if self.unresolvable_builders:
                issues.append(f"{len(self.unresolvable_builders)} unresolvable builders")
            if self.config_errors:
                total_errors = sum(len(errors) for errors in self.config_errors.values())
                issues.append(f"{total_errors} config errors")
            if self.dependency_issues:
                issues.append(f"{len(self.dependency_issues)} dependency issues")
            
            summary = f"❌ Validation failed: {', '.join(issues)}"
        
        return summary
    
    def detailed_report(self) -> str:
        """Detailed validation report with recommendations."""
        lines = [self.summary(), ""]
        
        if self.missing_configs:
            lines.append("Missing Configurations:")
            for config in self.missing_configs:
                lines.append(f"  - {config}")
            lines.append("")
        
        if self.unresolvable_builders:
            lines.append("Unresolvable Step Builders:")
            for builder in self.unresolvable_builders:
                lines.append(f"  - {builder}")
            lines.append("")
        
        if self.config_errors:
            lines.append("Configuration Errors:")
            for config_name, errors in self.config_errors.items():
                lines.append(f"  {config_name}:")
                for error in errors:
                    lines.append(f"    - {error}")
            lines.append("")
        
        if self.dependency_issues:
            lines.append("Dependency Issues:")
            for issue in self.dependency_issues:
                lines.append(f"  - {issue}")
            lines.append("")
        
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        # Add recommendations
        if not self.is_valid:
            lines.append("Recommendations:")
            
            if self.missing_configs:
                lines.append("  - Add missing configuration instances to your config file")
                lines.append("  - Ensure DAG node names match configuration identifiers")
                lines.append("  - Use job_type attributes to distinguish similar configs")
            
            if self.unresolvable_builders:
                lines.append("  - Register missing step builders in StepBuilderRegistry")
                lines.append("  - Use supported configuration types")
                lines.append("  - Check that config class names follow naming conventions")
            
            if self.config_errors:
                lines.append("  - Fix configuration validation errors")
                lines.append("  - Check required fields and value constraints")
            
            if self.dependency_issues:
                lines.append("  - Review DAG structure for dependency conflicts")
                lines.append("  - Ensure all dependencies can be resolved")
        
        return "\n".join(lines)


@dataclass
class ResolutionPreview:
    """Preview of how DAG nodes will be resolved."""
    node_config_map: Dict[str, str]  # node -> config type
    config_builder_map: Dict[str, str]  # config type -> builder type
    resolution_confidence: Dict[str, float]  # confidence scores
    ambiguous_resolutions: List[str]
    recommendations: List[str]
    
    def display(self) -> str:
        """Display-friendly resolution preview."""
        lines = ["Resolution Preview", "=" * 50, ""]
        
        # Node to config mappings
        lines.append("Node → Configuration Mappings:")
        for node, config_type in self.node_config_map.items():
            confidence = self.resolution_confidence.get(node, 1.0)
            confidence_indicator = "🟢" if confidence >= 0.9 else "🟡" if confidence >= 0.7 else "🔴"
            lines.append(f"  {confidence_indicator} {node} → {config_type} (confidence: {confidence:.2f})")
        lines.append("")
        
        # Config to builder mappings
        lines.append("Configuration → Builder Mappings:")
        for config_type, builder_type in self.config_builder_map.items():
            lines.append(f"  ✓ {config_type} → {builder_type}")
        lines.append("")
        
        # Ambiguous resolutions
        if self.ambiguous_resolutions:
            lines.append("⚠️  Ambiguous Resolutions:")
            for ambiguous in self.ambiguous_resolutions:
                lines.append(f"  - {ambiguous}")
            lines.append("")
        
        # Recommendations
        if self.recommendations:
            lines.append("💡 Recommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")
        
        return "\n".join(lines)


@dataclass
class ConversionReport:
    """Report generated after successful pipeline conversion."""
    pipeline_name: str
    steps: List[str]
    resolution_details: Dict[str, Dict[str, Any]]
    avg_confidence: float
    warnings: List[str]
    metadata: Dict[str, Any]
    
    def summary(self) -> str:
        """Summary of conversion results."""
        return (f"Pipeline '{self.pipeline_name}' created successfully with "
                f"{len(self.steps)} steps (avg confidence: {self.avg_confidence:.2f})")
    
    def detailed_report(self) -> str:
        """Detailed conversion report."""
        lines = [
            f"Pipeline Conversion Report",
            "=" * 50,
            f"Pipeline Name: {self.pipeline_name}",
            f"Steps Created: {len(self.steps)}",
            f"Average Confidence: {self.avg_confidence:.2f}",
            ""
        ]
        
        # Step details
        lines.append("Step Resolution Details:")
        for step in self.steps:
            details = self.resolution_details.get(step, {})
            config_type = details.get('config_type', 'Unknown')
            builder_type = details.get('builder_type', 'Unknown')
            confidence = details.get('confidence', 0.0)
            
            lines.append(f"  {step}:")
            lines.append(f"    Config: {config_type}")
            lines.append(f"    Builder: {builder_type}")
            lines.append(f"    Confidence: {confidence:.2f}")
        
        lines.append("")
        
        # Warnings
        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
            lines.append("")
        
        # Metadata
        if self.metadata:
            lines.append("Additional Metadata:")
            for key, value in self.metadata.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class ValidationEngine:
    """Engine for validating DAG-config compatibility."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_dag_compatibility(
        self,
        dag_nodes: List[str],
        available_configs: Dict[str, Any],
        config_map: Dict[str, Any],
        builder_registry: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate DAG-config compatibility.
        
        Args:
            dag_nodes: List of DAG node names
            available_configs: Available configuration instances
            config_map: Resolved node-to-config mapping
            builder_registry: Available step builders
            
        Returns:
            ValidationResult with detailed validation information
        """
        missing_configs = []
        unresolvable_builders = []
        config_errors = {}
        dependency_issues = []
        warnings = []
        
        # Check for missing configurations
        for node in dag_nodes:
            if node not in config_map:
                missing_configs.append(node)
        
        # Check for unresolvable builders
        for node, config in config_map.items():
            config_type = type(config).__name__
            
            # First try to get canonical step name from registry
            if config_type in CONFIG_STEP_REGISTRY:
                step_type = CONFIG_STEP_REGISTRY[config_type]
            else:
                # Fall back to simplified transformation only if not in registry
                self.logger.warning(f"Config class '{config_type}' not found in CONFIG_STEP_REGISTRY, using fallback logic")
                step_type = config_type.replace('Config', '').replace('Step', '')
                
                # Handle special cases for backward compatibility
                if step_type == "CradleDataLoad":
                    step_type = "CradleDataLoading"
                elif step_type == "PackageStep" or step_type == "Package":
                    step_type = "Package"  # Use canonical name
                elif step_type == "Payload":
                    step_type = "Payload"  # Use canonical name
            
            # Check for job type variants
            job_type = getattr(config, 'job_type', None)
            node_job_type = None
            
            # Extract job type from node name if present
            match = re.match(r'^([A-Za-z]+[A-Za-z0-9]*)_([a-z]+)$', node)
            if match:
                _, node_job_type = match.groups()
            
            # Try with job type first if available
            if job_type or node_job_type:
                effective_job_type = job_type or node_job_type
                job_type_step = f"{step_type}_{effective_job_type}"
                
                # Check if the builder registry contains the step type with job type
                if job_type_step in builder_registry:
                    continue
            
            # Check if step type is in builder registry or legacy aliases
            legacy_aliases = StepBuilderRegistry.LEGACY_ALIASES
            
            if step_type in builder_registry:
                continue
            elif step_type in legacy_aliases:
                canonical_step_type = legacy_aliases[step_type]
                if canonical_step_type in builder_registry:
                    continue
            
            # Special handling for known steps with legacy naming
            if step_type == "Package" and "MIMSPackaging" in builder_registry:
                continue
            elif step_type == "Payload" and "MIMSPayload" in builder_registry:
                continue
            elif step_type == "Registration" and "ModelRegistration" in builder_registry:
                continue
            
            # If we get here, builder not found
            unresolvable_builders.append(f"{node} ({step_type})")
        
        # Validate individual configurations
        for node, config in config_map.items():
            try:
                # Call config validation if available
                if hasattr(config, 'validate_config'):
                    config.validate_config()
            except Exception as e:
                if node not in config_errors:
                    config_errors[node] = []
                config_errors[node].append(str(e))
        
        # Check for potential dependency issues
        # This is a placeholder - actual dependency validation would be more complex
        for node in dag_nodes:
            if node in config_map:
                config = config_map[node]
                # Add any dependency-specific validation here
                pass
        
        # Generate warnings for low-confidence resolutions
        # This would be populated by the resolution engine
        
        is_valid = not (missing_configs or unresolvable_builders or config_errors or dependency_issues)
        
        return ValidationResult(
            is_valid=is_valid,
            missing_configs=missing_configs,
            unresolvable_builders=unresolvable_builders,
            config_errors=config_errors,
            dependency_issues=dependency_issues,
            warnings=warnings
        )
