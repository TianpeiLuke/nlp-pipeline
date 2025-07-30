"""
Step Builder Registry for the Pipeline API.

This module provides a centralized registry mapping configuration types
to step builder classes, enabling automatic resolution during pipeline construction.
"""

from typing import Dict, Type, List, Optional
import logging

from ..pipeline_steps.builder_step_base import StepBuilderBase
from ..pipeline_steps.config_base import BasePipelineConfig

# Import all step builders
from ..pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder
from ..pipeline_steps.builder_tabular_preprocessing_step import TabularPreprocessingStepBuilder
from ..pipeline_steps.builder_training_step_xgboost import XGBoostTrainingStepBuilder
from ..pipeline_steps.builder_model_eval_step_xgboost import XGBoostModelEvalStepBuilder
from ..pipeline_steps.builder_mims_packaging_step import MIMSPackagingStepBuilder
from ..pipeline_steps.builder_mims_payload_step import MIMSPayloadStepBuilder
from ..pipeline_steps.builder_mims_registration_step import ModelRegistrationStepBuilder
from ..pipeline_steps.builder_training_step_pytorch import PyTorchTrainingStepBuilder
from ..pipeline_steps.builder_model_step_pytorch import PyTorchModelStepBuilder
from ..pipeline_steps.builder_model_step_xgboost import XGBoostModelStepBuilder
from ..pipeline_steps.builder_batch_transform_step import BatchTransformStepBuilder
from ..pipeline_steps.builder_model_calibration_step import ModelCalibrationStepBuilder
from ..pipeline_steps.builder_currency_conversion_step import CurrencyConversionStepBuilder
from ..pipeline_steps.builder_risk_table_mapping_step import RiskTableMappingStepBuilder
from ..pipeline_steps.builder_dummy_training_step import DummyTrainingStepBuilder
from ..pipeline_steps.builder_hyperparameter_prep_step import HyperparameterPrepStepBuilder

from .exceptions import RegistryError

logger = logging.getLogger(__name__)


class StepBuilderRegistry:
    """
    Centralized registry mapping configuration types to step builders.
    
    This registry maintains the mapping between configuration classes
    and their corresponding step builder classes, enabling automatic
    resolution during pipeline construction.
    """
    
    # Core registry mapping step types to builders
    BUILDER_REGISTRY = {
        "CradleDataLoading": CradleDataLoadingStepBuilder,
        "TabularPreprocessing": TabularPreprocessingStepBuilder,
        "XGBoostTraining": XGBoostTrainingStepBuilder,
        "XGBoostModelEval": XGBoostModelEvalStepBuilder,
        "MIMSPackaging": MIMSPackagingStepBuilder,
        "MIMSPayload": MIMSPayloadStepBuilder,
        "ModelRegistration": ModelRegistrationStepBuilder,
        "PyTorchTraining": PyTorchTrainingStepBuilder,
        "PyTorchModel": PyTorchModelStepBuilder,
        "XGBoostModel": XGBoostModelStepBuilder,
        "BatchTransform": BatchTransformStepBuilder,
        "ModelCalibration": ModelCalibrationStepBuilder,
        "CurrencyConversion": CurrencyConversionStepBuilder,
        "RiskTableMapping": RiskTableMappingStepBuilder,
        "DummyTraining": DummyTrainingStepBuilder,
        "HyperparameterPrep": HyperparameterPrepStepBuilder,
    }
    
    def __init__(self):
        """Initialize the registry."""
        self._custom_builders = {}
        self.logger = logging.getLogger(__name__)
    
    def get_builder_map(self) -> Dict[str, Type[StepBuilderBase]]:
        """
        Get the complete builder registry.
        
        Returns:
            Dictionary mapping step types to builder classes
        """
        # Combine default and custom builders
        builder_map = self.BUILDER_REGISTRY.copy()
        builder_map.update(self._custom_builders)
        return builder_map
    
    def get_builder_for_config(self, config: BasePipelineConfig) -> Type[StepBuilderBase]:
        """
        Get step builder class for a specific configuration.
        
        Args:
            config: Configuration instance
            
        Returns:
            Step builder class
            
        Raises:
            RegistryError: If no builder found for config type
        """
        config_class_name = type(config).__name__
        
        # Convert config class name to step type
        step_type = self._config_class_to_step_type(config_class_name)
        
        builder_map = self.get_builder_map()
        if step_type not in builder_map:
            available_types = list(builder_map.keys())
            raise RegistryError(
                f"No step builder found for config type '{config_class_name}' (step type: '{step_type}')",
                unresolvable_types=[step_type],
                available_builders=available_types
            )
        
        return builder_map[step_type]
    
    def get_builder_for_step_type(self, step_type: str) -> Type[StepBuilderBase]:
        """
        Get step builder class for a specific step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            Step builder class
            
        Raises:
            RegistryError: If no builder found for step type
        """
        builder_map = self.get_builder_map()
        if step_type not in builder_map:
            available_types = list(builder_map.keys())
            raise RegistryError(
                f"No step builder found for step type '{step_type}'",
                unresolvable_types=[step_type],
                available_builders=available_types
            )
        
        return builder_map[step_type]
    
    def register_builder(self, step_type: str, builder_class: Type[StepBuilderBase]) -> None:
        """
        Register a new step builder (for extensibility).
        
        Args:
            step_type: Step type name
            builder_class: Step builder class
        """
        if not issubclass(builder_class, StepBuilderBase):
            raise ValueError(f"Builder class must extend StepBuilderBase: {builder_class}")
        
        self._custom_builders[step_type] = builder_class
        self.logger.info(f"Registered custom builder: {step_type} -> {builder_class.__name__}")
    
    def unregister_builder(self, step_type: str) -> None:
        """
        Unregister a custom step builder.
        
        Args:
            step_type: Step type name
        """
        if step_type in self._custom_builders:
            del self._custom_builders[step_type]
            self.logger.info(f"Unregistered custom builder: {step_type}")
        else:
            self.logger.warning(f"Attempted to unregister non-existent custom builder: {step_type}")
    
    def list_supported_step_types(self) -> List[str]:
        """
        List all supported step types.
        
        Returns:
            List of supported step type names
        """
        return list(self.get_builder_map().keys())
    
    def is_step_type_supported(self, step_type: str) -> bool:
        """
        Check if a step type is supported.
        
        Args:
            step_type: Step type name
            
        Returns:
            True if supported, False otherwise
        """
        return step_type in self.get_builder_map()
    
    def get_config_types_for_step_type(self, step_type: str) -> List[str]:
        """
        Get possible configuration class names for a step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            List of possible configuration class names
        """
        # This is a reverse mapping - given a step type, what config classes could produce it
        possible_configs = []
        
        # Standard patterns
        possible_configs.append(f"{step_type}Config")
        possible_configs.append(f"{step_type}StepConfig")
        
        # Special cases
        if step_type == "CradleDataLoading":
            possible_configs.append("CradleDataLoadConfig")
        elif step_type == "TabularPreprocessing":
            possible_configs.append("TabularPreprocessingConfig")
        elif step_type == "XGBoostTraining":
            possible_configs.append("XGBoostTrainingConfig")
        elif step_type == "XGBoostModelEval":
            possible_configs.append("XGBoostModelEvalConfig")
        elif step_type == "ModelRegistration":
            possible_configs.append("ModelRegistrationConfig")
        
        return possible_configs
    
    def _config_class_to_step_type(self, config_class_name: str) -> str:
        """
        Convert configuration class name to step type.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            Step type name
        """
        # Use the same logic as BasePipelineConfig.get_step_name()
        # Remove common suffixes
        step_type = config_class_name
        
        # Remove 'Config' suffix
        if step_type.endswith('Config'):
            step_type = step_type[:-6]
        
        # Remove 'Step' suffix if present
        if step_type.endswith('Step'):
            step_type = step_type[:-4]
        
        # Handle special cases
        if step_type == "CradleDataLoad":
            return "CradleDataLoading"
        elif step_type == "TabularPreprocessing":
            return "TabularPreprocessing"
        elif step_type == "XGBoostTraining":
            return "XGBoostTraining"
        elif step_type == "XGBoostModelEval":
            return "XGBoostModelEval"
        elif step_type == "ModelRegistration":
            return "ModelRegistration"
        elif step_type == "PackageStep" or step_type == "Package":
            return "MIMSPackaging"
        elif step_type == "Payload":
            return "MIMSPayload"
        
        return step_type
    
    def validate_registry(self) -> Dict[str, List[str]]:
        """
        Validate the registry for consistency.
        
        Returns:
            Dictionary with validation results:
            - 'valid': List of valid mappings
            - 'invalid': List of invalid mappings with reasons
        """
        results = {'valid': [], 'invalid': []}
        
        builder_map = self.get_builder_map()
        for step_type, builder_class in builder_map.items():
            try:
                # Check if builder class is valid
                if not issubclass(builder_class, StepBuilderBase):
                    results['invalid'].append(f"{step_type}: Not a StepBuilderBase subclass")
                    continue
                
                # Check if builder can be instantiated (basic check)
                # Note: We can't fully instantiate without a config, but we can check the class
                if not hasattr(builder_class, '__init__'):
                    results['invalid'].append(f"{step_type}: Missing __init__ method")
                    continue
                
                results['valid'].append(f"{step_type} -> {builder_class.__name__}")
                
            except Exception as e:
                results['invalid'].append(f"{step_type}: {str(e)}")
        
        return results
    
    def get_registry_stats(self) -> Dict[str, int]:
        """
        Get statistics about the registry.
        
        Returns:
            Dictionary with registry statistics
        """
        builder_map = self.get_builder_map()
        return {
            'total_builders': len(builder_map),
            'default_builders': len(self.BUILDER_REGISTRY),
            'custom_builders': len(self._custom_builders),
        }


# Global registry instance
_global_registry = None


def get_global_registry() -> StepBuilderRegistry:
    """
    Get the global step builder registry instance.
    
    Returns:
        Global StepBuilderRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = StepBuilderRegistry()
    return _global_registry


def register_global_builder(step_type: str, builder_class: Type[StepBuilderBase]) -> None:
    """
    Register a builder in the global registry.
    
    Args:
        step_type: Step type name
        builder_class: Step builder class
    """
    registry = get_global_registry()
    registry.register_builder(step_type, builder_class)


def list_global_step_types() -> List[str]:
    """
    List all step types in the global registry.
    
    Returns:
        List of supported step type names
    """
    registry = get_global_registry()
    return registry.list_supported_step_types()
