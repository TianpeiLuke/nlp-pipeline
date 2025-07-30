"""
Step Builder Registry for the Pipeline API.

This module provides a centralized registry mapping configuration types
to step builder classes, enabling automatic resolution during pipeline construction.
This implementation uses the step_names registry as the single source of truth
for step naming and supports auto-discovery of step builders.
"""

from typing import Dict, Type, List, Optional, Any, Callable
import logging
import importlib
import inspect
import pkgutil
import sys

from ..pipeline_steps.builder_step_base import StepBuilderBase
from ..pipeline_steps.config_base import BasePipelineConfig
from .step_names import STEP_NAMES, get_all_step_names, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES

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

# Import the RegistryError from pipeline_api
from ..pipeline_api.exceptions import RegistryError

logger = logging.getLogger(__name__)


# Decorator for auto-registering step builders
def register_builder(step_type: str = None):
    """
    Decorator to automatically register a step builder class.
    
    Args:
        step_type: Optional step type name. If not provided,
                  will be derived from the class name.
    """
    def decorator(cls):
        if not issubclass(cls, StepBuilderBase):
            raise TypeError(f"@register_builder can only be used on StepBuilderBase subclasses: {cls.__name__}")
        
        # Determine step type if not provided
        nonlocal step_type
        if step_type is None:
            if cls.__name__.endswith('StepBuilder'):
                derived_type = cls.__name__[:-11]  # Remove 'StepBuilder'
            else:
                derived_type = cls.__name__
            step_type = derived_type
        
        # Register the class
        StepBuilderRegistry.register_builder_class(step_type, cls)
        return cls
    
    return decorator


class StepBuilderRegistry:
    """
    Centralized registry mapping step types to builder classes.
    
    This registry maintains the mapping between step types and their
    corresponding step builder classes, enabling automatic resolution
    during pipeline construction. It uses the step_names registry as 
    the single source of truth for step naming.
    """
    
    # Core registry mapping step types to builders - auto-populated during initialization
    BUILDER_REGISTRY = {}  
    
    # Legacy aliases for backward compatibility
    LEGACY_ALIASES = {
        "MIMSPackaging": "Package", 
        "MIMSPayload": "Payload",
        "ModelRegistration": "Registration",
        "PyTorchTraining": "PytorchTraining",
        "PyTorchModel": "PytorchModel",
    }
    
    @classmethod
    def register_builder_class(cls, step_type: str, builder_class: Type[StepBuilderBase]) -> None:
        """
        Register a builder class directly in the registry.
        
        Args:
            step_type: Step type name
            builder_class: Step builder class
        """
        if not issubclass(builder_class, StepBuilderBase):
            raise ValueError(f"Builder class must extend StepBuilderBase: {builder_class}")
        
        cls.BUILDER_REGISTRY[step_type] = builder_class
        logging.getLogger(__name__).info(f"Registered builder: {step_type} -> {builder_class.__name__}")
    
    @classmethod
    def discover_builders(cls):
        """
        Automatically discover and register step builders.
        
        Returns:
            Dict[str, Type[StepBuilderBase]]: Dictionary of discovered builders
        """
        from ..pipeline_steps import builder_step_base
        
        discovered_builders = {}
        
        # Get the package containing step builders
        try:
            import src.pipeline_steps as steps_package
            
            # Walk through all modules in the package
            for _, module_name, _ in pkgutil.iter_modules(steps_package.__path__):
                if module_name.startswith('builder_'):
                    try:
                        # Import the module
                        module = importlib.import_module(f"src.pipeline_steps.{module_name}")
                        
                        # Find builder classes in the module
                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, builder_step_base.StepBuilderBase) and 
                                obj != builder_step_base.StepBuilderBase):
                                
                                # Get step type from class name (remove 'StepBuilder' suffix)
                                if name.endswith('StepBuilder'):
                                    step_type = name[:-11]  # Remove 'StepBuilder'
                                else:
                                    step_type = name
                                
                                # Find canonical step name from step registry if possible
                                canonical_name = None
                                for s_name, info in STEP_NAMES.items():
                                    builder_name = info.get("builder_step_name", "")
                                    if builder_name == f"{step_type}Step" or builder_name == name:
                                        canonical_name = s_name
                                        break
                                
                                # Register with canonical name if found, otherwise use derived step_type
                                key = canonical_name or step_type
                                discovered_builders[key] = obj
                                logging.getLogger(__name__).debug(f"Discovered builder: {key} -> {name}")
                    
                    except (ImportError, AttributeError) as e:
                        logging.getLogger(__name__).warning(f"Error discovering builders in {module_name}: {e}")
        
        except ImportError:
            logging.getLogger(__name__).warning("Could not import src.pipeline_steps for builder discovery")
        
        # Manual import and registration of known step builders to ensure backward compatibility
        cls._register_known_builders(discovered_builders)
        
        return discovered_builders
    
    @classmethod
    def _register_known_builders(cls, builder_map: Dict[str, Type[StepBuilderBase]]) -> None:
        """Register known step builders to ensure backward compatibility."""
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
        
        # Core registry with canonical step names from the central step registry
        known_builders = {
            "CradleDataLoading": CradleDataLoadingStepBuilder,
            "TabularPreprocessing": TabularPreprocessingStepBuilder,
            "XGBoostTraining": XGBoostTrainingStepBuilder,
            "XGBoostModelEval": XGBoostModelEvalStepBuilder, 
            "Package": MIMSPackagingStepBuilder,  # Canonical: Package (not MIMSPackaging)
            "Payload": MIMSPayloadStepBuilder,    # Canonical: Payload (not MIMSPayload)
            "Registration": ModelRegistrationStepBuilder,  # Canonical: Registration (not ModelRegistration)
            "PytorchTraining": PyTorchTrainingStepBuilder,  # Canonical: PytorchTraining (not PyTorchTraining)
            "PytorchModel": PyTorchModelStepBuilder,  # Canonical: PytorchModel (not PyTorchModel)
            "XGBoostModel": XGBoostModelStepBuilder,
            "BatchTransform": BatchTransformStepBuilder,
            "ModelCalibration": ModelCalibrationStepBuilder,
            "CurrencyConversion": CurrencyConversionStepBuilder,
            "RiskTableMapping": RiskTableMappingStepBuilder,
            "DummyTraining": DummyTrainingStepBuilder,
            "HyperparameterPrep": HyperparameterPrepStepBuilder,
        }
        
        # Add any known builders that weren't discovered automatically
        for step_type, builder_class in known_builders.items():
            if step_type not in builder_map:
                builder_map[step_type] = builder_class
                logging.getLogger(__name__).debug(f"Added known builder: {step_type} -> {builder_class.__name__}")
    
    def __init__(self):
        """Initialize the registry."""
        self._custom_builders = {}
        self.logger = logging.getLogger(__name__)
        
        # Populate the registry if empty (first initialization)
        if not self.__class__.BUILDER_REGISTRY:
            # Get core builders through discovery
            discovered = self.__class__.discover_builders()
            self.__class__.BUILDER_REGISTRY = discovered
            
            # Log discovery results
            self.logger.info(f"Discovered {len(discovered)} step builders")
    
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
        # Check if the step_type is a legacy alias and convert to canonical name
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        
        builder_map = self.get_builder_map()
        if canonical_step_type not in builder_map:
            available_types = list(builder_map.keys())
            raise RegistryError(
                f"No step builder found for step type '{step_type}' (canonical: '{canonical_step_type}')",
                unresolvable_types=[step_type],
                available_builders=available_types
            )
        
        return builder_map[canonical_step_type]
    
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
        canonical_types = list(self.get_builder_map().keys())
        # Include legacy aliases for backward compatibility
        all_types = canonical_types + list(self.LEGACY_ALIASES.keys())
        return sorted(all_types)
    
    def is_step_type_supported(self, step_type: str) -> bool:
        """
        Check if a step type is supported.
        
        Args:
            step_type: Step type name
            
        Returns:
            True if supported, False otherwise
        """
        # Check both canonical names and legacy aliases
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        return canonical_step_type in self.get_builder_map()
    
    def get_config_types_for_step_type(self, step_type: str) -> List[str]:
        """
        Get possible configuration class names for a step type.
        
        Args:
            step_type: Step type name
            
        Returns:
            List of possible configuration class names
        """
        # First, check if this is a legacy alias
        canonical_step_type = self.LEGACY_ALIASES.get(step_type, step_type)
        
        # Try to use the central registry from step_names.py
        possible_configs = []
        
        # Look up config class in step registry
        if canonical_step_type in STEP_NAMES:
            config_class = STEP_NAMES[canonical_step_type].get("config_class")
            if config_class:
                possible_configs.append(config_class)
        
        # Add standard patterns as fallback
        if not possible_configs:
            self.logger.warning(f"Step type '{step_type}' not found in STEP_NAMES registry, using fallback patterns")
            possible_configs.append(f"{step_type}Config")
            possible_configs.append(f"{step_type}StepConfig")
        
        return possible_configs
    
    def _config_class_to_step_type(self, config_class_name: str) -> str:
        """
        Convert configuration class name to step type.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            Step type name
        """
        # Use the central registry from step_names.py - already imported
        if config_class_name in CONFIG_STEP_REGISTRY:
            canonical_step_name = CONFIG_STEP_REGISTRY[config_class_name]
            # Check if this is a new canonical name that has a legacy builder name
            for legacy_name, canonical_name in self.LEGACY_ALIASES.items():
                if canonical_name == canonical_step_name:
                    return legacy_name  # Return legacy name for backward compatibility
            return canonical_step_name
            
        # Fallback to the old conversion logic for compatibility
        self.logger.warning(f"Config class '{config_class_name}' not found in CONFIG_STEP_REGISTRY, using fallback logic")
        
        # Remove common suffixes
        step_type = config_class_name
        
        # Remove 'Config' suffix
        if step_type.endswith('Config'):
            step_type = step_type[:-6]
        
        # Remove 'Step' suffix if present
        if step_type.endswith('Step'):
            step_type = step_type[:-4]
        
        # Handle special cases for backward compatibility
        if step_type == "CradleDataLoad":
            return "CradleDataLoading"
        elif step_type == "PackageStep" or step_type == "Package":
            return "Package"  # Use canonical name
        elif step_type == "Payload":
            return "Payload"  # Use canonical name
        
        return step_type
    
    def validate_registry(self) -> Dict[str, List[str]]:
        """
        Validate the registry for consistency.
        
        Returns:
            Dictionary with validation results:
            - 'valid': List of valid mappings
            - 'invalid': List of invalid mappings with reasons
            - 'missing': List of step names in step registry but missing builders
        """
        results = {'valid': [], 'invalid': [], 'missing': []}
        
        builder_map = self.get_builder_map()
        
        # Validate existing mappings
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
        
        # Check for missing step names from the registry
        for step_name in get_all_step_names():
            if step_name not in builder_map and step_name not in self.LEGACY_ALIASES.values():
                results['missing'].append(f"{step_name}: No builder registered")
        
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
            'legacy_aliases': len(self.LEGACY_ALIASES),
            'step_registry_names': len(get_all_step_names()),
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
