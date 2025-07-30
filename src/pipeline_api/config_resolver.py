"""
Configuration Resolver for the Pipeline API.

This module provides intelligent matching of DAG nodes to configuration instances
using multiple resolution strategies.
"""

from typing import Dict, List, Optional, Any, Tuple
import re
import logging
from difflib import SequenceMatcher

from ..pipeline_steps.config_base import BasePipelineConfig
from .exceptions import ConfigurationError, AmbiguityError, ResolutionError

logger = logging.getLogger(__name__)


class StepConfigResolver:
    """
    Resolves DAG nodes to configuration instances using intelligent matching.
    
    This class implements multiple resolution strategies to match DAG node
    names to configuration instances from the loaded configuration file.
    """
    
    # Pattern mappings for step type detection
    STEP_TYPE_PATTERNS = {
        r'.*data_load.*': ['CradleDataLoading'],
        r'.*preprocess.*': ['TabularPreprocessing'],
        r'.*train.*': ['XGBoostTraining', 'PyTorchTraining', 'DummyTraining'],
        r'.*eval.*': ['XGBoostModelEval'],
        r'.*model.*': ['XGBoostModel', 'PyTorchModel'],
        r'.*calibrat.*': ['ModelCalibration'],
        r'.*packag.*': ['MIMSPackaging'],
        r'.*payload.*': ['MIMSPayload'],
        r'.*regist.*': ['ModelRegistration'],
        r'.*transform.*': ['BatchTransform'],
        r'.*currency.*': ['CurrencyConversion'],
        r'.*risk.*': ['RiskTableMapping'],
        r'.*hyperparam.*': ['HyperparameterPrep'],
    }
    
    # Job type keywords for matching
    JOB_TYPE_KEYWORDS = {
        'train': ['training', 'train'],
        'calib': ['calibration', 'calib'],
        'eval': ['evaluation', 'eval', 'test'],
        'inference': ['inference', 'infer', 'predict'],
        'validation': ['validation', 'valid'],
    }
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the config resolver.
        
        Args:
            confidence_threshold: Minimum confidence score for automatic resolution
        """
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def resolve_config_map(
        self,
        dag_nodes: List[str],
        available_configs: Dict[str, BasePipelineConfig]
    ) -> Dict[str, BasePipelineConfig]:
        """
        Resolve DAG nodes to configuration instances.
        
        Resolution strategies (in order of preference):
        1. Direct name matching
        2. Job type + config type matching
        3. Semantic similarity matching
        4. Pattern-based matching
        
        Args:
            dag_nodes: List of DAG node names
            available_configs: Available configuration instances
            
        Returns:
            Dictionary mapping node names to configuration instances
            
        Raises:
            ConfigurationError: If nodes cannot be resolved
            AmbiguityError: If multiple configs match with similar confidence
        """
        config_map = {}
        resolution_details = {}
        failed_nodes = []
        
        for node in dag_nodes:
            try:
                config, confidence, method = self._resolve_single_node(node, available_configs)
                config_map[node] = config
                resolution_details[node] = {
                    'confidence': confidence,
                    'method': method,
                    'config_type': type(config).__name__
                }
                
                self.logger.debug(f"Resolved node '{node}' to {type(config).__name__} "
                                f"(confidence: {confidence:.2f}, method: {method})")
                
            except (ConfigurationError, AmbiguityError) as e:
                self.logger.warning(f"Failed to resolve node '{node}': {e}")
                failed_nodes.append(node)
        
        if failed_nodes:
            available_config_names = list(available_configs.keys())
            raise ConfigurationError(
                f"Failed to resolve {len(failed_nodes)} DAG nodes to configurations",
                missing_configs=failed_nodes,
                available_configs=available_config_names
            )
        
        return config_map
    
    def _resolve_single_node(
        self,
        node_name: str,
        available_configs: Dict[str, BasePipelineConfig]
    ) -> Tuple[BasePipelineConfig, float, str]:
        """
        Resolve a single DAG node to a configuration.
        
        Args:
            node_name: DAG node name
            available_configs: Available configuration instances
            
        Returns:
            Tuple of (config, confidence_score, resolution_method)
            
        Raises:
            ConfigurationError: If no suitable config found
            AmbiguityError: If multiple configs match with similar confidence
        """
        candidates = []
        
        # Strategy 1: Direct name matching
        direct_match = self._direct_name_matching(node_name, available_configs)
        if direct_match:
            candidates.append((direct_match, 1.0, 'direct_name'))
        
        # Strategy 2: Job type + config type matching
        job_type_matches = self._job_type_matching(node_name, available_configs)
        candidates.extend(job_type_matches)
        
        # Strategy 3: Semantic similarity matching
        semantic_matches = self._semantic_matching(node_name, available_configs)
        candidates.extend(semantic_matches)
        
        # Strategy 4: Pattern-based matching
        pattern_matches = self._pattern_matching(node_name, available_configs)
        candidates.extend(pattern_matches)
        
        if not candidates:
            raise ConfigurationError(
                f"No configuration found for node '{node_name}'",
                missing_configs=[node_name],
                available_configs=list(available_configs.keys())
            )
        
        # Sort by confidence and select best match
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_config, best_confidence, best_method = candidates[0]
        
        # Check for ambiguity
        if len(candidates) > 1:
            second_confidence = candidates[1][1]
            if abs(best_confidence - second_confidence) < 0.1:  # Very close confidence scores
                candidate_info = []
                for config, confidence, method in candidates[:3]:  # Top 3 candidates
                    job_type = getattr(config, 'job_type', 'N/A')
                    candidate_info.append({
                        'config_type': type(config).__name__,
                        'confidence': confidence,
                        'job_type': job_type,
                        'method': method
                    })
                
                raise AmbiguityError(
                    f"Multiple configurations match node '{node_name}' with similar confidence",
                    node_name=node_name,
                    candidates=candidate_info
                )
        
        # Check confidence threshold
        if best_confidence < self.confidence_threshold:
            raise ConfigurationError(
                f"Best match for node '{node_name}' has low confidence: {best_confidence:.2f} "
                f"(threshold: {self.confidence_threshold})"
            )
        
        return best_config, best_confidence, best_method
    
    def _direct_name_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> Optional[BasePipelineConfig]:
        """
        Match node name directly to config identifier.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            Matching configuration or None
        """
        # Exact match
        if node_name in configs:
            return configs[node_name]
        
        # Case-insensitive match
        for config_name, config in configs.items():
            if config_name.lower() == node_name.lower():
                return config
        
        return None
    
    def _job_type_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Match based on job_type attribute and node naming patterns.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        node_lower = node_name.lower()
        
        # Extract potential job type from node name
        detected_job_type = None
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in node_lower:
                    detected_job_type = job_type
                    break
            if detected_job_type:
                break
        
        if not detected_job_type:
            return matches
        
        # Find configs with matching job_type
        for config_name, config in configs.items():
            if hasattr(config, 'job_type'):
                config_job_type = getattr(config, 'job_type', '').lower()
                
                # Check for job type match
                job_type_keywords = self.JOB_TYPE_KEYWORDS.get(detected_job_type, [])
                if any(keyword in config_job_type for keyword in job_type_keywords):
                    # Calculate confidence based on how well the node name matches the config type
                    config_type_confidence = self._calculate_config_type_confidence(node_name, config)
                    total_confidence = 0.7 + (config_type_confidence * 0.3)  # Job type match + config type match
                    matches.append((config, total_confidence, 'job_type'))
        
        return matches
    
    def _semantic_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Use semantic similarity to match node names to config types.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        
        for config_name, config in configs.items():
            confidence = self._calculate_semantic_similarity(node_name, config)
            if confidence >= 0.5:  # Minimum semantic similarity threshold
                matches.append((config, confidence, 'semantic'))
        
        return matches
    
    def _pattern_matching(
        self,
        node_name: str,
        configs: Dict[str, BasePipelineConfig]
    ) -> List[Tuple[BasePipelineConfig, float, str]]:
        """
        Use regex patterns to match node names to config types.
        
        Args:
            node_name: DAG node name
            configs: Available configurations
            
        Returns:
            List of (config, confidence, method) tuples
        """
        matches = []
        node_lower = node_name.lower()
        
        # Find matching patterns
        matching_step_types = []
        for pattern, step_types in self.STEP_TYPE_PATTERNS.items():
            if re.match(pattern, node_lower):
                matching_step_types.extend(step_types)
        
        if not matching_step_types:
            return matches
        
        # Find configs that match the detected step types
        for config_name, config in configs.items():
            config_type = type(config).__name__
            
            # Convert config class name to step type
            step_type = self._config_class_to_step_type(config_type)
            
            if step_type in matching_step_types:
                # Base confidence for pattern match
                confidence = 0.6
                
                # Boost confidence if there are additional matches
                if hasattr(config, 'job_type'):
                    job_type_boost = self._calculate_job_type_boost(node_name, config)
                    confidence += job_type_boost * 0.2
                
                matches.append((config, min(confidence, 0.9), 'pattern'))
        
        return matches
    
    def _calculate_config_type_confidence(
        self,
        node_name: str,
        config: BasePipelineConfig
    ) -> float:
        """
        Calculate confidence based on how well node name matches config type.
        
        Args:
            node_name: DAG node name
            config: Configuration instance
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        config_type = type(config).__name__.lower()
        node_lower = node_name.lower()
        
        # Remove common suffixes for comparison
        config_base = config_type.replace('config', '').replace('step', '')
        
        # Check for substring matches
        if config_base in node_lower or any(part in node_lower for part in config_base.split('_')):
            return 0.8
        
        # Use sequence matching for similarity
        similarity = SequenceMatcher(None, node_lower, config_base).ratio()
        return similarity
    
    def _calculate_semantic_similarity(
        self,
        node_name: str,
        config: BasePipelineConfig
    ) -> float:
        """
        Calculate semantic similarity between node name and config.
        
        Args:
            node_name: DAG node name
            config: Configuration instance
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        config_type = type(config).__name__.lower()
        node_lower = node_name.lower()
        
        # Define semantic mappings
        semantic_mappings = {
            'data': ['cradle', 'load', 'loading'],
            'preprocess': ['preprocessing', 'process', 'clean'],
            'train': ['training', 'fit', 'learn'],
            'eval': ['evaluation', 'evaluate', 'test', 'assess'],
            'model': ['model', 'create', 'build'],
            'calibrat': ['calibration', 'calibrate', 'adjust'],
            'packag': ['packaging', 'package', 'bundle'],
            'regist': ['registration', 'register', 'deploy'],
        }
        
        max_similarity = 0.0
        
        for semantic_key, synonyms in semantic_mappings.items():
            if semantic_key in config_type:
                for synonym in synonyms:
                    if synonym in node_lower:
                        # Calculate similarity based on how well the synonym matches
                        similarity = SequenceMatcher(None, node_lower, synonym).ratio()
                        max_similarity = max(max_similarity, similarity * 0.8)  # Scale down semantic matches
        
        return max_similarity
    
    def _calculate_job_type_boost(
        self,
        node_name: str,
        config: BasePipelineConfig
    ) -> float:
        """
        Calculate confidence boost based on job type matching.
        
        Args:
            node_name: DAG node name
            config: Configuration instance
            
        Returns:
            Boost score (0.0 to 1.0)
        """
        if not hasattr(config, 'job_type'):
            return 0.0
        
        config_job_type = getattr(config, 'job_type', '').lower()
        node_lower = node_name.lower()
        
        # Check for job type keywords in node name
        for job_type, keywords in self.JOB_TYPE_KEYWORDS.items():
            if any(keyword in config_job_type for keyword in keywords):
                if any(keyword in node_lower for keyword in keywords):
                    return 1.0
        
        return 0.0
    
    def _config_class_to_step_type(self, config_class_name: str) -> str:
        """
        Convert configuration class name to step type.
        
        Args:
            config_class_name: Configuration class name
            
        Returns:
            Step type name
        """
        # Use the same logic as in builder_registry
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
        elif step_type == "PackageStep" or step_type == "Package":
            return "MIMSPackaging"
        
        return step_type
    
    def preview_resolution(
        self,
        dag_nodes: List[str],
        available_configs: Dict[str, BasePipelineConfig]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Preview resolution candidates for each DAG node.
        
        Args:
            dag_nodes: List of DAG node names
            available_configs: Available configuration instances
            
        Returns:
            Dictionary mapping node names to lists of candidate information
        """
        preview = {}
        
        for node in dag_nodes:
            candidates = []
            
            # Get all possible matches
            try:
                # Direct name matching
                direct_match = self._direct_name_matching(node, available_configs)
                if direct_match:
                    candidates.append({
                        'config': direct_match,
                        'config_type': type(direct_match).__name__,
                        'confidence': 1.0,
                        'method': 'direct_name',
                        'job_type': getattr(direct_match, 'job_type', 'N/A')
                    })
                
                # Job type matching
                job_matches = self._job_type_matching(node, available_configs)
                for config, confidence, method in job_matches:
                    candidates.append({
                        'config': config,
                        'config_type': type(config).__name__,
                        'confidence': confidence,
                        'method': method,
                        'job_type': getattr(config, 'job_type', 'N/A')
                    })
                
                # Semantic matching
                semantic_matches = self._semantic_matching(node, available_configs)
                for config, confidence, method in semantic_matches:
                    candidates.append({
                        'config': config,
                        'config_type': type(config).__name__,
                        'confidence': confidence,
                        'method': method,
                        'job_type': getattr(config, 'job_type', 'N/A')
                    })
                
                # Pattern matching
                pattern_matches = self._pattern_matching(node, available_configs)
                for config, confidence, method in pattern_matches:
                    candidates.append({
                        'config': config,
                        'config_type': type(config).__name__,
                        'confidence': confidence,
                        'method': method,
                        'job_type': getattr(config, 'job_type', 'N/A')
                    })
                
                # Remove duplicates and sort by confidence
                unique_candidates = {}
                for candidate in candidates:
                    key = (id(candidate['config']), candidate['method'])
                    if key not in unique_candidates or candidate['confidence'] > unique_candidates[key]['confidence']:
                        unique_candidates[key] = candidate
                
                sorted_candidates = sorted(unique_candidates.values(), key=lambda x: x['confidence'], reverse=True)
                preview[node] = sorted_candidates
                
            except Exception as e:
                self.logger.warning(f"Error previewing resolution for node '{node}': {e}")
                preview[node] = []
        
        return preview
