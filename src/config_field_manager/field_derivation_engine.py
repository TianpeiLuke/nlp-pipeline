"""
FieldDerivationEngine module.

This module defines the FieldDerivationEngine class which is responsible for deriving
configuration fields from essential user inputs and system defaults (Tier 3 fields).
"""

import inspect
import logging
import re
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Callable

logger = logging.getLogger(__name__)


class FieldDerivationEngine:
    """
    Engine for deriving configuration fields from essential inputs and system defaults.
    
    This class implements the derivation logic for Tier 3 fields in the three-tier
    configuration architecture. It automatically generates field values based on
    established relationships and dependencies between fields.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the field derivation engine
        
        Args:
            logger: Optional logger for reporting derivation actions
        """
        self.logger = logger
        # Track which fields have been derived
        self.derived_fields = set()
        # Track derivation relationships for diagnostics
        self.derivation_graph = {}
        
    def derive_fields(self, config: Any) -> Any:
        """
        Derive all applicable fields for a configuration object
        
        Args:
            config: Configuration object to derive fields for
            
        Returns:
            The modified configuration object
        """
        # Reset tracking for this config
        self.derived_fields = set()
        self.derivation_graph = {}
        
        # Apply all applicable derivation methods
        config_type = config.__class__.__name__
        
        # Get all methods that start with "derive_" 
        derivation_methods = [
            method for name, method in inspect.getmembers(self, inspect.ismethod)
            if name.startswith("derive_") and name != "derive_fields"
        ]
        
        # Apply each applicable derivation method
        derived_count = 0
        for method in derivation_methods:
            # Check if this method applies to this config type
            applicable_types = self._get_applicable_types(method)
            if applicable_types and config_type not in applicable_types:
                continue
                
            # Apply the derivation method
            try:
                result = method(config)
                if result:
                    derived_count += len(result)
                    self.derived_fields.update(result)
            except Exception as e:
                if self.logger:
                    self.logger.warning(
                        f"Error in {method.__name__} for {config_type}: {str(e)}"
                    )
                    
        if self.logger and derived_count > 0:
            self.logger.info(
                f"Derived {derived_count} fields for {config_type}: {sorted(self.derived_fields)}"
            )
            
        return config
        
    def derive_fields_for_multiple(self, configs: List[Any]) -> List[Any]:
        """
        Derive fields for multiple configuration objects
        
        This method handles cross-configuration dependencies by processing
        configs in multiple passes if necessary.
        
        Args:
            configs: List of configuration objects
            
        Returns:
            The list of modified configuration objects
        """
        # First pass - derive fields within each config independently
        for config in configs:
            self.derive_fields(config)
            
        # Second pass - handle cross-configuration dependencies
        config_map = {config.__class__.__name__: config for config in configs}
        
        for config in configs:
            self._apply_cross_config_derivations(config, config_map)
            
        return configs
        
    def _get_applicable_types(self, method: Callable) -> Optional[Set[str]]:
        """
        Get the configuration types a derivation method applies to
        
        Args:
            method: The derivation method
            
        Returns:
            Set of applicable configuration type names or None if applicable to all
        """
        doc = method.__doc__
        if not doc:
            return None
            
        # Look for "Applicable to:" in docstring
        match = re.search(r"Applicable to:\s*(.*?)(?:\n|$)", doc)
        if not match:
            return None
            
        # Parse the types list
        types_str = match.group(1).strip()
        if types_str.lower() == "all":
            return None
            
        return {t.strip() for t in types_str.split(",")}
        
    def _apply_cross_config_derivations(self, config: Any, config_map: Dict[str, Any]) -> Any:
        """
        Apply derivations that depend on other configurations
        
        Args:
            config: The configuration object to derive fields for
            config_map: Dictionary mapping config type names to instances
            
        Returns:
            The modified configuration object
        """
        # Example implementation for specific cross-config derivations
        config_type = config.__class__.__name__
        
        # Handle specific cross-configuration derivations
        if config_type == "ModelEvaluationConfig" and "XGBoostTrainingConfig" in config_map:
            # Copy framework version from training config
            training_config = config_map["XGBoostTrainingConfig"]
            if (hasattr(training_config, "framework_version") and 
                not hasattr(config, "xgboost_framework_version")):
                config.xgboost_framework_version = training_config.framework_version
                self.derived_fields.add("xgboost_framework_version")
                
        # Add other cross-config derivations as needed
                
        return config
    
    # ==========================================
    # Field derivation methods
    # ==========================================
    
    def derive_input_dimensions(self, config: Any) -> Dict[str, Any]:
        """
        Derive input dimensions from field lists
        
        Applicable to: ModelHyperparameters, XGBoostHyperparameters
        
        Derives:
        - input_tab_dim: Number of tabular fields
        
        Depends on:
        - tab_field_list: List of tabular fields
        """
        derived = {}
        
        # Derive input_tab_dim from tab_field_list
        if hasattr(config, "tab_field_list") and config.tab_field_list is not None:
            if not hasattr(config, "input_tab_dim") or config.input_tab_dim is None:
                config.input_tab_dim = len(config.tab_field_list)
                derived["input_tab_dim"] = config.input_tab_dim
                
        return derived
        
    def derive_classification_type(self, config: Any) -> Dict[str, Any]:
        """
        Derive classification type fields
        
        Applicable to: ModelHyperparameters, XGBoostHyperparameters
        
        Derives:
        - is_binary: Boolean indicating binary classification
        - num_classes: Number of classes
        - multiclass_categories: List of class values
        
        Depends on:
        - multiclass_categories: List of class values (if available)
        - is_binary: Boolean indicating binary classification (if available)
        - num_classes: Number of classes (if available)
        """
        derived = {}
        
        # Case 1: Derive from multiclass_categories
        if hasattr(config, "multiclass_categories") and config.multiclass_categories is not None:
            categories = config.multiclass_categories
            
            # Derive num_classes
            if not hasattr(config, "num_classes") or config.num_classes is None:
                config.num_classes = len(categories)
                derived["num_classes"] = config.num_classes
                
            # Derive is_binary
            if not hasattr(config, "is_binary") or config.is_binary is None:
                config.is_binary = (config.num_classes == 2)
                derived["is_binary"] = config.is_binary
                
        # Case 2: Derive from num_classes
        elif hasattr(config, "num_classes") and config.num_classes is not None:
            # Derive is_binary
            if not hasattr(config, "is_binary") or config.is_binary is None:
                config.is_binary = (config.num_classes == 2)
                derived["is_binary"] = config.is_binary
                
            # Derive multiclass_categories if missing
            if (not hasattr(config, "multiclass_categories") or 
                config.multiclass_categories is None):
                config.multiclass_categories = list(range(config.num_classes))
                derived["multiclass_categories"] = config.multiclass_categories
                
        # Case 3: Derive from is_binary
        elif hasattr(config, "is_binary") and config.is_binary is not None:
            # Derive num_classes
            if not hasattr(config, "num_classes") or config.num_classes is None:
                config.num_classes = 2 if config.is_binary else 3  # Default to 3 for multi-class
                derived["num_classes"] = config.num_classes
                
            # Derive multiclass_categories if missing
            if (not hasattr(config, "multiclass_categories") or 
                config.multiclass_categories is None):
                config.multiclass_categories = [0, 1] if config.is_binary else [0, 1, 2]
                derived["multiclass_categories"] = config.multiclass_categories
        
        return derived
        
    def derive_xgboost_specific(self, config: Any) -> Dict[str, Any]:
        """
        Derive XGBoost-specific parameters
        
        Applicable to: XGBoostHyperparameters
        
        Derives:
        - objective: XGBoost objective function
        - eval_metric: XGBoost evaluation metrics
        
        Depends on:
        - is_binary: Boolean indicating binary classification
        """
        derived = {}
        
        if hasattr(config, "is_binary"):
            # Derive objective function
            if not hasattr(config, "objective") or config.objective is None:
                config.objective = "binary:logistic" if config.is_binary else "multi:softmax"
                derived["objective"] = config.objective
                
            # Derive evaluation metrics
            if not hasattr(config, "eval_metric") or config.eval_metric is None:
                config.eval_metric = ['logloss', 'auc'] if config.is_binary else ['mlogloss', 'merror']
                derived["eval_metric"] = config.eval_metric
                
        return derived
        
    def derive_mds_field_list(self, config: Any) -> Dict[str, bool]:
        """
        Derive MDS field list from other field lists
        
        Applicable to: CradleDataLoadConfig
        
        Derives:
        - mds_field_list: Combined field list for MDS
        
        Depends on:
        - full_field_list: Full list of fields
        - cat_field_list: Categorical fields
        - tab_field_list: Tabular fields
        """
        derived = {}
        
        if ((hasattr(config, "tab_field_list") and config.tab_field_list is not None) or
            (hasattr(config, "cat_field_list") and config.cat_field_list is not None) or
            (hasattr(config, "full_field_list") and config.full_field_list is not None)):
            
            # Start with core fields always needed
            core_fields = ['objectId', 'transactionDate', 
                          'Abuse.currency_exchange_rate_inline.exchangeRate', 'baseCurrency']
            all_fields = set(core_fields)
            
            # Add fields from available lists
            if hasattr(config, "full_field_list") and config.full_field_list:
                all_fields.update(config.full_field_list)
            if hasattr(config, "cat_field_list") and config.cat_field_list:
                all_fields.update(config.cat_field_list)
            if hasattr(config, "tab_field_list") and config.tab_field_list:
                all_fields.update(config.tab_field_list)
                
            # Set the mds_field_list if not already set
            if not hasattr(config, "mds_field_list") or config.mds_field_list is None:
                config.mds_field_list = sorted(list(all_fields))
                derived["mds_field_list"] = True
                
        return derived
        
    def derive_output_schema(self, config: Any) -> Dict[str, bool]:
        """
        Derive output schema from field list
        
        Applicable to: CradleDataLoadConfig
        
        Derives:
        - output_schema: Schema definition for output
        
        Depends on:
        - mds_field_list: Combined field list
        """
        derived = {}
        
        if hasattr(config, "mds_field_list") and config.mds_field_list is not None:
            if not hasattr(config, "output_schema") or config.output_schema is None:
                # Create schema with all fields as STRING type
                config.output_schema = [
                    {'field_name': field, 'field_type': 'STRING'} 
                    for field in config.mds_field_list
                ]
                derived["output_schema"] = True
                
        return derived
        
    def derive_transform_sql(self, config: Any) -> Dict[str, bool]:
        """
        Derive SQL transformation based on data configuration
        
        Applicable to: CradleDataLoadConfig
        
        Derives:
        - transform_spec.transform_sql: SQL for data transformation
        
        Depends on:
        - region: Region code
        - label_name: Label field name
        """
        derived = {}
        
        if hasattr(config, "region") and config.region is not None:
            # Create transform_spec if needed
            if not hasattr(config, "transform_spec"):
                config.transform_spec = {}
                
            # Only derive SQL if not already set
            if not config.transform_spec.get("transform_sql"):
                # Build basic SQL that joins MDS and tag tables
                sql = f"""
                SELECT
                    mds.*, tags.{getattr(config, 'label_name', 'is_abuse')}, 
                    tags.marketplace_id
                FROM RAW_MDS_{config.region} mds
                JOIN TAGS tags
                ON mds.objectId=tags.order_id
                """
                
                config.transform_spec["transform_sql"] = sql
                derived["transform_spec.transform_sql"] = True
                
        return derived
        
    def derive_etl_job_id(self, config: Any) -> Dict[str, Any]:
        """
        Derive ETL job ID based on region
        
        Applicable to: CradleDataLoadConfig
        
        Derives:
        - etl_job_id: ETL job ID for the region
        
        Depends on:
        - region: Region code
        - etl_job_id_dict: Dictionary mapping regions to job IDs (if available)
        """
        derived = {}
        
        if hasattr(config, "region") and config.region is not None:
            # Skip if etl_job_id is already set
            if hasattr(config, "etl_job_id") and config.etl_job_id is not None:
                return derived
                
            # Try to get from etl_job_id_dict if available
            if hasattr(config, "etl_job_id_dict") and config.etl_job_id_dict:
                if config.region in config.etl_job_id_dict:
                    config.etl_job_id = config.etl_job_id_dict[config.region]
                    derived["etl_job_id"] = config.etl_job_id
                    return derived
            
            # Default mappings if dict not available
            default_etl_jobs = {
                "NA": "24292902",
                "EU": "24292903",
                "FE": "24292904"
            }
            
            if config.region in default_etl_jobs:
                config.etl_job_id = default_etl_jobs[config.region]
                derived["etl_job_id"] = config.etl_job_id
                
        return derived
        
    def derive_pipeline_fields(self, config: Any) -> Dict[str, Any]:
        """
        Derive pipeline-related fields
        
        Applicable to: BasePipelineConfig
        
        Derives:
        - pipeline_name: Name of the pipeline
        - pipeline_subdirectory: Subdirectory for pipeline files
        - pipeline_s3_loc: S3 location for pipeline
        
        Depends on:
        - author: Pipeline author
        - service_name: Service name
        - region: Region code
        - pipeline_version: Pipeline version
        - bucket: S3 bucket name
        """
        derived = {}
        
        if all(hasattr(config, attr) for attr in ["author", "service_name", "region"]):
            # Derive pipeline_name if not set
            if not hasattr(config, "pipeline_name") or config.pipeline_name is None:
                config.pipeline_name = f"{config.author}-{config.service_name}-XGBoostModel-{config.region}"
                derived["pipeline_name"] = config.pipeline_name
                
            # Derive pipeline_description if not set
            if not hasattr(config, "pipeline_description") or config.pipeline_description is None:
                config.pipeline_description = f"{config.service_name} XGBoost Model {config.region}"
                derived["pipeline_description"] = config.pipeline_description
                
            # Derive pipeline_subdirectory if not set
            if hasattr(config, "pipeline_version") and config.pipeline_version is not None:
                if not hasattr(config, "pipeline_subdirectory") or config.pipeline_subdirectory is None:
                    config.pipeline_subdirectory = f"{config.pipeline_name}_{config.pipeline_version}"
                    derived["pipeline_subdirectory"] = config.pipeline_subdirectory
                    
                # Derive pipeline_s3_loc if not set
                if hasattr(config, "bucket") and config.bucket is not None:
                    if not hasattr(config, "pipeline_s3_loc") or config.pipeline_s3_loc is None:
                        config.pipeline_s3_loc = f"s3://{config.bucket}/MODS/{config.pipeline_subdirectory}"
                        derived["pipeline_s3_loc"] = config.pipeline_s3_loc
                
        return derived
        
    def derive_aws_region(self, config: Any) -> Dict[str, Any]:
        """
        Derive AWS region from region code
        
        Applicable to: BasePipelineConfig
        
        Derives:
        - aws_region: AWS region name
        
        Depends on:
        - region: Region code
        """
        derived = {}
        
        if hasattr(config, "region") and config.region is not None:
            # Skip if aws_region is already set
            if hasattr(config, "aws_region") and config.aws_region is not None:
                return derived
                
            # Map region code to AWS region
            region_map = {
                "NA": "us-east-1",
                "EU": "eu-west-1",
                "FE": "us-west-2"
            }
            
            if config.region in region_map:
                config.aws_region = region_map[config.region]
                derived["aws_region"] = config.aws_region
                
        return derived
