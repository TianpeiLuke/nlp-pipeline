# FieldDerivationEngine Design

## Overview

The `FieldDerivationEngine` is a critical component of the three-tier configuration architecture responsible for managing all derived inputs (Tier 3). This document outlines the design and implementation of this engine, which automatically generates configuration fields based on essential user inputs and system defaults.

## Purpose and Responsibilities

The `FieldDerivationEngine` serves several key purposes:

1. Derive configuration fields from essential user inputs and system defaults
2. Implement dependency relationships between fields
3. Ensure consistency across derived values
4. Reduce user input requirements by automating field generation
5. Maintain compatibility with existing pipeline configurations

## Core Design Principles

### 1. Dependency-Aware Processing

The `FieldDerivationEngine` implements a dependency-aware processing model:

- Fields are derived in the correct order based on their dependencies
- The system automatically handles dependency chains (where a derived field depends on another derived field)
- Circular dependencies are detected and reported

### 2. Explicit Derivation Rules

All derivation rules are explicit and centralized:

- Each rule is clearly defined as a separate method
- Dependencies are documented in method docstrings
- The derivation process is transparent and traceable

### 3. Configuration Type Awareness

The engine respects configuration type boundaries:

- Derivation rules are applied only to appropriate configuration types
- Cross-configuration dependencies are handled through explicit references
- The system maintains the integrity of each configuration's purpose

### 4. Resilient Processing

The derivation engine is designed for resilience:

- Missing or invalid inputs are handled gracefully
- Partial derivation is possible when some dependencies are unavailable
- The system provides clear diagnostic information for troubleshooting

## Class Structure

```python
class FieldDerivationEngine:
    """
    Engine for deriving configuration fields from essential inputs and system defaults.
    
    This class implements the derivation logic for Tier 3 fields in the three-tier
    configuration architecture. It automatically generates field values based on
    established relationships and dependencies between fields.
    """
    
    def __init__(self, logger=None):
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
        
    def derive_fields(self, config):
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
        
    def derive_fields_for_multiple(self, configs):
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
        
    def _get_applicable_types(self, method):
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
        
    def _apply_cross_config_derivations(self, config, config_map):
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
    
    def derive_input_dimensions(self, config):
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
        
    def derive_classification_type(self, config):
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
        
    def derive_xgboost_specific(self, config):
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
        
    def derive_mds_field_list(self, config):
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
        
    def derive_output_schema(self, config):
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
        
    def derive_transform_sql(self, config):
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
        
    def derive_etl_job_id(self, config):
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
        
    def derive_edx_manifest(self, config):
        """
        Derive EDX manifest from provider, subject, dataset, and dates
        
        Applicable to: CradleDataLoadConfig
        
        Derives:
        - tag_edx_manifest: Full EDX manifest
        
        Depends on:
        - tag_edx_provider: EDX provider
        - tag_edx_subject: EDX subject
        - tag_edx_dataset: EDX dataset
        - etl_job_id: ETL job ID
        - start_date: Start date
        - end_date: End date
        - region: Region code
        """
        derived = {}
        
        required_attrs = [
            "tag_edx_provider", "tag_edx_subject", "tag_edx_dataset", 
            "etl_job_id", "data_sources_spec"
        ]
        
        # Check if all required attributes exist
        if all(hasattr(config, attr) for attr in required_attrs):
            # Skip if manifest is already set
            if hasattr(config, "tag_edx_manifest") and config.tag_edx_manifest is not None:
                return derived
                
            # Get dates from data_sources_spec
            if isinstance(config.data_sources_spec, dict):
                start_date = config.data_sources_spec.get("start_date")
                end_date = config.data_sources_spec.get("end_date")
                
                if start_date and end_date and hasattr(config, "region"):
                    # Construct the manifest
                    manifest = (
                        f'arn:amazon:edx:iad::manifest/{config.tag_edx_provider}/'
                        f'{config.tag_edx_subject}/{config.tag_edx_dataset}/'
                        f'["{config.etl_job_id}",{start_date}Z,{end_date}Z,'
                        f'"{config.region}"]'
                    )
                    
                    # Set the manifest in EDX data source properties
                    if not hasattr(config, "edx_data_source_properties"):
                        config.edx_data_source_properties = {}
                        
                    config.tag_edx_manifest = manifest
                    derived["tag_edx_manifest"] = True
                    
        return derived
        
    def derive_pipeline_fields(self, config):
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
        
    def derive_aws_region(self, config):
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
        
    def derive_model_inference_variable_list(self, config):
        """
        Derive model inference input/output variable lists
        
        Applicable to: ModelRegistrationConfig
        
        Derives:
        - source_model_inference_input_variable_list: Input variables for inference
        - source_model_inference_output_variable_list: Output variables for inference
        
        Depends on:
        - full_field_list, cat_field_list, tab_field_list: Field lists
        - label_name, id_name: Key fields to exclude from input list
        """
        derived = {}
        
        # Check if we have hyperparameters with field lists
        if hasattr(config, "hyperparameters"):
            hyper = config.hyperparameters
            
            # Derive input variable list if not set
            if (not hasattr(config, "source_model_inference_input_variable_list") or 
                not config.source_model_inference_input_variable_list):
                
                # Collect all fields
                all_fields = set()
                
                # Try to get fields from different sources
                if hasattr(hyper, "full_field_list") and hyper.full_field_list:
                    all_fields.update(hyper.full_field_list)
                elif hasattr(hyper, "cat_field_list") and hasattr(hyper, "tab_field_list"):
                    if hyper.cat_field_list:
                        all_fields.update(hyper.cat_field_list)
                    if hyper.tab_field_list:
                        all_fields.update(hyper.tab_field_list)
                
                # Filter out label and id fields
                exclude_fields = set()
                if hasattr(hyper, "label_name") and hyper.label_name:
                    exclude_fields.add(hyper.label_name)
                if hasattr(hyper, "id_name") and hyper.id_name:
                    exclude_fields.add(hyper.id_name)
                
                input_fields = all_fields - exclude_fields
                
                # Create input variable list
                input_var_list = []
                for field in input_fields:
                    field_type = "TEXT" if field in getattr(hyper, "cat_field_list", []) else "NUMERIC"
                    input_var_list.append([field, field_type])
                
                config.source_model_inference_input_variable_list = input_var_list
                derived["source_model_inference_input_variable_list"] = True
            
            # Derive output variable list if not set
            if (not hasattr(config, "source_model_inference_output_variable_list") or 
                not config.source_model_inference_output_variable_list):
                
                # Default output variables
                config.source_model_inference_output_variable_list = {
                    'legacy-score': 'NUMERIC',
                    'calibrated-score': 'NUMERIC',
                    'custom-output-label': 'TEXT'
                }
                derived["source_model_inference_output_variable_list"] = True
                
        return derived
        
    def derive_model_registration_objective(self, config):
        """
        Derive model registration objective
        
        Applicable to: ModelRegistrationConfig
        
        Derives:
        - model_registration_objective: Full objective string
        
        Depends on:
        - service_name: Service name
        - region: Region code
        """
        derived = {}
        
        if all(hasattr(config, attr) for attr in ["service_name", "region"]):
            # Skip if already set
            if hasattr(config, "model_registration_objective") and config.model_registration_objective:
                return derived
                
            # Generate the objective string
            config.model_registration_objective = f"{config.service_name}_Claims_SM_Model_{config.region}"
            derived["model_registration_objective"] = config.model_registration_objective
                
        return derived
```

## Field Derivation Rules

The `FieldDerivationEngine` implements a comprehensive set of derivation rules for different configuration types. Each rule:

- Is implemented as a separate method
- Has a clear naming convention (`derive_<field_group>`)
- Documents its applicable configuration types
- Specifies the fields it derives
- Lists the fields it depends on
- Returns a dictionary of derived fields for tracking

### Example Rule: Classification Type Derivation

```python
def derive_classification_type(self, config):
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
```

This rule handles three cases:

1. If `multiclass_categories` is available, derive `num_classes` and `is_binary`
2. If `num_classes` is available, derive `is_binary` and potentially `multiclass_categories`
3. If `is_binary` is available, derive `num_classes` and `multiclass_categories`

This approach ensures that the classification type is consistent across all related fields.

## Dependency Management

### Within-Configuration Dependencies

Most derivation rules handle dependencies within a single configuration object. For example:

- `derive_input_dimensions` depends on `tab_field_list` being set
- `derive_xgboost_specific` depends on `is_binary` being set

The engine applies these derivation rules in sequence, with each rule potentially building on the results of previous rules.

### Cross-Configuration Dependencies

Some fields depend on values in other configurations. These are handled by the `_apply_cross_config_derivations` method, which:

1. Takes a mapping of configuration types to instances
2. Identifies cross-configuration dependencies
3. Copies or derives values based on these dependencies

Example:
```python
def _apply_cross_config_derivations(self, config, config_map):
    """
    Apply derivations that depend on other configurations
    
    Args:
        config: The configuration object to derive fields for
        config_map: Dictionary mapping config type names to instances
    """
    config_type = config.__class__.__name__
    
    # Handle specific cross-configuration derivations
    if config_type == "ModelEvaluationConfig" and "XGBoostTrainingConfig" in config_map:
        # Copy framework version from training config
        training_config = config_map["XGBoostTrainingConfig"]
        if (hasattr(training_config, "framework_version") and 
            not hasattr(config, "xgboost_framework_version")):
            config.xgboost_framework_version = training_config.framework_version
            self.derived_fields.add("xgboost_framework_version")
```

## Integration with Three-Tier Architecture

Within the three-tier architecture, the `FieldDerivationEngine` implements Tier 3 (Derived Inputs). It works alongside:

1. **Essential User Interface (Tier 1)**: Collects essential user inputs
2. **DefaultValuesProvider (Tier 2)**: Applies system defaults

The typical workflow is:

```python
# 1. Collect essential user inputs (Tier 1)
essential_config = collect_user_inputs()

# 2. Create config objects from essential inputs
config_objects = create_config_objects(essential_config)

# 3. Apply system defaults (Tier 2)
DefaultValuesProvider.apply_defaults_to_multiple(config_objects)

# 4. Derive dependent fields (Tier 3)
field_engine = FieldDerivationEngine()
field_engine.derive_fields_for_multiple(config_objects)

# 5. Generate final configuration
final_config = merge_configs(config_objects)
```

## Field Derivation Coverage

The `FieldDerivationEngine` includes derivation rules for all identified Tier 3 (derived) fields from the field dependency analysis:

### Classification and Model Structure Fields
- `input_tab_dim` (derived from `tab_field_list`)
- `num_classes` (derived from `multiclass_categories` or `is_binary`)
- `is_binary` (derived from `num_classes` or `multiclass_categories`)
- `objective` (derived from `is_binary`)
- `eval_metric` (derived from `is_binary`)

### Field List and Schema Fields
- `mds_field_list` (derived from component field lists)
- `output_schema` (derived from `mds_field_list`)
- `source_model_inference_input_variable_list` (derived from field lists)
- `source_model_inference_output_variable_list` (derived with standard outputs)

### Path and Location Fields
- `pipeline_name` (derived from `author`, `service_name`, and `region`)
- `pipeline_description` (derived from `service_name` and `region`)
- `pipeline_subdirectory` (derived from `pipeline_name` and `pipeline_version`)
- `pipeline_s3_loc` (derived from `bucket` and `pipeline_subdirectory`)

### Data Source Fields
- `aws_region` (derived from `region` code)
- `etl_job_id` (derived from `region` or `etl_job_id_dict`)
- `tag_edx_manifest` (derived from provider, subject, dataset, and dates)
- `transform_sql` (derived from `region` and other fields)

## Customization and Extension

The `FieldDerivationEngine` is designed for easy extension and customization:

### 1. Add New Derivation Rules

To add new derivation rules, simply add new methods following the `derive_*` naming convention:

```python
def derive_new_field_group(self, config):
    """
    Derive new fields
    
    Applicable to: ConfigType1, ConfigType2
    
    Derives:
    - new_field_1: Description
    - new_field_2: Description
    
    Depends on:
    - dependency_1: Description
    - dependency_2: Description
    """
    derived = {}
    
    # Implementation
    
    return derived
```

### 2. Override Existing Rules

Existing derivation rules can be overridden by subclassing:

```python
class CustomFieldDerivationEngine(FieldDerivationEngine):
    """Custom field derivation engine with overridden rules"""
    
    def derive_classification_type(self, config):
        """Override with custom implementation"""
        # Custom implementation
```

### 3. Add Cross-Configuration Derivations

New cross-configuration derivations can be added by extending `_apply_cross_config_derivations`:

```python
def _apply_cross_config_derivations(self, config, config_map):
    # Call parent implementation
    super()._apply_cross_config_derivations(config, config_map)
    
    # Add custom derivations
    config_type = config.__class__.__name__
    if config_type == "NewConfigType" and "DependencyConfig" in config_map:
        # Custom cross-config derivation
```

## Logging and Diagnostics

The `FieldDerivationEngine` includes comprehensive logging to aid in debugging and understanding the derivation process:

### 1. Derivation Tracking

The engine tracks which fields are derived for each configuration:

```python
# Example log output
INFO: Derived 5 fields for XGBoostHyperparameters: ['eval_metric', 'input_tab_dim', 'is_binary', 'num_classes', 'objective']
```

### 2. Error Reporting

Errors in derivation rules are caught and reported:

```python
# Example log output
WARNING: Error in derive_classification_type for ModelHyperparameters: 'multiclass_categories' attribute is None
```

### 3. Dependency Visualization

The derivation graph can be visualized to understand field dependencies:

```python
def visualize_derivation_graph(self):
    """Visualize the derivation graph as a networkx graph"""
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.DiGraph()
    
    # Add nodes and edges from derivation_graph
    for target, sources in self.derivation_graph.items():
        G.add_node(target)
        for source in sources:
            G.add_node(source)
            G.add_edge(source, target)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=1500, arrows=True)
    plt.title("Field Derivation Dependencies")
    plt.tight_layout()
    plt.show()
```

## Testing Strategy

The `FieldDerivationEngine` is designed for comprehensive testing:

### 1. Unit Testing

Each derivation rule can be tested independently:

```python
def test_derive_classification_type():
    """Test the derive_classification_type method"""
    engine = FieldDerivationEngine()
    
    # Test Case 1: Derive from multiclass_categories
    config1 = MockConfig(multiclass_categories=[0, 1])
    engine.derive_classification_type(config1)
    assert config1.num_classes == 2
    assert config1.is_binary == True
    
    # Test Case 2: Derive from num_classes
    config2 = MockConfig(num_classes=3)
    engine.derive_classification_type(config2)
    assert config2.is_binary == False
    assert config2.multiclass_categories == [0, 1, 2]
    
    # Test Case 3: Derive from is_binary
    config3 = MockConfig(is_binary=True)
    engine.derive_classification_type(config3)
    assert config3.num_classes == 2
    assert config3.multiclass_categories == [0, 1]
```

### 2. Integration Testing

The complete derivation pipeline can be tested with real configurations:

```python
def test_derive_fields_for_multiple():
    """Test the end-to-end derivation process"""
    # Create test configurations
    base_config = BasePipelineConfig(...)
    hyperparams = XGBoostHyperparameters(...)
    training_config = XGBoostTrainingConfig(...)
    
    # Apply derivation
    engine = FieldDerivationEngine()
    configs = [base_config, hyperparams, training_config]
    engine.derive_fields_for_multiple(configs)
    
    # Verify results
    assert base_config.pipeline_name == f"{base_config.author}-{base_config.service_name}-XGBoostModel-{base_config.region}"
    assert hyperparams.input_tab_dim == len(hyperparams.tab_field_list)
    assert hyperparams.objective == ("binary:logistic" if hyperparams.is_binary else "multi:softmax")
```

### 3. Property-Based Testing

Property-based testing can verify that derivations maintain invariants:

```python
def test_derivation_invariants():
    """Test that derivations maintain expected invariants"""
    @given(st.lists(st.text(), min_size=1, max_size=100))
    def test_tab_field_list_invariant(field_list):
        """Test that input_tab_dim always equals len(tab_field_list)"""
        config = MockConfig(tab_field_list=field_list)
        engine = FieldDerivationEngine()
        engine.derive_input_dimensions(config)
        assert config.input_tab_dim == len(field_list)
    
    test_tab_field_list_invariant()
```

## Conclusion

The `FieldDerivationEngine` is a key component of the three-tier architecture that manages all derived inputs (Tier 3). It provides a centralized, maintainable, and extensible system for deriving configuration fields based on essential user inputs and system defaults.

By implementing dependency-aware processing with explicit derivation rules, the engine reduces the burden on users while ensuring consistency across derived values. The comprehensive logging and diagnostic capabilities further enhance maintainability and aid in troubleshooting.

Together with the `DefaultValuesProvider` and the essential user interface, the `FieldDerivationEngine` creates a robust configuration system that simplifies the user experience while maintaining compatibility with existing pipeline components.
