# Essential Inputs Notebook Design

## Executive Summary

The current `template_config_xgb_eval_v2.ipynb` notebook requires excessive user input across multiple configuration sections. This design document outlines a streamlined approach that focuses only on essential user inputs while automating all other aspects. By concentrating on just three critical areas - data loading, model training, and model registration - we can dramatically improve the user experience while maintaining full pipeline functionality.

## Problem Analysis

### Current Pain Points

1. **Excessive Input Requirements:** Users must manually configure 10+ separate sections, most of which could be automated
2. **High Cognitive Load:** Understanding all configuration options requires deep pipeline knowledge
3. **Time-Consuming Process:** The current approach requires extensive user interaction and decision-making
4. **Error-Prone:** Numerous manual inputs increase the likelihood of configuration errors
5. **Redundant Information:** Many parameters are duplicated or could be derived from other inputs

### User Requirements Analysis

Based on user feedback, the essential configuration areas are:

1. **Data Loading:** Configuration of data sources, date ranges, and field selections
2. **Model Training:** Core model hyperparameters and training settings
3. **Model Registration:** Information required for deploying the trained model

All other configuration sections should be automated with sensible defaults derived from these essential inputs.

## Design Approach

### 1. Three-Tier Configuration Architecture

Based on the field dependency analysis in [essential_inputs_field_dependency_analysis.md](./essential_inputs_field_dependency_analysis.md) and the comprehensive field categorization in [xgboost_pipeline_field_dependency_table.md](./xgboost_pipeline_field_dependency_table.md), we've established that only 23% of fields (19 out of 83) need direct user input while 77% can be derived or preset. This insight leads us to implement the Three-Tier Configuration Architecture detailed in [config_field_categorization_three_tier.md](./config_field_categorization_three_tier.md):

#### Tier 1: Essential User Inputs (23% of fields)
This layer focuses solely on collecting the 19 essential user inputs that represent true business decisions through a streamlined interface in three clearly defined sections:

##### Section 1: Data Configuration
- Region selection (`region_list`, `region_selection`)
- Training and calibration date ranges (`training_start_datetime`, `training_end_datetime`, etc.)
- Feature group selection (mapped to `full_field_list`, `cat_field_list`, `tab_field_list`)
- Data source configuration (`tag_edx_provider`, `tag_edx_subject`, `tag_edx_dataset`)

##### Section 2: Model Training Configuration
- Model type selection (`multiclass_categories` for binary/multiclass)
- Key hyperparameters (`num_round`, `max_depth`, `min_child_weight`)
- Target variable and ID fields (`label_name`, `id_name`, `marketplace_id_col`)
- Class weighting configuration (`class_weights`)

##### Section 3: Model Registration Configuration
- Model identification (`model_owner`, `model_registration_domain`)
- Performance requirements (`expected_tps`, `max_latency_in_millisecond`)
- Model versioning (`pipeline_version`)

#### Tier 2: System Inputs (Fixed Values)
This layer manages all standardized system inputs that don't require user configuration but may be adjusted by administrators:

- Instance types and resource allocations (`processing_instance_type_large`, etc.)
- Processing framework versions (`processing_framework_version`, `py_version`)
- Default hyperparameters (`device`, `batch_size`, `lr`, `max_epochs`, `optimizer`)
- Entry points for processing steps (`processing_entry_point`, etc.)
- Standard metrics (`metric_choices`, `eval_metric_choices`)
- Default thresholds (`max_acceptable_error_rate`)

These values are maintained in the `ConfigFieldTierRegistry` and `DefaultValuesProvider` systems described in the three-tier design document.

#### Tier 3: Derived Inputs (Generated Values)
This layer automatically generates dependent fields using the `FieldDerivationEngine`:

1. Field derivatives (`input_tab_dim`, `num_classes`, `is_binary`)
2. SQL transformations (`schema_list`, `training_transform_sql`)
3. Path constructions (`pipeline_subdirectory`, `pipeline_s3_loc`)
4. Regional mappings (`aws_region`)
5. Output specifications (`training_output_path`, `output_schema`)

The complete configuration processing pipeline:
1. Takes essential user inputs from Tier 1
2. Incorporates system defaults from Tier 2
3. Applies derivation logic for all Tier 3 fields
4. Produces a configuration compatible with existing pipelines

### 2. Feature Group Management

Instead of selecting individual fields, users will select logical feature groups:

- **Buyer Profile Metrics**: Account age, history metrics
- **Order Behavior Metrics**: Order patterns, value statistics
- **Refund and Claims Metrics**: Return history, claim patterns
- **Message and Communication Metrics**: Buyer-seller interaction statistics
- **Shipping and Delivery Metrics**: Delivery status patterns
- **Abuse Pattern Indicators**: Previous abuse flags, warnings

Each group contains related fields with clear business descriptions, dramatically simplifying the selection process.

### 3. Multi-Level Configuration Preview

To maintain user confidence while reducing direct control:

1. **Summary View**: High-level overview of key configuration decisions
2. **Detailed View**: Expandable sections showing all derived fields
3. **Advanced View**: Technical JSON export for expert users

Users can review the generated configuration at their preferred level of detail before proceeding.

### 3. Technical Implementation Strategy

#### Smart Defaults System

Create a defaults generator that produces appropriate values based on essential inputs:

```python
class SmartDefaultsGenerator:
    def __init__(self, essential_config):
        self.config = essential_config
        
    def generate_preprocessing_config(self):
        """Generate preprocessing configuration based on essential inputs"""
        # Implementation details
        
    def generate_training_config(self):
        """Generate training configuration based on essential inputs"""
        # Implementation details
        
    # Additional generators for other configuration sections
```

#### Configuration Transformation Pipeline

Implement a pipeline that transforms essential inputs into the complete configuration:

1. **Collection Phase:** Gather essential user inputs
2. **Validation Phase:** Verify input consistency and completeness
3. **Expansion Phase:** Generate derived configurations
4. **Integration Phase:** Combine all configurations into the final structure
5. **Verification Phase:** Validate the complete configuration

```python
def transform_essential_to_full_config(essential_config):
    # Validate essential inputs
    validate_essential_inputs(essential_config)
    
    # Generate derived configurations
    defaults_generator = SmartDefaultsGenerator(essential_config)
    preprocessing_config = defaults_generator.generate_preprocessing_config()
    training_config = defaults_generator.generate_training_config()
    # Generate other configurations...
    
    # Integrate into final configuration
    full_config = integrate_configurations(
        essential_config,
        preprocessing_config,
        training_config,
        # Other configurations...
    )
    
    # Verify complete configuration
    verify_full_config(full_config)
    
    return full_config
```

#### Feature Group System

Organize field selection through pre-defined feature groups rather than individual field selection:

```python
feature_groups = {
    "buyer_behavior": {
        "name": "Buyer Behavior Metrics",
        "description": "Metrics related to buyer purchasing patterns",
        "fields": [
            "Abuse.completed_afn_orders_by_customer_marketplace.n_afn_order_count_last_365_days",
            "Abuse.completed_afn_orders_by_customer_marketplace.n_afn_unit_amount_last_365_days",
            # Additional fields...
        ]
    },
    "refund_metrics": {
        "name": "Refund and Return Metrics",
        "description": "Metrics related to refunds and returns",
        "fields": [
            "Abuse.mfn_refunds_by_customer_marketplace.n_mfn_refund_order_count_last_365_days",
            "Abuse.mfn_refunds_by_customer_marketplace.n_mfn_refund_unit_amount_last_365_days",
            # Additional fields...
        ]
    },
    # Additional feature groups...
}
```

#### Config Preview System

Provide users with visibility into the generated configuration with options to override:

```python
def display_config_preview(full_config):
    """Display a preview of the generated configuration with override options"""
    # Implementation details for displaying a readable summary
    
    # Option to view detailed configuration
    if user_wants_details:
        display_detailed_config(full_config)
        
    # Option to override specific sections
    if user_wants_override:
        sections_to_override = get_user_override_sections()
        for section in sections_to_override:
            full_config[section] = collect_user_override(section)
```

### 4. Notebook Structure

The redesigned notebook will have the following structure:

#### Cell Group 1: Introduction and Setup
- Notebook overview and instructions
- Environment setup and library imports
- Optional: Template selection

#### Cell Group 2: Data Configuration
- Region selection
- Date range configuration
- Data source parameters
- Feature group selection

#### Cell Group 3: Model Configuration
- Model type selection
- Core hyperparameter configuration
- Target variable and evaluation metrics

#### Cell Group 4: Registration Configuration
- Model metadata
- Deployment parameters

#### Cell Group 5: Configuration Generation
- Generate complete configuration from essential inputs
- Preview generated configuration
- Optional: Override specific configuration sections

#### Cell Group 6: Pipeline Execution
- Execute pipeline with generated configuration
- Display progress and results

## Feature Group Definitions

To simplify field selection, we will organize fields into logical feature groups:

### Buyer Profile Metrics
- Account age and status
- Historical order patterns
- Lifetime value metrics

### Order Behavior Metrics
- Order frequency and volume
- Order value statistics
- Payment method information

### Refund and Claim Metrics
- Return and refund history
- Claim frequency and amounts
- Dispute patterns

### Message and Communication Metrics
- Buyer-seller message statistics
- Communication timing patterns
- Message content indicators

### Shipping and Delivery Metrics
- Delivery status patterns
- Shipping time statistics
- Address verification metrics

### Abuse Pattern Indicators
- Previous abuse flags
- Warning and solicitation history
- Risk indicators

## Implementation Plan

### Phase 1: Essential Input Collection System
- Develop the streamlined input cells for the three core sections
- Create validation functions for essential inputs
- Implement basic defaults for non-essential parameters

### Phase 2: Smart Defaults Generation System
- Develop the defaults generator for all configuration sections
- Implement intelligent derivation rules based on essential inputs
- Create validation system for generated configurations

### Phase 3: User Interface Enhancements
- Implement feature group selection interface
- Create configuration preview system
- Develop override mechanisms for advanced users

### Phase 4: Testing and Documentation
- Test with various configuration scenarios
- Document derivation rules and default values
- Create user guide for the new notebook

### Phase 5: Deployment and Feedback
- Release the new notebook alongside the existing version
- Collect user feedback on usability and effectiveness
- Iterate based on feedback

## Technical Details

### Essential Input Schemas

#### Data Configuration Schema
```python
class DataConfig(BaseModel):
    region: str = Field(..., description="Region code (NA, EU, FE)")
    training_period: DateRangePeriod
    calibration_period: DateRangePeriod
    mds_service_name: str = "AtoZ"
    mds_org_id: int = 0
    edx_provider: str = "trms-abuse-analytics"
    edx_subject: str = "qingyuye-notr-exp"
    edx_dataset: str = "atoz-tag"
    etl_job_id: Optional[str] = None
    feature_groups: Dict[str, bool] = Field(
        default_factory=lambda: {
            "buyer_profile": True,
            "order_behavior": True,
            "refund_claims": True,
            "messages": True,
            "shipping": True,
            "abuse_patterns": True
        }
    )
    custom_fields: List[str] = Field(default_factory=list)
```

#### Model Configuration Schema
```python
class ModelConfig(BaseModel):
    is_binary: bool = True
    label_name: str = "is_abuse"
    id_name: str = "order_id"
    marketplace_id_col: str = "marketplace_id"
    class_weights: Optional[List[float]] = None
    metric_choices: List[str] = Field(default_factory=lambda: ["f1_score", "auroc"])
    num_round: int = 300
    max_depth: int = 10
    min_child_weight: int = 1
    objective: Optional[str] = None  # Will be derived from is_binary
```

#### Registration Configuration Schema
```python
class RegistrationConfig(BaseModel):
    model_owner: str = "amzn1.abacus.team.djmdvixm5abr3p75c5ca"
    model_registration_domain: str = "AtoZ"
    model_registration_objective: Optional[str] = None  # Will be derived
    expected_tps: int = 2
    max_latency_ms: int = 800
    max_error_rate: float = 0.2
```

### Default Value Derivation Examples

#### Deriving XGBoost Objective
```python
def derive_objective(is_binary: bool) -> str:
    """Derive the appropriate XGBoost objective function"""
    return "binary:logistic" if is_binary else "multi:softmax"
```

#### Deriving Evaluation Metrics
```python
def derive_eval_metrics(is_binary: bool) -> List[str]:
    """Derive appropriate evaluation metrics based on model type"""
    return ["logloss", "auc"] if is_binary else ["mlogloss", "merror"]
```

#### Deriving Field Lists
```python
def derive_field_lists(feature_groups: Dict[str, bool], 
                       custom_fields: List[str]) -> Dict[str, List[str]]:
    """Derive full, categorical, and tabular field lists from feature groups"""
    all_fields = []
    cat_fields = []
    tab_fields = []
    
    # Add fields from selected feature groups
    for group_name, is_selected in feature_groups.items():
        if is_selected:
            group_fields = FEATURE_GROUP_DEFINITIONS[group_name]["fields"]
            all_fields.extend(group_fields)
            
            # Categorize fields
            for field in group_fields:
                if field in CATEGORICAL_FIELD_REGISTRY:
                    cat_fields.append(field)
                else:
                    tab_fields.append(field)
    
    # Add custom fields
    all_fields.extend(custom_fields)
    
    # Categorize custom fields (simplified logic)
    for field in custom_fields:
        if field.startswith("s_") or field.endswith("_cat"):
            cat_fields.append(field)
        else:
            tab_fields.append(field)
    
    return {
        "full_field_list": all_fields,
        "cat_field_list": cat_fields,
        "tab_field_list": tab_fields
    }
```

## User Experience Benefits

### Quantitative Improvements

1. **Reduced Input Fields:** From 50+ input parameters to ~15 essential inputs (70%+ reduction)
2. **Time Savings:** Estimated 75% reduction in configuration time
3. **Error Reduction:** Estimated 80% reduction in configuration errors

### Qualitative Improvements

1. **Lower Cognitive Load:** Users focus only on meaningful business decisions
2. **Increased Confidence:** Auto-generation with best practices increases trust
3. **Easier Onboarding:** New users can create valid configurations quickly
4. **Focus on Value:** Users spend time on model quality, not configuration details

## Conclusion

This essential inputs notebook design provides a streamlined, user-friendly approach to pipeline configuration. By focusing on the three most important areas of user input and automating everything else, we can dramatically improve the user experience while maintaining the full functionality and flexibility of the pipeline. The design balances simplicity for typical users with advanced options for experts, ensuring that all use cases are supported.

The feature group approach to field selection further simplifies the configuration process by allowing users to think in terms of logical data categories rather than individual fields. Combined with smart defaults and configuration previews, this design delivers a substantial improvement over the current approach.
