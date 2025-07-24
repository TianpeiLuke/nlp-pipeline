# Essential Inputs Implementation Guide

This document provides practical guidance for implementing the Essential Inputs approach to the XGBoost evaluation pipeline configuration using the three-tier architecture. It covers concrete implementation steps, code examples, and integration points for transforming the current complex configuration notebook into a streamlined user experience.

## Prerequisites

Before implementing this approach, ensure you have reviewed:

1. [Essential Inputs Notebook Design](./essential_inputs_notebook_design.md) - High-level design document
2. [Essential Inputs Field Dependency Analysis](./essential_inputs_field_dependency_analysis.md) - Detailed field analysis
3. [Essential Inputs Implementation Strategy](./essential_inputs_implementation_strategy.md) - Technical implementation details
4. [XGBoost Pipeline Field Dependency Table](./xgboost_pipeline_field_dependency_table.md) - Field categorization analysis

## Three-Tier Architecture Implementation

The implementation involves building these three distinct layers:

### Tier 1: Essential User Interface Layer
- **Purpose**: Collect only the 19 essential user inputs through a streamlined interface
- **Components**:
  - Essential Input Models - Core Pydantic models defining required user inputs
  - Feature Group Registry - Organized field groupings with business descriptions
  - User Interface Components - Jupyter widgets for simplified input

### Tier 2: System Configuration Layer
- **Purpose**: Maintain standardized values for all fixed system inputs
- **Components**:
  - System Config Repository - Version-controlled configuration files
  - Default Value Registry - Central registry of standard values
  - Infrastructure Templates - Resource allocation patterns

### Tier 3: Configuration Generation Layer
- **Purpose**: Generate complete configurations from essential inputs and system defaults
- **Components**:
  - Smart Defaults Generator - System to derive all dependent values
  - Configuration Transformation Pipeline - Process to produce full config
  - Configuration Preview System - Multi-level preview interface

## Step 1: Create Essential Input Models

Start by defining the Pydantic models for the essential user inputs:

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime

class DateRangePeriod(BaseModel):
    """Model for a date range period"""
    start_date: str
    end_date: str

class FeatureGroup(BaseModel):
    """Model for a feature group"""
    name: str
    description: str
    fields: List[str]

class DataConfig(BaseModel):
    """Essential data configuration"""
    region: str
    training_period: DateRangePeriod
    calibration_period: DateRangePeriod
    service_name: str = "AtoZ"
    feature_groups: Dict[str, bool]
    custom_fields: List[str] = Field(default_factory=list)

class ModelConfig(BaseModel):
    """Essential model configuration"""
    is_binary: bool = True
    label_name: str = "is_abuse"
    id_name: str = "order_id"
    marketplace_id_col: str = "marketplace_id"
    num_round: int = 300
    max_depth: int = 10
    min_child_weight: int = 1
    metric_choices: List[str] = Field(default_factory=lambda: ["f1_score", "auroc"])

class RegistrationConfig(BaseModel):
    """Essential registration configuration"""
    model_owner: str = "amzn1.abacus.team.djmdvixm5abr3p75c5ca"
    model_registration_domain: str = "AtoZ"
    expected_tps: int = 2
    max_latency_ms: int = 800
    max_error_rate: float = 0.2

class EssentialConfig(BaseModel):
    """Container for all essential configuration sections"""
    data: DataConfig
    model: ModelConfig
    registration: RegistrationConfig
```

## Step 2: Implement Feature Group Registry

Create a feature group registry that organizes fields into logical categories:

```python
def get_feature_groups(region_lower):
    """Get feature group definitions with region-specific field names"""
    return {
        "buyer_profile": {
            "name": "Buyer Profile Metrics",
            "description": "Account age, history metrics",
            "fields": [
                "COMP_DAYOB",
                "claimantInfo_allClaimCount365day",
                "claimantInfo_lifetimeClaimCount",
                "claimantInfo_pendingClaimCount",
                "claimantInfo_status"
            ]
        },
        "order_behavior": {
            "name": "Order Behavior Metrics",
            "description": "Order frequency, value statistics",
            "fields": [
                "Abuse.completed_afn_orders_by_customer_marketplace.n_afn_order_count_last_365_days",
                "Abuse.completed_afn_orders_by_customer_marketplace.n_afn_unit_amount_last_365_days",
                "Abuse.completed_afn_orders_by_customer_marketplace.n_afn_unit_count_last_365_days",
                "Abuse.completed_mfn_orders_by_customer_marketplace.n_mfn_order_count_last_365_days",
                "Abuse.completed_mfn_orders_by_customer_marketplace.n_mfn_unit_amount_last_365_days",
                "Abuse.completed_mfn_orders_by_customer_marketplace.n_mfn_unit_count_last_365_days",
                "Abuse.order_to_execution_time_from_eventvariables.n_order_to_execution",
                "PAYMETH"
            ]
        },
        "refund_claims": {
            "name": "Refund and Claims Metrics",
            "description": "Return history, claim patterns",
            "fields": [
                "Abuse.dnr_by_customer_marketplace.n_dnr_amount_si_last_365_days",
                "Abuse.dnr_by_customer_marketplace.n_dnr_order_count_last_365_days",
                "Abuse.mfn_a2z_claims_by_customer_na.n_mfn_claims_amount_last_365_days",
                "Abuse.mfn_a2z_claims_by_customer_na.n_mfn_claims_count_last_365_days",
                "claimAmount_value",
                "claim_reason"
                # Additional fields...
            ]
        },
        "shipping": {
            "name": "Shipping Status Metrics",
            "description": "Delivery status patterns",
            "fields": [
                "Abuse.shiptrack_flag_by_order.n_any_delivered",
                "Abuse.shiptrack_flag_by_order.n_any_available_for_pickup",
                "Abuse.shiptrack_flag_by_order.n_any_partial_delivered",
                "Abuse.shiptrack_flag_by_order.n_any_undeliverable",
                "Abuse.shiptrack_flag_by_order.n_any_returning",
                "Abuse.shiptrack_flag_by_order.n_any_returned",
                "shipments_status"
            ]
        },
        "messages": {
            "name": "Message Metrics",
            "description": "Buyer-seller communication stats",
            "fields": [
                f"Abuse.bsm_stats_for_evaluated_mfn_concessions_by_customer_{region_lower}.n_max_buyer_order_message_time_gap",
                f"Abuse.bsm_stats_for_evaluated_mfn_concessions_by_customer_{region_lower}.n_max_order_message_time_gap",
                f"Abuse.bsm_stats_for_evaluated_mfn_concessions_by_customer_{region_lower}.n_total_buyer_message_count",
                f"Abuse.bsm_stats_for_evaluated_mfn_concessions_by_customer_{region_lower}.n_total_message_count",
                # Additional message fields...
            ]
        },
        "abuse_patterns": {
            "name": "Abuse Pattern Indicators",
            "description": "Previous abuse flags, warnings",
            "fields": [
                f"Abuse.abuse_fap_action_by_customer_inline_transform_{region_lower}.n_claims_solicit_count_last_365_days",
                f"Abuse.abuse_fap_action_by_customer_inline_transform_{region_lower}.n_claims_warn_count_last_365_days",
                f"Abuse.abuse_fap_action_by_customer_inline_transform_{region_lower}.n_concession_solicit_count_last_365_days",
                f"Abuse.abuse_fap_action_by_customer_inline_transform_{region_lower}.n_concession_warn_count_last_365_days"
            ]
        }
    }
```

## Step 3: Implement Smart Defaults Generator

Create the SmartDefaultsGenerator class that derives all configuration values:

```python
class SmartDefaultsGenerator:
    """Generates derived configurations from essential inputs"""
    
    def __init__(self, essential_config: EssentialConfig):
        self.config = essential_config
        self.region = self.config.data.region
        self.region_lower = self.region.lower()
        self.is_binary = self.config.model.is_binary
        
        # Set up region mapping
        self.aws_region_map = {
            "NA": "us-east-1",
            "EU": "eu-west-1",
            "FE": "us-west-2"
        }
        self.aws_region = self.aws_region_map[self.region]
        
        # Set up feature groups
        self.feature_groups = get_feature_groups(self.region_lower)
        
        # Set up categorical field registry
        self.categorical_fields = set([
            "PAYMETH",
            "claim_reason",
            "claimantInfo_status",
            "shipments_status"
        ])
    
    def derive_field_lists(self):
        """Derive field lists based on feature group selection"""
        all_fields = []
        cat_fields = []
        tab_fields = []
        
        # Add selected feature group fields
        for group_name, is_selected in self.config.data.feature_groups.items():
            if is_selected and group_name in self.feature_groups:
                group_fields = self.feature_groups[group_name]["fields"]
                all_fields.extend(group_fields)
                
                # Categorize fields
                for field in group_fields:
                    if field in self.categorical_fields:
                        cat_fields.append(field)
                    else:
                        tab_fields.append(field)
        
        # Add custom fields
        all_fields.extend(self.config.data.custom_fields)
        
        # Categorize custom fields (simplified logic)
        for field in self.config.data.custom_fields:
            if field in self.categorical_fields:
                cat_fields.append(field)
            else:
                tab_fields.append(field)
        
        # Always include core fields
        core_fields = [
            self.config.model.id_name,
            self.config.model.marketplace_id_col, 
            self.config.model.label_name,
            "baseCurrency",
            "Abuse.currency_exchange_rate_inline.exchangeRate"
        ]
        
        for field in core_fields:
            if field not in all_fields:
                all_fields.append(field)
        
        return {
            "full_field_list": list(set(all_fields)),
            "cat_field_list": list(set(cat_fields)),
            "tab_field_list": list(set(tab_fields))
        }
    
    def generate_transform_sql(self, data_source_name, tag_source_name, field_lists):
        """Generate SQL transformation"""
        select_variable_text_list = []
        
        for field in field_lists["full_field_list"]:
            if field != self.config.model.label_name and field != self.config.model.id_name:
                field_dot_replaced = field.replace('.', '__DOT__')
                select_variable_text_list.append(
                    f'{data_source_name}.{field_dot_replaced}'
                )
        
        # Add label and ID fields from tag source
        select_variable_text_list.append(f'{tag_source_name}.{self.config.model.label_name}')
        select_variable_text_list.append(f'{tag_source_name}.{self.config.model.id_name}')
        
        schema_list = ',\n'.join(select_variable_text_list)
        
        transform_sql = f"""
        SELECT
        {schema_list}
        FROM {data_source_name}
        JOIN {tag_source_name} 
        ON {data_source_name}.objectId={tag_source_name}.{self.config.model.id_name}
        """
        
        return transform_sql
    
    def generate_full_config(self):
        """Generate the complete configuration structure"""
        # Implementation details...
        # This would generate all configuration objects and return them
        pass
```

## Step 4: Create UI Components for Input Collection

Implement Jupyter widgets for easy user input:

```python
import ipywidgets as widgets
from IPython.display import display, clear_output

def create_data_config_widgets(feature_groups):
    """Create widgets for data configuration"""
    # Region selection
    region_dropdown = widgets.Dropdown(
        options=[('North America (NA)', 'NA'), ('Europe (EU)', 'EU'), ('Far East (FE)', 'FE')],
        value='NA',
        description='Region:',
    )
    
    # Service name
    service_name = widgets.Text(
        value='AtoZ',
        description='Service Name:',
        disabled=False
    )
    
    # Date pickers
    training_start = widgets.Text(description='Training Start:', value='2025-01-01T00:00:00')
    training_end = widgets.Text(description='Training End:', value='2025-04-17T00:00:00')
    calibration_start = widgets.Text(description='Calibration Start:', value='2025-04-17T00:00:00')
    calibration_end = widgets.Text(description='Calibration End:', value='2025-04-28T00:00:00')
    
    # Feature group selection
    feature_group_widgets = {}
    for group_name, group_info in feature_groups.items():
        feature_group_widgets[group_name] = widgets.Checkbox(
            value=True,
            description=group_info['name'],
            style={'description_width': 'initial'},
            layout={'width': 'auto'}
        )
    
    # Custom fields
    custom_field = widgets.Text(description='Custom Field:')
    add_button = widgets.Button(description='Add')
    custom_field_list = widgets.SelectMultiple(
        options=[],
        description='Custom Fields:',
        disabled=False
    )
    
    def add_custom_field(_):
        if custom_field.value and custom_field.value not in custom_field_list.options:
            custom_field_list.options = custom_field_list.options + (custom_field.value,)
            custom_field.value = ''
    
    add_button.on_click(add_custom_field)
    
    # Organize widgets
    date_box = widgets.VBox([
        widgets.HBox([training_start, training_end]),
        widgets.HBox([calibration_start, calibration_end])
    ])
    
    feature_group_box = widgets.VBox([widget for widget in feature_group_widgets.values()])
    
    custom_field_box = widgets.VBox([
        widgets.HBox([custom_field, add_button]),
        custom_field_list
    ])
    
    # Return all widgets
    return {
        'region': region_dropdown,
        'service_name': service_name,
        'dates': date_box,
        'date_widgets': {
            'training_start': training_start,
            'training_end': training_end,
            'calibration_start': calibration_start,
            'calibration_end': calibration_end,
        },
        'feature_groups': feature_group_widgets,
        'feature_group_box': feature_group_box,
        'custom_fields': custom_field_list,
        'custom_field_box': custom_field_box
    }

def create_model_config_widgets():
    """Create widgets for model configuration"""
    # Model type
    model_type = widgets.Dropdown(
        options=[('Binary Classification', True), ('Multi-class Classification', False)],
        value=True,
        description='Model Type:',
    )
    
    # Label and ID fields
    label_field = widgets.Text(value='is_abuse', description='Label Field:')
    id_field = widgets.Text(value='order_id', description='ID Field:')
    marketplace_field = widgets.Text(value='marketplace_id', description='Marketplace ID:')
    
    # Hyperparameters
    num_round = widgets.IntSlider(value=300, min=100, max=1000, step=50, description='Rounds:')
    max_depth = widgets.IntSlider(value=10, min=3, max=20, description='Max Depth:')
    min_child_weight = widgets.IntSlider(value=1, min=1, max=10, description='Min Child Weight:')
    
    # Metrics
    metrics = widgets.SelectMultiple(
        options=[
            ('F1 Score', 'f1_score'),
            ('AUC-ROC', 'auroc'),
            ('Precision', 'precision'),
            ('Recall', 'recall'),
            ('Accuracy', 'accuracy')
        ],
        value=['f1_score', 'auroc'],
        description='Metrics:',
    )
    
    # Organize widgets
    fields_box = widgets.VBox([label_field, id_field, marketplace_field])
    params_box = widgets.VBox([num_round, max_depth, min_child_weight])
    
    # Return all widgets
    return {
        'model_type': model_type,
        'fields': fields_box,
        'field_widgets': {
            'label_field': label_field,
            'id_field': id_field,
            'marketplace_field': marketplace_field
        },
        'hyperparams': params_box,
        'hyperparam_widgets': {
            'num_round': num_round,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight
        },
        'metrics': metrics
    }

def create_registration_config_widgets():
    """Create widgets for registration configuration"""
    # Owner and domain
    model_owner = widgets.Text(
        value='amzn1.abacus.team.djmdvixm5abr3p75c5ca',
        description='Model Owner:',
    )
    model_domain = widgets.Text(value='AtoZ', description='Model Domain:')
    
    # Performance requirements
    expected_tps = widgets.IntSlider(value=2, min=1, max=10, description='Expected TPS:')
    max_latency = widgets.IntSlider(
        value=800,
        min=100,
        max=2000,
        step=100,
        description='Max Latency (ms):',
    )
    error_rate = widgets.FloatSlider(
        value=0.2,
        min=0.01,
        max=0.5,
        step=0.01,
        description='Max Error Rate:',
    )
    
    # Organize widgets
    owner_box = widgets.VBox([model_owner, model_domain])
    perf_box = widgets.VBox([expected_tps, max_latency, error_rate])
    
    # Return all widgets
    return {
        'owner': owner_box,
        'owner_widgets': {
            'model_owner': model_owner,
            'model_domain': model_domain
        },
        'performance': perf_box,
        'performance_widgets': {
            'expected_tps': expected_tps,
            'max_latency': max_latency,
            'error_rate': error_rate
        }
    }
```

## Step 5: Create Configuration Preview System

Implement a preview system to show the derived configuration:

```python
def generate_config_preview(full_config):
    """Generate a user-friendly preview of the configuration"""
    preview = {
        "Region": full_config["base_config"]["region"],
        "Pipeline Name": full_config["base_config"]["pipeline_name"],
        "Data Sources": [
            f"MDS: {full_config['training_cradle_data_load_config']['data_sources_spec']['data_sources'][0]['data_source_name']}",
            f"EDX: {full_config['training_cradle_data_load_config']['data_sources_spec']['data_sources'][1]['edx_data_source_properties']['edx_dataset']}"
        ],
        "Date Ranges": {
            "Training": f"{full_config['training_cradle_data_load_config']['data_sources_spec']['start_date']} to {full_config['training_cradle_data_load_config']['data_sources_spec']['end_date']}",
            "Calibration": f"{full_config['calibration_cradle_data_load_config']['data_sources_spec']['start_date']} to {full_config['calibration_cradle_data_load_config']['data_sources_spec']['end_date']}"
        },
        "Model Type": "Binary Classification" if full_config["xgb_train_config"]["hyperparameters"]["is_binary"] else "Multi-class Classification",
        "XGBoost Parameters": {
            "num_round": full_config["xgb_train_config"]["hyperparameters"]["num_round"],
            "max_depth": full_config["xgb_train_config"]["hyperparameters"]["max_depth"],
            "min_child_weight": full_config["xgb_train_config"]["hyperparameters"]["min_child_weight"]
        },
        "Feature Selection": {
            "Tabular Fields": len(full_config["xgb_train_config"]["hyperparameters"]["tab_field_list"]),
            "Categorical Fields": len(full_config["xgb_train_config"]["hyperparameters"]["cat_field_list"])
        },
        "Registration": {
            "Model Objective": full_config["model_registration_config"]["model_registration_objective"],
            "Expected TPS": full_config["payload_config"]["expected_tps"],
            "Max Latency": f"{full_config['payload_config']['max_latency_in_millisecond']}ms"
        }
    }
    
    # Format the preview as HTML or markdown
    preview_html = "<h2>Configuration Preview</h2>"
    preview_html += "<ul>"
    
    for key, value in preview.items():
        preview_html += f"<li><b>{key}:</b> "
        
        if isinstance(value, dict):
            preview_html += "<ul>"
            for k, v in value.items():
                preview_html += f"<li>{k}: {v}</li>"
            preview_html += "</ul>"
        elif isinstance(value, list):
            preview_html += "<ul>"
            for item in value:
                preview_html += f"<li>{item}</li>"
            preview_html += "</ul>"
        else:
            preview_html += f"{value}"
        
        preview_html += "</li>"
    
    preview_html += "</ul>"
    
    return preview_html

def display_config_preview(full_config):
    """Display a preview of the generated configuration with override options"""
    from IPython.display import HTML
    
    preview = generate_config_preview(full_config)
    display(HTML(preview))
    
    # Add button to show advanced options
    show_advanced = widgets.Button(description='Show Advanced Options')
    
    def on_show_advanced_click(_):
        # Code to display advanced configuration sections
        pass
    
    show_advanced.on_click(on_show_advanced_click)
    display(show_advanced)
```

## Step 6: Integrating the Components

Create the final notebook structure integrating all components:

```python
def collect_essential_inputs(data_widgets, model_widgets, reg_widgets):
    """Collect all inputs from widgets into EssentialConfig"""
    # Extract data config
    feature_groups_dict = {
        name: widget.value 
        for name, widget in data_widgets['feature_groups'].items()
    }
    
    data_config = DataConfig(
        region=data_widgets['region'].value,
        training_period=DateRangePeriod(
            start_date=data_widgets['date_widgets']['training_start'].value,
            end_date=data_widgets['date_widgets']['training_end'].value
        ),
        calibration_period=DateRangePeriod(
            start_date=data_widgets['date_widgets']['calibration_start'].value,
            end_date=data_widgets['date_widgets']['calibration_end'].value
        ),
        service_name=data_widgets['service_name'].value,
        feature_groups=feature_groups_dict,
        custom_fields=list(data_widgets['custom_fields'].options)
    )
    
    # Extract model config
    model_config = ModelConfig(
        is_binary=model_widgets['model_type'].value,
        label_name=model_widgets['field_widgets']['label_field'].value,
        id_name=model_widgets['field_widgets']['id_field'].value,
        marketplace_id_col=model_widgets['field_widgets']['marketplace_field'].value,
        num_round=model_widgets['hyperparam_widgets']['num_round'].value,
        max_depth=model_widgets['hyperparam_widgets']['max_depth'].value,
        min_child_weight=model_widgets['hyperparam_widgets']['min_child_weight'].value,
        metric_choices=list(model_widgets['metrics'].value)
    )
    
    # Extract registration config
    reg_config = RegistrationConfig(
        model_owner=reg_widgets['owner_widgets']['model_owner'].value,
        model_registration_domain=reg_widgets['owner_widgets']['model_domain'].value,
        expected_tps=reg_widgets['performance_widgets']['expected_tps'].value,
        max_latency_ms=reg_widgets['performance_widgets']['max_latency'].value,
        max_error_rate=reg_widgets['performance_widgets']['error_rate'].value
    )
    
    # Create essential config
    return EssentialConfig(
        data=data_config,
        model=model_config,
        registration=reg_config
    )

def initialize_notebook():
    """Set up the full notebook with all components"""
    # Initialize feature groups
    region = 'na'  # Default region for initial rendering
    feature_groups = get_feature_groups(region)
    
    # Create widget sets
    data_widgets = create_data_config_widgets(feature_groups)
    model_widgets = create_model_config_widgets()
    reg_widgets = create_registration_config_widgets()
    
    # Function to handle region change
    def on_region_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            new_region = change['new'].lower()
            new_feature_groups = get_feature_groups(new_region)
            
            # Update feature group widgets
            for group_name, widget in data_widgets['feature_groups'].items():
                if group_name in new_feature_groups:
                    widget.description = new_feature_groups[group_name]['name']
                else:
                    widget.disabled = True
    
    # Connect region change handler
    data_widgets['region'].observe(on_region_change)
    
    # Create generate button
    generate_button = widgets.Button(description='Generate Configuration')
    
    # Function to handle generate button click
    def on_generate_click(_):
        # Collect inputs
        essential_config = collect_essential_inputs(data_widgets, model_widgets, reg_widgets)
        
        # Create generator
        generator = SmartDefaultsGenerator(essential_config)
        
        # Generate full config
        full_config = generator.generate_full_config()
        
        # Display preview
        clear_output(wait=True)
        display_config_preview(full_config)
        
        # Add save button
        save_button = widgets.Button(description='Save Configuration')
        
        def on_save_click(_):
            # Code to save configuration
            pass
        
        save_button.on_click(on_save_click)
        display(save_button)
    
    # Connect generate button handler
    generate_button.on_click(on_generate_click)
    
    # Create tabs for organization
    tab_data = widgets.VBox([
        widgets.HBox([data_widgets['region'], data_widgets['service_name']]),
        widgets.Label('Date Ranges:'),
        data_widgets['dates'],
        widgets.Label('Feature Groups:'),
        data_widgets['feature_group_box'],
        widgets.Label('Custom Fields:'),
        data_widgets['custom_field_box']
    ])
    
    tab_model = widgets.VBox([
        model_widgets['model_type'],
        widgets.Label('Fields:'),
        model_widgets['fields'],
        widgets.Label('Hyperparameters:'),
        model_widgets['hyperparams'],
        widgets.Label('Evaluation Metrics:'),
        model_widgets['metrics']
    ])
    
    tab_reg = widgets.VBox([
        widgets.Label('Model Ownership:'),
        reg_widgets['owner'],
        widgets.Label('Performance Requirements:'),
        reg_widgets['performance']
    ])
    
    tabs = widgets.Tab()
    tabs.children = [tab_data, tab_model, tab_reg]
    tabs.set_title(0, 'Data Configuration')
    tabs.set_title(1, 'Model Configuration')
    tabs.set_title(2, 'Registration Configuration')
    
    # Display tabs and generate button
    display(tabs)
    display(generate_button)
```

## Step 7: Converting Notebook to the New Format

To convert the existing `template_config_xgb_eval_v2.ipynb` notebook to the Essential Inputs approach:

1. Create a new notebook file: `template_config_xgb_eval_simplified.ipynb`
2. Add standard imports and session setup from the original notebook
3. Add the code for Essential Input Models, Feature Group Registry, and Smart Defaults Generator
4. Add the UI components for input collection
5. Add the configuration preview and save functionality

Here's a basic structure for the new notebook:

```python
# Import standard libraries
import os
import json
import pandas as pd
import boto3
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Union

# Import session setup
from sagemaker import Session
from secure_ai_sandbox_python_lib.session import Session as SaisSession
# ... other imports from original notebook

# Import widgets
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

# Essential Input Models
# ... paste code for essential input models

# Feature Group Registry
# ... paste code for feature group registry

# Smart Defaults Generator
# ... paste code for smart defaults generator

# UI Components
# ... paste code for UI components

# Configuration Preview System
# ... paste code for configuration preview system

# Initialize the notebook UI
initialize_notebook()
```

## Step 8: Testing and Validation

To ensure the new implementation works correctly:

1. **Unit Testing**: Test each component individually
   - Verify that the SmartDefaultsGenerator produces correct configurations
   - Check that UI components collect and display inputs correctly
   - Validate the configuration preview displays accurately

2. **Integration Testing**: Test the complete notebook
   - Verify that the full configuration flow works end-to-end
   - Compare generated configurations with manually created ones
   - Ensure pipeline execution with the generated configuration succeeds

3. **User Testing**: Have users test the new interface
   - Gather feedback on usability and clarity
   - Identify any missing essential inputs
   - Verify the time savings compared to the original approach

## Step 9: Deployment and Documentation

1. **Deploy the Notebook**: Make the new notebook available to users
   - Add to existing project repository
   - Create a link from the original notebook to the new one
   
2. **Document the Approach**:
   - Update user guides with the new approach
   - Create examples of common configurations
   - Document any advanced customization options

3. **Monitor Usage**:
   - Track adoption rates of the new approach
   - Collect feedback for future improvements
   - Measure time savings an
