"""
Helper functions for creating CradleDataLoadConfig objects with minimal inputs.

This module provides utilities to simplify the creation of complex CradleDataLoadConfig
objects by deriving nested configurations from essential user inputs.

Author: Luke Xie
Date: July 26, 2025
"""

from typing import List, Dict, Any, Optional, Union
import uuid
from pathlib import Path

from src.pipeline_steps.config_data_load_step_cradle import (
    CradleDataLoadConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    DataSourceConfig,
    DataSourcesSpecificationConfig,
    JobSplitOptionsConfig,
    TransformSpecificationConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig
)

# Default values
DEFAULT_TAG_SCHEMA = [
    'order_id',
    'marketplace_id',
    'tag_date',
    'is_abuse',
    'abuse_type',
    'concession_type'
]

DEFAULT_MDS_BASE_FIELDS = [
    'objectId', 
    'transactionDate', 
    'Abuse.currency_exchange_rate_inline.exchangeRate', 
    'baseCurrency'
]


def _map_region_to_aws_region(region: str) -> str:
    """
    Map marketplace region to AWS region.
    
    Args:
        region (str): Marketplace region ('NA', 'EU', 'FE')
        
    Returns:
        str: AWS region name
    """
    region_mapping = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
    if region not in region_mapping:
        raise ValueError(f"Invalid region: {region}. Must be one of {list(region_mapping.keys())}")
        
    return region_mapping[region]


def _create_field_schema(fields: List[str]) -> List[Dict[str, str]]:
    """
    Convert a list of field names to schema dictionaries.
    
    Args:
        fields (List[str]): List of field names
        
    Returns:
        List[Dict[str, str]]: List of schema dictionaries
    """
    return [{'field_name': field, 'field_type': 'STRING'} for field in fields]


def _create_edx_manifest(
    provider: str,
    subject: str,
    dataset: str,
    etl_job_id: str,
    start_date: str,
    end_date: str,
    region: str
) -> str:
    """
    Create an EDX manifest ARN with date components.
    
    Args:
        provider (str): EDX provider name
        subject (str): EDX subject
        dataset (str): EDX dataset name
        etl_job_id (str): ETL job ID
        start_date (str): Start date string
        end_date (str): End date string
        region (str): Region code
        
    Returns:
        str: Properly formatted EDX manifest ARN
    """
    # Ensure the date strings do not already have 'Z' appended
    start_date_clean = start_date.rstrip('Z')
    end_date_clean = end_date.rstrip('Z')
    
    return (
        f'arn:amazon:edx:iad::manifest/'
        f'{provider}/{subject}/{dataset}/'
        f'["{etl_job_id}",{start_date_clean}Z,{end_date_clean}Z,"{region}"]'
    )


def _create_edx_manifest_from_key(
    provider: str,
    subject: str,
    dataset: str,
    manifest_key: str
) -> str:
    """
    Create an EDX manifest ARN from a provided manifest key.
    
    Args:
        provider (str): EDX provider name
        subject (str): EDX subject
        dataset (str): EDX dataset name
        manifest_key (str): The complete manifest key portion (e.g., '["xxx",...]')
        
    Returns:
        str: Properly formatted EDX manifest ARN
    """
    return (
        f'arn:amazon:edx:iad::manifest/'
        f'{provider}/{subject}/{dataset}/{manifest_key}'
    )


def _generate_transform_sql(
    mds_source_name: str,
    edx_source_name: str,
    mds_field_list: List[str],
    tag_schema: List[str],
    mds_join_key: str = 'objectId',
    edx_join_key: str = 'order_id',
    join_type: str = 'JOIN'
) -> str:
    """
    Generate a SQL query to join MDS and EDX data with configurable join keys.
    
    This function ensures there are no duplicate fields in the SELECT clause
    by checking for fields that appear in both MDS and tag schema.
    The join is performed using the specified keys and join type.
    
    Args:
        mds_source_name (str): Logical name for MDS source
        edx_source_name (str): Logical name for EDX source
        mds_field_list (List[str]): List of fields from MDS
        tag_schema (List[str]): List of fields from EDX tags
        
    Returns:
        str: SQL query string
    """
    # Build the select column list
    select_variable_text_list = []
    
    # Track fields that have been added to avoid duplicates
    added_fields = set()
    
    # Add MDS fields, replacing dots with __DOT__
    for field in mds_field_list:
        field_dot_replaced = field.replace('.', '__DOT__')
        select_variable_text_list.append(f'{mds_source_name}.{field_dot_replaced}')
        added_fields.add(field.lower())
    
    # Add tag fields, skipping any that have already been added from MDS
    # (except for the join field 'order_id' which we need from the tag source)
    for var in tag_schema:
        # Always include 'order_id' from tags, but skip other duplicates
        if var.lower() != 'order_id' and var.lower() in added_fields:
            continue
        
        select_variable_text_list.append(f'{edx_source_name}.{var}')
        added_fields.add(var.lower())
    
    # Join into a comma-separated list
    schema_list = ',\n'.join(select_variable_text_list)
    
    # Create the final SQL
    transform_sql = f"""
SELECT
{schema_list}
FROM {mds_source_name}
{join_type} {edx_source_name} 
ON {mds_source_name}.{mds_join_key}={edx_source_name}.{edx_join_key}
"""
    
    return transform_sql


def _get_all_fields(
    mds_fields: List[str],
    tag_fields: List[str]
) -> List[str]:
    """
    Get a combined list of all fields from MDS and EDX sources.
    
    Args:
        mds_fields (List[str]): List of MDS fields
        tag_fields (List[str]): List of tag fields
        
    Returns:
        List[str]: Combined and deduplicated list of fields
    """
    # Combine and deduplicate
    return sorted(list(set(mds_fields + tag_fields)))


def create_cradle_data_load_config(
    # Base pipeline essentials
    role: str,
    region: str,
    pipeline_s3_loc: str,
    
    # Job configuration
    job_type: str,  # 'training' or 'calibration'
    
    # Field lists (from hyperparameters)
    full_field_list: List[str],
    tab_field_list: List[str],
    cat_field_list: List[str],
    label_name: str,
    id_name: str,
    
    # Data timeframe
    start_date: str,
    end_date: str,
    
    # MDS data source
    service_name: str,
    
    # EDX data source
    tag_edx_provider: str,
    tag_edx_subject: str,
    tag_edx_dataset: str,
    etl_job_id: str,
    
    # Infrastructure configuration
    cradle_account: str = "Buyer-Abuse-RnD-Dev",
    aws_region: Optional[str] = None,
    current_date: Optional[str] = "2025-07-26",
    
    # Optional overrides with reasonable defaults
    cluster_type: str = "STANDARD",
    output_format: str = "PARQUET",
    output_save_mode: str = "ERRORIFEXISTS",
    split_job: bool = False,
    days_per_split: int = 7,
    merge_sql: Optional[str] = None,
    s3_input_override: Optional[str] = None,
    transform_sql: Optional[str] = None,  # Auto-generated if not provided
    tag_schema: Optional[List[str]] = None,  # Default provided if not specified
    
    # Join configuration
    mds_join_key: str = 'objectId',
    edx_join_key: str = 'order_id',
    join_type: str = 'JOIN'
) -> CradleDataLoadConfig:
    """
    Create a CradleDataLoadConfig with minimal required inputs.
    
    This helper function simplifies the creation of a CradleDataLoadConfig
    by handling the generation of nested configurations from essential user inputs.
    
    Parameters:
        role (str): IAM role to use for the pipeline
        region (str): Marketplace region ('NA', 'EU', 'FE')
        pipeline_s3_loc (str): S3 location for pipeline artifacts
        
        job_type (str): Type of job ('training' or 'calibration')
        
        full_field_list (List[str]): Complete list of fields used in the model
        tab_field_list (List[str]): List of tabular (numerical) fields
        cat_field_list (List[str]): List of categorical fields
        label_name (str): Name of the label field
        id_name (str): Name of the ID field
        
        start_date (str): Start date for data pull (format: YYYY-MM-DDT00:00:00)
        end_date (str): End date for data pull (format: YYYY-MM-DDT00:00:00)
        
        service_name (str): Name of the MDS service
        
        tag_edx_provider (str): EDX provider for tags
        tag_edx_subject (str): EDX subject for tags
        tag_edx_dataset (str): EDX dataset for tags
        etl_job_id (str): ETL job ID for the EDX manifest
        
        cradle_account (str): Cradle account name (default: "Buyer-Abuse-RnD-Dev")
        aws_region (str, optional): AWS region, derived from region if not provided
        current_date (str, optional): Current date string for metadata
        
        cluster_type (str): Cradle cluster type (default: "STANDARD")
        output_format (str): Output format (default: "PARQUET")
        output_save_mode (str): Output save mode (default: "ERRORIFEXISTS")
        split_job (bool): Whether to split the job (default: False)
        days_per_split (int): Days per split if splitting (default: 7)
        merge_sql (str, optional): SQL to merge split results, required if split_job=True
        s3_input_override (str, optional): S3 input override
        transform_sql (str, optional): Custom transform SQL, auto-generated if not provided
        tag_schema (List[str], optional): Schema for tag data, default provided if not specified
    
    Returns:
        CradleDataLoadConfig: A fully configured CradleDataLoadConfig object
    """
    # 1. Derive values and set defaults
    
    # Use default tag schema if not provided
    if tag_schema is None:
        tag_schema = DEFAULT_TAG_SCHEMA
        
    # Derive AWS region from marketplace region if not provided
    if aws_region is None:
        aws_region = _map_region_to_aws_region(region)
    
    # If split_job is True, ensure merge_sql is provided
    if split_job and merge_sql is None:
        merge_sql = "SELECT * FROM INPUT"  # Default merge SQL
    
    # 2. Create MDS Data Source Config
    
    # Create MDS field list by combining base fields with tabular and categorical fields
    mds_field_list = list(set(DEFAULT_MDS_BASE_FIELDS + tab_field_list + cat_field_list))
    mds_field_list = sorted(mds_field_list)
    
    # Create MDS schema
    mds_output_schema = _create_field_schema(mds_field_list)
    
    # Create MDS data source inner config
    mds_data_source_inner_config = MdsDataSourceConfig(
        service_name=service_name,
        region=region,
        output_schema=mds_output_schema,
        org_id=0  # Default for regional MDS bucket
    )
    
    # 3. Create EDX Data Source Config
    
    # Create EDX manifest key
    edx_manifest_key = f'["{etl_job_id}",{start_date},{end_date},"{region}"]'
    
    # Create EDX schema overrides
    edx_schema_overrides = _create_field_schema(tag_schema)
    
    # Create EDX data source inner config
    edx_source_inner_config = EdxDataSourceConfig(
        edx_provider=tag_edx_provider,
        edx_subject=tag_edx_subject,
        edx_dataset=tag_edx_dataset,
        edx_manifest_key=edx_manifest_key,
        schema_overrides=edx_schema_overrides
    )
    
    # 4. Create Data Source Configs
    
    # MDS data source
    mds_data_source = DataSourceConfig(
        data_source_name=f"RAW_MDS_{region}",
        data_source_type="MDS",
        mds_data_source_properties=mds_data_source_inner_config
    )
    
    # EDX data source
    edx_data_source = DataSourceConfig(
        data_source_name="TAGS",
        data_source_type="EDX",
        edx_data_source_properties=edx_source_inner_config
    )
    
    # 5. Create Data Sources Specification
    
    data_sources_spec = DataSourcesSpecificationConfig(
        start_date=start_date,
        end_date=end_date,
        data_sources=[mds_data_source, edx_data_source]
    )
    
    # 6. Create Job Split Options
    
    job_split_options = JobSplitOptionsConfig(
        split_job=split_job,
        days_per_split=days_per_split,
        merge_sql=merge_sql
    )
    
    # 7. Create Transform Specification
    
    # Generate transform SQL if not provided
    if transform_sql is None:
        transform_sql = _generate_transform_sql(
            mds_source_name=mds_data_source.data_source_name,
            edx_source_name=edx_data_source.data_source_name,
            mds_field_list=mds_field_list,
            tag_schema=tag_schema,
            mds_join_key=mds_join_key,
            edx_join_key=edx_join_key,
            join_type=join_type
        )
    
    transform_spec = TransformSpecificationConfig(
        transform_sql=transform_sql,
        job_split_options=job_split_options
    )
    
    # 8. Create Output Specification
    
    # Combine all fields from both sources
    output_fields = _get_all_fields(mds_field_list, tag_schema)
    
    # Generate a unique output directory
    output_dir = f'cradle_download_output/{uuid.uuid4()}'
    output_path = f'{pipeline_s3_loc}/{output_dir}/{job_type}'
    
    output_spec = OutputSpecificationConfig(
        output_schema=output_fields,
        job_type=job_type,
        output_format=output_format,
        output_save_mode=output_save_mode,
        keep_dot_in_output_schema=False,
        include_header_in_s3_output=True
    )
    
    # 9. Create Cradle Job Specification
    
    cradle_job_spec = CradleJobSpecificationConfig(
        cluster_type=cluster_type,
        cradle_account=cradle_account,
        job_retry_count=4  # Default to 4 retries
    )
    
    # 10. Create the final CradleDataLoadConfig
    
    cradle_data_load_config = CradleDataLoadConfig(
        # Base pipeline fields
        role=role,
        region=region,
        aws_region=aws_region,
        pipeline_s3_loc=pipeline_s3_loc,
        current_date=current_date,
        
        # Step-specific fields
        job_type=job_type,
        data_sources_spec=data_sources_spec,
        transform_spec=transform_spec,
        output_spec=output_spec,
        cradle_job_spec=cradle_job_spec,
        s3_input_override=s3_input_override
    )
    
    # Initialize derived fields
    cradle_data_load_config.initialize_derived_fields()
    
    return cradle_data_load_config


def create_training_and_calibration_configs(
    # Base fields
    role: str,
    region: str,
    pipeline_s3_loc: str,
    
    # Field lists (from hyperparameters)
    full_field_list: List[str],
    tab_field_list: List[str],
    cat_field_list: List[str],
    label_name: str,
    id_name: str,
    
    # MDS data source
    service_name: str,
    
    # EDX data source
    tag_edx_provider: str,
    tag_edx_subject: str,
    tag_edx_dataset: str,
    etl_job_id: str,
    
    # Data timeframes
    training_start_date: str,
    training_end_date: str,
    calibration_start_date: str,
    calibration_end_date: str,
    
    # Optional shared configuration
    cradle_account: str = "Buyer-Abuse-RnD-Dev",
    aws_region: Optional[str] = None,
    current_date: Optional[str] = "2025-07-26",
    cluster_type: str = "STANDARD",
    output_format: str = "PARQUET",
    output_save_mode: str = "ERRORIFEXISTS",
    split_job: bool = False,
    days_per_split: int = 7,
    merge_sql: Optional[str] = None,
    transform_sql: Optional[str] = None,
    tag_schema: Optional[List[str]] = None
) -> Dict[str, CradleDataLoadConfig]:
    """
    Create both training and calibration CradleDataLoadConfig objects with consistent settings.
    
    Args:
        role (str): IAM role to use for the pipeline
        region (str): Marketplace region ('NA', 'EU', 'FE')
        pipeline_s3_loc (str): S3 location for pipeline artifacts
        
        full_field_list (List[str]): Complete list of fields used in the model
        tab_field_list (List[str]): List of tabular (numerical) fields
        cat_field_list (List[str]): List of categorical fields
        label_name (str): Name of the label field
        id_name (str): Name of the ID field
        
        service_name (str): Name of the MDS service
        tag_edx_provider (str): EDX provider for tags
        tag_edx_subject (str): EDX subject for tags
        tag_edx_dataset (str): EDX dataset for tags
        etl_job_id (str): ETL job ID for the EDX manifest
        
        training_start_date (str): Training data start date
        training_end_date (str): Training data end date
        calibration_start_date (str): Calibration data start date
        calibration_end_date (str): Calibration data end date
        
        cradle_account (str): Cradle account name (default: "Buyer-Abuse-RnD-Dev")
        aws_region (str, optional): AWS region, derived from region if not provided
        current_date (str, optional): Current date string for metadata
        cluster_type (str): Cradle cluster type (default: "STANDARD")
        output_format (str): Output format (default: "PARQUET")
        output_save_mode (str): Output save mode (default: "ERRORIFEXISTS")
        split_job (bool): Whether to split the job (default: False)
        days_per_split (int): Days per split if splitting (default: 7)
        merge_sql (str, optional): SQL to merge split results
        transform_sql (str, optional): Custom transform SQL
        tag_schema (List[str], optional): Schema for tag data
        
    Returns:
        Dict[str, CradleDataLoadConfig]: Dictionary with 'training' and 'calibration' configs
    """
    # Create training config
    training_config = create_cradle_data_load_config(
        role=role,
        region=region,
        pipeline_s3_loc=pipeline_s3_loc,
        job_type="training",
        full_field_list=full_field_list,
        tab_field_list=tab_field_list,
        cat_field_list=cat_field_list,
        label_name=label_name,
        id_name=id_name,
        start_date=training_start_date,
        end_date=training_end_date,
        service_name=service_name,
        tag_edx_provider=tag_edx_provider,
        tag_edx_subject=tag_edx_subject,
        tag_edx_dataset=tag_edx_dataset,
        etl_job_id=etl_job_id,
        cradle_account=cradle_account,
        aws_region=aws_region,
        current_date=current_date,
        cluster_type=cluster_type,
        output_format=output_format,
        output_save_mode=output_save_mode,
        split_job=split_job,
        days_per_split=days_per_split,
        merge_sql=merge_sql,
        transform_sql=transform_sql,
        tag_schema=tag_schema
    )
    
    # Create calibration config
    calibration_config = create_cradle_data_load_config(
        role=role,
        region=region,
        pipeline_s3_loc=pipeline_s3_loc,
        job_type="calibration",
        full_field_list=full_field_list,
        tab_field_list=tab_field_list,
        cat_field_list=cat_field_list,
        label_name=label_name,
        id_name=id_name,
        start_date=calibration_start_date,
        end_date=calibration_end_date,
        service_name=service_name,
        tag_edx_provider=tag_edx_provider,
        tag_edx_subject=tag_edx_subject,
        tag_edx_dataset=tag_edx_dataset,
        etl_job_id=etl_job_id,
        cradle_account=cradle_account,
        aws_region=aws_region,
        current_date=current_date,
        cluster_type=cluster_type,
        output_format=output_format,
        output_save_mode=output_save_mode,
        split_job=split_job,
        days_per_split=days_per_split,
        merge_sql=merge_sql,
        transform_sql=transform_sql,
        tag_schema=tag_schema
    )
    
    return {
        "training": training_config,
        "calibration": calibration_config
    }
