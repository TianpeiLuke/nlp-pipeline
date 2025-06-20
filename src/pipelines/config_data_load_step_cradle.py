# File: pipelines/config_cradle_data_load.py

from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator

from .config_base import BasePipelineConfig


class MdsDataSourceConfig(BaseModel):
    """
    Corresponds to MdsDataSourceProperties:
      - service_name
      - org_id: integer organization ID
      - region: e.g. "NA", "EU", or "FE"
      - output_schema: list of Field-like dicts (each with field_name & field_type)
      - use_hourly_edx_data_set: bool
    """
    service_name: str = Field(
        ...,
        description="Name of the MDS service"
    )
    org_id: int = Field(
        default=0,
        description="Organization ID (integer) for MDS. Default as 0 for regional MDS bucket."
    )
    region: str = Field(
        ...,
        description="Region code for MDS (e.g. 'NA', 'EU', 'FE')"
    )
    output_schema: List[Dict[str, Any]] = Field(
        ...,
        description="List of dictionaries describing each output column, "
                    "e.g. [{'field_name':'objectId','field_type':'STRING'}, …]"
    )
    use_hourly_edx_data_set: bool = Field(
        default=False,
        description="Whether to use the hourly EDX dataset flag in MDS"
    )

    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        valid = {"NA", "EU", "FE"}
        if v not in valid:
            raise ValueError(f"region must be one of {valid}, got '{v}'")
        return v


class EdxDataSourceConfig(BaseModel):
    """
    Corresponds to EdxDataSourceProperties, but now:
      - edx_manifest: must begin with
          "arn:amazon:edx:iad::manifest/{edx_provider}/{edx_subject}/{edx_dataset}/"
      - edx_provider: part of ARN path
      - edx_subject: part of ARN path
      - edx_dataset: part of ARN path
      - schema_overrides: list of Field-like dicts (each with field_name & field_type)
    """
    edx_provider: str = Field(
        ...,
        description="Provider portion of the EDX manifest ARN"
    )
    edx_subject: str = Field(
        ...,
        description="Subject portion of the EDX manifest ARN"
    )
    edx_dataset: str = Field(
        ...,
        description="Dataset portion of the EDX manifest ARN"
    )
    edx_manifest: str = Field(
        ...,
        description=(
            "Full ARN of the EDX manifest. Must begin with "
            "'arn:amazon:edx:iad::manifest/{edx_provider}/{edx_subject}/{edx_dataset}/…'"
        )
    )
    schema_overrides: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "List of dicts overriding the EDX schema, e.g. "
            "[{'field_name':'order_id','field_type':'STRING'}, …]"
        )
    )

    @model_validator(mode="after")
    @classmethod
    def check_manifest_prefix(cls, model: "EdxDataSourceConfig") -> "EdxDataSourceConfig":
        """
        Ensure edx_manifest starts with the exact prefix:
            "arn:amazon:edx:iad::manifest/{edx_provider}/{edx_subject}/{edx_dataset}/"
        """
        prefix = (
            f"arn:amazon:edx:iad::manifest/"
            f"{model.edx_provider}/{model.edx_subject}/{model.edx_dataset}/"
        )
        if not model.edx_manifest.startswith(prefix):
            raise ValueError(
                f"edx_manifest must begin with '{prefix}', got '{model.edx_manifest}'"
            )
        return model

    
class AndesDataSourceConfig(BaseModel):
    """
    Configuration for Andes Data Source Properties.
    
    Attributes:
        provider: Andes provider ID (32-digit UUID or 'booker')
        table_name: Name of the Andes table
        andes3_enabled: Whether the table uses Andes 3.0
    """
    provider: str = Field(
        ...,
        description="Andes provider ID (32-digit UUID or 'booker')"
    )
    
    table_name: str = Field(
        ...,
        description="Name of the Andes table"
    )
    
    andes3_enabled: bool = Field(
        default=False,
        description="Whether the table uses Andes 3.0 with latest version"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """
        Validate that the provider is either:
        1. A valid 32-character UUID
        2. The special case 'booker'
        """
        if v == 'booker':
            return v
            
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        )
        
        if not uuid_pattern.match(v.lower()):
            raise ValueError(
                "provider must be either 'booker' or a valid 32-digit UUID "
                "(8-4-4-4-12 format). "
                "Verify provider validity at: "
                f"https://datacentral.a2z.com/hoot/providers/{v}"
            )
            
        return v

    @field_validator("table_name")
    @classmethod
    def validate_table_name(cls, v: str) -> str:
        """
        Validate that the table name is not empty and follows valid format.
        """
        if not v or not v.strip():
            raise ValueError("table_name cannot be empty")
            
        # Add any specific table name format validation rules here
        # For example, if table names must be lowercase and hyphenated:
        if not re.match(r'^[a-z0-9-]+$', v):
            raise ValueError(
                "table_name must contain only lowercase letters, numbers, and hyphens"
            )
            
        return v

    @model_validator(mode='after')
    def validate_andes_config(self) -> 'AndesDataSourceConfig':
        """
        Additional validation for the complete Andes configuration.
        """
        # Log warning if Andes 3.0 is enabled
        if self.andes3_enabled:
            logger.warning(
                f"Andes 3.0 is enabled for table '{self.table_name}'. "
                "Ensure all features are compatible with Andes 3.0."
            )
            
        # Add any cross-field validations here
        return self

    class Config:
        """Pydantic model configuration."""
        frozen = True  # Make the config immutable
        extra = "forbid"  # Prevent additional attributes
        str_strip_whitespace = True  # Strip whitespace from string values

    def __str__(self) -> str:
        """String representation of the Andes config."""
        return (
            f"AndesDataSourceConfig(provider='{self.provider}', "
            f"table_name='{self.table_name}', "
            f"andes3_enabled={self.andes3_enabled})"
        )
    

class DataSourceConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.datasource.DataSource:
      - data_source_name: e.g. 'RAW_MDS_NA' or 'TAGS'
      - data_source_type: one of 'MDS', 'EDX', or 'ANDES'
      - one of mds_data_source_properties, edx_data_source_properties, 
        or andes_data_source_properties must be present
    """
    data_source_name: str = Field(
        ...,
        description="Logical name for this data source (e.g. 'RAW_MDS_NA' or 'TAGS')"
    )
    
    data_source_type: str = Field(
        ...,
        description="One of 'MDS', 'EDX', or 'ANDES'"
    )
    
    mds_data_source_properties: Optional[MdsDataSourceConfig] = Field(
        default=None,
        description="If data_source_type=='MDS', this must be provided"
    )
    
    edx_data_source_properties: Optional[EdxDataSourceConfig] = Field(
        default=None,
        description="If data_source_type=='EDX', this must be provided"
    )
    
    andes_data_source_properties: Optional[AndesDataSourceConfig] = Field(
        default=None,
        description="If data_source_type=='ANDES', this must be provided"
    )

    @field_validator("data_source_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed: Set[str] = {"MDS", "EDX", "ANDES"}
        if v not in allowed:
            raise ValueError(f"data_source_type must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    @classmethod
    def check_properties(cls, model: "DataSourceConfig") -> "DataSourceConfig":
        """
        Ensure the appropriate properties are set based on data_source_type
        and that only one set of properties is provided.
        """
        t = model.data_source_type
        
        # Check required properties are present
        if t == "MDS" and model.mds_data_source_properties is None:
            raise ValueError("mds_data_source_properties must be set when data_source_type=='MDS'")
        if t == "EDX" and model.edx_data_source_properties is None:
            raise ValueError("edx_data_source_properties must be set when data_source_type=='EDX'")
        if t == "ANDES" and model.andes_data_source_properties is None:
            raise ValueError("andes_data_source_properties must be set when data_source_type=='ANDES'")
            
        # Ensure only one set of properties is provided
        properties_count = sum(
            1 for prop in [
                model.mds_data_source_properties,
                model.edx_data_source_properties,
                model.andes_data_source_properties
            ] if prop is not None
        )
        
        if properties_count > 1:
            raise ValueError(
                "Only one of mds_data_source_properties, edx_data_source_properties, "
                "or andes_data_source_properties should be provided"
            )
            
        return model

    class Config:
        """Pydantic model configuration."""
        frozen = True
        extra = "forbid"


class DataSourcesSpecificationConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.datasourcesspecification.DataSourcesSpecification:
      - start_date (exact format 'YYYY-mm-DDTHH:MM:SS')
      - end_date (exact format 'YYYY-mm-DDTHH:MM:SS')
      - data_sources: list of DataSourceConfig
    """
    start_date: str = Field(
        ...,
        description="Start timestamp exactly 'YYYY-mm-DDTHH:MM:SS', e.g. '2025-01-01T00:00:00'"
    )
    end_date: str = Field(
        ...,
        description="End timestamp exactly 'YYYY-mm-DDTHH:MM:SS', e.g. '2025-04-17T00:00:00'"
    )
    data_sources: List[DataSourceConfig] = Field(
        ...,
        description="List of DataSourceConfig objects (both MDS and EDX)"
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_exact_datetime_format(cls, v: str, field) -> str:
        """
        Must match exactly "%Y-%m-%dT%H:%M:%S"
        """
        try:
            parsed = datetime.strptime(v, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            raise ValueError(
                f"{field.name!r} must be in format YYYY-mm-DD'T'HH:MM:SS "
                f"(e.g. '2025-01-01T00:00:00'), got {v!r}"
            )
        if parsed.strftime("%Y-%m-%dT%H:%M:%S") != v:
            raise ValueError(
                f"{field.name!r} does not match the required format exactly; got {v!r}"
            )
        return v


class JobSplitOptionsConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.jobsplitoptions.JobSplitOptions:
      - split_job: bool
      - days_per_split: int
      - merge_sql: str
    """
    split_job: bool = Field(
        default=False,
        description="Whether to split the Cradle job into multiple daily runs"
    )
    days_per_split: int = Field(
        default=7,
        description="Number of days per split (only used if split_job=True)"
    )
    merge_sql: Optional[str] = Field(
        default=None,
        description="SQL to run after merging split results (if split_job=True). "
                    "For example: 'SELECT * FROM INPUT'."
    )

    @field_validator("days_per_split")
    @classmethod
    def days_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("days_per_split must be ≥ 1")
        return v

    @model_validator(mode="after")
    @classmethod
    def require_merge_sql_if_split(cls, model: "JobSplitOptionsConfig") -> "JobSplitOptionsConfig":
        if model.split_job and not model.merge_sql:
            raise ValueError("If split_job=True, merge_sql must be provided")
        return model


class TransformSpecificationConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.transformspecification.TransformSpecification:
      - transform_sql: str
      - job_split_options: JobSplitOptionsConfig
    """
    transform_sql: str = Field(
        ...,
        description="The SQL string used to join MDS and TAGS (or do any other transformation)."
    )
    job_split_options: JobSplitOptionsConfig = Field(
        ...,
        description="Options for splitting the Cradle job into multiple runs"
    )


class OutputSpecificationConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.outputspecification.OutputSpecification:
      - output_schema: List[str]
      - output_path: str (S3 URI)
      - output_format: str (e.g. 'PARQUET', 'CSV', etc.)
      - output_save_mode: str (e.g. 'ERRORIFEXISTS', 'OVERWRITE', 'APPEND', 'IGNORE')
      - output_file_count: int (0 means “auto”)
      - keep_dot_in_output_schema: bool
      - include_header_in_s3_output: bool
    """
    output_schema: List[str] = Field(
        ...,
        description="List of column names to emit (e.g. ['objectId','transactionDate',…])."
    )
    output_path: str = Field(
        ...,
        description="Target S3 URI for output data (e.g. 's3://my-bucket/output-folder')"
    )
    output_format: str = Field(
        default="PARQUET",
        description="Format for Cradle output: one of ['CSV','UNESCAPED_TSV','JSON','ION','PARQUET']"
    )
    output_save_mode: str = Field(
        default="ERRORIFEXISTS",
        description="One of ['ERRORIFEXISTS','OVERWRITE','APPEND','IGNORE']"
    )
    output_file_count: int = Field(
        default=0,
        ge=0,
        description="Number of output files (0 means auto‐split)"
    )
    keep_dot_in_output_schema: bool = Field(
        default=False,
        description="If False, replace '.' with '__DOT__' in the output header"
    )
    include_header_in_s3_output: bool = Field(
        default=True,
        description="Whether to write the header row in S3 output files"
    )

    @field_validator("output_path")
    @classmethod
    def validate_s3_uri(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("output_path must start with 's3://'")
        return v

    @field_validator("output_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        allowed = {"CSV", "UNESCAPED_TSV", "JSON", "ION", "PARQUET"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return v

    @field_validator("output_save_mode")
    @classmethod
    def validate_save_mode(cls, v: str) -> str:
        allowed = {"ERRORIFEXISTS", "OVERWRITE", "APPEND", "IGNORE"}
        if v not in allowed:
            raise ValueError(f"output_save_mode must be one of {allowed}")
        return v


class CradleJobSpecificationConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.cradlejobspecification.CradleJobSpecification:
      - cluster_type: str (e.g. 'SMALL', 'MEDIUM', 'LARGE')
      - cradle_account: str
      - extra_spark_job_arguments: Optional[str]
      - job_retry_count: int
    """
    cluster_type: str = Field(
        default='STANDARD',
        description="Cluster size for Cradle job (e.g. 'STANDARD', 'SMALL', 'MEDIUM', 'LARGE')"
    )
    cradle_account: str = Field(
        ...,
        description="Cradle account name (e.g. 'Buyer-Abuse-RnD-Dev')"
    )
    extra_spark_job_arguments: Optional[str] = Field(
        default="",
        description="Any extra Spark driver options (string or blank)"
    )
    job_retry_count: int = Field(
        default=1,
        ge=0,
        description="Number of times to retry on failure (default=1)"
    )

    @field_validator("cluster_type")
    @classmethod
    def validate_cluster_type(cls, v: str) -> str:
        allowed = {"STANDARD", "SMALL", "MEDIUM", "LARGE"}
        if v not in allowed:
            raise ValueError(f"cluster_type must be one of {allowed}, got '{v}'")
        return v


class CradleDataLoadConfig(BasePipelineConfig):
    """
    Top‐level Pydantic config for creating a CreateCradleDataLoadJobRequest.

    Instead of requiring each subfield directly, the user now provides:
      - job_type: str, one of ["training","validation","test","calibration"]
      - data_sources_spec: DataSourcesSpecificationConfig
      - transform_spec: TransformSpecificationConfig
      - output_spec: OutputSpecificationConfig
      - cradle_job_spec: CradleJobSpecificationConfig
      - (optional) s3_input_override
    """
    job_type: str = Field(
        ...,
        description="One of ['training','validation','testing','calibration'] to indicate which dataset this job is pulling"
    )
    data_sources_spec: DataSourcesSpecificationConfig = Field(
        ...,
        description="Full data‐sources specification (start/end dates plus list of sources)."
    )
    transform_spec: TransformSpecificationConfig = Field(
        ...,
        description="Transform specification: SQL + job‐split options."
    )
    output_spec: OutputSpecificationConfig = Field(
        ...,
        description="Output specification: schema, path, format, save mode, etc."
    )
    cradle_job_spec: CradleJobSpecificationConfig = Field(
        ...,
        description="Cradle job specification: cluster type, account, retry count, etc."
    )
    s3_input_override: Optional[str] = Field(
        default=None,
        description="If set, skip Cradle data pull and use this S3 prefix directly"
    )

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        allowed = {"training", "validation", "testing", "calibration"}
        if v not in allowed:
            raise ValueError(f"job_type must be one of {allowed}, got '{v}'")
        return v

    @model_validator(mode="after")
    @classmethod
    def check_split_and_override(cls, model: "CradleDataLoadConfig") -> "CradleDataLoadConfig":
        # (1) If splitting is enabled, merge_sql must be provided
        if model.transform_spec.job_split_options.split_job \
           and not model.transform_spec.job_split_options.merge_sql:
            raise ValueError("When split_job=True, merge_sql must be provided")

        # (2) If user supplied s3_input_override, they can skip transform or data sources,
        #     but we don't enforce that here. We simply allow s3_input_override to bypass usage.
        #     No extra checks are necessary—downstream code should look at s3_input_override first.

        return model