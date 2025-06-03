from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field, validator

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
        ...,
        description="Organization ID (integer) for MDS"
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

    @validator("region")
    def validate_region(cls, v: str) -> str:
        valid = {"NA", "EU", "FE"}
        if v not in valid:
            raise ValueError(f"region must be one of {valid}, got '{v}'")
        return v


class EdxDataSourceConfig(BaseModel):
    """
    Corresponds to EdxDataSourceProperties:
      - edx_arn: string ARN for the EDX manifest
      - schema_overrides: list of Field-like dicts (each with field_name & field_type)
    """
    edx_arn: str = Field(
        ...,
        description="ARN string for the EDX manifest (e.g. 'arn:amazon:edx:…')"
    )
    schema_overrides: List[Dict[str, Any]] = Field(
        ...,
        description="List of dicts overriding the EDX schema, "
                    "e.g. [{'field_name':'order_id','field_type':'STRING'}, …]"
    )


class DataSourceConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.datasource.DataSource:
      - data_source_name: e.g. 'RAW_MDS_NA' or 'TAGS'
      - data_source_type: either 'MDS' or 'EDX'
      - one of mds_data_source_properties or edx_data_source_properties must be present
    """
    data_source_name: str = Field(
        ...,
        description="Logical name for this data source (e.g. 'RAW_MDS_NA' or 'TAGS')"
    )
    data_source_type: str = Field(
        ...,
        description="Either 'MDS' or 'EDX'"
    )
    mds_data_source_properties: Optional[MdsDataSourceConfig] = Field(
        default=None,
        description="If data_source_type=='MDS', this must be provided"
    )
    edx_data_source_properties: Optional[EdxDataSourceConfig] = Field(
        default=None,
        description="If data_source_type=='EDX', this must be provided"
    )

    @validator("data_source_type")
    def validate_type(cls, v: str) -> str:
        allowed = {"MDS", "EDX"}
        if v not in allowed:
            raise ValueError(f"data_source_type must be one of {allowed}, got '{v}'")
        return v

    @validator("mds_data_source_properties", always=True)
    def check_mds_props_for_mds_type(cls, v, values):
        t = values.get("data_source_type")
        if t == "MDS" and v is None:
            raise ValueError("mds_data_source_properties must be set when data_source_type=='MDS'")
        return v

    @validator("edx_data_source_properties", always=True)
    def check_edx_props_for_edx_type(cls, v, values):
        t = values.get("data_source_type")
        if t == "EDX" and v is None:
            raise ValueError("edx_data_source_properties must be set when data_source_type=='EDX'")
        return v


class DataSourcesSpecificationConfig(BaseModel):
    """
    Corresponds to com.amazon.secureaisandboxproxyservice.models.datasourcesspecification.DataSourcesSpecification:
      - start_date (ISO string)
      - end_date (ISO string)
      - data_sources: list of DataSourceConfig
    """
    start_date: str = Field(
        ...,
        description="ISO‐8601 start date/time for data pull (e.g. '2025-01-01T00:00:00')"
    )
    end_date: str = Field(
        ...,
        description="ISO‐8601 end date/time for data pull (e.g. '2025-04-17T00:00:00')"
    )
    data_sources: List[DataSourceConfig] = Field(
        ...,
        description="List of DataSourceConfig objects (both MDS and EDX)"
    )

    @validator("start_date", "end_date")
    def validate_iso8601(cls, v: str) -> str:
        # A quick check: must parse as datetime
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"'{v}' is not a valid ISO‐8601 timestamp")
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

    @validator("days_per_split")
    def days_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("days_per_split must be ≥ 1")
        return v

    @validator("merge_sql", always=True)
    def require_merge_sql_if_split(cls, v, values):
        if values.get("split_job") and not v:
            raise ValueError("If split_job=True, merge_sql must be provided")
        return v


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

    @validator("output_path")
    def validate_s3_uri(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("output_path must start with 's3://'")
        return v

    @validator("output_format")
    def validate_format(cls, v: str) -> str:
        allowed = {"CSV", "UNESCAPED_TSV", "JSON", "ION", "PARQUET"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return v

    @validator("output_save_mode")
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
        ...,
        description="Cluster size for Cradle job (e.g. 'SMALL', 'MEDIUM', 'LARGE')"
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

    @validator("cluster_type")
    def validate_cluster_type(cls, v: str) -> str:
        allowed = {"SMALL", "MEDIUM", "LARGE"}
        if v not in allowed:
            raise ValueError(f"cluster_type must be one of {allowed}, got '{v}'")
        return v


class CradleDataLoadConfig(BasePipelineConfig):
    """
    Top‐level Pydantic config for creating a CreateCradleDataLoadJobRequest.
    In addition to BasePipelineConfig fields (bucket, author, etc.), it defines:
      - mds_source: MdsDataSourceConfig
      - tag_source: EdxDataSourceConfig
      - start_date, end_date
      - transform_sql, job_split_options
      - output_schema, output_path, output_format, output_save_mode, ...
      - cluster_type, cradle_account, extra_spark_job_arguments, job_retry_count
    """
    # 1) MDS + EDX source configs
    mds_source: MdsDataSourceConfig = Field(
        ...,
        description="Configuration for the MDS data source"
    )
    tag_source: EdxDataSourceConfig = Field(
        ...,
        description="Configuration for the EDX (tag) data source"
    )

    # 2) DataSourcesSpecification fields
    start_date: str = Field(
        ...,
        description="ISO‐8601 start timestamp (e.g. '2025-01-01T00:00:00')"
    )
    end_date: str = Field(
        ...,
        description="ISO‐8601 end timestamp (e.g. '2025-04-17T00:00:00')"
    )

    # 3) TransformSpecification fields
    transform_sql: str = Field(
        ...,
        description="SQL string that joins MDS and TAG tables"
    )
    split_job: bool = Field(
        default=False,
        description="Whether to split the Cradle job into daily runs"
    )
    days_per_split: int = Field(
        default=7,
        description="Number of days per split if split_job=True"
    )
    merge_sql: Optional[str] = Field(
        default=None,
        description="SQL used to merge split outputs if split_job=True (e.g. 'SELECT * FROM INPUT')"
    )

    # 4) OutputSpecification fields
    output_schema: List[str] = Field(
        ...,
        description="List of output columns to write (matching transform_sql SELECT clause)"
    )
    output_path: str = Field(
        ...,
        description="S3 URI where Cradle should write its output (e.g. 's3://my-bucket/output/…')"
    )
    output_format: str = Field(
        default="PARQUET",
        description="Cradle output format (one of 'CSV','UNESCAPED_TSV','JSON','ION','PARQUET')"
    )
    output_save_mode: str = Field(
        default="ERRORIFEXISTS",
        description="Cradle save mode (one of 'ERRORIFEXISTS','OVERWRITE','APPEND','IGNORE')"
    )
    output_file_count: int = Field(
        default=0,
        ge=0,
        description="Number of files to produce (0 means auto)"
    )
    keep_dot_in_output_schema: bool = Field(
        default=False,
        description="If False, Cradle replaces '.' with '__DOT__' in output column names"
    )
    include_header_in_s3_output: bool = Field(
        default=True,
        description="Whether to write header row in S3 output files"
    )

    # 5) CradleJobSpecification fields
    cluster_type: str = Field(
        ...,
        description="Cluster size for Cradle job (one of 'SMALL','MEDIUM','LARGE')"
    )
    cradle_account: str = Field(
        ...,
        description="Cradle account name (e.g. 'Buyer-Abuse-RnD-Dev')"
    )
    extra_spark_job_arguments: Optional[str] = Field(
        default="",
        description="Any extra Spark driver arguments (string or blank)"
    )
    job_retry_count: int = Field(
        default=1,
        ge=0,
        description="Number of times to retry the Cradle job on failure (≥0)"
    )

    # 6) (Optional) An override for s3 path if you already have raw data downloaded
    s3_input_override: Optional[str] = Field(
        default=None,
        description="If set, skip Cradle data pull and use this S3 prefix directly"
    )

    @validator("start_date", "end_date")
    def check_iso_format(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            raise ValueError(f"{v!r} is not a valid ISO‐8601 timestamp")
        return v

    @validator("merge_sql", always=True)
    def require_merge_if_split(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if values.get("split_job") and not v:
            raise ValueError("When split_job=True, merge_sql must be provided")
        return v

    @validator("output_path")
    def validate_output_path_s3(cls, v: str) -> str:
        if not v.startswith("s3://"):
            raise ValueError("output_path must start with 's3://'")
        return v

    @validator("output_format")
    def validate_output_format(cls, v: str) -> str:
        allowed = {"CSV", "UNESCAPED_TSV", "JSON", "ION", "PARQUET"}
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}")
        return v

    @validator("output_save_mode")
    def validate_output_save_mode(cls, v: str) -> str:
        allowed = {"ERRORIFEXISTS", "OVERWRITE", "APPEND", "IGNORE"}
        if v not in allowed:
            raise ValueError(f"output_save_mode must be one of {allowed}")
        return v

    @validator("cluster_type")
    def validate_cluster_type(cls, v: str) -> str:
        allowed = {"SMALL", "MEDIUM", "LARGE"}
        if v not in allowed:
            raise ValueError(f"cluster_type must be one of {allowed}")
        return v