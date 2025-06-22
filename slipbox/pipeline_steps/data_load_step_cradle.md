# Cradle Data Load Step

## Task Summary
The Cradle Data Load Step loads data from various sources (MDS, EDX, or ANDES) using the Cradle service. This step:

1. Configures a data loading job with specified data sources, time ranges, and transformation SQL
2. Executes the job in the Cradle service to extract and transform data
3. Outputs the processed data to S3 in the specified format
4. Optionally splits large jobs into smaller chunks for better performance

## Input and Output Format

### Input
- **Data Sources**: Configuration for MDS, EDX, or ANDES data sources
- **Time Range**: Start and end dates for data extraction
- **Transform SQL**: SQL query to transform and join data from sources
- **Optional Dependencies**: List of pipeline steps that must complete before this step runs

### Output
- **Processed Data**: Transformed data in the specified format (PARQUET, CSV, etc.) stored in S3
- **Metadata**: Information about the data load job
- **Signature**: Data signatures for verification
- **CradleDataLoadingStep**: A configured SageMaker pipeline step that can be added to a pipeline

## Configuration Parameters

### Main Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| job_type | Dataset type ('training', 'validation', 'testing', 'calibration') | Required |
| data_sources_spec | Data sources specification | Required |
| transform_spec | Transform specification | Required |
| output_spec | Output specification | Required |
| cradle_job_spec | Cradle job specification | Required |
| s3_input_override | Optional S3 prefix to use instead of Cradle data pull | None |

### Data Sources Specification
| Parameter | Description | Default |
|-----------|-------------|---------|
| start_date | Start timestamp (YYYY-MM-DDTHH:MM:SS) | Required |
| end_date | End timestamp (YYYY-MM-DDTHH:MM:SS) | Required |
| data_sources | List of data source configurations | Required |

### Data Source Configuration
| Parameter | Description | Default |
|-----------|-------------|---------|
| data_source_name | Logical name for the data source | Required |
| data_source_type | Type of data source ('MDS', 'EDX', 'ANDES') | Required |
| mds_data_source_properties | MDS-specific properties (required if type is 'MDS') | None |
| edx_data_source_properties | EDX-specific properties (required if type is 'EDX') | None |
| andes_data_source_properties | ANDES-specific properties (required if type is 'ANDES') | None |

### Transform Specification
| Parameter | Description | Default |
|-----------|-------------|---------|
| transform_sql | SQL query for data transformation | Required |
| job_split_options | Options for splitting the job | Required |

### Output Specification
| Parameter | Description | Default |
|-----------|-------------|---------|
| output_schema | List of column names to emit | Required |
| output_path | Target S3 URI for output data | Required |
| output_format | Format for output ('CSV', 'PARQUET', etc.) | PARQUET |
| output_save_mode | Save mode ('ERRORIFEXISTS', 'OVERWRITE', etc.) | ERRORIFEXISTS |
| output_file_count | Number of output files (0 = auto) | 0 |
| keep_dot_in_output_schema | Whether to keep dots in column names | False |
| include_header_in_s3_output | Whether to include header row | True |

### Cradle Job Specification
| Parameter | Description | Default |
|-----------|-------------|---------|
| cluster_type | Cluster size ('STANDARD', 'SMALL', 'MEDIUM', 'LARGE') | STANDARD |
| cradle_account | Cradle account name | Required |
| extra_spark_job_arguments | Extra Spark driver options | "" |
| job_retry_count | Number of retries on failure | 1 |

## Validation Rules
- job_type must be one of: 'training', 'validation', 'testing', 'calibration'
- start_date and end_date must be in exact format 'YYYY-MM-DDTHH:MM:SS'
- start_date must be strictly before end_date
- At least one data source must be provided
- For each data source, the appropriate properties must be set based on type
- When split_job=True, merge_sql must be provided
- output_path must be a valid S3 URI
- output_format must be one of: 'CSV', 'UNESCAPED_TSV', 'JSON', 'ION', 'PARQUET'
- output_save_mode must be one of: 'ERRORIFEXISTS', 'OVERWRITE', 'APPEND', 'IGNORE'
- cluster_type must be one of: 'STANDARD', 'SMALL', 'MEDIUM', 'LARGE'

## Usage Example
```python
from src.pipeline_steps.config_data_load_step_cradle import (
    CradleDataLoadConfig, DataSourcesSpecificationConfig, DataSourceConfig,
    MdsDataSourceConfig, TransformSpecificationConfig, JobSplitOptionsConfig,
    OutputSpecificationConfig, CradleJobSpecificationConfig
)
from src.pipeline_steps.builder_data_load_step_cradle import CradleDataLoadingStepBuilder

# Create MDS data source config
mds_config = MdsDataSourceConfig(
    service_name="MyService",
    region="NA",
    output_schema=[
        {"field_name": "objectId", "field_type": "STRING"},
        {"field_name": "transactionDate", "field_type": "TIMESTAMP"}
    ]
)

# Create data source config
data_source = DataSourceConfig(
    data_source_name="RAW_MDS_NA",
    data_source_type="MDS",
    mds_data_source_properties=mds_config
)

# Create data sources specification
data_sources_spec = DataSourcesSpecificationConfig(
    start_date="2025-01-01T00:00:00",
    end_date="2025-01-31T23:59:59",
    data_sources=[data_source]
)

# Create transform specification
transform_spec = TransformSpecificationConfig(
    transform_sql="SELECT * FROM RAW_MDS_NA",
    job_split_options=JobSplitOptionsConfig(
        split_job=False
    )
)

# Create output specification
output_spec = OutputSpecificationConfig(
    output_schema=["objectId", "transactionDate"],
    output_path="s3://my-bucket/training-data/",
    output_format="PARQUET"
)

# Create Cradle job specification
cradle_job_spec = CradleJobSpecificationConfig(
    cluster_type="STANDARD",
    cradle_account="My-Cradle-Account"
)

# Create main config
config = CradleDataLoadConfig(
    job_type="training",
    data_sources_spec=data_sources_spec,
    transform_spec=transform_spec,
    output_spec=output_spec,
    cradle_job_spec=cradle_job_spec
)

# Create builder and step
builder = CradleDataLoadingStepBuilder(config=config)
data_load_step = builder.create_step()

# Add to pipeline
pipeline.add_step(data_load_step)

# Get output locations
outputs = builder.get_step_outputs(data_load_step)
data_location = outputs["DATA"]
```
