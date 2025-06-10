from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
from datetime import datetime

import os
import importlib
from dotenv import load_dotenv
# Load environment variables from .env file
# This will search for a .env file in the current directory or parent directories.
# If your .env file is in `pipelines/.env` and this script is run from `pipelines/`
# or `root/`, it should find it.
# For robustness, you can specify the path: load_dotenv(dotenv_path=Path('.') / '.env')
# or ensure your execution environment (like a SageMaker container) has these vars set.
load_dotenv()


logger = logging.getLogger(__name__)


# --- Dynamically importing sensitive classes/modules ---

# Group 1: com.amazon.secureaisandboxproxyservice.models
SECUREAI_PROXY_MODELS_BASE_PATH = os.environ.get("SECUREAI_PROXY_MODELS_BASE")
Field = None
DataSource = None
MdsDataSourceProperties = None
EdxDataSourceProperties = None
DataSourcesSpecification = None
JobSplitOptions = None
TransformSpecification = None
OutputSpecification = None
CradleJobSpecification = None
CreateCradleDataLoadJobRequest = None

if SECUREAI_PROXY_MODELS_BASE_PATH:
    try:
        # Attempt to import the main 'models' module first
        # The original imports suggest classes are in submodules like 'field', 'datasource'
        # e.g., from com.amazon.secureaisandboxproxyservice.models.field import Field

        field_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.field")
        Field = field_module.Field

        datasource_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.datasource")
        DataSource = datasource_module.DataSource

        mdsprops_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.mdsdatasourceproperties")
        MdsDataSourceProperties = mdsprops_module.MdsDataSourceProperties

        edxprops_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.edxdatasourceproperties")
        EdxDataSourceProperties = edxprops_module.EdxDataSourceProperties
        
        dsspec_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.datasourcesspecification")
        DataSourcesSpecification = dsspec_module.DataSourcesSpecification

        jsopts_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.jobsplitoptions")
        JobSplitOptions = jsopts_module.JobSplitOptions

        transspec_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.transformspecification")
        TransformSpecification = transspec_module.TransformSpecification

        outspec_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.outputspecification")
        OutputSpecification = outspec_module.OutputSpecification

        cradlejobspec_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.cradlejobspecification")
        CradleJobSpecification = cradlejobspec_module.CradleJobSpecification

        createcradle_module = importlib.import_module(f"{SECUREAI_PROXY_MODELS_BASE_PATH}.createcradledataloadjobrequest")
        CreateCradleDataLoadJobRequest = createcradle_module.CreateCradleDataLoadJobRequest
        
        logger.info("Successfully imported classes from SECUREAI_PROXY_MODELS_BASE.")
    except ImportError as e:
        logger.error(f"Could not import one or more modules from base '{SECUREAI_PROXY_MODELS_BASE_PATH}': {e}")
    except AttributeError as e:
        logger.error(f"Attribute error while importing from SECUREAI_PROXY_MODELS_BASE: {e}")
else:
    logger.warning("SECUREAI_PROXY_MODELS_BASE environment variable not set. Secure AI Proxy Service models will not be available.")


# Group 2: secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step
SECUREAI_CRADLE_LOADING_STEP_MODULE_PATH = os.environ.get("SECUREAI_CRADLE_LOADING_STEP_MODULE")
CradleDataLoadingStep = None
if SECUREAI_CRADLE_LOADING_STEP_MODULE_PATH:
    try:
        cradle_step_module = importlib.import_module(SECUREAI_CRADLE_LOADING_STEP_MODULE_PATH)
        CradleDataLoadingStep = cradle_step_module.CradleDataLoadingStep
        logger.info(f"Successfully imported CradleDataLoadingStep from {SECUREAI_CRADLE_LOADING_STEP_MODULE_PATH}.")
    except ImportError:
        logger.error(f"Could not import module '{SECUREAI_CRADLE_LOADING_STEP_MODULE_PATH}' for CradleDataLoadingStep.")
    except AttributeError:
        logger.error(f"'CradleDataLoadingStep' not found in module '{SECUREAI_CRADLE_LOADING_STEP_MODULE_PATH}'.")
else:
    logger.warning("SECUREAI_CRADLE_LOADING_STEP_MODULE environment variable not set. CradleDataLoadingStep will not be available.")


# Group 3: secure_ai_sandbox_python_lib.utils (importing coral_utils)
SECUREAI_LIB_UTILS_MODULE_PATH = os.environ.get("SECUREAI_LIB_UTILS_MODULE")
coral_utils = None
if SECUREAI_LIB_UTILS_MODULE_PATH:
    try:
        utils_module = importlib.import_module(SECUREAI_LIB_UTILS_MODULE_PATH)
        coral_utils = utils_module.coral_utils # Assuming coral_utils is an attribute (object/submodule)
        logger.info(f"Successfully imported coral_utils from {SECUREAI_LIB_UTILS_MODULE_PATH}.")
    except ImportError:
        logger.error(f"Could not import utils module: {SECUREAI_LIB_UTILS_MODULE_PATH}")
    except AttributeError:
        logger.error(f"'coral_utils' not found in module '{SECUREAI_LIB_UTILS_MODULE_PATH}'.")
else:
    logger.warning("SECUREAI_LIB_UTILS_MODULE environment variable not set. coral_utils will not be available.")


from .config_data_load_step_cradle import (
    CradleDataLoadConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    DataSourcesSpecificationConfig,
    DataSourceConfig as InnerDataSourceConfig,
    TransformSpecificationConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig,
)
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)


class CradleDataLoadingStepBuilder(StepBuilderBase):
    """
    Builder for a CradleDataLoadingStep. Takes a CradleDataLoadConfig, validates it,
    constructs a CreateCradleDataLoadJobRequest, and returns a CradleDataLoadingStep.
    """

    def __init__(
        self,
        config: CradleDataLoadConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Args:
            config: An instance of CradleDataLoadConfig (Pydantic‐validated).
            sagemaker_session: A SageMaker PipelineSession (optional).
            role: IAM role ARN for the Cradle Data Loading job.
            notebook_root: If running locally, used to validate local paths.
        """
        if not isinstance(config, CradleDataLoadConfig):
            raise ValueError("CradleDataLoadingStepBuilder requires a CradleDataLoadConfig instance.")
        super().__init__(config=config, sagemaker_session=sagemaker_session, role=role, notebook_root=notebook_root)
        self.config: CradleDataLoadConfig = config

    def validate_configuration(self) -> None:
        """
        Called by StepBuilderBase.__init__(). Ensures required fields are set
        and in the correct format.

        In particular:
          - job_type ∈ {'training','validation','testing'}
          - At least one data source in data_sources_spec
          - Each MDS/EDX config is present if indicated
          - start_date and end_date must exactly match 'YYYY-mm-DDTHH:MM:SS'
          - start_date < end_date
        """
        logger.info("Validating CradleDataLoadConfig…")

        # (1) job_type is already validated by Pydantic, but double-check presence:
        if not self.config.job_type:
            raise ValueError("job_type must be provided (e.g. 'training','validation','testing').")

        # (2) data_sources_spec must have at least one entry
        ds_list = self.config.data_sources_spec.data_sources
        if not ds_list or len(ds_list) == 0:
            raise ValueError("At least one DataSourceConfig must be provided in data_sources_spec.data_sources")

        # (3) For each data_source, check that required subfields are present
        for idx, ds_cfg in enumerate(ds_list):
            if ds_cfg.data_source_type == "MDS":
                mds_props: MdsDataSourceConfig = ds_cfg.mds_data_source_properties  # type: ignore
                if mds_props is None:
                    raise ValueError(f"DataSource #{idx} is MDS but mds_data_source_properties was not provided.")
                # MdsDataSourceConfig validators have already run.
            elif ds_cfg.data_source_type == "EDX":
                edx_props: EdxDataSourceConfig = ds_cfg.edx_data_source_properties  # type: ignore
                if edx_props is None:
                    raise ValueError(f"DataSource #{idx} is EDX but edx_data_source_properties was not provided.")
                # Check EDX manifest
                if not edx_props.edx_manifest:
                    raise ValueError(f"DataSource #{idx} EDX manifest must be a nonempty string.")
            else:
                raise ValueError(f"DataSource #{idx} has invalid type: {ds_cfg.data_source_type}")

        # (4) Check that start_date & end_date match exact format YYYY-mm-DDTHH:MM:SS
        for field_name in ("start_date", "end_date"):
            value = getattr(self.config.data_sources_spec, field_name)
            try:
                parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
            except Exception:
                raise ValueError(
                    f"'{field_name}' must be in format YYYY-mm-DD'T'HH:MM:SS "
                    f"(e.g. '2025-01-01T00:00:00'), got: {value!r}"
                )
            if parsed.strftime("%Y-%m-%dT%H:%M:%S") != value:
                raise ValueError(
                    f"'{field_name}' does not match the required format exactly; got {value!r}"
                )

        # (5) Also ensure start_date < end_date
        s = datetime.strptime(self.config.data_sources_spec.start_date, "%Y-%m-%dT%H:%M:%S")
        e = datetime.strptime(self.config.data_sources_spec.end_date, "%Y-%m-%dT%H:%M:%S")
        if s >= e:
            raise ValueError("start_date must be strictly before end_date.")

        # (6) Everything else (output_path S3 URI, output_format, cluster_type, etc.) 
        #     is validated by Pydantic already.

        logger.info("CradleDataLoadConfig validation succeeded.")

    def _build_request(self) -> CreateCradleDataLoadJobRequest:
        """
        Convert self.config → a CreateCradleDataLoadJobRequest instance under the hood.
        """

        # (a) Build each DataSource from data_sources_spec.data_sources
        data_source_models: List[DataSource] = []
        for ds_cfg in self.config.data_sources_spec.data_sources:
            if ds_cfg.data_source_type == "MDS":
                mds_props_cfg: MdsDataSourceConfig = ds_cfg.mds_data_source_properties  # type: ignore
                mds_props = MdsDataSourceProperties(
                    service_name=mds_props_cfg.service_name,
                    org_id=mds_props_cfg.org_id,
                    region=mds_props_cfg.region,
                    output_schema=[
                        Field(field_name=f["field_name"], field_type=f["field_type"])
                        for f in mds_props_cfg.output_schema
                    ],
                    use_hourly_edx_data_set=mds_props_cfg.use_hourly_edx_data_set,
                )
                data_source_models.append(
                    DataSource(
                        data_source_name=ds_cfg.data_source_name,
                        data_source_type="MDS",
                        mds_data_source_properties=mds_props,
                        edx_data_source_properties=None,
                    )
                )

            elif ds_cfg.data_source_type == "EDX":
                edx_props_cfg: EdxDataSourceConfig = ds_cfg.edx_data_source_properties  # type: ignore
                edx_props = EdxDataSourceProperties(
                    edx_arn=edx_props_cfg.edx_manifest,
                    schema_overrides=[
                        Field(field_name=f["field_name"], field_type=f["field_type"])
                        for f in edx_props_cfg.schema_overrides
                    ],
                )
                data_source_models.append(
                    DataSource(
                        data_source_name=ds_cfg.data_source_name,
                        data_source_type="EDX",
                        mds_data_source_properties=None,
                        edx_data_source_properties=edx_props,
                    )
                )

        # (b) DataSourcesSpecification
        ds_spec_cfg: DataSourcesSpecificationConfig = self.config.data_sources_spec
        data_sources_spec = DataSourcesSpecification(
            start_date=ds_spec_cfg.start_date,
            end_date=ds_spec_cfg.end_date,
            data_sources=data_source_models,
        )

        # (c) TransformSpecification
        transform_spec_cfg: TransformSpecificationConfig = self.config.transform_spec
        jso = transform_spec_cfg.job_split_options
        split_opts = JobSplitOptions(
            split_job=jso.split_job,
            days_per_split=jso.days_per_split,
            merge_sql=jso.merge_sql or "",
        )
        transform_spec = TransformSpecification(
            transform_sql=transform_spec_cfg.transform_sql,
            job_split_options=split_opts,
        )

        # (d) OutputSpecification
        output_spec_cfg: OutputSpecificationConfig = self.config.output_spec
        output_spec = OutputSpecification(
            output_schema=output_spec_cfg.output_schema,
            output_path=output_spec_cfg.output_path,
            output_format=output_spec_cfg.output_format,
            output_save_mode=output_spec_cfg.output_save_mode,
            output_file_count=output_spec_cfg.output_file_count,
            keep_dot_in_output_schema=output_spec_cfg.keep_dot_in_output_schema,
            include_header_in_s3_output=output_spec_cfg.include_header_in_s3_output,
        )

        # (e) CradleJobSpecification
        cradle_job_spec_cfg: CradleJobSpecificationConfig = self.config.cradle_job_spec
        cradle_job_spec = CradleJobSpecification(
            cluster_type=cradle_job_spec_cfg.cluster_type,
            cradle_account=cradle_job_spec_cfg.cradle_account,
            extra_spark_job_arguments=cradle_job_spec_cfg.extra_spark_job_arguments or "",
            job_retry_count=cradle_job_spec_cfg.job_retry_count,
        )

        # (f) Build the final CreateCradleDataLoadJobRequest
        request = CreateCradleDataLoadJobRequest(
            data_sources=data_sources_spec,
            transform_specification=transform_spec,
            output_specification=output_spec,
            cradle_job_specification=cradle_job_spec,
        )

        return request

    def create_step(
        self,
        dependencies: Optional[List] = None
    ) -> CradleDataLoadingStep:
        """
        Build the CradleDataLoadingStep.  
        This step will run a Data Load job in the Cradle service using the parameters
        from self.config. Returns the fully‐configured CradleDataLoadingStep.
        """
        logger.info("Creating CradleDataLoadingStep…")

        # Build the CreateCradleDataLoadJobRequest
        request = self._build_request()
        request_dict = coral_utils.convert_coral_to_dict(request)
        logger.debug("CradleDataLoad request (dict): %s", request_dict)

        # Instantiate the actual CradleDataLoadingStep
        # Append capitalized job_type (e.g. "Training", "Validation", "Test") to distinguish steps
        job_type_cap = self.config.job_type.capitalize()
        step_name = f"{self._get_step_name('CradleDataLoading')}-{job_type_cap}"

        step = CradleDataLoadingStep(
            step_name=step_name,
            role=self.role,
            sagemaker_session=self.session
        )

        logger.info("Created CradleDataLoadingStep with name: %s", step.name)
        return step

    def get_request_dict(self) -> Dict[str, Any]:
        """
        Return the CradleDataLoad request as a plain Python dict
        (after converting via coral_utils).
        Useful for logging or for passing to StepOperator.
        """
        request = self._build_request()
        return coral_utils.convert_coral_to_dict(request)
