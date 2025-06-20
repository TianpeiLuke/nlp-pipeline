from typing import Optional, Dict, Any, List
from pathlib import Path
import os, importlib
import logging
from dotenv import load_dotenv
from datetime import datetime

# Pull in .env or environment variables
load_dotenv()
logger = logging.getLogger(__name__)

# --- Dynamically import the Coral utilities ---
SECUREAI_LIB_UTILS_MODULE_PATH = os.environ.get("SECUREAI_LIB_UTILS_MODULE")
coral_utils = None
if SECUREAI_LIB_UTILS_MODULE_PATH:
    try:
        utils_module = importlib.import_module(SECUREAI_LIB_UTILS_MODULE_PATH)
        coral_utils = utils_module.coral_utils
        logger.info(f"Imported coral_utils from {SECUREAI_LIB_UTILS_MODULE_PATH}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Could not import coral_utils: {e}")
else:
    logger.warning("SECUREAI_LIB_UTILS_MODULE not set; coral_utils unavailable.")

# --- Dynamically import the Cradle service models ---
SECUREAI_PROXY_MODELS_BASE = os.environ.get("SECUREAI_PROXY_MODELS_BASE")
Field = DataSource = MdsDataSourceProperties = EdxDataSourceProperties = None
DataSourcesSpecification = JobSplitOptions = TransformSpecification = None
OutputSpecification = CradleJobSpecification = CreateCradleDataLoadJobRequest = None

if SECUREAI_PROXY_MODELS_BASE:
    try:
        mod = SECUREAI_PROXY_MODELS_BASE
        Field = importlib.import_module(f"{mod}.field").Field
        DataSource = importlib.import_module(f"{mod}.datasource").DataSource
        MdsDataSourceProperties = importlib.import_module(f"{mod}.mdsdatasourceproperties").MdsDataSourceProperties
        EdxDataSourceProperties = importlib.import_module(f"{mod}.edxdatasourceproperties").EdxDataSourceProperties
        AndesDataSourceProperties = importlib.import_module(f"{mod}.andesdatasourceproperties").AndesDataSourceProperties
        DataSourcesSpecification = importlib.import_module(f"{mod}.datasourcesspecification").DataSourcesSpecification
        JobSplitOptions = importlib.import_module(f"{mod}.jobsplitoptions").JobSplitOptions
        TransformSpecification = importlib.import_module(f"{mod}.transformspecification").TransformSpecification
        OutputSpecification = importlib.import_module(f"{mod}.outputspecification").OutputSpecification
        CradleJobSpecification = importlib.import_module(f"{mod}.cradlejobspecification").CradleJobSpecification
        CreateCradleDataLoadJobRequest = importlib.import_module(f"{mod}.createcradledataloadjobrequest").CreateCradleDataLoadJobRequest
        logger.info(f"Imported Cradle proxy models from {mod}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed dynamic import of Cradle models from {mod}: {e}")
else:
    logger.warning("SECUREAI_PROXY_MODELS_BASE not set; CradleDataLoadRequest classes unavailable.")

# --- Dynamically import the actual CradleDataLoadingStep class ---
SECUREAI_CRADLE_LOADING_STEP_MODULE = os.environ.get("SECUREAI_CRADLE_LOADING_STEP_MODULE")
CradleDataLoadingStep = None
if SECUREAI_CRADLE_LOADING_STEP_MODULE:
    try:
        step_mod = importlib.import_module(SECUREAI_CRADLE_LOADING_STEP_MODULE)
        CradleDataLoadingStep = step_mod.CradleDataLoadingStep
        logger.info(f"Imported CradleDataLoadingStep from {SECUREAI_CRADLE_LOADING_STEP_MODULE}")
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed import of CradleDataLoadingStep from {SECUREAI_CRADLE_LOADING_STEP_MODULE}: {e}")
else:
    logger.warning("SECUREAI_CRADLE_LOADING_STEP_MODULE not set; CradleDataLoadingStep unavailable.")

SECUREAI_PIPELINE_CONSTANTS_MODULE = os.environ.get("SECUREAI_PIPELINE_CONSTANTS_MODULE")
OUTPUT_TYPE_DATA = OUTPUT_TYPE_METADATA = OUTPUT_TYPE_SIGNATURE = None
if SECUREAI_PIPELINE_CONSTANTS_MODULE:
    try:
        const_mod = importlib.import_module(SECUREAI_PIPELINE_CONSTANTS_MODULE)
        OUTPUT_TYPE_DATA      = getattr(const_mod, "OUTPUT_TYPE_DATA",      None)
        OUTPUT_TYPE_METADATA  = getattr(const_mod, "OUTPUT_TYPE_METADATA",  None)
        OUTPUT_TYPE_SIGNATURE = getattr(const_mod, "OUTPUT_TYPE_SIGNATURE", None)
        logger.info(f"Imported pipeline constants from {SECUREAI_PIPELINE_CONSTANTS_MODULE}")
    except ImportError as e:
        logger.error(f"Could not import pipeline constants: {e}")
else:
    logger.warning(
        "SECUREAI_PIPELINE_CONSTANTS_MODULE not set; "
        "pipeline constants (DATA, METADATA, SIGNATURE) unavailable."
    )


from .config_data_load_step_cradle import (
    CradleDataLoadConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    AndesDataSourceConfig,
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
          - job_type ∈ {'training','validation','testing','calibration'}
          - At least one data source in data_sources_spec
          - Each MDS/EDX/ANDES config is present if indicated
          - start_date and end_date must exactly match 'YYYY-mm-DDTHH:MM:SS'
          - start_date < end_date
        """
        logger.info("Validating CradleDataLoadConfig…")

        # (1) job_type is already validated by Pydantic, but double-check presence:
        valid_job_types = {'training', 'validation', 'testing', 'calibration'}
        if not self.config.job_type:
            raise ValueError("job_type must be provided (e.g. 'training','validation','testing','calibration').")
        if self.config.job_type.lower() not in valid_job_types:
            raise ValueError(f"job_type must be one of: {valid_job_types}")


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
            elif ds_cfg.data_source_type == "ANDES":
                andes_props: AndesDataSourceConfig = ds_cfg.andes_data_source_properties  # type: ignore
                if andes_props is None:
                    raise ValueError(f"DataSource #{idx} is ANDES but andes_data_source_properties was not provided.")
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
            elif ds_cfg.data_source_type == "ANDES":
                andes_props_cfg: AndesDataSourceConfig = ds_cfg.andes_data_source_properties  # type: ignore
                if andes_props_cfg.andes3_enabled:
                    logger.warning(f"ANDES 3.0 is enabled for table {andes_props_cfg.table_name}")
                andes_props = AndesDataSourceProperties(
                    provider=andes_props_cfg.provider,
                    table_name=andes_props_cfg.table_name,
                    andes3_enabled=andes_props_cfg.andes3_enabled,
                )
                data_source_models.append(
                    DataSource(
                        data_source_name=ds_cfg.data_source_name,
                        data_source_type="ANDES",
                        mds_data_source_properties=None,
                        edx_data_source_properties=None,
                        andes_data_source_properties=andes_props,
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

    def create_cradle_data_loading_step(
        self,
        dependencies: Optional[List] = None
    ) -> CradleDataLoadingStep:
        """Backwards compatible method for creating the data loading step."""
        logger.warning("create_cradle_data_loading_step is deprecated, use create_step instead.")
        return self.create_step(dependencies)
    
    def get_request_dict(self) -> Dict[str, Any]:
        """
        Return the CradleDataLoad request as a plain Python dict
        (after converting via coral_utils).
        Useful for logging or for passing to StepOperator.
        """
        request = self._build_request()
        return coral_utils.convert_coral_to_dict(request)

    def get_step_outputs(self, step: CradleDataLoadingStep) -> Dict[str, str]:
        """
        Get the output locations from a created CradleDataLoadingStep.

        Args:
            step (CradleDataLoadingStep): The CradleDataLoadingStep created by this builder

        Returns:
            Dict[str, str]: Dictionary mapping output types to their S3 locations:
                          - OUTPUT_TYPE_DATA: Main data output location
                          - OUTPUT_TYPE_METADATA: Metadata output location
                          - OUTPUT_TYPE_SIGNATURE: Signature output location

        Raises:
            ValueError: If the step is not a CradleDataLoadingStep instance or 
                      wasn't created by this builder

        Example:
            ```python
            builder = CradleDataLoadingStepBuilder(config)
            step = builder.create_step()
            outputs = builder.get_step_outputs(step)
            data_location = outputs[OUTPUT_TYPE_DATA]
            metadata_location = outputs[OUTPUT_TYPE_METADATA]
            signature_location = outputs[OUTPUT_TYPE_SIGNATURE]
            ```
        """
        if not isinstance(step, CradleDataLoadingStep):
            raise ValueError("Argument must be a CradleDataLoadingStep instance")

        if step.name != f"{self._get_step_name('CradleDataLoading')}-{self.config.job_type.capitalize()}":
            raise ValueError("Step was not created by this builder")

        # Get all output locations from the step
        output_locations = step.get_output_locations()
        
        if not output_locations:
            raise ValueError("No output locations found in the step")

        # Validate that all required output types are present
        required_outputs = {OUTPUT_TYPE_DATA, OUTPUT_TYPE_METADATA, OUTPUT_TYPE_SIGNATURE}
        missing_outputs = required_outputs - set(output_locations.keys())
        if missing_outputs:
            raise ValueError(f"Missing required output types: {missing_outputs}")

        return output_locations
