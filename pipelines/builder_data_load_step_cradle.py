from typing import Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime

from com.amazon.secureaisandboxproxyservice.models.field import Field
from com.amazon.secureaisandboxproxyservice.models.datasource import DataSource
from com.amazon.secureaisandboxproxyservice.models.mdsdatasourceproperties import MdsDataSourceProperties
from com.amazon.secureaisandboxproxyservice.models.edxdatasourceproperties import EdxDataSourceProperties
from com.amazon.secureaisandboxproxyservice.models.datasourcesspecification import DataSourcesSpecification
from com.amazon.secureaisandboxproxyservice.models.jobsplitoptions import JobSplitOptions
from com.amazon.secureaisandboxproxyservice.models.transformspecification import TransformSpecification
from com.amazon.secureaisandboxproxyservice.models.outputspecification import OutputSpecification
from com.amazon.secureaisandboxproxyservice.models.cradlejobspecification import CradleJobSpecification
from com.amazon.secureaisandboxproxyservice.models.createcradledataloadjobrequest import CreateCradleDataLoadJobRequest

from secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step import (
    CradleDataLoadingStep,
)
from secure_ai_sandbox_python_lib.utils import coral_utils

from .config_data_load_step_cradle import (
    CradleDataLoadConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig
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
          - job_type ∈ {'training','validation','test'}
          - mds_source & tag_source must exist
          - start_date and end_date must exactly match 'YYYY-mm-DDTHH:MM:SS'
          - start_date < end_date
        """
        logger.info("Validating CradleDataLoadConfig…")

        # (1) job_type is already validated by Pydantic, but double-check presence:
        if not self.config.job_type:
            raise ValueError("job_type must be provided (e.g. 'training','validation','test').")

        # (2) MDS & EDX source must both exist
        if not self.config.mds_source:
            raise ValueError("mds_source must be provided.")
        if not self.config.tag_source:
            raise ValueError("tag_source must be provided.")

        # (3) Verify that the EDX ARN is a nonempty string
        if not self.config.tag_source.edx_arn:
            raise ValueError("tag_source.edx_arn must be provided.")

        # (4) Check that start_date & end_date match exact format YYYY-mm-DDTHH:MM:SS
        for field_name in ("start_date", "end_date"):
            value = getattr(self.config, field_name)
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
        s = datetime.strptime(self.config.start_date, "%Y-%m-%dT%H:%M:%S")
        e = datetime.strptime(self.config.end_date, "%Y-%m-%dT%H:%M:%S")
        if s >= e:
            raise ValueError("start_date must be strictly before end_date.")

        # (6) Everything else (S3 URI, output_format, cluster_type, etc.) is validated by Pydantic already.

        logger.info("CradleDataLoadConfig validation succeeded.")

    def _build_request(self) -> CreateCradleDataLoadJobRequest:
        """
        Convert self.config → a CreateCradleDataLoadJobRequest instance under the hood.
        """

        # (a) MDS DataSource
        mds_cfg: MdsDataSourceConfig = self.config.mds_source
        mds_props = MdsDataSourceProperties(
            service_name=mds_cfg.service_name,
            org_id=mds_cfg.org_id,
            region=mds_cfg.region,
            output_schema=[Field(field_name=f["field_name"], field_type=f["field_type"]) for f in mds_cfg.output_schema],
            use_hourly_edx_data_set=mds_cfg.use_hourly_edx_data_set,
        )
        mds_data_source = DataSource(
            data_source_name=f"RAW_MDS_{mds_cfg.region}",
            data_source_type="MDS",
            mds_data_source_properties=mds_props,
        )

        # (b) EDX (tag) DataSource
        edx_cfg: EdxDataSourceConfig = self.config.tag_source
        edx_props = EdxDataSourceProperties(
            edx_arn=edx_cfg.edx_arn,
            schema_overrides=[Field(field_name=f["field_name"], field_type=f["field_type"]) for f in edx_cfg.schema_overrides],
        )
        tag_data_source = DataSource(
            data_source_name="TAGS",
            data_source_type="EDX",
            edx_data_source_properties=edx_props,
        )

        # (c) DataSourcesSpecification
        data_sources_spec = DataSourcesSpecification(
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            data_sources=[mds_data_source, tag_data_source],
        )

        # (d) TransformSpecification
        split_opts = JobSplitOptions(
            split_job=self.config.split_job,
            days_per_split=self.config.days_per_split,
            merge_sql=self.config.merge_sql or "",
        )
        transform_spec = TransformSpecification(
            transform_sql=self.config.transform_sql,
            job_split_options=split_opts,
        )

        # (e) OutputSpecification
        output_spec = OutputSpecification(
            output_schema=self.config.output_schema,
            output_path=self.config.output_path,
            output_format=self.config.output_format,
            output_save_mode=self.config.output_save_mode,
            output_file_count=self.config.output_file_count,
            keep_dot_in_output_schema=self.config.keep_dot_in_output_schema,
            include_header_in_s3_output=self.config.include_header_in_s3_output,
        )

        # (f) CradleJobSpecification
        cradle_job_spec = CradleJobSpecification(
            cluster_type=self.config.cluster_type,
            cradle_account=self.config.cradle_account,
            extra_spark_job_arguments=self.config.extra_spark_job_arguments or "",
            job_retry_count=self.config.job_retry_count,
        )

        # (g) Build the final request
        request = CreateCradleDataLoadJobRequest(
            data_sources=data_sources_spec,
            transform_specification=transform_spec,
            output_specification=output_spec,
            cradle_job_specification=cradle_job_spec,
        )

        return request

    def create_step(
        self,
        dependencies: Optional[list] = None
    ) -> CradleDataLoadingStep:
        """
        Build the CradleDataLoadingStep.  
        This step will run a Data Load job in the Cradle service using the parameters
        from self.config. Returns the fully‐configured CradleDataLoadingStep.
        """
        logger.info("Creating CradleDataLoadingStep…")

        # Build the CreateCradleDataLoadJobRequest
        request = self._build_request()

        # Convert to a dict (for logging or for passing into the Step param)
        request_dict = coral_utils.convert_coral_to_dict(request)
        logger.debug("CradleDataLoad request (dict): %s", request_dict)

        # Instantiate the actual CradleDataLoadingStep
        step_name = f"{self._get_step_name('CradleDataLoading')}-{self.config.job_type.capitalize()}"
        step = CradleDataLoadingStep(
            step_name=step_name,
            role=self.role,
            sagemaker_session=self.session
        )

        logger.info("Created CradleDataLoadingStep with name: %s", step.step_name)
        return step

    def get_request_dict(self) -> Dict[str, Any]:
        """
        Return the CradleDataLoad request as a plain Python dict
        (after converting via coral_utils).
        Useful for logging or for passing to StepOperator.
        """
        request = self._build_request()
        return coral_utils.convert_coral_to_dict(request)
