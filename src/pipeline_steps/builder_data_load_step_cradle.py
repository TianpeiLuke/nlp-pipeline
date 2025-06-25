from typing import Dict, Optional, Any, List, Set, Union
from pathlib import Path
import logging
import os
import json
import importlib
from datetime import datetime

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep, Step

# Import CradleDataLoadingStep
from secure_ai_sandbox_workflow_python_sdk.cradle_data_loading.cradle_data_loading_step import (
    CradleDataLoadingStep,
)

# Import Cradle models for request building
try:
    from com.amazon.secureaisandboxproxyservice.models.field import Field
    from com.amazon.secureaisandboxproxyservice.models.datasource import DataSource
    from com.amazon.secureaisandboxproxyservice.models.mdsdatasourceproperties import MdsDataSourceProperties
    from com.amazon.secureaisandboxproxyservice.models.edxdatasourceproperties import EdxDataSourceProperties
    from com.amazon.secureaisandboxproxyservice.models.andesdatasourceproperties import AndesDataSourceProperties
    from com.amazon.secureaisandboxproxyservice.models.datasourcesspecification import DataSourcesSpecification
    from com.amazon.secureaisandboxproxyservice.models.jobsplitoptions import JobSplitOptions
    from com.amazon.secureaisandboxproxyservice.models.transformspecification import TransformSpecification
    from com.amazon.secureaisandboxproxyservice.models.outputspecification import OutputSpecification
    from com.amazon.secureaisandboxproxyservice.models.cradlejobspecification import CradleJobSpecification
    from com.amazon.secureaisandboxproxyservice.models.createcradledataloadjobrequest import CreateCradleDataLoadJobRequest
    CRADLE_MODELS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Cradle models not available. _build_request and get_request_dict will not work.")
    CRADLE_MODELS_AVAILABLE = False

# Import coral utils for request conversion
try:
    from secure_ai_sandbox_python_lib.utils import coral_utils
    CORAL_UTILS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("coral_utils not available. get_request_dict will not work.")
    CORAL_UTILS_AVAILABLE = False

from .config_data_load_step_cradle import CradleDataLoadConfig
from .builder_step_base import StepBuilderBase

logger = logging.getLogger(__name__)

# Import constants from the same module used by the data loading step
try:
    from secure_ai_sandbox_workflow_python_sdk.utils.constants import (
        OUTPUT_TYPE_DATA,
        OUTPUT_TYPE_METADATA,
        OUTPUT_TYPE_SIGNATURE,
    )
except ImportError:
    # Fallback to dynamic import if the direct import fails
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


class CradleDataLoadingStepBuilder(StepBuilderBase):
    """
    Builder for a Cradle Data Loading ProcessingStep.
    This class is responsible for configuring and creating a SageMaker ProcessingStep
    that executes the Cradle data loading script.
    """

    def __init__(
        self,
        config: CradleDataLoadConfig,
        sagemaker_session=None,
        role: Optional[str] = None,
        notebook_root: Optional[Path] = None,
    ):
        """
        Initializes the builder with a specific configuration for the data loading step.

        Args:
            config: A CradleDataLoadConfig instance containing all necessary settings.
            sagemaker_session: The SageMaker session object to manage interactions with AWS.
            role: The IAM role ARN to be used by the SageMaker Processing Job.
            notebook_root: The root directory of the notebook environment, used for resolving
                         local paths if necessary.
        """
        if not isinstance(config, CradleDataLoadConfig):
            raise ValueError(
                "CradleDataLoadingStepBuilder requires a CradleDataLoadConfig instance."
            )
        super().__init__(
            config=config,
            sagemaker_session=sagemaker_session,
            role=role,
            notebook_root=notebook_root
        )
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
        
    def get_input_requirements(self) -> Dict[str, str]:
        """
        Get the input requirements for this step builder.
        
        Returns:
            Dictionary mapping input parameter names to descriptions
        """
        # This step doesn't require any inputs from previous steps
        input_reqs = {
            "outputs": f"Dictionary containing {', '.join([f'{v}' for v in (self.config.output_names or {}).values()])} S3 paths",
            "enable_caching": self.COMMON_PROPERTIES["enable_caching"]
        }
        return input_reqs
    
    def get_output_properties(self) -> Dict[str, str]:
        """
        Get the output properties this step provides.
        
        Returns:
            Dictionary mapping output property names to descriptions
        """
        # Get output properties from config's output_names
        output_props = {}
        
        # Map output names to their descriptions
        output_descriptions = {
            "data_output": "S3 URI of the loaded data",
            "metadata_output": "S3 URI of the metadata",
            "signature_output": "S3 URI of the signature"
        }
        
        # Create output properties dictionary
        for output_name, output_key in (self.config.output_names or {}).items():
            if output_name in output_descriptions:
                output_props[output_key] = output_descriptions[output_name]
            else:
                output_props[output_key] = f"S3 URI of the {output_name}"
        
        # Add constants if available
        if OUTPUT_TYPE_DATA:
            output_props[OUTPUT_TYPE_DATA] = "S3 URI of the loaded data"
        if OUTPUT_TYPE_METADATA:
            output_props[OUTPUT_TYPE_METADATA] = "S3 URI of the metadata"
        if OUTPUT_TYPE_SIGNATURE:
            output_props[OUTPUT_TYPE_SIGNATURE] = "S3 URI of the signature"
        
        return output_props
        
    def _match_custom_properties(self, inputs: Dict[str, Any], input_requirements: Dict[str, str], 
                                prev_step: Step) -> Set[str]:
        """
        Match custom properties specific to CradleDataLoading step.
        
        Args:
            inputs: Dictionary to add matched inputs to
            input_requirements: Dictionary of input requirements
            prev_step: The dependency step
            
        Returns:
            Set of input names that were successfully matched
        """
        matched_inputs = set()
        
        # No custom properties to match for this step
        return matched_inputs
    
    def create_step(self, **kwargs) -> Step:
        """
        Creates a specialized CradleDataLoadingStep for Cradle data loading.
        
        This method creates a CradleDataLoadingStep that directly interacts with the
        Cradle service to load data.

        Args:
            **kwargs: Keyword arguments for configuring the step, including:
                - outputs: A dictionary mapping output channel names to their S3 destinations.
                - dependencies: Optional list of steps that this step depends on.
                - enable_caching: A boolean indicating whether to cache the results of this step.

        Returns:
            Step: A CradleDataLoadingStep instance.
            
        Raises:
            ValueError: If there's an error creating the CradleDataLoadingStep.
        """
            
        # Create the step name
        step_name = f"{self._get_step_name('CradleDataLoading')}-{self.config.job_type.capitalize()}"
        
        logger.info("Creating CradleDataLoadingStep...")
        try:
            # Create a CradleDataLoadingStep
            step = CradleDataLoadingStep(
                step_name=step_name,
                role=self.role,
                sagemaker_session=self.session
            )
            
            logger.info(f"Created CradleDataLoadingStep with name: {step.name}")
            
            # Get the output locations for logging
            output_locations = step.get_output_locations()
            logger.info(f"CradleDataLoadingStep output locations: {output_locations}")
            
            return step
            
        except Exception as e:
            logger.error(f"Error creating CradleDataLoadingStep: {e}")
            raise ValueError(f"Failed to create CradleDataLoadingStep: {e}") from e
        
    def _build_request(self) -> Any:
        """
        Convert self.config → a CreateCradleDataLoadJobRequest instance under the hood.
        
        This method builds a Cradle data load request from the configuration, which can be
        used to fill in the execution document or for logging purposes.
        
        Returns:
            CreateCradleDataLoadJobRequest: The request object for Cradle data loading
            
        Raises:
            ImportError: If the required Cradle models are not available
        """
        if not CRADLE_MODELS_AVAILABLE:
            raise ImportError("Cradle models not available. Cannot build request.")
            
        # Check if we have the necessary configuration attributes
        required_attrs = [
            'data_sources_spec',
            'transform_spec',
            'output_spec',
            'cradle_job_spec'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"CradleDataLoadConfig missing required attribute: {attr}")
        
        try:
            # (a) Build each DataSource from data_sources_spec.data_sources
            data_source_models: List[DataSource] = []
            for ds_cfg in self.config.data_sources_spec.data_sources:
                if ds_cfg.data_source_type == "MDS":
                    mds_props_cfg = ds_cfg.mds_data_source_properties
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
                    edx_props_cfg = ds_cfg.edx_data_source_properties
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
                    andes_props_cfg = ds_cfg.andes_data_source_properties
                    if andes_props_cfg.andes3_enabled:
                        logger.info(f"ANDES 3.0 is enabled for table {andes_props_cfg.table_name}")
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
            ds_spec_cfg = self.config.data_sources_spec
            data_sources_spec = DataSourcesSpecification(
                start_date=ds_spec_cfg.start_date,
                end_date=ds_spec_cfg.end_date,
                data_sources=data_source_models,
            )

            # (c) TransformSpecification
            transform_spec_cfg = self.config.transform_spec
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
            output_spec_cfg = self.config.output_spec
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
            cradle_job_spec_cfg = self.config.cradle_job_spec
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
            
        except Exception as e:
            logger.error(f"Error building Cradle request: {e}")
            raise ValueError(f"Failed to build Cradle request: {e}") from e
    
    def get_request_dict(self) -> Dict[str, Any]:
        """
        Return the CradleDataLoad request as a plain Python dict.
        
        This method is useful for logging or for passing to StepOperator.
        It builds the request using _build_request and then converts it to a dictionary.
        
        Returns:
            Dict[str, Any]: The request as a dictionary
            
        Raises:
            ImportError: If coral_utils is not available
            ValueError: If the request could not be built
        """
        if not CORAL_UTILS_AVAILABLE:
            raise ImportError("coral_utils not available. Cannot convert request to dict.")
            
        try:
            request = self._build_request()
            return coral_utils.convert_coral_to_dict(request)
        except Exception as e:
            logger.error(f"Error getting request dict: {e}")
            raise ValueError(f"Failed to get request dict: {e}") from e
            
    def get_step_outputs(self, step: CradleDataLoadingStep, output_type: str = None) -> Union[Dict[str, str], str]:
        """
        Get the output locations from a created CradleDataLoadingStep.
        
        This method retrieves the S3 locations where the Cradle data loading step will store its outputs.
        These locations can be used as inputs to subsequent steps in the pipeline.

        Args:
            step (CradleDataLoadingStep): The CradleDataLoadingStep created by this builder
            output_type (str, optional): Specific output type to retrieve. If None, returns all output types.
                                       Valid values are OUTPUT_TYPE_DATA, OUTPUT_TYPE_METADATA, OUTPUT_TYPE_SIGNATURE.

        Returns:
            Union[Dict[str, str], str]: 
                - If output_type is None: Dictionary mapping output types to their S3 locations
                - If output_type is specified: S3 location for the specified output type

        Raises:
            ValueError: If the step is not a CradleDataLoadingStep instance, wasn't created by this builder,
                      or if the requested output_type is not valid

        Example:
            ```python
            # Get all output locations
            builder = CradleDataLoadingStepBuilder(config)
            step = builder.create_step()
            all_outputs = builder.get_step_outputs(step)
            
            # Get specific output location
            data_location = builder.get_step_outputs(step, OUTPUT_TYPE_DATA)
            
            # Connect to downstream step
            preprocessing_step = preprocessing_builder.build(
                dependencies=[step],
                inputs={
                    "input_data": builder.get_step_outputs(step, OUTPUT_TYPE_DATA)
                }
            )
            ```
        """
        if not isinstance(step, CradleDataLoadingStep):
            raise ValueError("Argument must be a CradleDataLoadingStep instance")

        expected_step_name = f"{self._get_step_name('CradleDataLoading')}-{self.config.job_type.capitalize()}"
        if step.name != expected_step_name:
            raise ValueError(f"Step was not created by this builder. Expected name: {expected_step_name}, got: {step.name}")

        try:
            # Get output locations based on whether output_type is specified
            if output_type is None:
                # Get all output locations
                output_locations = step.get_output_locations()
                
                if not output_locations:
                    raise ValueError("No output locations found in the step")
                
                # Validate that all required output types are present
                required_outputs = {OUTPUT_TYPE_DATA, OUTPUT_TYPE_METADATA, OUTPUT_TYPE_SIGNATURE}
                missing_outputs = required_outputs - set(output_locations.keys())
                if missing_outputs:
                    raise ValueError(f"Missing required output types: {missing_outputs}")
                
                return output_locations
            else:
                # Get specific output location
                return step.get_output_locations(output_type)
                
        except Exception as e:
            logger.error(f"Error getting output locations from step: {e}")
            raise ValueError(f"Failed to get output locations from step: {e}") from e
    
    def get_output_location(self, step: CradleDataLoadingStep, output_type: str) -> str:
        """
        Get a specific output location from a created CradleDataLoadingStep.
        
        This is a convenience method that calls get_step_outputs with a specific output_type.
        It's useful for connecting the output of this step to the input of a downstream step.

        Args:
            step (CradleDataLoadingStep): The CradleDataLoadingStep created by this builder
            output_type (str): The output type to retrieve. Valid values are:
                             - OUTPUT_TYPE_DATA: Main data output location
                             - OUTPUT_TYPE_METADATA: Metadata output location
                             - OUTPUT_TYPE_SIGNATURE: Signature output location

        Returns:
            str: S3 location for the specified output type

        Raises:
            ValueError: If the step is not a CradleDataLoadingStep instance, wasn't created by this builder,
                      or if the requested output_type is not valid

        Example:
            ```python
            # Connect to downstream step
            preprocessing_step = preprocessing_builder.build(
                dependencies=[step],
                inputs={
                    "input_data": builder.get_output_location(step, OUTPUT_TYPE_DATA)
                }
            )
            ```
        """
        return self.get_step_outputs(step, output_type)
