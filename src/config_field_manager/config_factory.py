"""
Configuration Factory Module.

This module provides a factory for creating complete configuration objects
from essential user inputs (Tier 1) using the three-tier configuration architecture.

It leverages default_values_provider for Tier 2 inputs and field_derivation_engine
for Tier 3 inputs to minimize the information required from users while maintaining
compatibility with the existing configuration system.
"""

import logging
from typing import List, Dict, Any, Optional, Type
from datetime import datetime
from pathlib import Path

from src.config_field_manager.essential_input_models import (
    EssentialInputs,
    DataConfig,
    ModelConfig,
    RegistrationConfig,
    DateRangePeriod,
)
from src.config_field_manager.default_values_provider import DefaultValuesProvider
from src.config_field_manager.field_derivation_engine import FieldDerivationEngine
from src.pipeline_steps.config_base import BasePipelineConfig
from src.pipeline_steps.config_processing_step_base import ProcessingStepConfigBase
from src.pipeline_steps.config_data_load_step_cradle import (
    CradleDataLoadConfig,
    MdsDataSourceConfig,
    EdxDataSourceConfig,
    DataSourceConfig,
    DataSourcesSpecificationConfig,
    JobSplitOptionsConfig,
    TransformSpecificationConfig,
    OutputSpecificationConfig,
    CradleJobSpecificationConfig,
)
from src.pipeline_steps.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.pipeline_steps.config_training_step_xgboost import XGBoostTrainingConfig
from src.pipeline_steps.config_model_calibration_step import ModelCalibrationConfig
from src.pipeline_steps.config_model_eval_step_xgboost import XGBoostModelEvalConfig
from src.pipeline_steps.config_mims_packaging_step import PackageStepConfig
from src.pipeline_steps.config_mims_registration_step import ModelRegistrationConfig
from src.pipeline_steps.config_mims_payload_step import PayloadConfig

logger = logging.getLogger(__name__)


class ConfigFactory:
    """
    Base configuration factory class that handles the creation of configuration objects
    from essential user inputs (Tier 1).
    
    This class provides the foundation for specific factory implementations
    that target different model types and pipeline structures.
    """
    
    def __init__(self, essential_inputs: EssentialInputs):
        """
        Initialize the factory with essential inputs.
        
        Args:
            essential_inputs: Essential user inputs (Tier 1)
        """
        self.inputs = essential_inputs
        self.default_provider = DefaultValuesProvider()
        self.derivation_engine = FieldDerivationEngine(logger=logger)
        
    def create_config_list(self) -> List[BasePipelineConfig]:
        """
        Create a complete list of configuration objects with defaults and derived fields applied.
        
        This is the main entry point for configuration generation and should be
        implemented by specific factory subclasses.
        
        Returns:
            List of configuration objects
        """
        raise NotImplementedError("This method must be implemented by subclasses")
    
    def _apply_defaults_and_derive(self, config_list: List[BasePipelineConfig]) -> List[BasePipelineConfig]:
        """
        Apply defaults and derive fields for all configs.
        
        Args:
            config_list: List of configuration objects
            
        Returns:
            List of configuration objects with defaults and derived fields applied
        """
        # Apply defaults to all configs
        self.default_provider.apply_defaults_to_multiple(config_list)
        
        # Derive fields for all configs
        self.derivation_engine.derive_fields_for_multiple(config_list)
        
        return config_list


class XGBoostConfigFactory(ConfigFactory):
    """
    Factory for creating XGBoost pipeline configurations.
    
    This class generates all necessary configuration objects for an XGBoost
    pipeline with training, evaluation, and registration steps.
    """
    
    def create_config_list(self) -> List[BasePipelineConfig]:
        """
        Create a complete list of XGBoost pipeline configuration objects.
        
        Returns:
            List of configuration objects for the XGBoost pipeline
        """
        # Create all required config objects with base values
        base_config = self._create_base_config()
        processing_base_config = self._create_processing_base_config(base_config)
        training_cradle_config = self._create_training_cradle_config(base_config)
        calibration_cradle_config = self._create_calibration_cradle_config(base_config)
        training_preprocessing_config = self._create_training_preprocessing_config(processing_base_config)
        calibration_preprocessing_config = self._create_calibration_preprocessing_config(processing_base_config)
        training_config = self._create_training_config(base_config)
        calibration_config = self._create_calibration_config(processing_base_config)
        model_eval_config = self._create_model_eval_config(processing_base_config)
        packaging_config = self._create_packaging_config(processing_base_config)
        registration_config = self._create_registration_config(base_config)
        payload_config = self._create_payload_config(processing_base_config)
        
        # Combine all configs
        config_list = [
            base_config,
            processing_base_config,
            training_cradle_config,
            calibration_cradle_config,
            training_preprocessing_config,
            calibration_preprocessing_config,
            training_config,
            calibration_config,
            model_eval_config,
            packaging_config,
            registration_config,
            payload_config,
        ]
        
        # Apply defaults and derive fields
        return self._apply_defaults_and_derive(config_list)
    
    def _create_base_config(self) -> BasePipelineConfig:
        """
        Create the base configuration.
        
        Returns:
            Base pipeline configuration
        """
        data = self.inputs.data
        model = self.inputs.model
        
        # Generate essential base config values
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Map region to AWS region
        region_mapping = {
            "NA": "us-east-1",
            "EU": "eu-west-1",
            "FE": "us-west-2"
        }
        aws_region = region_mapping.get(data.region, "us-east-1")
        
        # Create pipeline naming
        pipeline_name = f"{data.author}-{data.service_name}-XGBoostModel-{data.region}"
        pipeline_description = f"{data.service_name} XGBoost Model {data.region}"
        pipeline_subdirectory = f"{pipeline_name}_{data.pipeline_version}"
        pipeline_s3_loc = f"s3://{data.bucket}/MODS/{pipeline_subdirectory}"
        
        # Framework information
        framework_version = "1.7-1"  # Default for XGBoost
        py_version = "py3"
        
        # Source directory (should be provided by environment or defaults)
        current_dir = Path.cwd()
        source_dir = Path(current_dir) / 'dockers' / 'xgboost_atoz'
        
        return BasePipelineConfig(
            bucket=data.bucket,
            current_date=current_date,
            region=data.region,
            aws_region=aws_region,
            author=data.author,
            pipeline_name=pipeline_name,
            pipeline_description=pipeline_description,
            pipeline_version=data.pipeline_version,
            pipeline_s3_loc=pipeline_s3_loc,
            framework_version=framework_version,
            py_version=py_version,
            source_dir=str(source_dir)
        )
    
    def _create_processing_base_config(self, base_config: BasePipelineConfig) -> ProcessingStepConfigBase:
        """
        Create the base processing configuration.
        
        Args:
            base_config: Base pipeline configuration
            
        Returns:
            Base processing configuration
        """
        processing_dict = {
            'processing_instance_type_large': 'ml.m5.4xlarge',
            'processing_instance_type_small': 'ml.m5.xlarge',
            'processing_volume_size': 500,
            'processing_instance_count': 1,
            'processing_source_dir': f"{base_config.source_dir}/pipeline_scripts",
            'processing_framework_version': '1.2-1'  # Support for Python 3.8
        }
        
        return ProcessingStepConfigBase(
            **base_config.model_dump(),
            **processing_dict
        )
    
    def _create_training_cradle_config(self, base_config: BasePipelineConfig) -> CradleDataLoadConfig:
        """
        Create the training data loading configuration.
        
        Args:
            base_config: Base pipeline configuration
            
        Returns:
            Training data loading configuration
        """
        data = self.inputs.data
        model = self.inputs.model
        
        # MDS data source configuration
        mds_field_list = ['objectId', 'transactionDate', 'Abuse.currency_exchange_rate_inline.exchangeRate', 'baseCurrency'] + model.tab_field_list + model.cat_field_list
        mds_field_list = sorted(list(set(mds_field_list)))
        
        output_schema = [{'field_name': field, 'field_type': 'STRING'} for field in mds_field_list]
        
        mds_data_source_inner_config = MdsDataSourceConfig(
            service_name=data.service_name,
            org_id=0,  # Default organization ID
            region=data.region,
            output_schema=output_schema
        )
        
        mds_data_source = DataSourceConfig(
            data_source_name=f"RAW_MDS_{data.region}",
            data_source_type="MDS",
            mds_data_source_properties=mds_data_source_inner_config
        )
        
        # EDX data source configuration for training
        tag_schema = [
            'order_id',
            'marketplace_id',
            'tag_date',
            model.label_name,
            'abuse_type',
            'concession_type',
        ]
        
        edx_schema_overrides = [{'field_name': field, 'field_type': 'STRING'} for field in tag_schema]
        
        # ETL job ID mapping
        etl_job_id_dict = {
            'NA': '24292902',
            'EU': '24292941',
            'FE': '25782074',
        }
        etl_job_id = etl_job_id_dict.get(data.region, '24292902')
        
        # Format dates for EDX manifest
        training_start = data.training_period.start_date.isoformat()
        training_end = data.training_period.end_date.isoformat()
        
        # Create EDX manifest
        tag_edx_provider = "trms-abuse-analytics"
        tag_edx_subject = "qingyuye-notr-exp"
        tag_edx_dataset = "atoz-tag"
        training_tag_edx_manifest = f'arn:amazon:edx:iad::manifest/{tag_edx_provider}/{tag_edx_subject}/{tag_edx_dataset}/["{etl_job_id}",{training_start}Z,{training_end}Z,"{data.region}"]'
        
        training_edx_source_inner_config = EdxDataSourceConfig(
            edx_provider=tag_edx_provider,
            edx_subject=tag_edx_subject,
            edx_dataset=tag_edx_dataset,
            edx_manifest=training_tag_edx_manifest,
            schema_overrides=edx_schema_overrides
        )
        
        training_edx_data_source = DataSourceConfig(
            data_source_name="TAGS",
            data_source_type="EDX",
            edx_data_source_properties=training_edx_source_inner_config
        )
        
        # Data sources specification
        training_data_sources_spec = DataSourcesSpecificationConfig(
            start_date=training_start,
            end_date=training_end,
            data_sources=[mds_data_source, training_edx_data_source]
        )
        
        # Job split options
        job_split_options = JobSplitOptionsConfig(
            split_job=False,
            days_per_split=7,
            merge_sql="SELECT * from INPUT"
        )
        
        # Transform SQL
        select_variable_text_list = []
        for field in mds_field_list:
            field_dot_replaced = field.replace('.', '__DOT__')
            select_variable_text_list.append(
                f'{mds_data_source.data_source_name}.{field_dot_replaced}'
            )
        
        for var in tag_schema:
            select_variable_text_list.append(f'{training_edx_data_source.data_source_name}.{var}')
        
        schema_list = ',\n'.join(select_variable_text_list)
        
        transform_sql_template = '''
        SELECT
        {schema_list}
        FROM {data_source_name}
        JOIN {tag_source_name} 
        ON {data_source_name}.objectId={tag_source_name}.order_id
        '''
        
        training_transform_sql = transform_sql_template.format(
            schema_list=schema_list,
            data_source_name=mds_data_source.data_source_name,
            tag_source_name=training_edx_data_source.data_source_name
        )
        
        training_transform_spec = TransformSpecificationConfig(
            transform_sql=training_transform_sql,
            job_split_options=job_split_options
        )
        
        # Output specification
        import uuid
        output_dir = f'cradle_download_output/{uuid.uuid4()}'
        training_output_path = f's3://{data.bucket}/{output_dir}/train'
        
        training_output_fields = self._get_all_fields(training_data_sources_spec)
        
        training_output_spec = OutputSpecificationConfig(
            output_schema=training_output_fields,
            output_path=training_output_path,
            output_format='PARQUET',
            output_save_mode='ERRORIFEXISTS',
            keep_dot_in_output_schema=False,
            include_header_in_s3_output=True
        )
        
        # Cradle job specification
        cradle_job_spec = CradleJobSpecificationConfig(
            cluster_type='STANDARD',
            cradle_account='Buyer-Abuse-RnD-Dev',
            job_retry_count=4
        )
        
        return CradleDataLoadConfig(
            **base_config.model_dump(),
            job_type='training',
            data_sources_spec=training_data_sources_spec,
            transform_spec=training_transform_spec,
            output_spec=training_output_spec,
            cradle_job_spec=cradle_job_spec
        )
    
    def _create_calibration_cradle_config(self, base_config: BasePipelineConfig) -> CradleDataLoadConfig:
        """
        Create the calibration data loading configuration.
        
        Args:
            base_config: Base pipeline configuration
            
        Returns:
            Calibration data loading configuration
        """
        data = self.inputs.data
        model = self.inputs.model
        
        # MDS data source configuration
        mds_field_list = ['objectId', 'transactionDate', 'Abuse.currency_exchange_rate_inline.exchangeRate', 'baseCurrency'] + model.tab_field_list + model.cat_field_list
        mds_field_list = sorted(list(set(mds_field_list)))
        
        output_schema = [{'field_name': field, 'field_type': 'STRING'} for field in mds_field_list]
        
        mds_data_source_inner_config = MdsDataSourceConfig(
            service_name=data.service_name,
            org_id=0,  # Default organization ID
            region=data.region,
            output_schema=output_schema
        )
        
        mds_data_source = DataSourceConfig(
            data_source_name=f"RAW_MDS_{data.region}",
            data_source_type="MDS",
            mds_data_source_properties=mds_data_source_inner_config
        )
        
        # EDX data source configuration for calibration
        tag_schema = [
            'order_id',
            'marketplace_id',
            'tag_date',
            model.label_name,
            'abuse_type',
            'concession_type',
        ]
        
        edx_schema_overrides = [{'field_name': field, 'field_type': 'STRING'} for field in tag_schema]
        
        # ETL job ID mapping
        etl_job_id_dict = {
            'NA': '24292902',
            'EU': '24292941',
            'FE': '25782074',
        }
        etl_job_id = etl_job_id_dict.get(data.region, '24292902')
        
        # Format dates for EDX manifest
        calibration_start = data.calibration_period.start_date.isoformat()
        calibration_end = data.calibration_period.end_date.isoformat()
        
        # Create EDX manifest
        tag_edx_provider = "trms-abuse-analytics"
        tag_edx_subject = "qingyuye-notr-exp"
        tag_edx_dataset = "atoz-tag"
        calibration_tag_edx_manifest = f'arn:amazon:edx:iad::manifest/{tag_edx_provider}/{tag_edx_subject}/{tag_edx_dataset}/["{etl_job_id}",{calibration_start}Z,{calibration_end}Z,"{data.region}"]'
        
        calibration_edx_source_inner_config = EdxDataSourceConfig(
            edx_provider=tag_edx_provider,
            edx_subject=tag_edx_subject,
            edx_dataset=tag_edx_dataset,
            edx_manifest=calibration_tag_edx_manifest,
            schema_overrides=edx_schema_overrides
        )
        
        calibration_edx_data_source = DataSourceConfig(
            data_source_name="TAGS",
            data_source_type="EDX",
            edx_data_source_properties=calibration_edx_source_inner_config
        )
        
        # Data sources specification
        calibration_data_sources_spec = DataSourcesSpecificationConfig(
            start_date=calibration_start,
            end_date=calibration_end,
            data_sources=[mds_data_source, calibration_edx_data_source]
        )
        
        # Job split options
        job_split_options = JobSplitOptionsConfig(
            split_job=False,
            days_per_split=7,
            merge_sql="SELECT * from INPUT"
        )
        
        # Transform SQL
        select_variable_text_list = []
        for field in mds_field_list:
            field_dot_replaced = field.replace('.', '__DOT__')
            select_variable_text_list.append(
                f'{mds_data_source.data_source_name}.{field_dot_replaced}'
            )
        
        for var in tag_schema:
            select_variable_text_list.append(f'{calibration_edx_data_source.data_source_name}.{var}')
        
        schema_list = ',\n'.join(select_variable_text_list)
        
        transform_sql_template = '''
        SELECT
        {schema_list}
        FROM {data_source_name}
        JOIN {tag_source_name} 
        ON {data_source_name}.objectId={tag_source_name}.order_id
        '''
        
        calibration_transform_sql = transform_sql_template.format(
            schema_list=schema_list,
            data_source_name=mds_data_source.data_source_name,
            tag_source_name=calibration_edx_data_source.data_source_name
        )
        
        calibration_transform_spec = TransformSpecificationConfig(
            transform_sql=calibration_transform_sql,
            job_split_options=job_split_options
        )
        
        # Output specification
        import uuid
        output_dir = f'cradle_download_output/{uuid.uuid4()}'
        calibration_output_path = f's3://{data.bucket}/{output_dir}/calibration'
        
        calibration_output_fields = self._get_all_fields(calibration_data_sources_spec)
        
        calibration_output_spec = OutputSpecificationConfig(
            output_schema=calibration_output_fields,
            output_path=calibration_output_path,
            output_format='PARQUET',
            output_save_mode='ERRORIFEXISTS',
            keep_dot_in_output_schema=False,
            include_header_in_s3_output=True
        )
        
        # Cradle job specification
        cradle_job_spec = CradleJobSpecificationConfig(
            cluster_type='STANDARD',
            cradle_account='Buyer-Abuse-RnD-Dev',
            job_retry_count=4
        )
        
        return CradleDataLoadConfig(
            **base_config.model_dump(),
            job_type='calibration',
            data_sources_spec=calibration_data_sources_spec,
            transform_spec=calibration_transform_spec,
            output_spec=calibration_output_spec,
            cradle_job_spec=cradle_job_spec
        )
    
    def _create_training_preprocessing_config(self, processing_base_config: ProcessingStepConfigBase) -> TabularPreprocessingConfig:
        """
        Create the training preprocessing configuration.
        
        Args:
            processing_base_config: Base processing configuration
            
        Returns:
            Training preprocessing configuration
        """
        model = self.inputs.model
        
        from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
        
        base_hyperparameter = ModelHyperparameters(
            full_field_list=model.full_field_list,
            cat_field_list=model.cat_field_list,
            tab_field_list=model.tab_field_list,
            label_name=model.label_name,
            id_name=model.id_name,
            input_tab_dim=len(model.tab_field_list),
            is_binary=model.is_binary,
            num_classes=model.num_classes,
            multiclass_categories=model.multiclass_categories,
            class_weights=model.class_weights,
            header=0,
            device=-1,
            optimizer='SGD',
            batch_size=4,
            lr=3e-05,
            max_epochs=3,
            metric_choices=['f1_score', 'auroc']
        )
        
        training_tabular_preprocessing_dict = processing_base_config.model_dump()
        training_tabular_preprocessing_dict['processing_entry_point'] = "tabular_preprocess.py"
        training_tabular_preprocessing_dict['job_type'] = "training"
        training_tabular_preprocessing_dict['hyperparameters'] = base_hyperparameter
        training_tabular_preprocessing_dict['test_val_ratio'] = 0.5
        
        return TabularPreprocessingConfig(
            **training_tabular_preprocessing_dict
        )
    
    def _create_calibration_preprocessing_config(self, processing_base_config: ProcessingStepConfigBase) -> TabularPreprocessingConfig:
        """
        Create the calibration preprocessing configuration.
        
        Args:
            processing_base_config: Base processing configuration
            
        Returns:
            Calibration preprocessing configuration
        """
        model = self.inputs.model
        
        from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
        
        base_hyperparameter = ModelHyperparameters(
            full_field_list=model.full_field_list,
            cat_field_list=model.cat_field_list,
            tab_field_list=model.tab_field_list,
            label_name=model.label_name,
            id_name=model.id_name,
            input_tab_dim=len(model.tab_field_list),
            is_binary=model.is_binary,
            num_classes=model.num_classes,
            multiclass_categories=model.multiclass_categories,
            class_weights=model.class_weights,
            header=0,
            device=-1,
            optimizer='SGD',
            batch_size=4,
            lr=3e-05,
            max_epochs=3,
            metric_choices=['f1_score', 'auroc']
        )
        
        calibration_tabular_preprocessing_dict = processing_base_config.model_dump()
        calibration_tabular_preprocessing_dict['processing_entry_point'] = "tabular_preprocess.py"
        calibration_tabular_preprocessing_dict['job_type'] = "calibration"
        calibration_tabular_preprocessing_dict['hyperparameters'] = base_hyperparameter
        calibration_tabular_preprocessing_dict['test_val_ratio'] = 0.5
        
        return TabularPreprocessingConfig(
            **calibration_tabular_preprocessing_dict
        )
    
    def _create_training_config(self, base_config: BasePipelineConfig) -> XGBoostTrainingConfig:
        """
        Create the training configuration.
        
        Args:
            base_config: Base pipeline configuration
            
        Returns:
            XGBoost training configuration
        """
        model = self.inputs.model
        
        from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters
        from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
        
        base_hyperparameter = ModelHyperparameters(
            full_field_list=model.full_field_list,
            cat_field_list=model.cat_field_list,
            tab_field_list=model.tab_field_list,
            label_name=model.label_name,
            id_name=model.id_name,
            input_tab_dim=len(model.tab_field_list),
            is_binary=model.is_binary,
            num_classes=model.num_classes,
            multiclass_categories=model.multiclass_categories,
            class_weights=model.class_weights,
            header=0,
            device=-1,
            optimizer='SGD',
            batch_size=4,
            lr=3e-05,
            max_epochs=3,
            metric_choices=['f1_score', 'auroc']
        )
        
        # Determine objective and eval_metric based on classification type
        if model.is_binary:
            objective = "binary:logistic"
            eval_metric = ['logloss', 'auc']
        else:
            objective = "multi:softmax"
            eval_metric = ['mlogloss', 'merror']
        
        model_params = {
            "num_round": model.num_round,
            "max_depth": model.max_depth,
            "min_child_weight": model.min_child_weight,
            "objective": objective,
            "eval_metric": eval_metric
        }
        
        xgb_hyperparams = XGBoostModelHyperparameters(
            **base_hyperparameter.model_dump(),
            **model_params
        )
        
        train_dict = {
            'training_instance_type': 'ml.m5.4xlarge',
            'training_entry_point': "train_xgb.py",
            'training_volume_size': 800,
            'training_instance_count': 1,
            'hyperparameters': xgb_hyperparams
        }
        
        return XGBoostTrainingConfig(
            **base_config.model_dump(),
            **train_dict
        )
    
    def _create_calibration_config(self, processing_base_config: ProcessingStepConfigBase) -> ModelCalibrationConfig:
        """
        Create the model calibration configuration.
        
        Args:
            processing_base_config: Base processing configuration
            
        Returns:
            Model calibration configuration
        """
        model = self.inputs.model
        
        base_processing_config_dict = processing_base_config.model_dump()
        base_processing_config_dict['processing_entry_point'] = 'model_calibration.py'
        
        calibration_method = "gam"  # Generative Additive Model
        score_field = 'prob_class_1'
        score_field_prefix = 'prob_class_'
        multiclass_categories = list(range(model.num_classes))
        
        return ModelCalibrationConfig(
            **base_processing_config_dict,
            calibration_method=calibration_method,
            label_field=model.label_name,
            score_field=score_field,
            is_binary=model.is_binary,
            num_classes=model.num_classes,
            score_field_prefix=score_field_prefix,
            multiclass_categories=multiclass_categories
        )
    
    def _create_model_eval_config(self, processing_base_config: ProcessingStepConfigBase) -> XGBoostModelEvalConfig:
        """
        Create the model evaluation configuration.
        
        Args:
            processing_base_config: Base processing configuration
            
        Returns:
            XGBoost model evaluation configuration
        """
        model = self.inputs.model
        
        from src.pipeline_steps.hyperparameters_xgboost import XGBoostModelHyperparameters
        from src.pipeline_steps.hyperparameters_base import ModelHyperparameters
        
        base_hyperparameter = ModelHyperparameters(
            full_field_list=model.full_field_list,
            cat_field_list=model.cat_field_list,
            tab_field_list=model.tab_field_list,
            label_name=model.label_name,
            id_name=model.id_name,
            input_tab_dim=len(model.tab_field_list),
            is_binary=model.is_binary,
            num_classes=model.num_classes,
            multiclass_categories=model.multiclass_categories,
            class_weights=model.class_weights,
            header=0,
            device=-1,
            optimizer='SGD',
            batch_size=4,
            lr=3e-05,
            max_epochs=3,
            metric_choices=['f1_score', 'auroc']
        )
        
        # Determine objective and eval_metric based on classification type
        if model.is_binary:
            objective = "binary:logistic"
            eval_metric = ['logloss', 'auc']
        else:
            objective = "multi:softmax"
            eval_metric = ['mlogloss', 'merror']
        
        model_params = {
            "num_round": model.num_round,
            "max_depth": model.max_depth,
            "min_child_weight": model.min_child_weight,
            "objective": objective,
            "eval_metric": eval_metric
        }
        
        xgb_hyperparams = XGBoostModelHyperparameters(
            **base_hyperparameter.model_dump(),
            **model_params
        )
        
        previous_processing_config = processing_base_config.model_dump()
        previous_processing_config['processing_entry_point'] = 'model_eval_xgb.py'
        previous_processing_config['use_large_processing_instance'] = True
        
        return XGBoostModelEvalConfig(
            **previous_processing_config,
            job_type='calibration',
            hyperparameters=xgb_hyperparams,
            eval_metric_choices=["auc", "average_precision", "f1_score"],
            xgboost_framework_version=processing_base_config.framework_version
        )
    
    def _create_packaging_config(self, processing_base_config: ProcessingStepConfigBase) -> PackageStepConfig:
        """
        Create the packaging configuration.
        
        Args:
            processing_base_config: Base processing configuration
            
        Returns:
            Packaging configuration
        """
        processing_base_dict = processing_base_config.model_dump()
        processing_base_dict['processing_entry_point'] = 'mims_package.py'
        processing_base_dict['use_large_processing_instance'] = True
        
        return PackageStepConfig(
            **processing_base_dict
        )
    
    def _create_registration_config(self, base_config: BasePipelineConfig) -> ModelRegistrationConfig:
        """
        Create the model registration configuration.
        
        Args:
            base_config: Base pipeline configuration
            
        Returns:
            Model registration configuration
        """
        model = self.inputs.model
        registration = self.inputs.registration
        
        # Create the model variable list
        def create_model_variable_list(full_fields, tab_fields, cat_fields, label_name, id_name):
            model_var_list = []
            
            for field in full_fields:
                # Skip label and id fields
                if field in [label_name, id_name]:
                    continue
                
                # Determine field type
                if field in tab_fields:
                    field_type = 'NUMERIC'
                elif field in cat_fields:
                    field_type = 'TEXT'
                else:
                    # For fields not explicitly categorized, default to TEXT
                    field_type = 'TEXT'
                
                # Add [field_name, field_type] pair
                model_var_list.append([field, field_type])
            
            return model_var_list
        
        # Adjusted field list to remove non-feature fields
        adjusted_full_field_list = model.tab_field_list + model.cat_field_list
        
        # Create the model variable list
        source_model_inference_input_variable_list = create_model_variable_list(
            adjusted_full_field_list,
            model.tab_field_list,
            model.cat_field_list,
            model.label_name,
            model.id_name
        )
        
        # Default output variables
        source_model_inference_output_variable_list = {
            'legacy-score': 'NUMERIC',
            'calibrated-score': 'NUMERIC',
            'custom-output-label': 'TEXT'
        }
        
        source_model_inference_content_types = ["text/csv"]
        source_model_inference_response_types = ["application/json"]
        
        return ModelRegistrationConfig(
            **base_config.model_dump(),
            framework='xgboost',
            inference_entry_point='inference_xgb.py',
            inference_instance_type="ml.m5.4xlarge",
            model_owner=registration.model_owner,
            model_registration_domain=registration.model_registration_domain,
            model_registration_objective=f'AtoZ_Claims_SM_Model_{base_config.region}',
            source_model_inference_content_types=source_model_inference_content_types,
            source_model_inference_response_types=source_model_inference_response_types,
            source_model_inference_output_variable_list=source_model_inference_output_variable_list,
            source_model_inference_input_variable_list=source_model_inference_input_variable_list
        )
    
    def _create_payload_config(self, processing_base_config: ProcessingStepConfigBase) -> PayloadConfig:
        """
        Create the payload configuration.
        
        Args:
            processing_base_config: Base processing configuration
            
        Returns:
            Payload configuration
        """
        model = self.inputs.model
        registration = self.inputs.registration
        
        processing_base_dict = processing_base_config.model_dump()
        processing_base_dict['processing_entry_point'] = 'mims_payload.py'
        
        # Create the model variable list
        def create_model_variable_list(full_fields, tab_fields, cat_fields, label_name, id_name):
            model_var_list = []
            
            for field in full_fields:
                # Skip label and id fields
                if field in [label_name, id_name]:
                    continue
                
                # Determine field type
                if field in tab_fields:
                    field_type = 'NUMERIC'
                elif field in cat_fields:
                    field_type = 'TEXT'
                else:
                    # For fields not explicitly categorized, default to TEXT
                    field_type = 'TEXT'
                
                # Add [field_name, field_type] pair
                model_var_list.append([field, field_type])
            
            return model_var_list
        
        # Adjusted field list to remove non-feature fields
        adjusted_full_field_list = model.tab_field_list + model.cat_field_list
        
        # Create the model variable list
        source_model_inference_input_variable_list = create_model_variable_list(
            adjusted_full_field_list,
            model.tab_field_list,
            model.cat_field_list,
            model.label_name,
            model.id_name
        )
        
        # Default output variables
        source_model_inference_output_variable_list = {
            'legacy-score': 'NUMERIC',
            'calibrated-score': 'NUMERIC',
            'custom-output-label': 'TEXT'
        }
        
        source_model_inference_content_types = ["text/csv"]
        source_model_inference_response_types = ["application/json"]
        
        return PayloadConfig(
            **processing_base_dict,
            expected_tps=registration.expected_tps,
            max_latency_in_millisecond=registration.max_latency_ms,
            max_acceptable_error_rate=registration.max_error_rate,
            default_numeric_value=0.0,
            default_text_value='Default',
            special_field_values=None,
            model_owner=registration.model_owner,
            model_registration_domain=registration.model_registration_domain,
            model_registration_objective=f'AtoZ_Claims_SM_Model_{processing_base_config.region}',
            source_model_inference_content_types=source_model_inference_content_types,
            source_model_inference_response_types=source_model_inference_response_types,
            source_model_inference_output_variable_list=source_model_inference_output_variable_list,
            source_model_inference_input_variable_list=source_model_inference_input_variable_list
        )
        
    def _get_all_fields(self, ds_spec: DataSourcesSpecificationConfig) -> List[str]:
        """
        Return the full list of distinct field names across all data sources
        in the given DataSourcesSpecificationConfig.

        Args:
            ds_spec: An instance of DataSourcesSpecificationConfig.

        Returns:
            A sorted list of every field_name that appears in any MDS output_schema
            or any EDX schema_overrides.
        """
        collected = set()

        for ds_cfg in ds_spec.data_sources:
            if ds_cfg.data_source_type == "MDS":
                mds_props = ds_cfg.mds_data_source_properties
                for field_desc in mds_props.output_schema:
                    collected.add(field_desc["field_name"])
            elif ds_cfg.data_source_type == "EDX":
                edx_props = ds_cfg.edx_data_source_properties
                for field_desc in edx_props.schema_overrides:
                    collected.add(field_desc["field_name"])
            else:
                raise ValueError(f"Unsupported data_source_type: {ds_cfg.data_source_type}")

        return sorted(collected)
