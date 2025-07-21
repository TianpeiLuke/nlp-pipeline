#!/usr/bin/env python
"""Configuration for ModelCalibration step.

This module defines the configuration class for the ModelCalibration step,
which calibrates model prediction scores to accurate probabilities.
"""

from typing import Optional
from pydantic import model_validator

from .config_processing_step_base import ProcessingStepConfigBase


class ModelCalibrationConfig(ProcessingStepConfigBase):
    """Configuration for ModelCalibration step.
    
    This class defines the configuration parameters for the ModelCalibration step,
    which calibrates model prediction scores to accurate probabilities. Calibration
    ensures that model scores reflect true probabilities, which is crucial for
    risk-based decision-making and threshold setting.
    """
    
    def __init__(
        self,
        region: str,
        pipeline_s3_loc: str,
        processing_instance_type: str = "ml.m5.xlarge",
        processing_instance_count: int = 1,
        processing_volume_size: int = 30,
        max_runtime_seconds: int = 3600,
        pipeline_name: Optional[str] = None,
        calibration_method: str = "gam",
        monotonic_constraint: bool = True,
        gam_splines: int = 10,
        error_threshold: float = 0.05,
        label_field: str = "label",
        score_field: str = "prob_class_1",
        processing_entry_point: str = "model_calibration.py",
        processing_source_dir: str = "dockers/xgboost_atoz/pipeline_scripts"
    ):
        """Initialize ModelCalibration configuration.
        
        Args:
            region: AWS region
            pipeline_s3_loc: S3 location for pipeline artifacts
            processing_instance_type: SageMaker instance type for processing
            processing_instance_count: Number of processing instances
            processing_volume_size: EBS volume size in GB
            max_runtime_seconds: Maximum runtime in seconds
            pipeline_name: Name of the pipeline (optional)
            calibration_method: Method to use for calibration (gam, isotonic, platt)
            monotonic_constraint: Whether to enforce monotonicity in GAM
            gam_splines: Number of splines for GAM
            error_threshold: Acceptable calibration error threshold
            label_field: Name of the label column
            score_field: Name of the score column to calibrate
            processing_entry_point: Script entry point filename
            processing_source_dir: Directory containing the processing script
        """
        super().__init__(
            region=region,
            pipeline_s3_loc=pipeline_s3_loc,
            processing_instance_type=processing_instance_type,
            processing_instance_count=processing_instance_count,
            processing_volume_size=processing_volume_size,
            max_runtime_seconds=max_runtime_seconds,
            pipeline_name=pipeline_name,
            processing_entry_point=processing_entry_point,
            processing_source_dir=processing_source_dir
        )
        
        self.calibration_method = calibration_method
        self.monotonic_constraint = monotonic_constraint
        self.gam_splines = gam_splines
        self.error_threshold = error_threshold
        self.label_field = label_field
        self.score_field = score_field
    
    @model_validator(mode='after')
    def validate_config(self) -> 'ModelCalibrationConfig':
        """Validate configuration and ensure defaults are set.
        
        Returns:
            Self: The validated configuration object
            
        Raises:
            ValueError: If any validation fails
        """
        # Basic validation
        if not self.processing_entry_point:
            raise ValueError("ModelCalibrationConfig requires a processing_entry_point")
            
        if not self.processing_source_dir:
            raise ValueError("ModelCalibrationConfig requires a processing_source_dir")
            
        # Validate script contract - this will be the source of truth
        contract = self.get_script_contract()
        if not contract:
            raise ValueError("Failed to load script contract")
            
        # Validate input/output paths in contract
        required_input_paths = ["evaluation_data", "model_artifacts"]
        for path_name in required_input_paths:
            if path_name not in contract.expected_input_paths:
                raise ValueError(f"Script contract missing required input path: {path_name}")
                
        required_output_paths = ["calibration_output", "metrics_output", "calibrated_data"]
        for path_name in required_output_paths:
            if path_name not in contract.expected_output_paths:
                raise ValueError(f"Script contract missing required output path: {path_name}")
                
        # Validate calibration method
        valid_methods = ['gam', 'isotonic', 'platt']
        if self.calibration_method.lower() not in valid_methods:
            raise ValueError(f"Invalid calibration method: {self.calibration_method}. "
                            f"Must be one of: {valid_methods}")
        
        # Validate numeric parameters
        if self.gam_splines <= 0:
            raise ValueError(f"gam_splines must be > 0, got {self.gam_splines}")
            
        if not 0 <= self.error_threshold <= 1:
            raise ValueError(f"error_threshold must be between 0 and 1, got {self.error_threshold}")
            
        return self
        
    def get_script_contract(self):
        """Return the script contract for this step.
        
        Returns:
            ScriptContract: The contract for this step's script.
        """
        from ..pipeline_script_contracts.model_calibration_contract import MODEL_CALIBRATION_CONTRACT
        return MODEL_CALIBRATION_CONTRACT
        
    def get_script_path(self):
        """Return the script path relative to the source directory.
        
        Returns:
            str: The path to the processing script.
        """
        return self.processing_entry_point
