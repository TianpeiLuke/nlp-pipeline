from pydantic import BaseModel, Field, model_validator
from typing import Optional, Dict, List, Any, Union, TYPE_CHECKING
from pathlib import Path
from datetime import datetime
from enum import Enum

import json
import boto3
import tarfile
import tempfile
import os
import logging

logger = logging.getLogger(__name__)

from .config_processing_step_base import ProcessingStepConfigBase
from .config_mims_registration_step import VariableType

# Import the script contract
from ..pipeline_script_contracts.mims_payload_contract import MIMS_PAYLOAD_CONTRACT

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract


class PayloadConfig(ProcessingStepConfigBase):
    """Configuration for payload generation and testing."""
    
    # Model framework and registration fields (previously inherited from ModelRegistrationConfig)
    framework: str = Field(
        default="xgboost",
        description="ML framework used for the model"
    )
    # Note: We use processing_instance_type_large and processing_instance_type_small from ProcessingStepConfigBase
    # Note: We use processing_entry_point from ProcessingStepConfigBase
    model_owner: str = Field(
        default="team id",
        description="Team ID of model owner"
    )
    model_registration_domain: str = Field(
        default="BuyerSellerMessaging",
        description="Domain for model registration"
    )
    model_registration_objective: Optional[str] = Field(
        default=None,
        description="Objective of model registration"
    )
    
    # Content and response types
    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Content type for model inference input. Must be exactly ['text/csv'] or ['application/json']"
    )
    source_model_inference_response_types: List[str] = Field(
        default=["application/json"],
        description="Response type for model inference output. Must be exactly ['text/csv'] or ['application/json']"
    )

    # Variable lists for input and output
    source_model_inference_output_variable_list: Dict[str, VariableType] = Field(
        default={
            'legacy-score': VariableType.NUMERIC
        },
        description="Dictionary of output variables and their types (NUMERIC or TEXT)"
    )
    
    source_model_inference_input_variable_list: Union[Dict[str, Union[VariableType, str]], List[List[str]]] = Field(
        default_factory=dict,
        description="Input variables and their types. Can be either:\n"
                   "1. Dictionary: {'var1': 'NUMERIC', 'var2': 'TEXT'}\n"
                   "2. List of pairs: [['var1', 'NUMERIC'], ['var2', 'TEXT']]"
    )
    
    # Performance metrics
    expected_tps: int = Field(
        default=2,
        ge=1,
        description="Expected transactions per second"
    )
    max_latency_in_millisecond: int = Field(
        default=800,
        ge=100,
        le=10000,
        description="Maximum acceptable latency in milliseconds"
    )
    max_acceptable_error_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable error rate (0-1)"
    )
    
    # S3 path configuration
    sample_payload_s3_key: Optional[str] = Field(
        default=None,
        description="S3 key for sample payload file"
    )
    
    # Default values for payload generation
    default_numeric_value: float = Field(
        default=0.0,
        description="Default value for numeric fields"
    )
    default_text_value: str = Field(
        default="DEFAULT_TEXT",
        description="Default value for text fields"
    )
    
    # Special field values dictionary is now optional
    special_field_values: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional dictionary of special TEXT fields and their template values"
    )
    
    # Note: We use processing_script_arguments from ProcessingStepConfigBase instead of payload_script_arguments

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'
        json_encoders = {
            VariableType: lambda v: v.value,
            Path: str
        }
        
    @model_validator(mode='before')
    @classmethod
    def _preprocess_values(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess input values"""
        # Handle variable type conversion if needed
        if 'source_model_inference_input_variable_list' in values:
            input_vars = values['source_model_inference_input_variable_list']
            if isinstance(input_vars, dict):
                # Convert string values to VariableType enum for dictionary format
                values['source_model_inference_input_variable_list'] = {
                    k: VariableType(v) if isinstance(v, str) else v
                    for k, v in input_vars.items()
                }
            elif isinstance(input_vars, list):
                # Convert string values to VariableType enum for list format
                values['source_model_inference_input_variable_list'] = [
                    [name, VariableType(type_str) if isinstance(type_str, str) else type_str]
                    for name, type_str in input_vars
                ]
        
        if 'source_model_inference_output_variable_list' in values:
            output_vars = values['source_model_inference_output_variable_list']
            if isinstance(output_vars, dict):
                # Convert string values to VariableType enum
                values['source_model_inference_output_variable_list'] = {
                    k: VariableType(v) if isinstance(v, str) else v
                    for k, v in output_vars.items()
                }
        
        return values
    
    @model_validator(mode='after')
    def construct_payload_path(self) -> 'PayloadConfig':
        """Construct S3 key for payload if not provided"""
        # Construct S3 key for payload if not provided
        if not self.sample_payload_s3_key:
            payload_file_name = f'payload_{self.pipeline_name}_{self.pipeline_version}'
            if self.model_registration_objective:
                payload_file_name += f'_{self.model_registration_objective}'
            self.sample_payload_s3_key = f'mods/payload/{payload_file_name}.tar.gz'
        
        # Update model with sample payload S3 key
        self = self.model_copy(update={"sample_payload_s3_key": self.sample_payload_s3_key})
        
        return self
        
    @model_validator(mode='after')
    def validate_registration_configs(self) -> 'PayloadConfig':
        """Override the parent validator to validate registration-specific fields"""
        # Validate model registration objective
        if not self.model_registration_objective:
            raise ValueError("model_registration_objective must be provided")
        
        # Use the effective source directory instead of just source_dir
        effective_source_dir = self.get_effective_source_dir()
        if effective_source_dir and not effective_source_dir.startswith('s3://'):
            entry_point_path = Path(effective_source_dir) / self.processing_entry_point
            if not entry_point_path.exists():
                raise ValueError(f"Processing entry point script not found: {entry_point_path}")
        
        return self
        
    def ensure_payload_path(self) -> None:
        """Ensure S3 key for payload is set. This is a regular method that can be called directly."""
        if not self.sample_payload_s3_key:
            payload_file_name = f'payload_{self.pipeline_name}_{self.pipeline_version}'
            if self.model_registration_objective:
                payload_file_name += f'_{self.model_registration_objective}'
            self.sample_payload_s3_key = f'mods/payload/{payload_file_name}.tar.gz'

    @model_validator(mode='after')
    def validate_special_fields(self) -> 'PayloadConfig':
        """Validate special fields configuration if provided"""
        if not self.special_field_values:
            return self
            
        # Check if all special fields are in input variable list
        invalid_fields = []
        input_vars = self.source_model_inference_input_variable_list
        
        for field_name in self.special_field_values:
            if isinstance(input_vars, dict):
                if field_name not in input_vars:
                    invalid_fields.append(field_name)
                else:
                    field_type = input_vars[field_name]
                    if field_type != VariableType.TEXT:
                        raise ValueError(
                            f"Special field '{field_name}' must be of type TEXT, "
                            f"got {field_type}"
                        )
            else:  # List format
                field_found = False
                for var_name, var_type in input_vars:
                    if var_name == field_name:
                        field_found = True
                        if var_type != VariableType.TEXT:
                            raise ValueError(
                                f"Special field '{field_name}' must be of type TEXT, "
                                f"got {var_type}"
                            )
                        break
                if not field_found:
                    invalid_fields.append(field_name)
        
        if invalid_fields:
            raise ValueError(
                f"Special fields not found in input variable list: {invalid_fields}"
            )
            
        return self

    def get_full_payload_path(self) -> str:
        """Get full S3 path for payload"""
        return f"s3://{self.bucket}/{self.sample_payload_s3_key}"

    def get_field_default_value(self, field_name: str, var_type: VariableType) -> str:
        """Get default value for a field"""
        if var_type == VariableType.TEXT:
            if self.special_field_values and field_name in self.special_field_values:
                template = self.special_field_values[field_name]
                try:
                    return template.format(
                        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    )
                except KeyError as e:
                    raise ValueError(f"Invalid placeholder in template for field '{field_name}': {str(e)}")
            return self.default_text_value
        elif var_type == VariableType.NUMERIC:
            return str(self.default_numeric_value)
        else:
            raise ValueError(f"Unknown variable type: {var_type}")   
    
    def generate_csv_payload(self) -> str:
        """
        Generate CSV format payload following the order in source_model_inference_input_variable_list.
    
        Returns:
            Comma-separated string of values
        """
        values = []
        input_vars = self.source_model_inference_input_variable_list
        
        if isinstance(input_vars, dict):
            # Dictionary format
            for field_name, var_type in input_vars.items():
                values.append(self.get_field_default_value(field_name, var_type))
        else:
            # List format
            for field_name, var_type in input_vars:
                values.append(self.get_field_default_value(field_name, var_type))
                
        return ",".join(values)

    def generate_json_payload(self) -> str:
        """
        Generate JSON format payload using source_model_inference_input_variable_list.
    
        Returns:
            JSON string with field names and values
        """
        payload = {}
        input_vars = self.source_model_inference_input_variable_list
        
        if isinstance(input_vars, dict):
            # Dictionary format
            for field_name, var_type in input_vars.items():
                payload[field_name] = self.get_field_default_value(field_name, var_type)
        else:
            # List format
            for field_name, var_type in input_vars:
                payload[field_name] = self.get_field_default_value(field_name, var_type)
                
        return json.dumps(payload)
    
    def generate_sample_payloads(self) -> List[Dict[str, Union[str, dict]]]:
        """
        Generate sample payloads for each content type.
    
        Returns:
            List of dictionaries containing content type and payload
        """
        payloads = []
    
        for content_type in self.source_model_inference_content_types:
            payload_info = {
                "content_type": content_type,
                "payload": None
            }
        
            if content_type == "text/csv":
                payload_info["payload"] = self.generate_csv_payload()
            elif content_type == "application/json":
                payload_info["payload"] = self.generate_json_payload()
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
            
            payloads.append(payload_info)
        
        return payloads

    def save_payloads(self, output_dir: Path) -> List[Path]:
        """
        Save payloads to files.
        
        Args:
            output_dir: Directory to save payload files
            
        Returns:
            List of paths to created payload files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        payloads = self.generate_sample_payloads()
        
        for i, payload_info in enumerate(payloads):
            content_type = payload_info["content_type"]
            payload = payload_info["payload"]
            
            # Determine file extension and name
            ext = ".csv" if content_type == "text/csv" else ".json"
            file_name = f"payload_{content_type.replace('/', '_')}_{i}{ext}"
            file_path = output_dir / file_name
            
            # Save payload
            with open(file_path, "w") as f:
                f.write(payload)
                
            file_paths.append(file_path)
            logger.info(f"Created payload file: {file_path}")
            
        return file_paths

    def upload_payloads_to_s3(self, payload_files: List[Path]) -> str:
        """
        Create tar.gz archive of payload files and upload to S3.
    
        Args:
            payload_files: List of payload file paths to upload
        
        Returns:
            S3 URI of uploaded archive
        
        Raises:
            ValueError: If no payload files provided or S3 upload fails
        """
        if not payload_files:
            raise ValueError("No payload files provided for upload")
        
        if not self.bucket:
            raise ValueError("Bucket not specified in configuration")
        
        if not self.sample_payload_s3_key:
            raise ValueError("sample_payload_s3_key not specified in configuration")
        
        try:
            # Create temporary directory for tar.gz creation
            with tempfile.TemporaryDirectory() as temp_dir:
                archive_path = Path(temp_dir) / "payload.tar.gz"
            
                # Create tar.gz archive
                with tarfile.open(archive_path, "w:gz") as tar:
                    for file_path in payload_files:
                        # Add file to archive with its basename as name
                        tar.add(file_path, arcname=file_path.name)
            
                # Use bucket and key from config
                bucket = self.bucket
                key = self.sample_payload_s3_key
                s3_uri = f"s3://{bucket}/{key}"
            
                logger.info(f"Uploading payloads archive to bucket: {bucket}")
                logger.info(f"Using S3 key: {key}")
            
                # Upload to S3
                s3_client = boto3.client('s3')
                s3_client.upload_file(
                    str(archive_path),
                    bucket,
                    key,
                    #ExtraArgs={'ServerSideEncryption': 'aws:kms'}
                )
                
                logger.info(f"Successfully uploaded payloads to: {s3_uri}")
                return s3_uri
            
        except Exception as e:
            logger.error(f"Failed to upload payloads to S3: {str(e)}")
            raise

    def generate_and_upload_payloads(self) -> str:
        """
        Generate payloads, saveave them, and upload to S3.
        
        Returns:
            S3 URI of uploaded archive
            
        Raises:
            Exception: If any step fails
        """
        # Ensure S3 path is constructed
        if not self.sample_payload_s3_key:
            self.ensure_payload_path()
            logger.info(f"Constructed S3 key: {self.sample_payload_s3_key}")
            
        try:
            # Create temporary directory for payload files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save payloads to temporary directory
                logger.info("Generating and saving payload files...")
                payload_files = self.save_payloads(Path(temp_dir))
                
                # Upload to S3
                logger.info("Uploading payloads to S3...")
                s3_uri = self.upload_payloads_to_s3(payload_files)
                
                return s3_uri
                
        except Exception as e:
            logger.error(f"Failed to generate and upload payloads: {str(e)}")
            raise
            
    def get_effective_source_dir(self) -> Optional[str]:
        """Get the effective source directory"""
        return self.processing_source_dir or self.source_dir
        
    def get_script_path(self) -> str:
        """
        Get script path with priority order:
        1. Use processing_entry_point if provided
        2. Fall back to script_contract.entry_point if available
        
        Always combines with effective source directory.
        
        Returns:
            Script path or None if no entry point can be determined
        """
        # Determine which entry point to use
        entry_point = None
        
        # First priority: Use processing_entry_point if provided
        if self.processing_entry_point:
            entry_point = self.processing_entry_point
        # Second priority: Use contract entry point
        else:
            contract = self.get_script_contract()
            if contract and hasattr(contract, 'entry_point'):
                entry_point = contract.entry_point
        
        if not entry_point:
            return None
        
        # Get the effective source directory
        effective_source_dir = self.get_effective_source_dir()
        if not effective_source_dir:
            return entry_point  # No source dir, just return entry point
        
        # Combine source dir with entry point
        if effective_source_dir.startswith('s3://'):
            full_path = f"{effective_source_dir.rstrip('/')}/{entry_point}"
        else:
            full_path = str(Path(effective_source_dir) / entry_point)
        
        return full_path
        
    def get_script_contract(self) -> 'ScriptContract':
        """
        Get script contract for this configuration.
        
        Returns:
            The MIMS payload script contract
        """
        return MIMS_PAYLOAD_CONTRACT

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Custom serialization"""
        data = super().model_dump(**kwargs)
        # No special handling needed since we're using standard fields from ProcessingStepConfigBase
        return data
