from pydantic import BaseModel, Field, model_validator, field_validator, ValidationInfo
from typing import List, Optional, Dict, Any, ClassVar, TYPE_CHECKING
from pathlib import Path
import json
from datetime import datetime
import logging

# Import for type hints only
if TYPE_CHECKING:
    from ..pipeline_script_contracts.base_script_contract import ScriptContract
else:
    # Just for type hints, won't be used at runtime if not available
    ScriptContract = Any

logger = logging.getLogger(__name__)

# Import step registry from central source of truth
from ..pipeline_registry.step_names import CONFIG_STEP_REGISTRY as STEP_REGISTRY


class BasePipelineConfig(BaseModel):
    """Base configuration with shared pipeline attributes."""
    
    # Class variables using ClassVar for Pydantic
    REGION_MAPPING: ClassVar[Dict[str, str]] = {
        "NA": "us-east-1",
        "EU": "eu-west-1",
        "FE": "us-west-2"
    }
    
    STEP_NAMES: ClassVar[Dict[str, str]] = STEP_REGISTRY
    
    # Note: input_names and output_names have been removed in favor of script contracts
    
    # Shared basic info
    bucket: str = Field(description="S3 bucket name for pipeline artifacts and data.")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date, typically used for versioning or pathing."
    )
    region: str = Field(default='NA', description="Custom region code (NA, EU, FE) for internal logic.")
    aws_region: Optional[str] = Field(default=None, description="Derived AWS region (e.g., us-east-1).")
    author: str = Field(description="Author or owner of the pipeline.")

    # Overall pipeline identification
    pipeline_name: str = Field(description="Name of the SageMaker Pipeline.")
    pipeline_description: str = Field(description="Description for the SageMaker Pipeline.")
    pipeline_version: str = Field(description="Version string for the SageMaker Pipeline.")
    pipeline_s3_loc: str = Field(
        description="Root S3 location for storing pipeline definition and step artifacts.",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )

    # Common framework/scripting info (if shared across steps like train/inference)
    framework_version: str = Field(default='2.1.0', description="Default framework version (e.g., PyTorch).")
    py_version: str = Field(default='py310', description="Default Python version.")
    # General source_dir, specific entry points will be in step configs
    source_dir: Optional[str] = Field(default=None, description="Common source directory for scripts if applicable. Can be overridden by step configs.")


    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        # extra = 'forbid' # Or 'allow' / 'ignore' as per your preference
        protected_namespaces = ()

    @model_validator(mode='before')
    @classmethod
    def _construct_base_attributes(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sets up default values for core attributes if not provided.
        """
        # Resolve bucket and author using sais_session if not provided
        if 'bucket' not in values or values['bucket'] is None:
            values['bucket'] = 'my-default-bucket'
        if 'author' not in values or values['author'] is None:
            values['author'] = 'my-default-author'

        # Ensure current_date is set
        if 'current_date' not in values or values['current_date'] is None:
            values['current_date'] = datetime.now().strftime("%Y-%m-%d")
        
        # Derive aws_region from custom region code
        region_code = values.get('region', 'NA')
        values['region'] = region_code

        if 'aws_region' not in values or values['aws_region'] is None:
            region_code = values.get('region', 'NA')
            values['aws_region'] = cls.REGION_MAPPING.get(region_code, "us-east-1")

        # Construct pipeline name if not provided
        if not values.get('pipeline_name'):
            author = values.get('author')
            region = values.get('region')
            values['pipeline_name'] = f'{author}-BSM-RnR-{region}'

        # Construct pipeline description if not provided
        if not values.get('pipeline_description'):
            region = values.get('region')
            values['pipeline_description'] = f'BSM RnR {region}'

        # Set default pipeline version if not provided
        if not values.get('pipeline_version'):
            values['pipeline_version'] = '0.1.0'

        # Construct pipeline S3 location if not provided
        if not values.get('pipeline_s3_loc'):
            bucket = values.get('bucket')
            pipeline_name = values.get('pipeline_name')
            pipeline_version = values.get('pipeline_version')
            pipeline_subdirectory = 'MODS'
            pipeline_subsubdirectory = f"{pipeline_name}_{pipeline_version}"
            values['pipeline_s3_loc'] = f"s3://{bucket}/{pipeline_subdirectory}/{pipeline_subsubdirectory}"
        
        return values

    @model_validator(mode='after')
    def validate_dependencies(self) -> 'BasePipelineConfig':
        """Validate interdependent fields after all values are set"""
        # Ensure all required fields are present
        required_fields = ['bucket', 'author', 'region', 'pipeline_name', 
                         'pipeline_description', 'pipeline_version', 'pipeline_s3_loc']
        
        for field in required_fields:
            if not getattr(self, field):
                raise ValueError(f"Required field '{field}' is missing or empty")
        
        return self


    @field_validator('region')
    @classmethod
    def _validate_custom_region(cls, v: str) -> str:
        valid_regions = ['NA', 'EU', 'FE']
        if v not in valid_regions:
            raise ValueError(f"Invalid custom region code: {v}. Must be one of {valid_regions}")
        return v

    @field_validator('source_dir', check_fields=False) # check_fields=False if source_dir can be None
    @classmethod
    def _validate_source_dir_exists(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.startswith('s3://'): # Only validate local paths
            if not Path(v).exists():
                raise ValueError(f"Local source directory does not exist: {v}")
            if not Path(v).is_dir():
                raise ValueError(f"Local source_dir is not a directory: {v}")
        return v

    def get_script_contract(self) -> Optional['ScriptContract']:
        """
        Get script contract for this configuration.
        
        This base implementation returns None. Derived classes should override
        this method to return their specific script contract.
        
        Returns:
            Script contract instance or None if not available
        """
        # Check for hardcoded script_contract first (for backward compatibility)
        if hasattr(self, '_script_contract'):
            return self._script_contract
            
        # Otherwise attempt to load based on class and job_type
        try:
            class_name = self.__class__.__name__.replace('Config', '')
            
            # Try with job_type if available
            if hasattr(self, 'job_type') and self.job_type:
                module_name = f"..pipeline_script_contracts.{class_name.lower()}_{self.job_type.lower()}_contract"
                contract_name = f"{class_name.upper()}_{self.job_type.upper()}_CONTRACT"
                
                try:
                    contract_module = __import__(module_name, fromlist=[''])
                    if hasattr(contract_module, contract_name):
                        return getattr(contract_module, contract_name)
                except (ImportError, AttributeError):
                    pass
            
            # Try without job_type
            module_name = f"..pipeline_script_contracts.{class_name.lower()}_contract"
            contract_name = f"{class_name.upper()}_CONTRACT"
            
            try:
                contract_module = __import__(module_name, fromlist=[''])
                if hasattr(contract_module, contract_name):
                    return getattr(contract_module, contract_name)
            except (ImportError, AttributeError):
                pass
                
        except Exception as e:
            logger.debug(f"Error loading script contract: {e}")
            
        return None
        
    @property
    def script_contract(self) -> Optional['ScriptContract']:
        """
        Property accessor for script contract.
        
        Returns:
            Script contract instance or None if not available
        """
        return self.get_script_contract()
        
    def get_script_path(self, default_path: str = None) -> str:
        """
        Get script path, preferring contract-defined path if available.
        
        Args:
            default_path: Default script path to use if not found in contract
            
        Returns:
            Script path
        """
        # Try to get from contract
        contract = self.get_script_contract()
        if contract and hasattr(contract, 'script_path'):
            return contract.script_path
            
        # Fall back to default or hardcoded path
        if hasattr(self, 'script_path'):
            return self.script_path
            
        return default_path

    @classmethod
    def get_step_name(cls, config_class_name: str) -> str:
        """Get the step name for a configuration class"""
        return cls.STEP_NAMES.get(config_class_name, config_class_name)

    @classmethod
    def get_config_class_name(cls, step_name: str) -> str:
        """Get the configuration class name from a step name"""
        reverse_mapping = {v: k for k, v in cls.STEP_NAMES.items()}
        return reverse_mapping.get(step_name, step_name)
