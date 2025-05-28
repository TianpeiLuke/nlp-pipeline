from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime

from .hyperparameters_base import ModelHyperparameters

class BSMModelHyperparameters(ModelHyperparameters):
    """Hyperparameters for the BSM model training"""
    # Inherit all fields from ModelHyperparameters
    # Note: The ModelHyperparameters class should already define the necessary fields
    
    # Training parameters
    batch_size: int = Field(default=2, gt=0, le=256, description="Batch size for training")
    adam_epsilon: float = Field(default=1e-08, description="Epsilon for Adam optimizer")
    class_weights: List[float] = Field(default=[1.0, 1.0], description="Class weights for loss function")
    device: int = Field(default=-1, description="Device ID for training")
    dropout_keep: float = Field(default=0.1, description="Dropout keep probability")
    
    # Early stopping configuration
    early_stop_metric: str = Field(default='val_loss', description="Metric for early stopping")
    early_stop_patience: int = Field(default=3, gt=0, le=10, description="Patience for early stopping")
    
    # Model configuration
    fixed_tokenizer_length: bool = Field(default=True, description="Use fixed tokenizer length")
    header: int = Field(default=0,  description="Header row for CSV files")
    hidden_common_dim: int = Field(default=100, description="Common hidden dimension")
    input_tab_dim: int  = Field(default=6, description="Input tabular dimension")
    id_name: str = Field(default='order_id', description="ID field name")    
    is_binary: bool = Field(default=True, description="Binary classification flag")
    is_embeddings_trainable: bool = Field(default=True, description="Trainable embeddings flag")
    kernel_size: List[int] = Field(default=[3, 5, 7], description="Kernel sizes for convolutional layers")
    label_name: str = Field(default='llm_reversal_flag', description="Label field name")
    
    # Checkpoint and learning rate
    load_ckpt: bool = Field(default=False, description="Load checkpoint flag")
    lr: float = Field(default=3e-05, description="Learning rate")
    lr_decay: float = Field(default=0.05, description="Learning rate decay")
    
    # Training configuration
    max_epochs: int = Field(default=3, gt=0, le=10, description="Maximum epochs for training")
    max_sen_len: int = Field(default=512, description="Maximum sentence length")
    chunk_trancate: bool = Field(default=True, description="Chunk truncation flag")
    max_total_chunks: int = Field(default=3, description="Maximum total chunks")
    metric_choices: List[str] = Field(default=['f1_score', 'auroc'], description="Metric choices for evaluation")
    model_class: str = Field(default='multimodal_bert', description="Model class name")
    momentum: float = Field(default=0.9, description="Momentum for SGD optimizer")
    
    # Network architecture
    num_channels: List[int] = Field(default=[100, 100], description="Number of channels for convolutional layers")
    num_classes: int = Field(default=2, description="Number of classes for classification")
    multiclass_categories: List[int] = Field(default=[0, 1], description="Multiclass categories")
    num_layers: int = Field(default=2, description="Number of layers in the model")
    
    # Optimizer and model settings
    optimizer: str = Field(default='SGD', description="Optimizer type")
    pretrained_embedding: bool = Field(default=True, description="Use pretrained embeddings")
    reinit_layers: int = Field(default=2, description="Number of layers to reinitialize")
    reinit_pooler: bool = Field(default=True, description="Reinitialize pooler flag")
    run_scheduler: bool = Field(default=True, description="Run scheduler flag")
    
    # Field configurations
    tab_field_dim: int = Field(default=6, description="Tabular field dimension")
    text_field_overwrite: bool = Field(default=False, description="Overwrite text field flag")
    text_name: str = Field(default='dialogue', description="Text field name")
    text_input_ids_key: str = Field(default='input_ids', description="Input IDs key for text field")
    text_attention_mask_key: str = Field(default='attention_mask', description="Attention mask key for text field")
    tokenizer: str = Field(default='bert-base-multilingual-uncased', description="Tokenizer name")
    
    # Training schedule
    val_check_interval: float = Field(default=0.25, description="Validation check interval")
    warmup_steps: int = Field(default=300, gt=0, le=1000, description="Warmup steps for learning rate scheduler")
    weight_decay: float = Field(default=0, description="Weight decay for optimizer")
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'
        protected_namespaces = ()

    @model_validator(mode='after')
    def validate_dimensions(self) -> 'ModelHyperparameters':
        """Validate model dimensions and configurations"""
        # Validate class weights match number of classes
        if len(self.class_weights) != len(self.multiclass_categories):
            raise ValueError(
                f"The dimension of class_weights ({len(self.class_weights)}) "
                f"does not match with number of classes ({len(self.multiclass_categories)})"
            )
        
        # Validate tab field dimensions
        if self.tab_field_dim != len(self.tab_field_list):
            raise ValueError(
                f"tab_field_dim ({self.tab_field_dim}) does not match "
                f"length of tab_field_list ({len(self.tab_field_list)})"
            )
        
        # Validate input tab dimensions
        if self.input_tab_dim != len(self.tab_field_list):
            raise ValueError(
                f"input_tab_dim ({self.input_tab_dim}) does not match "
                f"length of tab_field_list ({len(self.tab_field_list)})"
            )
        
        # Validate binary classification setting
        if self.is_binary and len(self.multiclass_categories) != 2:
            raise ValueError(
                f"is_binary is True but number of classes is {len(self.multiclass_categories)}"
            )
        
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        return self.model_dump()

    def serialize_config(self) -> Dict[str, str]:
        """Serialize configuration for SageMaker"""
        config = self.get_config()
        return {
            k: json.dumps(v) if isinstance(v, (list, dict, bool)) else str(v)
            for k, v in config.items()
        }




class ModelConfig(BaseModel):
    """Primary model configuration"""
    # Required fields from config
    bucket: str = Field(description="S3 bucket name")
    current_date: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d"),
        description="Current date in YYYY-MM-DD format")
    region: str = Field(default='NA', description="region (NA, EU, FE)")
    aws_region: Optional[str] = Field(default=None, description="AWS region based on model registration region")

    # S3 paths with updated pattern
    input_path: str = Field(
        description="S3 path for input data",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )
    output_path: str = Field(
        description="S3 path for output data",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )
    checkpoint_path: Optional[str] = Field(
        default=None,
        description="Optional S3 path for model checkpoints",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )
    
    # Pipeline Config
    author: str = Field(description="Author of the pipeline")
    pipeline_name: str = Field(description="Name of the pipeline")
    pipeline_description: str = Field(description="Description of the pipeline")
    pipeline_version: str = Field(description="Version of the pipeline")
    pipeline_s3_loc: str = Field(
        description="S3 location for pipeline artifacts",
        pattern=r'^s3://[a-zA-Z0-9.-][a-zA-Z0-9.-]*(?:/[a-zA-Z0-9.-][a-zA-Z0-9._-]*)*$'
    )

    # Training Configuration
    instance_type: str = Field(default='ml.g5.12xlarge', description="Instance type for training")
    framework_version: str = Field(default='2.1.0', description="Framework version")
    py_version: str = Field(default='py310', description="Python version")
    volume_size: int = Field(default=500, ge=10, le=1000, description="Volume size in GB")
    entry_point: str = Field(default='train.py', description="Entry point for training script")
    source_dir: str = Field(default=None, description="Source directory for training script")
    instance_count: int = Field(default=1)
    
    # Endpoint Configuration
    inference_instance_type: str = Field(default='ml.m5.4xlarge', description="Instance type for inference")
    inference_entry_point: str = Field(default='inference.py',  description="Entry point for inference script")
    initial_instance_count: int = Field(default=1, ge=1, le=10, description="Initial instance count for inference")
    
    container_startup_health_check_timeout: int = Field(default=300, ge=0, le=3600, description="Timeout for container startup health check")
    container_memory_limit: int = Field(default=6144, ge=1024, le=61440, description="Memory limit for the container in MB")
    data_download_timeout: int = Field(default=900, ge=0, le=3600, description="Timeout for data download in seconds")
    inference_memory_limit: int = Field(default=6144, ge=1024, le=61440, description="Memory limit for inference in MB")
    max_concurrent_invocations: int = Field(default=1, ge=1, le=10, description="Max concurrent invocations for the endpoint")
    
    
    # Model Registration Configuration
    model_owner: str = Field(default="amzn1.abacus.team.djmdvixm5abr3p75c5ca", description="Team ID of abuse-analytics")
    model_registration_domain: str = Field(default="BuyerSellerMessaging", description="Domain of model registry")
    model_registration_objective: Optional[str] = Field(default=None, description="Objective of model registry")
    
    
    # Payload Test
    expected_tps: int = Field(default=2, description="Expected transactions per second")
    max_latency_in_millisecond: int = Field(default=800, description="Maximum latency in milliseconds")
    max_acceptable_error_rate: float = Field(default=0.2, description="Maximum acceptable error rate")
    max_payload_size: int = Field(default=6, ge=1, le=6, description="Max payload size for the endpoint in MB")
    sample_payload_s3_key: str = Field(
        default="",
        description="S3 key for the sample payload file"
    )
    model_var_list: List[str] = Field(default_factory=list, description="List of model variables")

    source_model_inference_output_variable_list: Dict[str, str] = Field(
        default={
            'legacy-score': 'NUMERIC'
        },
        description="Source model inference output variable list"
    )
    source_model_inference_input_variable_list: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of input variables and their types"
    )
    source_model_inference_content_types: List[str] = Field(
        default=["text/csv"],
        description="Allowed content types for model inference input"
    )
    source_model_inference_response_types: List[str] = Field(
        default=["application/json"],
        description="Allowed response types for model inference output"
    )


    # Processing Step Configuration
    processing_instance_type: str = Field(default='ml.m5.4xlarge', description="Instance type for processing jobs")
    processing_instance_count: int = Field(default=1, ge=1, le=10, description="Instance count for processing jobs")
    processing_volume_size: int = Field(default=500, ge=10, le=1000, description="Volume size for processing jobs in GB")

    # Add reference to hyperparameters
    hyperparameters: Optional[ModelHyperparameters] = Field(None, description="Model hyperparameters")

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = 'forbid'
        protected_namespaces = ()
    
    
    @model_validator(mode='before')
    @classmethod
    def construct_paths(cls, values: dict) -> dict:
        """Construct S3 paths before validation"""
        bucket = values.get('bucket', sais_session.team_owned_s3_bucket_name())

        current_date = values.get('current_date', datetime.now().strftime("%Y-%m-%d"))
        region = values.get('region', 'NA')
        author = values.get('author', sais_session.owner_alias())
        region_mapping = {
            'NA': "us-east-1",
            'EU': "eu-west-1",
            'FE': "us-west-2"
        }
        if 'aws_region' not in values:
            values['aws_region'] = region_mapping[region]

        # Construct paths if not provided
        if not values.get('input_path'):
            values['input_path'] = f"s3://{bucket}/train_test_val/{current_date}"
        
        if not values.get('output_path'):
            values['output_path'] = f"s3://{bucket}/models/{current_date}"
        
        if not values.get('checkpoint_path'):
            values['checkpoint_path'] = f"s3://{bucket}/checkpointing/{current_date}"

        # Construct pipeline name if not provided
        pipeline_name = values.get('pipeline_name')
        if not pipeline_name:
            pipeline_name = f'{author}-BSM-RnR-{region}'
            values['pipeline_name'] = pipeline_name

        # Construct pipeline description if not provided
        if not values.get('pipeline_description'):
            values['pipeline_description'] = f'BSM RnR {region}'

        # Set default pipeline version if not provided
        pipeline_version = values.get('pipeline_version')
        if not pipeline_version:
            pipeline_version = '0.1.0'
            values['pipeline_version'] = pipeline_version


        # Construct pipeline S3 location if not provided
        if not values.get('pipeline_s3_loc'):
            pipeline_subdirectory = 'MODS'  # You might want to make this configurable
            pipeline_subsubdirectory = f"{pipeline_name}_{pipeline_version}"
            values['pipeline_s3_loc'] = f"s3://{Path(bucket) / pipeline_subdirectory / pipeline_subsubdirectory}"
            
        # Construct sample payload S3 key
        model_objective= f'RnR_BSM_Model_{region}'
        payload_file_name = f'payload_{pipeline_name}_{pipeline_version}_{model_objective}'
        values['sample_payload_s3_key'] = f'mods/payload/{payload_file_name}.tar.gz'
        
        return values

    
    @model_validator(mode='after')
    def validate_paths(self) -> 'ModelConfig':
        """Validate all path relationships and requirements"""
        paths = {
            'input_path': self.input_path,
            'output_path': self.output_path
        }
        if self.checkpoint_path:
            paths['checkpoint_path'] = self.checkpoint_path

        # Check for uniqueness
        if len(set(paths.values())) != len(paths):
            raise ValueError("All paths (input, output, checkpoint) must be different")

        # Validate minimum path depths
        min_depth = 2
        for path_name, path in paths.items():
            depth = len(path.split('/')[3:])
            if depth < min_depth:
                raise ValueError(f"{path_name} must have at least {min_depth} levels of hierarchy")

        return self

    
    @model_validator(mode='after')
    def validate_field_lists(self) -> 'ModelConfig':
        """Validate field lists from hyperparameters"""
        if not self.hyperparameters:
            raise ValueError("hyperparameters must be provided")

        # Check if all fields in tab_field_list and cat_field_list are in full_field_list
        all_fields = set(self.hyperparameters.full_field_list)
        if not set(self.hyperparameters.tab_field_list).issubset(all_fields):
            raise ValueError("All fields in tab_field_list must be in full_field_list")
        if not set(self.hyperparameters.cat_field_list).issubset(all_fields):
            raise ValueError("All fields in cat_field_list must be in full_field_list")
        
        # Check if label_name and id_name are in full_field_list
        if self.hyperparameters.label_name not in all_fields:
            raise ValueError(f"label_name '{self.hyperparameters.label_name}' must be in full_field_list")
        if self.hyperparameters.id_name not in all_fields:
            raise ValueError(f"id_name '{self.hyperparameters.id_name}' must be in full_field_list")

        return self

    

    def get_checkpoint_uri(self) -> Optional[str]:
        """Returns the checkpoint URI if it exists"""
        return self.checkpoint_path

    def has_checkpoint(self) -> bool:
        """Returns whether a checkpoint path is configured"""
        return self.checkpoint_path is not None

    
    @field_validator('source_dir')
    @classmethod
    def validate_source_dir(cls, v: str) -> str:
        if not Path(v).exists():
            raise ValueError(f"Source directory does not exist: {v}")
        return v


    @field_validator('instance_type')
    @classmethod
    def validate_instance_type(cls, v: str) -> str:
        valid_instances = [
            "ml.g4dn.16xlarge", 
            "ml.g5.12xlarge", 
            "ml.g5.16xlarge",
            "ml.p3.8xlarge", 
            "ml.m5.12xlarge",
            "ml.p3.16xlarge"
        ]
        if v not in valid_instances:
            raise ValueError(
                f"Invalid training instance type: {v}. "
                f"Must be one of: {', '.join(valid_instances)}"
            )
        return v

    @field_validator('inference_memory_limit')
    @classmethod
    def validate_memory_limits(cls, v: int, info) -> int:
        container_memory_limit = info.data.get('container_memory_limit')
        if container_memory_limit and v > container_memory_limit:
            raise ValueError("Inference memory limit cannot exceed container memory limit")
        return v
    
    @field_validator('region')
    @classmethod
    def validate_region(cls, v: str) -> str:
        valid_regions = ['NA', 'EU', 'FE']
        if v not in valid_regions:
            raise ValueError(f"Invalid region: {v}. Must be one of {valid_regions}")
        return v

    @field_validator('max_acceptable_error_rate')
    @classmethod
    def validate_max_acceptable_error_rate(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"max_acceptable_error_rate must be between 0 and 1, got {v}")
        return v

    @field_validator('source_model_inference_content_types', 'source_model_inference_response_types')
    @classmethod
    def validate_types(cls, v):
        allowed_types = {"text/csv", "application/json"}
        if not set(v).issubset(allowed_types):
            raise ValueError(f"Only 'text/csv' and 'application/json' are allowed. Got: {v}")
        return v