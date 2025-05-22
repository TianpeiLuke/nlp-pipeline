from pydantic import BaseModel, Field, model_validator, field_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime


class ModelHyperparameters(BaseModel):
    # Field lists
    full_field_list: List[str] = Field(default=[
        'order_id', 'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag',
        'dialogue', 'llm_reversal_flag'
    ],  description="Full field list")
    
    cat_field_list: List[str] = Field(default=['dialogue'], description="Categorical fields")
    
    tab_field_list: List[str] = Field(default=[
        'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag'
    ], description="Tabular fields")

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
    pipeline_name: str = Field(default='pipeline', description="Pipeline name")

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

    # Rest of the configurations...
    instance_type: str = Field(default='ml.g5.12xlarge', description="Instance type for training")
    framework_version: str = Field(default='2.1.0', description="Framework version")
    py_version: str = Field(default='py310', description="Python version")
    volume_size: int = Field(default=500, ge=10, le=1000, description="Volume size in GB")
    entry_point: str = Field(default='train.py', description="Entry point for training script")
    source_dir: str = Field(default=None, description="Source directory for training script")
    instance_count: int = Field(default=1)
    
    inference_instance_type: str = Field(default='ml.m5.4xlarge', description="Instance type for inference")
    inference_entry_point: str = Field(default='inference.py',  description="Entry point for inference script")
    initial_instance_count: int = Field(default=1, ge=1, le=10, description="Initial instance count for inference")

    endpoint_name_prefix: Optional[str] = Field(
        default=None,
        description="Prefix for the endpoint name. If None, a random name will be generated."
    )
    tags: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of tags to apply to the SageMaker resources"
    )

    container_startup_health_check_timeout: int = Field(default=300, ge=0, le=3600, description="Timeout for container startup health check")
    container_memory_limit: int = Field(default=6144, ge=1024, le=61440, description="Memory limit for the container in MB")
    data_download_timeout: int = Field(default=900, ge=0, le=3600, description="Timeout for data download in seconds")
    inference_memory_limit: int = Field(default=6144, ge=1024, le=61440, description="Memory limit for inference in MB")
    max_concurrent_invocations: int = Field(default=1, ge=1, le=10, description="Max concurrent invocations for the endpoint")
    max_payload_size: int = Field(default=6, ge=1, le=6, description="Max payload size for the endpoint in MB")
    
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
        bucket = values.get('bucket')
        if not bucket:
            raise ValueError("Bucket name is required")

        current_date = values.get('current_date', datetime.now().strftime("%Y-%m-%d"))

        # Construct paths if not provided
        if not values.get('input_path'):
            values['input_path'] = f"s3://{bucket}/train_test_val/{current_date}"
        
        if not values.get('output_path'):
            values['output_path'] = f"s3://{bucket}/models/{current_date}"
        
        if not values.get('checkpoint_path'):
            values['checkpoint_path'] = f"s3://{bucket}/checkpointing/{current_date}"

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
    

def save_config_to_json(
    model_config: ModelConfig, 
    hyperparams: ModelHyperparameters, 
    config_path: str = "config/config.json"
) -> Path:
    """
    Save ModelConfig and ModelHyperparameters to JSON file
    
    Args:
        model_config: ModelConfig instance
        hyperparams: ModelHyperparameters instance
        config_path: Path to save the config file
    
    Returns:
        Path object of the saved config file
    """
    try:
        # Convert both models to dictionaries
        config_dict = model_config.model_dump()
        hyperparams_dict = hyperparams.model_dump()
        
        # Combine both dictionaries
        combined_config = {**config_dict, **hyperparams_dict}
        
        # Create config directory if it doesn't exist
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        with open(path, 'w') as f:
            json.dump(combined_config, f, indent=2, sort_keys=True)
            
        print(f"Configuration saved to: {path}")
        return path
        
    except Exception as e:
        raise ValueError(f"Failed to save config: {str(e)}")