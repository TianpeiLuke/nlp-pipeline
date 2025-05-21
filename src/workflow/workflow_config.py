from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Dict, Any
from pathlib import Path
import json


class ModelHyperparameters(BaseModel):
    # Field lists
    full_field_list: List[str] = Field(default=[
        'order_id', 'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag',
        'dialogue', 'llm_reversal_flag'
    ])
    
    cat_field_list: List[str] = Field(default=['dialogue'])
    
    tab_field_list: List[str] = Field(default=[
        'net_conc_amt', 'ttm_conc_amt', 'ttm_conc_count',
        'concsi', 'deliverable_flag', 'undeliverable_flag'
    ])

    # Training parameters
    batch_size: int = Field(gt=0)
    adam_epsilon: float = Field(default=1e-08)
    class_weights: List[float]
    device: int = Field(default=-1)
    dropout_keep: float = Field(default=0.1)
    
    # Early stopping configuration
    early_stop_metric: str = Field(default='val_loss')
    early_stop_patience: int = Field(default=3)
    
    # Model configuration
    fixed_tokenizer_length: bool = Field(default=True)
    header: int = Field(default=0)
    hidden_common_dim: int = Field(default=100)
    input_tab_dim: int
    id_name: str
    is_binary: bool
    is_embeddings_trainable: bool = Field(default=True)
    kernel_size: List[int] = Field(default=[3, 5, 7])
    label_name: str
    
    # Checkpoint and learning rate
    load_ckpt: bool = Field(default=False)
    lr: float = Field(default=3e-05)
    lr_decay: float = Field(default=0.05)
    
    # Training configuration
    max_epochs: int = Field(default=3)
    max_sen_len: int = Field(default=512)
    chunk_trancate: bool = Field(default=True)
    max_total_chunks: int = Field(default=3)
    metric_choices: List[str] = Field(default=['f1_score', 'auroc'])
    model_class: str
    momentum: float = Field(default=0.9)
    
    # Network architecture
    num_channels: List[int] = Field(default=[100, 100])
    num_classes: int
    multiclass_categories: List[int]
    num_layers: int = Field(default=2)
    
    # Optimizer and model settings
    optimizer: str = Field(default='SGD')
    pretrained_embedding: bool = Field(default=True)
    reinit_layers: int = Field(default=2)
    reinit_pooler: bool = Field(default=True)
    run_scheduler: bool = Field(default=True)
    
    # Field configurations
    tab_field_dim: int
    tab_field_list: List[str]
    text_field_overwrite: bool = Field(default=False)
    text_name: str
    text_input_ids_key: str = Field(default='input_ids')
    text_attention_mask_key: str = Field(default='attention_mask')
    tokenizer: str = Field(default='bert-base-multilingual-uncased')
    
    # Training schedule
    val_check_interval: float = Field(default=0.25)
    warmup_steps: int = Field(default=300)
    weight_decay: float = Field(default=0)
    
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
    region: str = Field(description="region (NA, EU, FE)")
    pipeline_name: str = Field(description="Pipeline name")

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
    instance_type: str = Field(default=training_instance_type)
    framework_version: str = Field(default=framework_version)
    py_version: str = Field(default=py_version)
    volume_size: int = Field(default=training_volume_size)
    entry_point: str = Field(default=entry_point)
    source_dir: str = Field(default=source_dir)
    instance_count: int = Field(default=training_instance_count)
    
    inference_instance_type: str = Field(default=inference_instance_type)
    container_startup_health_check_timeout: int = Field(default=300)
    container_memory_limit: int = Field(default=6144, ge=1024)
    data_download_timeout: int = Field(default=900)
    inference_memory_limit: int = Field(default=6144)
    max_concurrent_invocations: int = Field(default=1)
    max_payload_size: int = Field(default=6)
    
    processing_instance_type: str = Field(default=processing_sagemaker_instance_type_small)
    processing_instance_count: int = Field(default=processing_instance_count)
    processing_volume_size: int = Field(default=processing_volume_size)
    sklearn_version: str = Field(default="1.0-1")

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