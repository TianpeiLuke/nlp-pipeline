from pydantic import Field
from typing import List 

from .hyperparameters_base import ModelHyperparameters

class BSMModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for the BSM (Buyer Seller Messaging) model training,
    extending the base ModelHyperparameters.
    """
    
    # BSM-specific Training parameters / Overrides if defaults differ (not the case here for overrides)
    adam_epsilon: float = Field(default=1e-08, description="Epsilon for Adam optimizer")
    dropout_keep: float = Field(default=0.1, description="Dropout keep probability")
    
    # Early stopping configuration
    early_stop_metric: str = Field(default='val_loss', description="Metric for early stopping")
    early_stop_patience: int = Field(default=3, gt=0, le=10, description="Patience for early stopping")
    
    # BSM-specific Model configuration
    fixed_tokenizer_length: bool = Field(default=True, description="Use fixed tokenizer length")
    hidden_common_dim: int = Field(default=100, description="Common hidden dimension for multimodal model")
    is_embeddings_trainable: bool = Field(default=True, description="Trainable embeddings flag")
    kernel_size: List[int] = Field(default=[3, 5, 7], description="Kernel sizes for convolutional layers")
    
    # Checkpoint and learning rate decay (learning_rate itself is inherited)
    load_ckpt: bool = Field(default=False, description="Load checkpoint flag")
    lr_decay: float = Field(default=0.05, description="Learning rate decay")
    
    # BSM-specific Training configuration
    max_sen_len: int = Field(default=512, description="Maximum sentence length for tokenizer")
    chunk_trancate: bool = Field(default=True, description="Chunk truncation flag for long texts") # Typo 'trancate' kept as per original
    max_total_chunks: int = Field(default=3, description="Maximum total chunks for processing long texts")
    momentum: float = Field(default=0.9, description="Momentum for SGD optimizer (if SGD is chosen)")
    
    # BSM-specific Network architecture
    num_channels: List[int] = Field(default=[100, 100], description="Number of channels for convolutional layers")
    num_layers: int = Field(default=2, description="Number of layers in the model (e.g., BiLSTM, Transformer encoders)")
    
    # BSM-specific Optimizer and model settings
    pretrained_embedding: bool = Field(default=True, description="Use pretrained embeddings")
    reinit_layers: int = Field(default=2, description="Number of layers to reinitialize from pretrained model")
    reinit_pooler: bool = Field(default=True, description="Reinitialize pooler layer flag")
    run_scheduler: bool = Field(default=True, description="Run learning rate scheduler flag")
    
    # BSM-specific Field configurations for text processing
    text_field_overwrite: bool = Field(default=False, description="Overwrite text field if it exists (e.g. during feature engineering)")
    text_name: str = Field(default='dialogue', description="Name of the primary text field to be processed")
    text_input_ids_key: str = Field(default='input_ids', description="Key name for input_ids from tokenizer output")
    text_attention_mask_key: str = Field(default='attention_mask', description="Key name for attention_mask from tokenizer output")
    tokenizer: str = Field(default='bert-base-multilingual-uncased', description="Tokenizer name or path (e.g., from Hugging Face)")
    
    # BSM-specific Training schedule
    val_check_interval: float = Field(default=0.25, description="Validation check interval during training (float for fraction of epoch, int for steps)")
    warmup_steps: int = Field(default=300, gt=0, le=1000, description="Warmup steps for learning rate scheduler")
    weight_decay: float = Field(default=0.0, description="Weight decay for optimizer (L2 penalty)")