import os
import json
import traceback
from io import StringIO, BytesIO
import logging
from typing import List, Union, Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader
from flask import Response
from transformers import AutoTokenizer

from processing.processors import (
    Processor,
    HTMLNormalizerProcessor,
    EmojiRemoverProcessor,
    TextNormalizationProcessor,
    DialogueSplitterProcessor,
    DialogueChunkerProcessor,
)
from processing.bert_tokenize_processor import TokenizationProcessor
from processing.categorical_label_processor import CategoricalLabelProcessor
from processing.multiclass_label_processor import MultiClassLabelProcessor
from processing.bsm_datasets import BSMDataset
from processing.bsm_dataloader import build_collate_batch
from lightning_models.pl_tab_ae import TabAE
from lightning_models.pl_text_cnn import TextCNN
from lightning_models.pl_multimodal_cnn import MultimodalCNN
from lightning_models.pl_multimodal_bert import MultimodalBert
from lightning_models.pl_bert_classification import TextBertClassification
from lightning_models.pl_lstm import TextLSTM
from lightning_models.pl_train import (
    model_inference,
    model_online_inference,
    predict_stack_transform,
    load_model,
    load_artifacts,
    load_checkpoint,
    load_onnx_model
)
from lightning_models.pl_model_plots import (
    compute_metrics,
    roc_metric_plot,
    pr_metric_plot,
)
from lightning_models.dist_utils import get_rank, is_main_process
from pydantic import BaseModel, Field, ValidationError, field_validator  # For Config Validation

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


# ================== Model, Data and Hyperparameter Folder =================
prefix = "/opt/ml/"
input_path = os.path.join(prefix, "input/data")
output_path = os.path.join(prefix, "output")
model_path = os.path.join(prefix, "model")
hparam_path = os.path.join(prefix, "input/config/hyperparameters.json")
checkpoint_path = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
train_channel = "train"
train_path = os.path.join(input_path, train_channel)
val_channel = "val"
val_path = os.path.join(input_path, val_channel)
test_channel = "test"
test_path = os.path.join(input_path, test_channel)
#==========================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================================================================================
class Config(BaseModel):
    id_name: str = "order_id"
    text_name: str = "text"
    label_name: str = "label"
    batch_size: int = 32
    full_field_list: List[str] = Field(default_factory=list)
    cat_field_list: List[str] = Field(default_factory=list)
    tab_field_list: List[str] = Field(default_factory=list)
    categorical_features_to_encode: List[str] = Field(default_factory=list)
    header: int = 0
    max_sen_len: int = 512
    chunk_trancate: bool = False
    max_total_chunks: int = 5
    kernel_size: List[int] = Field(default_factory=lambda: [3, 5, 7])
    num_layers: int = 2
    num_channels: List[int] = Field(default_factory=lambda: [100, 100])
    hidden_common_dim: int = 100
    input_tab_dim: int = 11
    num_classes: int = 2
    is_binary: bool = True
    multiclass_categories: List[Union[int, str]] = Field(default_factory=lambda: [0, 1])
    max_epochs: int = 10
    lr: float = 0.02
    lr_decay: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0
    class_weights: List[int] = Field(default_factory=lambda: [1, 10])
    dropout_keep: float = 0.5
    optimizer: str = "SGD"
    fixed_tokenizer_length: bool = True
    is_embeddings_trainable: bool = True
    tokenizer: str = "bert-base-multilingual-cased"
    metric_choices: List[str] = Field(default_factory=lambda: ["auroc", "f1_score"])
    early_stop_metric: str = "val/f1_score"
    early_stop_patience: int = 3
    gradient_clip_val: float = 1.0
    model_class: str = "multimodal_bert"
    load_ckpt: bool = False
    val_check_interval: float = 0.25
    adam_epsilon: float = 1e-08
    fp16: bool = False
    run_scheduler: bool = True
    reinit_pooler: bool = True
    reinit_layers: int = 2
    warmup_steps: int = 300
    text_input_ids_key: str = "input_ids"  # Configurable text input key
    text_attention_mask_key: str = "attention_mask"  # Configurable attention mask key
    train_filename: Optional[str] = None
    val_filename: Optional[str] = None
    test_filename: Optional[str] = None
    embed_size: Optional[int] = None  # Added for type consistency
    model_path: str = "/opt/ml/model"  # Add model_path with a default value
    categorical_processor_mappings: Optional[Dict[str, Dict[str, int]]] = None  # Add this line
    label_to_id: Optional[Dict[str, int]] = None  # Added: label to ID mapping
    id_to_label: Optional[List[str]] = None  # Added: ID to label mapping
    
    
    def model_post_init(self, __context):
        # Validate consistency between multiclass_categories and num_classes
        if self.is_binary and self.num_classes != 2:
            raise ValueError("For binary classification, num_classes must be 2.")
        if not self.is_binary:
            if self.num_classes < 2:
                raise ValueError("For multiclass classification, num_classes must be >= 2.")
            if not self.multiclass_categories:
                raise ValueError("multiclass_categories must be provided for multiclass classification.")
            if len(self.multiclass_categories) != self.num_classes:
                raise ValueError(
                    f"num_classes={self.num_classes} does not match "
                    f"len(multiclass_categories)={len(self.multiclass_categories)}"
                )
            if len(set(self.multiclass_categories)) != len(self.multiclass_categories):
                raise ValueError("multiclass_categories must contain unique values.")
        else:
            # Optional: Warn if multiclass_categories is defined when binary
            if self.multiclass_categories and len(self.multiclass_categories) != 2:
                raise ValueError("For binary classification, multiclass_categories must contain exactly 2 items.")
                
        # New: validate class_weights length
        if self.class_weights and len(self.class_weights) != self.num_classes:
            raise ValueError(
                f"class_weights must have the same number of elements as num_classes "
                f"(expected {self.num_classes}, got {len(self.class_weights)})."


#=================== Helper Function ================
def data_preprocess_pipeline(config: Config) -> Tuple[AutoTokenizer, Dict[str, Processor]]:
    if not config.tokenizer:
        config.tokenizer = "bert-base-multilingual-cased"
    logger.info(f"Constructing tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    dialogue_pipeline = (
        HTMLNormalizerProcessor()
        >> EmojiRemoverProcessor()
        >> TextNormalizationProcessor()
        >> DialogueSplitterProcessor()
        >> DialogueChunkerProcessor(tokenizer=tokenizer, 
                                    max_tokens=config.max_sen_len,
                                    truncate=config.chunk_trancate,
                                    max_total_chunks=config.max_total_chunks
                                   )
        >> TokenizationProcessor(
            tokenizer,
            add_special_tokens=True,
            max_length = config.max_sen_len,
            input_ids_key=config.text_input_ids_key,  # Pass key names
            attention_mask_key=config.text_attention_mask_key,
        )
    )
    pipelines = {config.text_name: dialogue_pipeline}
    return tokenizer, pipelines


#=================== Model Function ======================
def model_fn(model_dir, context=None):
    model_filename = 'model.pth'
    model_artifact_name = 'model_artifacts.pth'

    load_config, embedding_mat, vocab, model_class, label_to_id, id_to_label = load_artifacts(
        os.path.join(model_path, model_artifact_name), device_l=device
    )
                
    config = Config(**load_config)
    
    ## load models
    model = load_model(
        os.path.join(model_path, model_filename),
        config.dict(),
        embedding_mat,
        model_class,
        device_l=device
    )
    model.eval()
    
    ## reconstruct pipelines
    tokenizer, pipelines = data_preprocess_pipeline(config)
                
    # === Add multiclass label processor if needed ===
    if not config.is_binary and config.num_classes > 2:
        if config.multiclass_categories and id_to_label:
            label_processor = MultiClassLabelProcessor(label_list=config.multiclass_categories, strict=True)
            pipelines[config.label_name] = label_processor
                
    return {
        "model": model,
        "config": config,
        "embedding_mat": embedding_mat,
        "vocab": vocab,
        "model_class": model_class,
        "pipelines": pipelines,
    }

                
                
# =================== Input Function ================================
def input_fn(request_body, request_content_type, context=None):
    """
    Deserialize the Invoke request body into an object we can perform prediction on.
    """
    try:
        if request_content_type == 'text/csv':
            return pd.read_csv(StringIO(request_body), header=None, index_col=None)
        elif request_content_type == 'application/json':
            return pd.read_json(f"[{StringIO(request_body).read()}]", orient='records')
        elif request_content_type == 'application/x-parquet':
            return pd.read_parquet(BytesIO(request_body))
        else:
            return Response(
                response='This predictor only supports CSV, JSON, or Parquet data',
                status=415,
                mimetype='text/plain'
            )
    except Exception as e:
        logger.error(f"Failed to parse input ({request_content_type}): {e}")
        return Response(
            response='Invalid input format or corrupted data',
            status=400,
            mimetype='text/plain'
        )              
                

# ================== Prediction Function ============================
def predict_fn(input_object, model_data, context=None):
    if not isinstance(input_object, pd.DataFrame):
        raise TypeError("input data type must be pandas.DataFrame")

    model = model_data["model"]
    config = model_data["config"]
    pipelines = model_data["pipelines"]

    config_predict = config.dict()
    label_field = config_predict.get('label_name', None)

    if label_field:
        config_predict['full_field_list'] = [col for col in config_predict['full_field_list'] if col != label_field]
        config_predict['cat_field_list'] = [col for col in config_predict['cat_field_list'] if col != label_field]

    dataset = BSMDataset(config_predict, dataframe=input_object)
    for field_name, pipeline in pipelines.items():
        dataset.add_pipeline(field_name, pipeline)

    bsm_collate_batch = build_collate_batch(
        input_ids_key=config.text_input_ids_key, 
        attention_mask_key=config.text_attention_mask_key
    )

    batch_size = min(config.batch_size, len(input_object))
    predict_dataloader = DataLoader(dataset, collate_fn=bsm_collate_batch, batch_size=batch_size)

    try:
        logger.info("Model prediction...")
        return model_online_inference(model, predict_dataloader)
    except Exception:
        logger.error("Model scoring error:\n" + traceback.format_exc())
        return [-4]

                
                
# ================== Output Function ================================
def output_fn(predictions, accept, context=None):
    logger.info("Score: {}".format(predictions))
    out = StringIO()

    if accept == 'text/csv':
        pd.DataFrame({'score': predictions}).to_csv(out, header=False, index=False)
        return out.getvalue()
    elif accept in ['application/json', 'application/jsonlines']:
        records_json = pd.DataFrame({'score': predictions}).to_json(orient='records')
        return json.dumps({"predictions": json.loads(records_json)})
    else:
        logger.error("Unsupported return format")
        return Response(
            response="Unsupported return format",
            status=406,
            mimetype='text/plain'
        )











