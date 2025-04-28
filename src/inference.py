import os
import json
import traceback
from io import StringIO, BytesIO
from pathlib import Path
import logging
from typing import List, Union, Dict, Tuple, Optional

import numpy as np
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

from lightning_models.pl_train import (
    model_inference,
    model_online_inference,
    load_model,
    load_artifacts,
    load_onnx_model
)
from lightning_models.dist_utils import get_rank, is_main_process
from pydantic import BaseModel, Field, ValidationError  # For Config Validation

# =================== Logging Setup =================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # <-- THIS LINE IS MISSING

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
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
    class_weights: List[float] = Field(default_factory=lambda: [1.0, 10.0])
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
            )


#=================== Helper Function ================
def data_preprocess_pipeline(config: Config) -> Tuple[AutoTokenizer, Dict[str, Processor]]:
    logger.info(f"Constructing tokenizer: {config.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    dialogue_pipeline = (
        ##HTMLNormalizerProcessor()
        ##>> 
        EmojiRemoverProcessor()
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
    onnx_model_path = os.path.join(model_dir, "model.onnx")


    load_config, embedding_mat, vocab, model_class = load_artifacts(os.path.join(model_path, model_artifact_name), device_l=device)
                
    config = Config(**load_config)
    
    # Load model based on file type
    if os.path.exists(onnx_model_path):
        logger.info("Detected ONNX model.")
        model = load_onnx_model(onnx_model_path)
    else:
        logger.info("Detected PyTorch model.")
        model = load_model(
                os.path.join(model_path, model_filename),
                config.model_dump(),
                embedding_mat,
                model_class,
                device_l=device
            )
        model.eval()
    
    ## reconstruct pipelines
    tokenizer, pipelines = data_preprocess_pipeline(config)
                
    # === Add multiclass label processor if needed ===
    if not config.is_binary and config.num_classes > 2:
        if config.multiclass_categories:
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
    logger.info(f"Received request with Content-Type: {request_content_type}") # Log content type
    try:
        if request_content_type == 'text/csv':
            logger.info("Processing content type: text/csv")
            decoded = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
            logger.debug(f"Decoded CSV data:\n{decoded[:500]}...") # Optional: Log decoded data (be careful with large data)
            try:
                df = pd.read_csv(StringIO(decoded), header=None, index_col=None)
                logger.info(f"Successfully parsed CSV into DataFrame. Shape: {df.shape}, Type: {type(df)}")
                return df # <--- Returns DataFrame here
            except Exception as parse_error:
                logger.error(f"Failed to parse CSV data: {parse_error}")
                # If parsing fails, it will fall through to the final except block
                raise # Re-raise the parsing error to be caught below

        elif request_content_type == 'application/json':
            logger.info("Processing content type: application/json")
            # ... your JSON handling ...
            # Ensure this branch also returns a DataFrame if called
            decoded = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
            try:
                if "\n" in decoded:
                    # Multi-record JSON (NDJSON) handling
                    records = [json.loads(line) for line in decoded.strip().splitlines() if line.strip()]
                    df = pd.DataFrame(records)
                else:
                    json_obj = json.loads(decoded)
                    if isinstance(json_obj, dict):
                        df = pd.DataFrame([json_obj])
                    elif isinstance(json_obj, list):
                        df = pd.DataFrame(json_obj)
                    else:
                        raise ValueError("Unsupported JSON structure")
                logger.info(f"Successfully parsed JSON into DataFrame. Shape: {df.shape}")
                return df
            except Exception as parse_error:
                logger.error(f"Failed to parse JSON data: {parse_error}")
                raise

        elif request_content_type == 'application/x-parquet':
            logger.info("Processing content type: application/x-parquet")
            # ... your Parquet handling ...
            # Ensure this branch also returns a DataFrame if called
            df = pd.read_parquet(BytesIO(request_body))
            logger.info(f"Successfully parsed Parquet into DataFrame. Shape: {df.shape}, Type: {type(df)}")
            return df # <--- Returns DataFrame here

        else:
            logger.warning(f"Unsupported content type: {request_content_type}")
            # THIS RETURNS A Response OBJECT, NOT A DataFrame
            return Response(
                response=f'This predictor only supports CSV, JSON, or Parquet data. Received: {request_content_type}',
                status=415,
                mimetype='text/plain'
            )
    except Exception as e:
        # THIS ALSO RETURNS A Response OBJECT, NOT A DataFrame
        logger.error(f"Failed to parse input ({request_content_type}). Error: {e}", exc_info=True) # Log full traceback
        return Response(
            response=f'Invalid input format or corrupted data. Error during parsing: {e}',
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

    config_predict = config.model_dump()
    label_field = config_predict.get('label_name', None)

    if label_field:
        config_predict['full_field_list'] = [col for col in config_predict['full_field_list'] if col != label_field]
        config_predict['cat_field_list'] = [col for col in config_predict['cat_field_list'] if col != label_field]

    dataset = BSMDataset(config_predict, dataframe=input_object)
    for feature_name, pipeline in pipelines.items():
        dataset.add_pipeline(feature_name, pipeline)

    bsm_collate_batch = build_collate_batch(
        input_ids_key=config.text_input_ids_key, 
        attention_mask_key=config.text_attention_mask_key
    )

    batch_size = len(input_object)
    predict_dataloader = DataLoader(dataset, collate_fn=bsm_collate_batch, batch_size=batch_size)

    try:
        logger.info("Model prediction...")
        return model_online_inference(model, predict_dataloader)
    except Exception:
        logger.error("Model scoring error:\n" + traceback.format_exc())
        return [-4]

                
                
# ================== Output Function ================================
def output_fn(prediction_output, accept='application/json'):
    """
    Serializes the multi-class prediction output.

    Args:
        prediction_output: The output from predict_fn, expected to be a
                           numpy array of shape (N, num_classes) or list of lists.
        accept: The requested response MIME type (e.g., 'application/json').

    Returns:
        tuple: (response_body, content_type)
    """
    logger.info(f"Received prediction output of type: {type(prediction_output)} for accept type: {accept}")

    scores_list = None

    # Step 1: Normalize input format into a list of lists
    if isinstance(prediction_output, np.ndarray):
        logger.info(f"Prediction output numpy array shape: {prediction_output.shape}")
        scores_list = prediction_output.tolist()
    elif isinstance(prediction_output, list):
        scores_list = prediction_output
    else:
        msg = f"Unsupported prediction output type: {type(prediction_output)}"
        logger.error(msg)
        raise ValueError(msg)
        
        
    try:
        is_multiclass = isinstance(scores_list[0], list)

        # Step 2: JSON output formatting
        # {
        #  "prob_01": ...
        #  "prob_02": ...
        # ...
        #  "prob_0k": ...
        #  "output-label"": ...
        # }
        if accept.lower() == 'application/json':
            output_records = []
            for probs in scores_list:
                probs = probs if isinstance(probs, list) else [probs]
                max_idx = probs.index(max(probs)) if probs else -1

                record = {
                    **{f"prob_{str(i+1).zfill(2)}": p for i, p in enumerate(probs)},
                    "output-label": f"class-{max_idx}" if max_idx >= 0 else "unknown"
                }
                output_records.append(record)

            response = json.dumps({"predictions": output_records})
            return response, 'application/json'

        # Step 3: CSV output formatting
        elif accept.lower() == 'text/csv':
            csv_lines = []
            for probs in scores_list:
                probs = probs if isinstance(probs, list) else [probs]
                max_idx = probs.index(max(probs)) if probs else -1
                formatted_probs = [round(float(p), 4) for p in probs]  # Output as numerical floats
                # Format list string without brackets and parentheses
                list_str = ",".join(f"{p:.4f}" for p in formatted_probs)
                
                line = [list_str] + [f"class-{max_idx}" if max_idx >= 0 else "unknown"]
                csv_lines.append(",".join(map(str, line)))

            response_body = "\n".join(csv_lines) + "\n"
            return response_body, 'text/csv'

        # Step 4: Unsupported content type
        else:
            logger.error(f"Unsupported accept type: {accept}")
            raise ValueError(f"Unsupported accept type: {accept}")

    # Step 5: Error handling
    except Exception as e:
        logger.error(f"Error during DataFrame creation or serialization in output_fn: {e}", exc_info=True)
        error_response = json.dumps({'error': f'Failed to serialize output: {e}'})
        return error_response, 'application/json'