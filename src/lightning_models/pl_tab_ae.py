import torch
import torch.nn as nn
import torch.optim as optim

import lightning.pytorch as pl  # Or torch.nn.Module if not training independently
from typing import Dict, Union, List
import lightning.pytorch as pl

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class TabularEmbeddingConfig(BaseModel):
    tab_field_list: List[str]
    hidden_common_dim: int
    input_tab_dim: int = Field(init=False)  # Calculated dynamically
    is_binary: bool = True  # Added for clarity (though not used)
    num_classes: int = 2  # Added for clarity (though not used)

    @property
    def output_tab_dim(self) -> int:
        return self.hidden_common_dim

    @field_validator("tab_field_list")
    @classmethod
    def validate_tab_field_list(cls, v: List[str], info: ValidationInfo) -> List[str]:
        if not v:
            raise ValueError("tab_field_list must not be empty")
        return v

    @field_validator("input_tab_dim")
    @classmethod
    def set_input_tab_dim(cls, v: int, info: ValidationInfo) -> int:
        tab_field_list = info.data.get("tab_field_list")
        if tab_field_list:
            return len(tab_field_list)
        return v

class TabularEmbeddingModule(pl.LightningModule): 
    def __init__(self, config: Union[Dict, TabularEmbeddingConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = TabularEmbeddingConfig(**config)
        self.config = config

        self.embedding_layer = nn.Sequential(
            nn.LayerNorm(self.config.input_tab_dim),
            nn.Linear(self.config.input_tab_dim, self.config.hidden_common_dim),
            nn.ReLU()
        )
        self.output_tab_dim = self.config.hidden_common_dim

        if isinstance(config, dict):
            self.save_hyperparameters(config)
        else:
            self.save_hyperparameters(config.dict())

    def combine_tab_data(self, batch: Dict[str, Union[torch.Tensor, List]]) -> torch.Tensor:
        """
        Combines tabular fields into a single tensor of shape [B, input_tab_dim]
        """
        features = []
        for field in self.config.tab_field_list:
            val = batch[field]
            if isinstance(val, list):
                val = torch.tensor(val, dtype=torch.float32, device=next(self.parameters()).device)
            elif isinstance(val, torch.Tensor):
                val = val.float()
            else:
                raise TypeError(f"Unsupported type for field {field}: {type(val)}")
            if val.dim() == 1:
                val = val.unsqueeze(1)
            features.append(val)
        return torch.cat(features, dim=1)

    def forward(self, inputs: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Returns embedding vector from tabular input.
        """
        if isinstance(inputs, dict):
            inputs = self.combine_tab_data(inputs)
        return self.embedding_layer(inputs)


class TabAE(TabularEmbeddingModule, pl.LightningModule):  # Inherit for combine_tab_data and config
    def __init__(self, config: Union[Dict, TabularEmbeddingConfig]):
        super().__init__(config)
        self.save_hyperparameters(config)  # Still useful in Lightning