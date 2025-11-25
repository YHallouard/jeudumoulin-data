from functools import singledispatch

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel


class ClassicEmbeddingConfig(BaseModel):
    model_type: str = "classic_embedding"
    embedding_dim: int = 32


class ClassicEmbedding(nn.Module):
    def __init__(self, config: ClassicEmbeddingConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(25, self.config.embedding_dim)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.embedding(positions)
        return result


class PositionalEmbeddingConfig(BaseModel):
    model_type: str = "positional_embedding"
    embedding_dim: int = 32


class PositionalEmbedding(nn.Module):
    def __init__(self, config: PositionalEmbeddingConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = self.config.embedding_dim

        self.level_embedding = nn.Embedding(4, self.config.embedding_dim // 4)
        self.position_in_level_embedding = nn.Embedding(9, self.config.embedding_dim // 4)

        self.register_buffer("sinusoidal_table", self._get_sinusoidal_encoding(25, self.config.embedding_dim // 2))

    def _get_sinusoidal_encoding(self, num_positions: int, dim: int) -> torch.Tensor:
        position = torch.arange(num_positions).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        return torch.cat([torch.sin(position * div_term), torch.cos(position * div_term)], dim=1)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        level = positions // 8
        position_in_level = positions % 8
        sinusoidal_emb = self.sinusoidal_table[positions]
        return torch.cat(
            [self.level_embedding(level), self.position_in_level_embedding(position_in_level), sinusoidal_emb], dim=1
        )


EmbeddingConfig = PositionalEmbeddingConfig | ClassicEmbeddingConfig


@singledispatch
def get_embedding(config: EmbeddingConfig) -> nn.Module:
    raise NotImplementedError(f"Unsupported positional embedding config type: {type(config)}")


@get_embedding.register
def _(config: ClassicEmbeddingConfig) -> nn.Module:
    return ClassicEmbedding(config)


@get_embedding.register
def _(config: PositionalEmbeddingConfig) -> nn.Module:
    return PositionalEmbedding(config)
