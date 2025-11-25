from typing import Literal

import torch
import torch.nn as nn
from pydantic import BaseModel


class DQNNetworkConfig(BaseModel):
    class BackboneConfig(BaseModel):
        input_dim: int = 77
        hidden_dim: int = 128
        output_dim: int = 128

    class QNetworkConfig(BaseModel):
        embedding_dim: int = 32
        hidden_dim: int = 256
        output_dim: int = 1

    model_type: Literal["dqn_network"] = "dqn_network"
    backbone: BackboneConfig
    q_network: QNetworkConfig


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with embedding layers for action representation.

    Architecture:
        State Input (77 floats) → State Backbone → State Embedding (128)
        Action Input (3 indices) → 3x nn.Embedding(25, 32) → Action Embedding (96)
        [State Emb (128), Action Emb (96)] → Q-Network → Q(s,a) (1 scalar)

    Key Innovation:
        Instead of using 72-dimensional one-hot encoded actions, we use 3 indices
        (from_position, to_position, removed_position) that are embedded into learned
        32-dimensional vectors. This allows the network to:
        - Learn dense position representations
        - Capture similarities between positions
        - Use 24x fewer input features
        - Generalize better across similar positions

    Args:
        state_dim: Dimension of state embedding (default: 77)
        action_emb_dim: Dimension of each action position embedding (default: 32)
        hidden_dim: Dimension of hidden layers (default: 256)
        state_emb_dim: Dimension of state embedding after backbone (default: 128)
    """

    def __init__(self, config: DQNNetworkConfig):
        super().__init__()

        self.state_backbone = nn.Sequential(
            nn.Linear(config.backbone.input_dim, config.backbone.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.backbone.hidden_dim, config.backbone.output_dim),
            nn.ReLU(),
        )

        # Action embeddings
        # Each position (0-24) gets a learned 32-dim embedding
        # Position 24 is used for "None" (no position)
        self.from_embedding = nn.Embedding(25, config.q_network.embedding_dim)
        self.to_embedding = nn.Embedding(25, config.q_network.embedding_dim)
        self.remove_embedding = nn.Embedding(25, config.q_network.embedding_dim)

        # Q-network
        # Takes concatenated state and action embeddings to predict Q(s,a)
        combined_dim = config.backbone.output_dim + (3 * config.q_network.embedding_dim)  # 128 + 96 = 224

        self.q_network = nn.Sequential(
            nn.Linear(combined_dim, config.q_network.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.q_network.hidden_dim, config.q_network.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.q_network.hidden_dim, config.q_network.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.q_network.hidden_dim, config.q_network.output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, state: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_backbone(state)  # (batch_size, 128)

        from_emb = self.from_embedding(action_indices[:, 0])  # (batch_size, 32)
        to_emb = self.to_embedding(action_indices[:, 1])  # (batch_size, 32)
        remove_emb = self.remove_embedding(action_indices[:, 2])  # (batch_size, 32)

        action_emb = torch.cat([from_emb, to_emb, remove_emb], dim=1)  # (batch_size, 96)

        combined = torch.cat([state_emb, action_emb], dim=1)  # (batch_size, 224)

        q_value: torch.Tensor = self.q_network(combined)  # (batch_size, 1)

        return q_value

    def predict_q_values(self, state: torch.Tensor, action_indices_batch: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, 77)

        num_actions = action_indices_batch.shape[0]
        state_expanded = state.expand(num_actions, -1)  # (num_actions, 77)

        q_values = self.forward(state_expanded, action_indices_batch)  # (num_actions, 1)

        return q_values.squeeze(1)  # (num_actions,)
