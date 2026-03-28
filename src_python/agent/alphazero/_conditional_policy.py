from functools import singledispatch
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from agent.alphazero._position import EmbeddingConfig, get_embedding


def _create_from_mask(legal_actions: list[list[int | None]], device: torch.device) -> torch.Tensor:
    mask = torch.full((25,), float("-inf"), device=device)
    for action in legal_actions:
        from_idx = action[0] if action[0] is not None else 24
        mask[from_idx] = 0.0
    return mask


def _create_batch_to_mask(legal_actions: list[list[int | None]], device: torch.device) -> torch.Tensor:
    num_actions = len(legal_actions)
    mask = torch.full((num_actions, 24), float("-inf"), device=device)
    from_to_map: dict[int, set[int]] = {}
    for action in legal_actions:
        from_idx = action[0] if action[0] is not None else 24
        to_idx = action[1] if action[1] is not None else 0
        from_to_map.setdefault(from_idx, set()).add(to_idx)
    for i, action in enumerate(legal_actions):
        from_idx = action[0] if action[0] is not None else 24
        for to_idx in from_to_map[from_idx]:
            mask[i, to_idx] = 0.0
    return mask


def _create_batch_remove_mask(legal_actions: list[list[int | None]], device: torch.device) -> torch.Tensor:
    num_actions = len(legal_actions)
    mask = torch.full((num_actions, 25), float("-inf"), device=device)
    pair_remove_map: dict[tuple[int, int], set[int]] = {}
    for action in legal_actions:
        from_idx = action[0] if action[0] is not None else 24
        to_idx = action[1] if action[1] is not None else 0
        remove_idx = action[2] if action[2] is not None else 24
        pair_remove_map.setdefault((from_idx, to_idx), set()).add(remove_idx)
    for i, action in enumerate(legal_actions):
        from_idx = action[0] if action[0] is not None else 24
        to_idx = action[1] if action[1] is not None else 0
        for remove_idx in pair_remove_map[(from_idx, to_idx)]:
            mask[i, remove_idx] = 0.0
    return mask


class SemiConditionalPolicyHeadConfig(BaseModel):
    model_type: Literal["semi_conditional_policy_head"] = "semi_conditional_policy_head"
    state_embedding_dim: int = 128
    embedding: EmbeddingConfig
    from_head_hidden_dim: int = 128
    to_head_hidden_dim: int = 128
    remove_head_hidden_dim: int = 128
    from_head_dropout_rate: float = 0.2
    to_head_dropout_rate: float = 0.2
    remove_head_dropout_rate: float = 0.2


class FullyConditionalPolicyHeadConfig(BaseModel):
    model_type: Literal["fully_conditional_policy_head"] = "fully_conditional_policy_head"
    state_embedding_dim: int = 128
    embedding: EmbeddingConfig
    from_head_hidden_dim: int = 128
    to_head_hidden_dim: int = 128
    remove_head_hidden_dim: int = 128
    from_head_dropout_rate: float = 0.2
    to_head_dropout_rate: float = 0.2
    remove_head_dropout_rate: float = 0.2


class GatedConditionalPolicyHeadConfig(BaseModel):
    model_type: Literal["gated_conditional_policy_head"] = "gated_conditional_policy_head"
    state_embedding_dim: int = 512
    embedding: EmbeddingConfig
    hidden_dim: int = 512
    dropout_rate: float = 0.2


ConditionalPolicyHeadConfig = (
    SemiConditionalPolicyHeadConfig | FullyConditionalPolicyHeadConfig | GatedConditionalPolicyHeadConfig
)


@singledispatch
def get_policy_head(config: ConditionalPolicyHeadConfig) -> nn.Module:
    raise NotImplementedError(f"Unsupported policy head config type: {type(config)}")


@get_policy_head.register
def _(config: SemiConditionalPolicyHeadConfig) -> nn.Module:
    return SemiConditionalPolicyHead(config)


@get_policy_head.register
def _(config: FullyConditionalPolicyHeadConfig) -> nn.Module:
    return FullyConditionalPolicyHead(config)


@get_policy_head.register
def _(config: GatedConditionalPolicyHeadConfig) -> nn.Module:
    return GatedConditionalPolicyHead(config)


class SemiConditionalPolicyHead(nn.Module):
    def __init__(self, config: SemiConditionalPolicyHeadConfig) -> None:
        super().__init__()

        self.state_embedding_dim = config.state_embedding_dim

        self.position_embedding = get_embedding(config.embedding)

        self.from_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim, config.from_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.from_head_dropout_rate),
            nn.Linear(config.from_head_hidden_dim, 25),
        )

        self.to_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim + config.embedding.embedding_dim, config.to_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.to_head_dropout_rate),
            nn.Linear(config.to_head_hidden_dim, 24),
        )

        self.remove_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim, config.remove_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.remove_head_dropout_rate),
            nn.Linear(config.remove_head_hidden_dim, 25),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.from_head, self.to_head, self.remove_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, state_embedding: torch.Tensor, legal_actions: list[list[int | None]]) -> torch.Tensor:
        device = state_embedding.device
        num_actions = len(legal_actions)

        if num_actions == 0:
            return torch.tensor([], device=device)

        from_logits = self.from_head(state_embedding)

        from_indices: list[int] = []
        to_indices: list[int] = []
        remove_indices: list[int] = []

        for action in legal_actions:
            from_idx = action[0] if action[0] is not None else 24
            to_idx = action[1] if action[1] is not None else 0
            remove_idx = action[2] if action[2] is not None else 24

            from_indices.append(from_idx)
            to_indices.append(to_idx)
            remove_indices.append(remove_idx)

        from_indices_tensor = torch.tensor(from_indices, device=device)
        to_indices_tensor = torch.tensor(to_indices, device=device)
        remove_indices_tensor = torch.tensor(remove_indices, device=device)

        from_embs = self.position_embedding(from_indices_tensor)

        state_emb_expanded = state_embedding.unsqueeze(0).expand(num_actions, -1)
        to_inputs = torch.cat([state_emb_expanded, from_embs], dim=-1)
        to_logits_all = self.to_head(to_inputs)  # [num_actions, 24]

        remove_logits = self.remove_head(state_embedding)  # [25]
        remove_logits_all = remove_logits.unsqueeze(0).expand(num_actions, -1)  # [num_actions, 25]

        from_mask = _create_from_mask(legal_actions, device)
        from_log_probs_all = F.log_softmax(from_logits + from_mask, dim=0)
        from_log_probs = from_log_probs_all[from_indices_tensor]

        to_batch_mask = _create_batch_to_mask(legal_actions, device)
        to_log_probs_all = F.log_softmax(to_logits_all + to_batch_mask, dim=1)
        to_log_probs = to_log_probs_all.gather(1, to_indices_tensor.unsqueeze(1)).squeeze(1)

        remove_batch_mask = _create_batch_remove_mask(legal_actions, device)
        remove_log_probs_all = F.log_softmax(remove_logits_all + remove_batch_mask, dim=1)
        remove_log_probs = remove_log_probs_all.gather(1, remove_indices_tensor.unsqueeze(1)).squeeze(1)

        log_probs = from_log_probs + to_log_probs + remove_log_probs

        return F.log_softmax(log_probs, dim=0)


class FullyConditionalPolicyHead(nn.Module):
    def __init__(self, config: FullyConditionalPolicyHeadConfig) -> None:
        super().__init__()

        self.state_embedding_dim = config.state_embedding_dim

        self.from_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim, config.from_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.from_head_dropout_rate),
            nn.Linear(config.from_head_hidden_dim, 25),
        )

        self.from_embedding = get_embedding(config.embedding)
        self.to_embedding = get_embedding(config.embedding)

        self.to_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim + config.embedding.embedding_dim, config.to_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.to_head_dropout_rate),
            nn.Linear(config.to_head_hidden_dim, 24),
        )

        self.remove_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim + config.embedding.embedding_dim * 2, config.remove_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.remove_head_dropout_rate),
            nn.Linear(config.remove_head_hidden_dim, 25),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.from_head, self.to_head, self.remove_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, state_embedding: torch.Tensor, legal_actions: list[list[int | None]]) -> torch.Tensor:
        device = state_embedding.device
        num_actions = len(legal_actions)

        if num_actions == 0:
            return torch.tensor([], device=device)

        from_logits = self.from_head(state_embedding)

        from_indices: list[int] = []
        to_indices: list[int] = []
        remove_indices: list[int] = []

        for action in legal_actions:
            from_idx = action[0] if action[0] is not None else 24
            to_idx = action[1] if action[1] is not None else 0
            remove_idx = action[2] if action[2] is not None else 24

            from_indices.append(from_idx)
            to_indices.append(to_idx)
            remove_indices.append(remove_idx)

        from_indices_tensor = torch.tensor(from_indices, device=device)
        to_indices_tensor = torch.tensor(to_indices, device=device)
        remove_indices_tensor = torch.tensor(remove_indices, device=device)

        from_embs = self.from_embedding(from_indices_tensor)
        to_embs = self.to_embedding(to_indices_tensor)

        state_emb_expanded = state_embedding.unsqueeze(0).expand(num_actions, -1)

        to_inputs = torch.cat([state_emb_expanded, from_embs], dim=-1)
        to_logits_all = self.to_head(to_inputs)  # [num_actions, 24]

        remove_inputs = torch.cat([state_emb_expanded, from_embs, to_embs], dim=-1)
        remove_logits_all = self.remove_head(remove_inputs)  # [num_actions, 25]

        from_mask = _create_from_mask(legal_actions, device)
        from_log_probs_all = F.log_softmax(from_logits + from_mask, dim=0)
        from_log_probs = from_log_probs_all[from_indices_tensor]

        to_batch_mask = _create_batch_to_mask(legal_actions, device)
        to_log_probs_all = F.log_softmax(to_logits_all + to_batch_mask, dim=1)
        to_log_probs = to_log_probs_all.gather(1, to_indices_tensor.unsqueeze(1)).squeeze(1)

        remove_batch_mask = _create_batch_remove_mask(legal_actions, device)
        remove_log_probs_all = F.log_softmax(remove_logits_all + remove_batch_mask, dim=1)
        remove_log_probs = remove_log_probs_all.gather(1, remove_indices_tensor.unsqueeze(1)).squeeze(1)

        log_probs = from_log_probs + to_log_probs + remove_log_probs

        return F.log_softmax(log_probs, dim=0)


class GatedBlock(nn.Module):
    def __init__(self, input_dim: int, condition_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.state_proj = nn.Linear(input_dim, hidden_dim)
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, state: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h_state = self.state_proj(state)
        gate = torch.sigmoid(self.condition_proj(condition))
        combined = self.activation(h_state * gate)
        combined = self.dropout(combined)
        combined = self.output_proj(combined)
        return combined  # type: ignore[no-any-return]


class GatedConditionalPolicyHead(nn.Module):
    def __init__(self, config: GatedConditionalPolicyHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = get_embedding(config.embedding)
        emb_dim = config.embedding.embedding_dim

        self.from_head = nn.Sequential(
            nn.Linear(config.state_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 25),
        )

        self.to_gate = GatedBlock(
            input_dim=config.state_embedding_dim,
            condition_dim=emb_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate,
        )
        self.to_output = nn.Linear(config.hidden_dim, 24)

        self.remove_gate = GatedBlock(
            input_dim=config.state_embedding_dim,
            condition_dim=emb_dim * 2,  # concatenation of from + to embeddings
            hidden_dim=config.hidden_dim,
            dropout=config.dropout_rate,
        )
        self.remove_output = nn.Linear(config.hidden_dim, 25)

        self._init_weights()

    def _init_weights(self) -> None:
        linear_layers: list[nn.Linear] = []
        for layer in self.from_head:
            if isinstance(layer, nn.Linear):
                linear_layers.append(layer)
        for gated_module in [self.to_gate, self.remove_gate]:
            for layer in gated_module.modules():
                if isinstance(layer, nn.Linear):
                    linear_layers.append(layer)
        linear_layers.extend([self.to_output, self.remove_output])
        for layer in linear_layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state_embedding: torch.Tensor, legal_actions: list[list[int | None]]) -> torch.Tensor:
        device = state_embedding.device
        num_actions = len(legal_actions)

        if num_actions == 0:
            return torch.tensor([], device=device)

        from_logits = self.from_head(state_embedding)

        from_indices: list[int] = []
        to_indices: list[int] = []
        remove_indices: list[int] = []

        for action in legal_actions:
            from_idx = action[0] if action[0] is not None else 24
            to_idx = action[1] if action[1] is not None else 0
            remove_idx = action[2] if action[2] is not None else 24

            from_indices.append(from_idx)
            to_indices.append(to_idx)
            remove_indices.append(remove_idx)

        from_indices_tensor = torch.tensor(from_indices, device=device)
        to_indices_tensor = torch.tensor(to_indices, device=device)
        remove_indices_tensor = torch.tensor(remove_indices, device=device)

        from_embs = self.embedding(from_indices_tensor)  # [num_actions, embedding_dim]
        to_embs = self.embedding(to_indices_tensor)  # [num_actions, embedding_dim]

        state_emb_expanded = state_embedding.unsqueeze(0).expand(num_actions, -1)  # [num_actions, state_embedding_dim]

        to_features = self.to_gate(state_emb_expanded, from_embs)  # [num_actions, hidden_dim]
        to_logits_all = self.to_output(to_features)  # [num_actions, 24]

        move_condition = torch.cat([from_embs, to_embs], dim=-1)  # [num_actions, embedding_dim * 2]
        remove_features = self.remove_gate(state_emb_expanded, move_condition)  # [num_actions, hidden_dim]
        remove_logits_all = self.remove_output(remove_features)  # [num_actions, 25]

        from_mask = _create_from_mask(legal_actions, device)  # [25]
        from_log_probs_all = F.log_softmax(from_logits + from_mask, dim=0)  # [25]
        from_log_probs = from_log_probs_all[from_indices_tensor]  # [num_actions]

        to_batch_mask = _create_batch_to_mask(legal_actions, device)
        to_log_probs_all = F.log_softmax(to_logits_all + to_batch_mask, dim=1)
        to_log_probs = to_log_probs_all.gather(1, to_indices_tensor.unsqueeze(1)).squeeze(1)

        remove_batch_mask = _create_batch_remove_mask(legal_actions, device)
        remove_log_probs_all = F.log_softmax(remove_logits_all + remove_batch_mask, dim=1)
        remove_log_probs = remove_log_probs_all.gather(1, remove_indices_tensor.unsqueeze(1)).squeeze(1)

        log_probs = from_log_probs + to_log_probs + remove_log_probs

        return F.log_softmax(log_probs, dim=0)
