import torch
import torch.nn as nn
from pydantic import BaseModel

from agent.alphazero._backbone import BackboneConfig, get_backbone
from agent.alphazero._base import Model
from agent.alphazero._conditional_policy import ConditionalPolicyHeadConfig, get_policy_head


class MLPDualNetConfig(BaseModel):
    class ValueHeadConfig(BaseModel):
        hidden_dim: int = 128
        dropout_rate: float = 0.2
        output_dim: int = 1

    model_type: str = "mlp_dual_net"
    backbone: BackboneConfig
    policy_head: ConditionalPolicyHeadConfig
    value_head: ValueHeadConfig


class MLPDualNet(Model):
    def __init__(self, config: MLPDualNetConfig) -> None:
        super().__init__()

        self.backbone = get_backbone(config.backbone)

        self.policy_head = get_policy_head(config.policy_head)

        self.value_head = nn.Sequential(
            nn.Linear(config.backbone.output_dim, config.value_head.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.value_head.dropout_rate),
            nn.Linear(config.value_head.hidden_dim, config.value_head.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.value_head.dropout_rate),
            nn.Linear(config.value_head.hidden_dim, config.value_head.output_dim),
            nn.Tanh(),
        )

        self._device: torch.device | None = None

    @property
    def device(self) -> torch.device:
        if self._device is None:
            self._device = next(self.parameters()).device
        return self._device

    def policy_value(
        self, state_embedding: list[float], legal_moves: list[list[int | None]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_tensor = torch.tensor(state_embedding, dtype=torch.float32, device=self.device, requires_grad=False)
        state_tensor = self.backbone(state_tensor)
        policy = self.policy_head(state_tensor, legal_moves)
        value = self.value_head(state_tensor)
        return policy, value
