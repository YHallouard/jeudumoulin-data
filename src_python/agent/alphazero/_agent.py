import json
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel
from safetensors.torch import load_model, save_model

from agent.alphazero._base import Agent
from agent.alphazero._models import MLPDualNet, MLPDualNetConfig


class AlphaZeroAgentConfig(BaseModel):
    model: MLPDualNetConfig
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class AlphaZeroAgent(Agent):
    def __init__(self, config: AlphaZeroAgentConfig) -> None:
        self.config = config
        self.model = MLPDualNet(config.model)
        self.model.to(config.device)
        self.model.eval()
        self._device = config.device

    def predict(
        self, state_embedding: list[float], legal_moves: list[list[int | None]]
    ) -> tuple[dict[int, float], float]:
        with torch.no_grad():
            policy, value = self.model.policy_value(state_embedding, legal_moves)

            if policy.device.type in ("mps", "cuda"):
                policy_cpu = policy.cpu()
                value_cpu = value.cpu()
            else:
                policy_cpu = policy
                value_cpu = value

            policy_dict = {i: float(policy_cpu[i].item()) for i in range(len(legal_moves))}
            value_float = float(value_cpu.squeeze().item())

            return policy_dict, value_float

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_model(self.model, str(path / "model.safetensors"))
        with open(path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device: str = "cpu") -> "AlphaZeroAgent":
        with open(Path(model_path) / "config.json") as f:
            config = AlphaZeroAgentConfig.model_validate_json(json.load(f))
        agent = cls(config=config)

        load_model(agent.model, model_path, device=device)
        return agent
