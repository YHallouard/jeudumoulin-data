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
        self,
        state_embeddings: list[list[float]],
        legal_moves_batch: list[list[list[int | None]]],
    ) -> list[tuple[dict[int, float], float]]:
        with torch.no_grad():
            policies, values = self.model.policy_value_batch(state_embeddings, legal_moves_batch)

            needs_cpu = values.device.type in ("mps", "cuda")
            if needs_cpu:
                values = values.cpu()

            results: list[tuple[dict[int, float], float]] = []
            for i, policy in enumerate(policies):
                policy_probs = torch.exp(policy)
                policy_cpu = policy_probs.cpu() if needs_cpu else policy_probs
                policy_dict = {j: float(policy_cpu[j].item()) for j in range(len(legal_moves_batch[i]))}
                value_float = float(values[i].squeeze().item())
                results.append((policy_dict, value_float))

            return results

    def save_pretrained(self, save_directory: str | Path) -> None:
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

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
