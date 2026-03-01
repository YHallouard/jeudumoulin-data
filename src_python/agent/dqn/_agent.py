import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from pydantic import BaseModel
from safetensors.torch import load_model, save_model

from agent.dqn._models import DQNNetwork, DQNNetworkConfig

if TYPE_CHECKING:
    import jdm_ru


class DQNAgentConfig(BaseModel):
    model: DQNNetworkConfig
    device: str = "cpu"


class DQNAgent:
    def __init__(self, config: DQNAgentConfig):
        """Initialize the DQN agent."""
        self.config = config
        self.model = DQNNetwork(config.model)
        self.model.to(config.device)
        self.model.eval()

    def select_action(
        self,
        state: list[float],
        legal_moves: list["jdm_ru.PyMove"],
        epsilon: float = 0.0,
    ) -> int:
        """
        Select an action using epsilon-greedy strategy.

        With probability epsilon, select a random action (exploration).
        Otherwise, select the action with highest Q-value (exploitation).

        Args:
            state: State embedding (77 floats)
            legal_moves: List of legal PyMove objects
            epsilon: Exploration probability (0 = greedy, 1 = random)

        Returns:
            Index of selected move in legal_moves list
        """
        if not legal_moves:
            raise ValueError("No legal moves available")  # noqa: TRY003

        if random.random() < epsilon:  # noqa: S311
            return random.randint(0, len(legal_moves) - 1)  # noqa: S311

        return self.select_best_action(state, legal_moves)

    def select_best_action(
        self,
        state: list[float],
        legal_moves: list["jdm_ru.PyMove"],
    ) -> int:
        """
        Select the action with highest Q-value (greedy policy).

        Args:
            state: State embedding (77 floats)
            legal_moves: List of legal PyMove objects

        Returns:
            Index of best move in legal_moves list
        """
        if not legal_moves:
            raise ValueError("No legal moves available")  # noqa: TRY003

        q_values = self.predict_q_values(state, legal_moves)

        # Return index of maximum Q-value
        best_idx = torch.argmax(q_values).item()
        return int(best_idx)

    def predict_q_values(
        self,
        state: list[float],
        legal_moves: list["jdm_ru.PyMove"],
    ) -> torch.Tensor:
        """
        Predict Q-values for all legal moves.

        Args:
            state: State embedding (77 floats)
            legal_moves: List of legal PyMove objects

        Returns:
            Tensor of Q-values, shape (num_legal_moves,)
        """
        if not legal_moves:
            return torch.tensor([], device=self.config.device)

        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.config.device)

        # Convert moves to action indices
        action_indices = []
        for move in legal_moves:
            indices = move.to_indices()
            action_indices.append(indices)

        action_indices_tensor = torch.tensor(action_indices, dtype=torch.long, device=self.config.device)

        # Predict Q-values
        with torch.no_grad():
            q_values = self.model.predict_q_values(state_tensor, action_indices_tensor)

        return q_values

    def get_q_value(
        self,
        state: list[float],
        action_indices: list[int],
    ) -> float:
        """
        Get Q-value for a specific state-action pair.

        Args:
            state: State embedding (77 floats)
            action_indices: Action as indices [from, to, remove]

        Returns:
            Q-value as float
        """
        state_tensor = torch.tensor([state], dtype=torch.float32, device=self.config.device)
        action_tensor = torch.tensor([action_indices], dtype=torch.long, device=self.config.device)

        with torch.no_grad():
            q_value = self.model(state_tensor, action_tensor)

        return float(q_value.detach().cpu().item())

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def save_pretrained(self, save_directory: str | Path) -> None:
        """
        Save the agent's model to disk.

        Args:
            path: Path to save the model (e.g., 'model.pt')
        """
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self.model, str(path / "model.safetensors"))
        with open(path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=4)

    @classmethod
    def from_pretrained(cls, model_path: str | Path, device: str = "cpu") -> "DQNAgent":
        """
        Load an agent from disk.

        Args:
            load_directory: Path to the saved model
            device: Device to load the model on

        Returns:
            Loaded DQNAgent instance
        """
        path = Path(model_path)
        with open(path / "config.json") as f:
            config_dict = json.load(f)
            config = DQNAgentConfig.model_validate(config_dict)

        config.device = device
        agent = cls(config=config)

        load_model(agent.model, str(path / "model.safetensors"), device=device)
        return agent

    def get_model_parameters(self) -> nn.Module:
        """
        Get the underlying model for optimizer setup.

        Returns:
            The neural network model
        """
        return self.model

    def get_epsilon_greedy_probabilities(
        self,
        state: list[float],
        legal_moves: list["jdm_ru.PyMove"],
        epsilon: float,
    ) -> list[float]:
        """
        Get action probabilities under epsilon-greedy policy.

        Useful for analysis and debugging.

        Args:
            state: State embedding
            legal_moves: Legal moves
            epsilon: Exploration probability

        Returns:
            List of probabilities for each legal move
        """
        num_actions = len(legal_moves)
        if num_actions == 0:
            return []

        # Get Q-values
        q_values = self.predict_q_values(state, legal_moves)
        best_idx = torch.argmax(q_values).item()

        # Calculate epsilon-greedy probabilities
        probs = [epsilon / num_actions] * num_actions
        probs[int(best_idx)] += 1.0 - epsilon

        return probs

    def get_statistics(self) -> dict:
        """
        Get agent statistics for monitoring.

        Returns:
            Dictionary with agent info
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_type": "DQN",
            "device": self.config.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "state_dim": self.config.model.backbone.input_dim,
            "action_emb_dim": self.config.model.q_network.embedding_dim,
            "hidden_dim": self.config.model.q_network.hidden_dim,
        }

    def __repr__(self) -> str:
        """String representation of the agent."""
        params = sum(p.numel() for p in self.model.parameters())
        return f"DQNAgent(params={params:,}, device={self.config.device})"
