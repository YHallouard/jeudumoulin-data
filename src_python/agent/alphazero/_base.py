from abc import ABC, abstractmethod

import torch


class Model(ABC, torch.nn.Module):
    @abstractmethod
    def policy_value(
        self, state_embedding: list[float], legal_moves: list[list[int | None]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy and value for a given state.

        Args:
            state_embedding: List of floats from Board.to_embed()
            legal_moves: List of legal moves as [from, to, removed] lists

        Returns:
            Tuple of (policy probabilities, value estimate)
        """
        pass


class Agent(ABC):
    model: Model

    @abstractmethod
    def predict(
        self, state_embedding: list[float], legal_moves: list[list[int | None]]
    ) -> tuple[dict[int, float], float]:
        """
        Predict policy and value for a state.

        Args:
            state_embedding: List of floats from Board.to_embed()
            legal_moves: List of legal moves as [from, to, removed] lists

        Returns:
            Tuple of (dict mapping move index to probability, value estimate)
        """
        pass
