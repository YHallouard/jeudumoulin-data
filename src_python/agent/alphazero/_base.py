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

    @abstractmethod
    def policy_value_batch(
        self,
        state_embeddings: list[list[float]],
        legal_moves_batch: list[list[list[int | None]]],
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        pass


class Agent(ABC):
    model: Model

    @abstractmethod
    def predict(
        self,
        state_embeddings: list[list[float]],
        legal_moves_batch: list[list[list[int | None]]],
    ) -> list[tuple[dict[int, float], float]]:
        pass
