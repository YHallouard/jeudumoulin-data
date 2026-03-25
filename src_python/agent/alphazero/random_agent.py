from agent.alphazero._base import Agent


class RandomAgent(Agent):
    def __init__(self) -> None:
        self.model = None  # type: ignore[assignment]

    def predict(
        self,
        state_embeddings: list[list[float]],
        legal_moves_batch: list[list[list[int | None]]],
    ) -> list[tuple[dict[int, float], float]]:
        return [({i: 1.0 / len(moves) for i in range(len(moves))}, 0.0) for moves in legal_moves_batch]
