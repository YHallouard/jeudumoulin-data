from agent.alphazero._base import Agent


class RandomAgent(Agent):
    def __init__(self) -> None:
        self.model = None  # type: ignore

    def predict(
        self, state_embedding: list[float], legal_moves: list[list[int | None]]
    ) -> tuple[dict[int, float], float]:
        policy_dict = {i: 1.0 / len(legal_moves) for i in range(len(legal_moves))}
        return policy_dict, 0.0
