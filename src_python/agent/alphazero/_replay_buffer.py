import random
from collections import deque


class AlphaZeroReplayBuffer:
    def __init__(self, max_size: int = 10000) -> None:
        self.buffer: deque = deque(maxlen=max_size)
        self.max_size = max_size

    def add_examples(
        self,
        state_embeddings: list[list[float]],
        legal_moves: list[list[list[int | None]]],
        policy_labels: list[list[float]],
        value_labels: list[float],
    ) -> None:
        for state, moves, policy, value in zip(state_embeddings, legal_moves, policy_labels, value_labels):
            self.buffer.append((state, moves, policy, value))

    def sample_batch(self, batch_size: int) -> tuple[list, list, list, list]:
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, moves, policies, values = zip(*batch)
        return list(states), list(moves), list(policies), list(values)

    def can_sample(self, batch_size: int) -> bool:
        return len(self.buffer) >= batch_size

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        self.buffer.clear()

    def get_statistics(self) -> dict:
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.max_size,
                "fill_ratio": 0.0,
                "avg_value": 0.0,
            }

        values = [t[3] for t in self.buffer]

        return {
            "size": len(self.buffer),
            "capacity": self.max_size,
            "fill_ratio": len(self.buffer) / self.max_size,
            "avg_value": sum(values) / len(values),
        }
