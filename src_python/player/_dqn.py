import jdm_ru
import torch
from agent.dqn._agent import DQNAgent

from player._base import Player


class DQNPlayer(Player):
    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def select_move(self, board: jdm_ru.PyBoard) -> tuple[jdm_ru.PyMove, float]:
        state_embedding = board.to_embed()
        legal_moves = board.legal_moves()
        q_values = self.agent.predict_q_values(state_embedding, legal_moves)
        best_move_idx: int = int(torch.argmax(q_values).item())
        move = legal_moves[best_move_idx]
        value: float = q_values[best_move_idx].item()
        return move, value
