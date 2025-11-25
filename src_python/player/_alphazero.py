import jdm_ru
from agent.alphazero._agent import Agent
from jdm_ru import PyMCTS

from player._base import Player


class AlphaZeroPlayer(Player):
    def __init__(self, agent: Agent, temperature: float = 1.0, num_simulations: int = 100) -> None:
        self.agent = agent
        self.mcts = PyMCTS(num_simulations=num_simulations, show_progress=False)
        self.temperature = temperature

    def select_move(self, board: jdm_ru.PyBoard) -> tuple[jdm_ru.PyMove, float]:
        root = self.mcts.run(self.agent, board, 0, None)
        action = root.select_action(self.temperature)
        value = root.children[action].value

        return action, value
