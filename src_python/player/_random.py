import random
from typing import TYPE_CHECKING

from player._base import Player

if TYPE_CHECKING:
    import jdm_ru


class RandomPlayer(Player):
    def select_move(self, board: "jdm_ru.PyBoard") -> tuple["jdm_ru.PyMove", float]:
        legal_moves = board.legal_moves()
        move: jdm_ru.PyMove = random.choice(legal_moves)  # noqa: S311
        return move, 0.0
