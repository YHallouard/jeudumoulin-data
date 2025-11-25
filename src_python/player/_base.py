from abc import ABC, abstractmethod

import jdm_ru


class Player(ABC):
    @abstractmethod
    def select_move(self, board: jdm_ru.PyBoard) -> tuple[jdm_ru.PyMove, float]:
        pass
