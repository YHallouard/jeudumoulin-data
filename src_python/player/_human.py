import jdm_ru

from player._base import Player


class HumanPlayer(Player):
    def select_move(self, board: jdm_ru.PyBoard) -> tuple[jdm_ru.PyMove, float]:
        legal_moves = board.legal_moves()

        print("\nLegal moves:")
        for i, move in enumerate(legal_moves):
            print(f"{i}: {move}")

        while True:
            try:
                choice = int(input("Enter move number: "))
                if 0 <= choice < len(legal_moves):
                    return legal_moves[choice], 0.0
                else:
                    print(f"Invalid choice. Please enter a number between 0 and {len(legal_moves) - 1}")
            except ValueError:
                print("Invalid input. Please enter a number.")
