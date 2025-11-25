import click
import jdm_ru
from player import PlayerConfig, get_player
from pydantic import BaseModel

from cli.utils import yaml_arg


class PlayConfig(BaseModel):
    player1: PlayerConfig
    player2: PlayerConfig


@click.command()
@click.option(
    "--config",
    type=yaml_arg,
    help="Path to a YAML config file for the game parameters",
)
def play(config: PlayConfig) -> None:
    print("Starting Jeu du Moulin game...")
    print(f"Player 1 (White): {config.player1}")
    print(f"Player 2 (Black): {config.player2}")

    player1_instance = get_player(config.player1)
    player2_instance = get_player(config.player2)

    board = jdm_ru.PyBoard()

    move_count = 0
    while not board.is_terminal():
        print(f"\n--- Move {move_count + 1} ---")
        board.print_board()

        current_player = board.current_player()

        if current_player == 1:
            print("Current player: White")
            move1, value1 = player1_instance.select_move(board)
            board = board.apply_move(move1)
            print(f"White selected: {move1}, value: {-value1}")
        else:
            print("Current player: Black")
            move2, value2 = player2_instance.select_move(board)
            board = board.apply_move(move2)
            print(f"Black selected: {move2}, value: {-value2}")

        move_count += 1

    print("\n=== Game Over ===")
    board.print_board()
    winner = board.winner()
    if winner == 1:
        print("Winner: White")
    elif winner == -1:
        print("Winner: Black")
    else:
        print("Draw")
