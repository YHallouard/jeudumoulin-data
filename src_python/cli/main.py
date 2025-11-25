"""Command-line interface for Jeu du Moulin."""

import click

from cli.play import play
from cli.train import train


@click.group(help="Jeu du Moulin - Nine Men's Morris game with AI agents")
def cli() -> None:
    pass


cli.add_command(play)
cli.add_command(train)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
