from functools import singledispatch
from typing import Any

import click
from pydantic import BaseModel

from cli.utils import yaml_arg

from ._train_alphazero import TrainAlphazeroConfig, train_alphazero
from ._train_dqn import TrainDQNConfig, train_dqn


class TrainConfig(BaseModel):
    config: TrainAlphazeroConfig | TrainDQNConfig


@singledispatch
def start_train(config: Any) -> None:
    raise NotImplementedError("Unsupported training configuration")


@start_train.register
def _(config: TrainAlphazeroConfig) -> None:
    train_alphazero(config)


@start_train.register
def _(config: TrainDQNConfig) -> None:
    train_dqn(config)


@click.command()
@click.option("--config", type=yaml_arg, required=True, help="Path to the training configuration file")
def train(config: dict[str, Any]) -> None:
    train_config = TrainConfig(**config)
    start_train(train_config.config)
