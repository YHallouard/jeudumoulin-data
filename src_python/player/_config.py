from functools import singledispatch
from pathlib import Path

from agent.alphazero._agent import AlphaZeroAgent
from agent.dqn._agent import DQNAgent
from pydantic import BaseModel
from typing_extensions import Literal

from player._alphazero import AlphaZeroPlayer
from player._base import Player
from player._dqn import DQNPlayer
from player._human import HumanPlayer


class HumanPlayerConfig(BaseModel):
    type: Literal["human"] = "human"


class AlphaZeroPlayerConfig(BaseModel):
    type: Literal["alphazero"] = "alphazero"
    model_path: Path
    temperature: float = 1.0
    num_simulations: int = 100


class DQNPlayerConfig(BaseModel):
    type: Literal["dqn"] = "dqn"
    model_path: Path


PlayerConfig = HumanPlayerConfig | AlphaZeroPlayerConfig | DQNPlayerConfig


@singledispatch
def get_player(config: PlayerConfig) -> Player:
    raise NotImplementedError(f"Unsupported player config type: {type(config)}")


@get_player.register
def _(config: HumanPlayerConfig) -> Player:
    return HumanPlayer()


@get_player.register
def _(config: AlphaZeroPlayerConfig) -> Player:
    alphazero_agent = AlphaZeroAgent.from_pretrained(config.model_path)
    return AlphaZeroPlayer(
        alphazero_agent,
        temperature=config.temperature,
        num_simulations=config.num_simulations,
    )


@get_player.register
def _(config: DQNPlayerConfig) -> Player:
    dqn_agent = DQNAgent.from_pretrained(config.model_path)
    return DQNPlayer(dqn_agent)
