import tempfile
from functools import singledispatch
from pathlib import Path
from typing import Any, Literal

import structlog
from agent.alphazero import AlphaZeroAgent, AlphaZeroTrainer
from agent.alphazero._agent import AlphaZeroAgentConfig
from agent.alphazero._trainer import LRSchedulerConfig
from pydantic import BaseModel
from utils.checkpoints import CheckpointHandler, CheckpointStorageConfig, S3CheckpointStorage, get_checkpoint_handler

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Init strategies
# ---------------------------------------------------------------------------


class NewModelInit(BaseModel):
    strategy: Literal["new"] = "new"
    agent: AlphaZeroAgentConfig
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class LocalCheckpointInit(BaseModel):
    strategy: Literal["local"] = "local"
    model_path: Path
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class S3CheckpointInit(BaseModel):
    strategy: Literal["s3"] = "s3"
    s3_prefix: str
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    load_buffer: bool = True
    load_optimizer: bool = True
    start_iteration: int = 0


TrainingInitConfig = NewModelInit | LocalCheckpointInit | S3CheckpointInit


# -------------------------------------------------------------------
# Agent factory — CLI path (no S3)
# ---------------------------------------------------------------------------


@singledispatch
def _create_agent_from_init(config: Any, **kwargs: Any) -> AlphaZeroAgent:
    raise NotImplementedError(f"Unsupported init strategy for CLI: {config.strategy}")


@_create_agent_from_init.register(NewModelInit)
def _(config: NewModelInit, **kwargs: Any) -> AlphaZeroAgent:
    agent_config = AlphaZeroAgentConfig(model=config.agent.model, device=config.device)
    return AlphaZeroAgent(config=agent_config)


@_create_agent_from_init.register(LocalCheckpointInit)
def _(config: LocalCheckpointInit, **kwargs: Any) -> AlphaZeroAgent:
    return AlphaZeroAgent.from_pretrained(config.model_path, device=config.device)


@_create_agent_from_init.register(S3CheckpointInit)
def _(init: S3CheckpointInit, **kwargs: Any) -> AlphaZeroAgent:
    handler: CheckpointHandler = kwargs.get("handler") or get_checkpoint_handler(S3CheckpointStorage())
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        handler.download_agent_files(init.s3_prefix, tmp_path)
        return AlphaZeroAgent.from_pretrained(tmp_path, device=init.device)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TrainAlphazeroConfig(BaseModel):
    class TrainingConfig(BaseModel):
        iterations: int
        episodes: int
        simulations: int
        max_episode_steps: int
        epochs: int
        replay_buffer_size: int
        batch_size: int
        learning_rate: float
        lr_scheduler_config: LRSchedulerConfig
        temperature: float
        save_frequency: int
        eval_frequency: int
        verbose: bool = True

    strategy: str = "alphazero"
    init: TrainingInitConfig
    training: TrainingConfig
    checkpoint_storage: CheckpointStorageConfig = S3CheckpointStorage()


# ---------------------------------------------------------------------------
# CLI train function
# ---------------------------------------------------------------------------


def train_alphazero(config: TrainAlphazeroConfig) -> None:
    training_config = config.training
    handler = get_checkpoint_handler(config.checkpoint_storage)
    agent = _create_agent_from_init(config.init, handler=handler)
    trainer = AlphaZeroTrainer(
        agent=agent,
        lr_scheduler_config=training_config.lr_scheduler_config,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        buffer_size=training_config.replay_buffer_size,
    )

    trainer.train(
        num_iterations=training_config.iterations,
        episodes_per_iteration=training_config.episodes,
        simulations_per_move=training_config.simulations,
        max_episode_steps=training_config.max_episode_steps,
        epochs_per_iteration=training_config.epochs,
        temperature=training_config.temperature,
        checkpoint_handler=handler,
        save_frequency=training_config.save_frequency,
        eval_frequency=training_config.eval_frequency,
        verbose=training_config.verbose,
    )
