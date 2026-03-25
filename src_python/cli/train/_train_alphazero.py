import tempfile
from functools import singledispatch
from pathlib import Path
from typing import Annotated, Any, Literal

import structlog
from agent.alphazero import AlphaZeroAgent, AlphaZeroTrainer
from agent.alphazero._agent import AlphaZeroAgentConfig
from agent.alphazero._trainer import LRSchedulerConfig
from pydantic import BaseModel, Field
from utils.checkpoints import download_agent_files

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


TrainingInitConfig = Annotated[
    NewModelInit | LocalCheckpointInit | S3CheckpointInit,
    Field(discriminator="strategy"),
]


# ---------------------------------------------------------------------------
# Agent factory — CLI path (no S3)
# ---------------------------------------------------------------------------


@singledispatch
def _create_agent_from_init(config: Any) -> AlphaZeroAgent:
    raise NotImplementedError(f"Unsupported init strategy for CLI: {config.strategy}")  # noqa: TRY003


@_create_agent_from_init.register(NewModelInit)
def _(config: NewModelInit) -> AlphaZeroAgent:
    agent_config = AlphaZeroAgentConfig(model=config.agent.model, device=config.device)
    return AlphaZeroAgent(config=agent_config)


@_create_agent_from_init.register(LocalCheckpointInit)
def _(config: LocalCheckpointInit) -> AlphaZeroAgent:
    return AlphaZeroAgent.from_pretrained(config.model_path, device=config.device)


@_create_agent_from_init.register(S3CheckpointInit)
def _(init: S3CheckpointInit) -> AlphaZeroAgent:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        download_agent_files(init.s3_prefix, tmp_path)
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
        save_folder: Path
        save_frequency: int
        eval_frequency: int
        verbose: bool = True

    strategy: str = "alphazero"
    init: TrainingInitConfig
    training: TrainingConfig


# ---------------------------------------------------------------------------
# CLI train function
# ---------------------------------------------------------------------------


def train_alphazero(config: TrainAlphazeroConfig) -> None:
    training_config = config.training
    agent = _create_agent_from_init(config.init)
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
        save_folder=training_config.save_folder,
        checkpoint_path=None,
        save_frequency=training_config.save_frequency,
        eval_frequency=training_config.eval_frequency,
        verbose=training_config.verbose,
    )

    final_model_path = training_config.save_folder / "final_model.safetensors"
    agent.save_pretrained(final_model_path)
    logger.info("final_model_saved", path=str(final_model_path))

    if training_config.verbose:
        print("\n" + "=" * 70)
        print("AlphaZero Training Complete!")
        print("=" * 70)
        print(f"Final model saved to: {final_model_path}")
        print("=" * 70 + "\n")
