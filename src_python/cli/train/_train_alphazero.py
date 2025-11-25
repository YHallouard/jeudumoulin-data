from functools import singledispatch
from pathlib import Path
from typing import Any, Literal

import structlog
from agent.alphazero import AlphaZeroAgent, AlphaZeroTrainer
from agent.alphazero._agent import AlphaZeroAgentConfig
from agent.alphazero._trainer import LRSchedulerConfig
from pydantic import BaseModel

logger = structlog.get_logger()


class FromPretrainedAlphazeroAgentConfig(BaseModel):
    model_path: Path
    device: Literal["cpu", "cuda", "mps"] = "cpu"


@singledispatch
def init_alphazero_agent(config: Any) -> AlphaZeroAgent:
    raise NotImplementedError(f"Initializing model from {config} is not implemented")


@init_alphazero_agent.register(FromPretrainedAlphazeroAgentConfig)
def _(config: FromPretrainedAlphazeroAgentConfig) -> AlphaZeroAgent:
    return AlphaZeroAgent.from_pretrained(config.model_path, device=config.device)


@init_alphazero_agent.register
def _(config: AlphaZeroAgentConfig) -> AlphaZeroAgent:
    return AlphaZeroAgent(config=config)


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
        checkpoint: Path | None = None
        verbose: bool = True

    strategy: str = "alphazero"

    training: TrainingConfig
    agent: FromPretrainedAlphazeroAgentConfig | AlphaZeroAgentConfig


def train_alphazero(
    config: TrainAlphazeroConfig,
) -> None:
    training_config = config.training
    agent = init_alphazero_agent(config.agent)
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
        checkpoint_path=training_config.checkpoint,
        save_frequency=training_config.save_frequency,
        eval_frequency=training_config.eval_frequency,
        verbose=training_config.verbose,
    )

    final_model_path = training_config.save_folder / "final_model.safetensors"
    agent.save_pretrained(final_model_path)
    logger.info("final_model_saved", path=str(final_model_path))

    if training_config.verbose:
        print("\n" + "=" * 70)
        print("🎉 AlphaZero Training Complete!")
        print("=" * 70)
        print(f"Final model saved to: {final_model_path}")
        print("=" * 70 + "\n")
