from functools import singledispatch
from pathlib import Path
from typing import Any, Literal

import structlog
from agent.dqn._agent import DQNAgent, DQNAgentConfig
from agent.dqn._trainer import DQNTrainer
from pydantic import BaseModel

logger = structlog.get_logger()


class FromPretrainedDQNAgentConfig(BaseModel):
    model_path: Path
    device: Literal["cpu", "cuda", "mps"] = "cpu"


@singledispatch
def init_dqn_agent(config: Any) -> DQNAgent:
    raise NotImplementedError(f"Initializing model from {config} is not implemented")


@init_dqn_agent.register(FromPretrainedDQNAgentConfig)
def _(config: FromPretrainedDQNAgentConfig) -> DQNAgent:
    return DQNAgent.from_pretrained(config.model_path, device=config.device)


@init_dqn_agent.register
def _(config: DQNAgentConfig) -> DQNAgent:
    return DQNAgent(config=config)


class TrainDQNConfig(BaseModel):
    class TrainingConfig(BaseModel):
        episodes: int = 1000
        learning_rate: float = 0.001
        gamma: float = 0.99
        batch_size: int = 64
        buffer_size: int = 10000
        epsilon_start: float = 1.0
        epsilon_end: float = 0.1
        epsilon_decay: float = 0.995
        opponent: str = "random"
        max_steps: int = 100
        use_prioritized_replay: bool = False
        save_folder: Path = Path("models/dqn")
        save_frequency: int | None = 100
        eval_frequency: int = 100
        eval_games: int = 50
        verbose: bool = True

    strategy: str = "dqn"
    agent: FromPretrainedDQNAgentConfig | DQNAgentConfig
    training: TrainingConfig


def train_dqn(
    config: TrainDQNConfig,
) -> None:
    """
    Train a DQN agent to play Jeu du Moulin (Nine Men's Morris).

    This command implements Deep Q-Learning with experience replay and optional
    Double DQN and prioritized replay features. The agent learns to play by
    maximizing cumulative rewards through self-play or against random opponents.

    Examples:

        # Basic training with default settings
        $ jeudumoulin train-dqn

        # Fast training with fewer episodes
        $ jeudumoulin train-dqn --episodes 100 --epsilon-decay 0.9

        # Advanced: Double DQN with prioritized replay
        $ jeudumoulin train-dqn --use-double-dqn --use-prioritized-replay \\
            --buffer-size 50000 --batch-size 128


        # Resume from checkpoint
        $ jeudumoulin train-dqn --checkpoint models/dqn/checkpoint_ep500.pt

        # Self-play training (more challenging)
        $ jeudumoulin train-dqn --opponent self --episodes 2000
    """
    training_config = config.training

    if training_config.use_prioritized_replay:
        logger.warning(
            "prioritized_replay_not_implemented",
            message="Prioritized replay buffer available but not integrated in trainer yet. Using standard replay.",
        )

    agent = init_dqn_agent(config.agent)

    if training_config.verbose:
        stats = agent.get_statistics()
        print("\n" + "=" * 70)
        print("DQN Agent Configuration")
        print("=" * 70)
        print(f"Model type: {stats['model_type']}")
        print(f"Total parameters: {stats['total_parameters']:,}")
        print(f"Device: {config.agent.device}")
        print("=" * 70 + "\n")

    trainer = DQNTrainer(
        agent=agent,
        learning_rate=training_config.learning_rate,
        gamma=training_config.gamma,
        batch_size=training_config.batch_size,
        buffer_size=training_config.buffer_size,
    )

    trainer.train(
        num_episodes=training_config.episodes,
        epsilon_start=training_config.epsilon_start,
        epsilon_end=training_config.epsilon_end,
        epsilon_decay=training_config.epsilon_decay,
        opponent=training_config.opponent,
        max_steps_per_episode=training_config.max_steps,
        verbose=training_config.verbose,
        save_frequency=training_config.save_frequency,
        save_folder=training_config.save_folder,
        eval_frequency=training_config.eval_frequency,
        eval_games=training_config.eval_games,
    )

    agent.save_pretrained(training_config.save_folder)
    logger.info("final_model_saved", path=str(training_config.save_folder))

    if training_config.verbose:
        print("\n" + "=" * 70)
        print("🎉 DQN Training Complete!")
        print("=" * 70)
        print(f"Final model saved to: {training_config.save_folder}")
        print(f"Total training steps: {trainer.total_steps:,}")
        print("=" * 70 + "\n")
