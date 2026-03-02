import signal
from typing import Any

from agent.alphazero._trainer import (
    AlphaZeroTrainer,
    evaluate_task,
    finish_mlflow_task,
    log_iteration_metrics_task,
    register_model_task,
    self_play_task,
    start_mlflow_task,
    train_on_buffer_task,
)
from agent.dqn._trainer import (
    DQNTrainer,
    dqn_episode_batch_task,
    dqn_evaluate_task,
    dqn_finish_mlflow_task,
    dqn_log_eval_metrics_task,
    dqn_register_model_task,
    dqn_start_mlflow_task,
)
from cli.train._train_alphazero import TrainAlphazeroConfig, init_alphazero_agent
from cli.train._train_dqn import TrainDQNConfig, init_dqn_agent
from prefect import flow

from workflows._tasks import detect_compute_device

DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.yannhallouard.com"


@flow(name="train-alphazero", log_prints=True)
def train_alphazero_flow(
    raw_config: dict[str, Any],
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict:
    device = detect_compute_device()
    raw_config["agent"]["device"] = device

    config = TrainAlphazeroConfig.model_validate(raw_config)
    agent = init_alphazero_agent(config.agent)
    training = config.training

    trainer = AlphaZeroTrainer(
        agent=agent,
        lr_scheduler_config=training.lr_scheduler_config,
        learning_rate=training.learning_rate,
        batch_size=training.batch_size,
        buffer_size=training.replay_buffer_size,
        device=config.agent.device,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    start_mlflow_task(
        trainer,
        num_iterations=training.iterations,
        episodes_per_iteration=training.episodes,
        simulations_per_move=training.simulations,
        max_episode_steps=training.max_episode_steps,
        epochs_per_iteration=training.epochs,
        temperature=training.temperature,
        save_frequency=training.save_frequency,
        eval_frequency=training.eval_frequency,
    )

    stop_requested = False

    def handle_stop_signal(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_sigterm_handler = signal.signal(signal.SIGTERM, handle_stop_signal)
    prev_sigint_handler = signal.signal(signal.SIGINT, handle_stop_signal)

    try:
        for iteration in range(training.iterations):
            if stop_requested:
                break

            self_play_task(
                trainer,
                episodes=training.episodes,
                simulations_per_move=training.simulations,
                max_episode_steps=training.max_episode_steps,
                temperature=training.temperature,
            )

            losses = train_on_buffer_task(trainer, epochs=training.epochs, verbose=training.verbose)

            eval_metrics = None
            if training.eval_frequency and iteration % training.eval_frequency == 0:
                eval_metrics = evaluate_task(trainer, num_games=50, opponent_simulations=2000, verbose=training.verbose)

            metrics = {
                "iteration": iteration + 1,
                "buffer_size": len(trainer.replay_buffer),
                "avg_policy_loss": losses["avg_policy_loss"],
                "avg_value_loss": losses["avg_value_loss"],
                "avg_total_loss": losses["avg_total_loss"],
                "eval_metrics": eval_metrics,
            }

            log_iteration_metrics_task(trainer, metrics, iteration)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm_handler)
        signal.signal(signal.SIGINT, prev_sigint_handler)
        register_model_task(trainer)
        finish_mlflow_task(trainer)

    return trainer._get_training_metrics()


@flow(name="train-dqn", log_prints=True)
def train_dqn_flow(
    raw_config: dict[str, Any],
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
) -> dict:
    device = detect_compute_device()
    raw_config["agent"]["device"] = device

    config = TrainDQNConfig.model_validate(raw_config)
    agent = init_dqn_agent(config.agent)
    training = config.training

    trainer = DQNTrainer(
        agent=agent,
        learning_rate=training.learning_rate,
        gamma=training.gamma,
        batch_size=training.batch_size,
        buffer_size=training.buffer_size,
        mlflow_tracking_uri=mlflow_tracking_uri,
    )

    dqn_start_mlflow_task(
        trainer,
        num_episodes=training.episodes,
        epsilon_start=training.epsilon_start,
        epsilon_end=training.epsilon_end,
        epsilon_decay=training.epsilon_decay,
        opponent=training.opponent,
        max_steps_per_episode=training.max_steps,
        save_frequency=training.save_frequency,
        eval_frequency=training.eval_frequency,
    )

    batch_size = training.eval_frequency if training.eval_frequency else training.episodes

    stop_requested = False

    def handle_stop_signal(signum: int, frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_sigterm_handler = signal.signal(signal.SIGTERM, handle_stop_signal)
    prev_sigint_handler = signal.signal(signal.SIGINT, handle_stop_signal)

    try:
        for batch_start in range(0, training.episodes, batch_size):
            if stop_requested:
                break

            actual_batch = min(batch_size, training.episodes - batch_start)

            dqn_episode_batch_task(
                trainer,
                num_episodes=actual_batch,
                epsilon_start=max(training.epsilon_end, training.epsilon_start * (training.epsilon_decay**batch_start)),
                epsilon_end=training.epsilon_end,
                epsilon_decay=training.epsilon_decay,
                opponent=training.opponent,
                max_steps_per_episode=training.max_steps,
                verbose=training.verbose,
            )

            if training.eval_frequency and (batch_start + actual_batch) % training.eval_frequency == 0:
                eval_metrics = dqn_evaluate_task(
                    trainer,
                    num_games=training.eval_games,
                    opponent=training.opponent,
                    verbose=training.verbose,
                )
                dqn_log_eval_metrics_task(trainer, eval_metrics, batch_start + actual_batch)
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm_handler)
        signal.signal(signal.SIGINT, prev_sigint_handler)
        dqn_register_model_task(trainer)
        dqn_finish_mlflow_task(trainer)

    return trainer._get_training_metrics()
