import io
import signal
from typing import Any

import mlflow
import structlog
from agent.alphazero._agent import AlphaZeroAgent, AlphaZeroAgentConfig
from agent.alphazero._trainer import (
    AlphaZeroTrainer,
    IterationMetrics,
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
from connectors.storage.minio import get_minio_client
from mypy_boto3_s3 import S3Client
from prefect import flow
from prefect.context import get_run_context
from prefect.deployments import run_deployment
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from settings import settings

from workflows._tasks import detect_compute_device

logger = structlog.get_logger(__name__)

DEFAULT_MLFLOW_TRACKING_URI = "https://mlflow.yannhallouard.com"
EVAL_DEPLOYMENT_NAME = "evaluate-alphazero/evaluate-alphazero-k8s"


def _get_s3_client() -> S3Client:
    return get_minio_client()


def _upload_weights(agent: AlphaZeroAgent, iteration: int, execution_id: str) -> str:
    state_dict = agent.model.state_dict()
    weights_bytes = safetensors_save({k: v.cpu() for k, v in state_dict.items()})

    s3_key = f"eval/{execution_id}/iter_{iteration:04d}.safetensors"
    s3 = _get_s3_client()

    s3.upload_fileobj(io.BytesIO(weights_bytes), settings.MINIO_BUCKET, s3_key)
    return s3_key


def _download_weights(agent: AlphaZeroAgent, s3_key: str) -> None:
    s3 = _get_s3_client()
    buffer = io.BytesIO()
    s3.download_fileobj(settings.MINIO_BUCKET, s3_key, buffer)
    buffer.seek(0)
    state_dict = safetensors_load(buffer.read())
    agent.model.load_state_dict(state_dict)


@flow(name="evaluate-alphazero", log_prints=True)
def evaluate_alphazero_flow(
    agent_config: dict[str, Any],
    weights_s3_key: str,
    iteration: int,
    mlflow_tracking_uri: str = DEFAULT_MLFLOW_TRACKING_URI,
    mlflow_run_id: str = "",
    num_games: int = 50,
    opponent_simulations: int = 2000,
) -> dict:
    config = AlphaZeroAgentConfig.model_validate(agent_config)
    config.device = "cpu"
    agent = AlphaZeroAgent(config)
    _download_weights(agent, weights_s3_key)

    from agent.alphazero._trainer import StepLRSchedulerConfig

    trainer = AlphaZeroTrainer(
        agent=agent,
        lr_scheduler_config=StepLRSchedulerConfig(step_size=1, gamma=1.0),
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment="alphazero",
    )

    eval_metrics = evaluate_task(
        trainer,
        num_games=num_games,
        opponent_simulations=opponent_simulations,
        verbose=True,
    )

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics(
            {
                "eval_win_rate": eval_metrics.win_rate,
                "eval_loss_rate": eval_metrics.loss_rate,
                "eval_draw_rate": eval_metrics.draw_rate,
                "eval_avg_steps": eval_metrics.avg_steps,
            },
            step=iteration,
        )

    _cleanup_weights(weights_s3_key)

    return eval_metrics.model_dump()


def _cleanup_weights(s3_key: str) -> None:
    try:
        s3 = _get_s3_client()
        s3.delete_object(Bucket=settings.MINIO_BUCKET, Key=s3_key)
    except Exception:
        logger.warning("failed_to_cleanup_weights", s3_key=s3_key)


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

    mlflow_run_id = trainer.mlflow_logger.run.info.run_id

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

            training_metrics = train_on_buffer_task(trainer, epochs=training.epochs, verbose=training.verbose)

            if training.eval_frequency and iteration % training.eval_frequency == 0:
                logger.info("submitting_async_evaluation", iteration=iteration + 1)
                execution_id = str(get_run_context().flow_run.id)
                s3_key = _upload_weights(trainer.agent, iteration + 1, execution_id)
                run_deployment(
                    name=EVAL_DEPLOYMENT_NAME,
                    parameters={
                        "agent_config": trainer.agent.config.model_dump(),
                        "weights_s3_key": s3_key,
                        "iteration": iteration + 1,
                        "mlflow_tracking_uri": mlflow_tracking_uri,
                        "mlflow_run_id": mlflow_run_id,
                    },
                    timeout=0,
                )

            metrics = IterationMetrics(
                iteration=iteration + 1,
                buffer_size=len(trainer.replay_buffer),
                training_metrics=training_metrics,
            )

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
