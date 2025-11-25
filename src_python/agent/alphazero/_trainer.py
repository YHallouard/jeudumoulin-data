import json
import random
from datetime import datetime
from functools import singledispatch
from pathlib import Path

import jdm_ru
import structlog
import torch
from jdm_ru import generate_train_examples
from monitoring import MLflowLogger
from pydantic import BaseModel
from safetensors.torch import save_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler, StepLR
from tqdm import tqdm  # type: ignore[import-untyped]

from agent.alphazero.random_agent import RandomAgent

from ._agent import AlphaZeroAgent
from ._conditional_policy import conditional_cross_entropy
from ._replay_buffer import AlphaZeroReplayBuffer

logger = structlog.get_logger(__name__)


class StepLRSchedulerConfig(BaseModel):
    model_type: str = "step_lr"
    step_size: int
    gamma: float


class CosineWarmRestartLRSchedulerConfig(BaseModel):
    model_type: str = "cosine_warm_rest_lr"
    T_0: int
    T_mult: int = 1
    min_lr: float
    last_epoch: int = -1


LRSchedulerConfig = StepLRSchedulerConfig | CosineWarmRestartLRSchedulerConfig


@singledispatch
def get_scheduler(config: LRSchedulerConfig, optimizer: torch.optim.Optimizer) -> LRScheduler:
    raise ValueError(f"Unsupported scheduler type: {config.model_type}")  # noqa: TRY003


@get_scheduler.register
def _(config: StepLRSchedulerConfig, optimizer: torch.optim.Optimizer) -> StepLR:
    return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)


@get_scheduler.register
def _(config: CosineWarmRestartLRSchedulerConfig, optimizer: torch.optim.Optimizer) -> CosineAnnealingWarmRestarts:
    return CosineAnnealingWarmRestarts(
        optimizer, T_0=config.T_0, T_mult=config.T_mult, eta_min=config.min_lr, last_epoch=config.last_epoch
    )


class AlphaZeroTrainer:
    def __init__(
        self,
        agent: AlphaZeroAgent,
        lr_scheduler_config: LRSchedulerConfig,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        buffer_size: int = 10000,
        device: str = "cpu",
        mlflow_tracking_uri: str = "http://localhost:5001",
        mlflow_experiment: str = "alphazero",
    ):
        self.agent = agent
        self.agent.model.to("cpu")
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device

        self.replay_buffer = AlphaZeroReplayBuffer(max_size=buffer_size)
        self.optimizer = torch.optim.Adam([
            {
                "params": [param for name, param in agent.model.named_parameters() if "backbone" in name],
                "lr": learning_rate,
            },
            {
                "params": [param for name, param in agent.model.named_parameters() if "policy" in name],
                "lr": learning_rate,
            },
            {
                "params": [param for name, param in agent.model.named_parameters() if "value" in name],
                "lr": learning_rate * 0.5,
            },
        ])
        self.lr_scheduler_config = lr_scheduler_config
        self.scheduler = get_scheduler(self.lr_scheduler_config, self.optimizer)

        self.iteration_metrics: list[dict] = []

        self.mlflow_logger = MLflowLogger(
            experiment_name=mlflow_experiment,
            run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tracking_uri=mlflow_tracking_uri,
            tags={"algorithm": "alphazero"},
        )

    def train(
        self,
        num_iterations: int,
        episodes_per_iteration: int,
        simulations_per_move: int,
        max_episode_steps: int,
        epochs_per_iteration: int,
        temperature: float,
        save_folder: Path,
        checkpoint_path: Path | None = None,
        save_frequency: int = 1,
        eval_frequency: int = 1,
        verbose: bool = True,
    ) -> dict:
        self.mlflow_logger.start()

        self.mlflow_logger.log_params({
            "num_iterations": num_iterations,
            "episodes_per_iteration": episodes_per_iteration,
            "simulations_per_move": simulations_per_move,
            "max_episode_steps": max_episode_steps,
            "epochs_per_iteration": epochs_per_iteration,
            "temperature": temperature,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "buffer_size": self.replay_buffer.max_size,
            "save_frequency": save_frequency,
            "eval_frequency": eval_frequency,
            **{f"lr_scheduler__{key}": value for key, value in self.lr_scheduler_config.model_dump().items()},
        })

        self.mlflow_logger.log_config_artifact(self.agent.config.model_dump(), "config.json")

        if verbose:
            logger.info(
                "starting_alphazero_training",
                num_iterations=num_iterations,
                episodes_per_iteration=episodes_per_iteration,
                simulations_per_move=simulations_per_move,
                epochs_per_iteration=epochs_per_iteration,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                buffer_size=self.replay_buffer.max_size,
                save_folder=str(save_folder),
            )

        save_folder.mkdir(parents=True, exist_ok=True)

        if checkpoint_path:
            logger.info("loading_checkpoint", path=str(checkpoint_path))
            start_iteration = 0
        else:
            start_iteration = 0

        for iteration in range(start_iteration, num_iterations):
            logger.info("iteration_start", iteration=iteration + 1, total=num_iterations)

            self.agent.model.eval()
            logger.info("generating_self_play_data", episodes=episodes_per_iteration)
            state_embeddings, legal_moves_data, policy_labels, value_labels = generate_train_examples(
                self.agent,
                simulations_per_move,
                episodes_per_iteration,
                max_episode_steps,
                temperature,
            )

            logger.info(
                "self_play_complete",
                num_examples=len(state_embeddings),
            )

            self.replay_buffer.add_examples(state_embeddings, legal_moves_data, policy_labels, value_labels)
            logger.info("replay_buffer_size", size=len(self.replay_buffer))

            self.agent.model.train()
            logger.info("training_network", epochs=epochs_per_iteration)
            iteration_losses = self._train_on_buffer(epochs_per_iteration, verbose)

            if save_frequency and iteration % save_frequency == 0:
                self._save_checkpoint(save_folder, iteration)

            if eval_frequency and iteration % eval_frequency == 0:
                eval_metrics = self.evaluate(num_games=50, opponent_simulations=2000, verbose=verbose)

                self.mlflow_logger.log_metrics(
                    {
                        "eval_win_rate": eval_metrics["win_rate"],
                        "eval_loss_rate": eval_metrics["loss_rate"],
                        "eval_draw_rate": eval_metrics["draw_rate"],
                        "eval_avg_steps": eval_metrics["avg_steps"],
                    },
                    step=iteration + 1,
                )

            metrics = {
                "iteration": iteration + 1,
                "num_examples": len(state_embeddings) / episodes_per_iteration,
                "buffer_size": len(self.replay_buffer),
                "avg_policy_loss": iteration_losses["avg_policy_loss"],
                "avg_value_loss": iteration_losses["avg_value_loss"],
                "avg_total_loss": iteration_losses["avg_total_loss"],
            }

            self.iteration_metrics.append(metrics)

            self.scheduler.step()

            self.mlflow_logger.log_metrics(
                {
                    "policy_loss": metrics["avg_policy_loss"],
                    "value_loss": metrics["avg_value_loss"],
                    "total_loss": metrics["avg_total_loss"],
                    "num_examples": metrics["num_examples"],
                    "buffer_size": metrics["buffer_size"],
                    "learning_rate": self.scheduler.get_last_lr()[0],
                },
                step=iteration + 1,
            )

            logger.info("iteration_complete", iteration=iteration + 1)

        logger.info("training_complete")

        self.mlflow_logger.finish()

        return self._get_training_metrics()

    def _train_on_buffer(self, epochs: int, verbose: bool) -> dict:
        epoch_policy_losses = []
        epoch_value_losses = []
        epoch_total_losses = []

        self.agent.model.to(self.device)

        with tqdm(range(epochs), desc="Epochs", disable=not verbose) as progress_bar:
            for _ in progress_bar:
                epoch_policy_loss = 0.0
                epoch_value_loss = 0.0
                epoch_total_loss = 0.0
                num_batches = 0

                num_training_batches = max(1, len(self.replay_buffer) // self.batch_size)

                for _ in range(num_training_batches):
                    states, moves, policies, values = self.replay_buffer.sample_batch(self.batch_size)

                    policy_loss, value_loss, total_loss = self._train_on_batch(states, moves, policies, values)

                    epoch_policy_loss += policy_loss
                    epoch_value_loss += value_loss
                    epoch_total_loss += total_loss
                    num_batches += 1

                avg_policy_loss = epoch_policy_loss / num_batches if num_batches > 0 else 0
                avg_value_loss = epoch_value_loss / num_batches if num_batches > 0 else 0
                avg_total_loss = epoch_total_loss / num_batches if num_batches > 0 else 0

                epoch_policy_losses.append(avg_policy_loss)
                epoch_value_losses.append(avg_value_loss)
                epoch_total_losses.append(avg_total_loss)

                progress_bar.set_postfix({
                    "policy_loss": f"{avg_policy_loss:.4f}",
                    "value_loss": f"{avg_value_loss:.4f}",
                    "total_loss": f"{avg_total_loss:.4f}",
                })

        self.agent.model.to("cpu")

        return {
            "avg_policy_loss": sum(epoch_policy_losses) / len(epoch_policy_losses) if epoch_policy_losses else 0.0,
            "avg_value_loss": sum(epoch_value_losses) / len(epoch_value_losses) if epoch_value_losses else 0.0,
            "avg_total_loss": sum(epoch_total_losses) / len(epoch_total_losses) if epoch_total_losses else 0.0,
        }

    def _train_on_batch(
        self,
        state_embeddings: list[list[float]],
        legal_moves_batch: list,
        policy_targets: list[list[float]],
        value_targets: list[float],
    ) -> tuple[float, float, float]:
        self.optimizer.zero_grad()

        batch_policy_loss = 0.0
        batch_value_loss = 0.0
        batch_size = len(state_embeddings)

        for state_emb, legal_moves, policy_target, value_target in zip(
            state_embeddings, legal_moves_batch, policy_targets, value_targets
        ):
            policy_pred, value_pred = self.agent.model.policy_value(state_emb, legal_moves)

            policy_target_tensor = torch.tensor(policy_target, device=policy_pred.device)
            policy_loss = conditional_cross_entropy(policy_pred, policy_target_tensor)

            value_target_tensor = torch.tensor([value_target], device=value_pred.device)
            value_loss = torch.nn.functional.mse_loss(value_pred, value_target_tensor)

            total_loss = policy_loss + value_loss
            total_loss.backward()

            batch_policy_loss += policy_loss.item()
            batch_value_loss += value_loss.item()

        self.optimizer.step()

        return (
            batch_policy_loss / batch_size,
            batch_value_loss / batch_size,
            (batch_policy_loss + batch_value_loss) / batch_size,
        )

    def _save_checkpoint(self, save_folder: Path, iteration: int) -> None:
        model_path = save_folder / "checkpoints" / f"model_iter{iteration:04d}.safetensors"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(self.agent.model, str(model_path))
        with open(model_path.parent.parent / "config.json", "w") as f:
            json.dump(self.agent.config.model_dump(), f, indent=4)
        logger.info("checkpoint_saved", path=str(model_path), iteration=iteration)

    def evaluate(
        self,
        num_games: int = 100,
        opponent_simulations: int = 2000,
        verbose: bool = True,
    ) -> dict:
        from player import AlphaZeroPlayer

        self.agent.model.eval()

        agent_player_instance = AlphaZeroPlayer(
            agent=self.agent,
            temperature=0,
            num_simulations=200,
        )

        opponent_agent = RandomAgent()
        opponent_player_instance = AlphaZeroPlayer(
            agent=opponent_agent,
            temperature=0,
            num_simulations=opponent_simulations,
        )

        wins = 0
        losses = 0
        draws = 0
        agent_total_steps = []

        for _ in range(num_games):
            board = jdm_ru.PyBoard()
            agent_color = random.choice([1, -1])
            step = 0
            max_steps = 300

            while not board.is_terminal() and step < max_steps:
                current_player = board.current_player()
                legal_moves = board.legal_moves()

                if not legal_moves:
                    break

                if current_player == agent_color:
                    move, _ = agent_player_instance.select_move(board)
                else:
                    move, _ = opponent_player_instance.select_move(board)

                board = board.apply_move(move)
                step += 1

            winner = board.winner()
            if winner is None:
                draws += 1
            elif winner == agent_color:
                wins += 1
            else:
                losses += 1

            agent_total_steps.append(step)

        metrics = {
            "num_games": num_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games,
            "loss_rate": losses / num_games,
            "draw_rate": draws / num_games,
            "avg_steps": sum(agent_total_steps) / len(agent_total_steps),
        }

        if verbose:
            logger.info(
                "evaluation_results",
                num_games=num_games,
                opponent_simulations=opponent_simulations,
                wins=wins,
                win_rate=metrics["win_rate"],
                losses=losses,
                loss_rate=metrics["loss_rate"],
                draws=draws,
                draw_rate=metrics["draw_rate"],
                avg_steps=metrics["avg_steps"],
            )

        return metrics

    def _get_training_metrics(self) -> dict:
        return {
            "iteration_metrics": self.iteration_metrics,
            "buffer_size": len(self.replay_buffer),
            "buffer_statistics": self.replay_buffer.get_statistics(),
        }

    def get_buffer_statistics(self) -> dict:
        return self.replay_buffer.get_statistics()
