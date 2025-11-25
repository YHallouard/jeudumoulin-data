import json
import random
from datetime import datetime
from pathlib import Path

import jdm_ru
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from monitoring import MLflowLogger
from player._dqn import DQNPlayer
from player._random import RandomPlayer
from reward.calculator import RewardCalculator
from safetensors.torch import save_model

from ._agent import DQNAgent
from ._replay_buffer import ExperienceReplayBuffer

logger = structlog.get_logger(__name__)


class DQNTrainer:
    """
    Trainer for DQN agents using Q-learning with experience replay.

    The trainer manages the full training loop:
    1. Generate episodes by playing against opponents
    2. Store transitions in replay buffer
    3. Sample batches and perform Q-learning updates
    4. Update target network periodically
    5. Track and log training metrics

    Args:
        agent: DQN agent to train
        learning_rate: Learning rate for optimizer (default: 0.001)
        gamma: Discount factor for future rewards (default: 0.99)
        batch_size: Batch size for training (default: 64)
        buffer_size: Size of replay buffer (default: 10000)
        target_update_frequency: Steps between target network updates (default: 100)
        device: Device for computation ('cpu' or 'cuda')

    Example:
        >>> agent = DQNAgent()
        >>> trainer = DQNTrainer(agent, learning_rate=0.001)
        >>>
        >>> # Train for 1000 episodes
        >>> metrics = trainer.train(
        ...     num_episodes=1000,
        ...     epsilon_start=1.0,
        ...     epsilon_end=0.05,
        ...     epsilon_decay=0.995,
        ...     opponent='random'
        ... )
    """

    def __init__(
        self,
        agent: DQNAgent,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 10000,
        mlflow_tracking_uri: str = "http://localhost:5001",
        mlflow_experiment: str = "dqn",
    ):
        """Initialize the trainer."""
        self.agent = agent
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = agent.config.device

        self.replay_buffer = ExperienceReplayBuffer(max_size=buffer_size, device=self.device)
        self.reward_calculator = RewardCalculator()

        self.optimizer = optim.Adam(agent.get_model_parameters().parameters(), lr=learning_rate)

        self.criterion = nn.MSELoss()

        self.total_steps = 0
        self.episode_rewards: list[float] = []
        self.episode_losses: list[float] = []
        self.win_counts = {"agent": 0, "opponent": 0, "draw": 0}

        self.mlflow_logger = MLflowLogger(
            experiment_name=mlflow_experiment,
            run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tracking_uri=mlflow_tracking_uri,
            tags={"algorithm": "dqn"},
        )

    def train(
        self,
        num_episodes: int,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        opponent: str = "random",
        max_steps_per_episode: int = 200,
        verbose: bool = True,
        save_frequency: int | None = None,
        save_folder: Path | None = None,
        eval_frequency: int = 0,
        eval_games: int = 50,
    ) -> dict:
        self.mlflow_logger.start()

        self.mlflow_logger.log_params({
            "num_episodes": num_episodes,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "opponent": opponent,
            "max_steps_per_episode": max_steps_per_episode,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "buffer_size": self.replay_buffer.max_size,
            "save_frequency": save_frequency,
            "eval_frequency": eval_frequency,
        })

        self.mlflow_logger.log_config_artifact(self.agent.config.model_dump(), "config.json")

        if verbose:
            logger.info(
                "starting_dqn_training",
                num_episodes=num_episodes,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                opponent=opponent,
                replay_buffer_size=self.replay_buffer.max_size,
                batch_size=self.batch_size,
            )

        if save_folder:
            save_folder.mkdir(parents=True, exist_ok=True)

        logger.info("training_with_evaluation", eval_frequency=eval_frequency, eval_games=eval_games)

        for episode_start in range(0, num_episodes):
            episodes_this_segment = min(eval_frequency, num_episodes - episode_start)

            logger.info(
                "training_segment",
                episode_range=(episode_start, episode_start + episodes_this_segment),
            )

            self._train_episodes(
                num_episodes=episodes_this_segment,
                epsilon_start=max(epsilon_end, epsilon_start * (epsilon_decay**episode_start)),
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                opponent=opponent,
                max_steps_per_episode=max_steps_per_episode,
                verbose=verbose,
                save_frequency=save_frequency,
                save_folder=save_folder,
            )

            if eval_frequency and (episode_start + episodes_this_segment) % eval_frequency == 0:
                logger.info("evaluating_agent", episode=episode_start + episodes_this_segment)
                eval_metrics = self.evaluate(
                    num_games=eval_games,
                    opponent=opponent,
                    verbose=verbose,
                )

                self.mlflow_logger.log_metrics(
                    {
                        "eval_win_rate": eval_metrics["win_rate"],
                        "eval_loss_rate": eval_metrics["loss_rate"],
                        "eval_draw_rate": eval_metrics["draw_rate"],
                        "eval_avg_reward": eval_metrics["avg_reward"],
                    },
                    step=episode_start + episodes_this_segment,
                )

                logger.info(
                    "evaluation_results",
                    episode=episode_start + episodes_this_segment,
                    win_rate=eval_metrics["win_rate"],
                    avg_reward=eval_metrics["avg_reward"],
                )

        if verbose:
            logger.info("training_complete")
            self._log_final_statistics()

        self.mlflow_logger.finish()

        return self._get_training_metrics()

    def _train_episodes(
        self,
        num_episodes: int,
        epsilon_start: float,
        epsilon_end: float,
        epsilon_decay: float,
        opponent: str,
        max_steps_per_episode: int,
        verbose: bool,
        save_frequency: int | None,
        save_folder: Path | None,
    ) -> None:
        epsilon = epsilon_start

        for episode in range(num_episodes):
            episode_reward, episode_steps, winner = self._play_episode(
                epsilon=epsilon, opponent=opponent, max_steps=max_steps_per_episode
            )

            self.episode_rewards.append(episode_reward)

            if winner == "agent":
                self.win_counts["agent"] += 1
            elif winner == "opponent":
                self.win_counts["opponent"] += 1
            else:
                self.win_counts["draw"] += 1

            if self.replay_buffer.can_sample(self.batch_size):
                num_training_steps = max(1, episode_steps // 2)
                episode_loss = 0.0

                for _ in range(num_training_steps):
                    loss = self._training_step()
                    episode_loss += loss
                    self.total_steps += 1

                avg_loss = episode_loss / num_training_steps
                self.episode_losses.append(avg_loss)
            else:
                self.episode_losses.append(0.0)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if verbose and (episode + 1) % 10 == 0:
                total_episodes = len(self.episode_rewards)
                self._log_progress(total_episodes, num_episodes, epsilon)

            if (episode + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_losses = [loss for loss in self.episode_losses[-10:] if loss > 0]

                self.mlflow_logger.log_metrics(
                    {
                        "episode_reward": episode_reward,
                        "avg_reward_last_10": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0,
                        "avg_loss_last_10": sum(recent_losses) / len(recent_losses) if recent_losses else 0,
                        "epsilon": epsilon,
                        "buffer_size": len(self.replay_buffer),
                        "win_rate": self.win_counts["agent"] / total_episodes if total_episodes > 0 else 0,
                    },
                    step=total_episodes,
                )

            if save_frequency and (episode + 1) % save_frequency == 0 and save_folder:
                total_episodes = len(self.episode_rewards)
                self._save_checkpoint(save_folder, total_episodes)

    def _play_episode(self, epsilon: float, opponent: str, max_steps: int) -> tuple[float, int, str]:
        """
        Play a single episode and store transitions in replay buffer.

        Args:
            epsilon: Exploration rate for agent
            opponent: Type of opponent
            max_steps: Maximum steps per episode

        Returns:
            Tuple of (total_reward, num_steps, winner)
            winner is 'agent', 'opponent', or 'draw'
        """
        board = jdm_ru.PyBoard()

        agent_player = random.choice([1, -1])  # 1 = White, -1 = Black

        total_reward = 0.0
        step = 0

        while not board.is_terminal() and step < max_steps:
            current_player = board.current_player()
            prev_board = board

            if current_player == agent_player:
                state = board.to_embed()
                legal_moves = board.legal_moves()

                if not legal_moves:
                    break

                action_idx = self.agent.select_action(state, legal_moves, epsilon)
                move = legal_moves[action_idx]
                action_indices = move.to_indices()

                board = board.apply_move(move)
                next_state = board.to_embed()
                done = board.is_terminal()

                reward = self.reward_calculator.calculate_reward(board, prev_board, move, agent_player)
                total_reward += reward

                self.replay_buffer.add(state, action_indices, reward, next_state, done)

            else:
                legal_moves = board.legal_moves()
                if not legal_moves:
                    break

                if opponent == "random":
                    move = random.choice(legal_moves)
                elif opponent == "self":
                    state = board.to_embed()
                    action_idx = self.agent.select_action(state, legal_moves, epsilon=0.1)
                    move = legal_moves[action_idx]
                else:
                    raise ValueError(f"Unknown opponent type: {opponent}")

                board = board.apply_move(move)

            step += 1

        winner_player = board.winner()
        if winner_player is None:
            winner = "draw"
        elif winner_player == agent_player:
            winner = "agent"
        else:
            winner = "opponent"

        return total_reward, step, winner

    def _training_step(self) -> float:
        """
        Perform one training step using a batch from replay buffer.

        Returns:
            Loss value
        """
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        self.agent.train_mode()

        current_q_values = self.agent.model(states, actions).squeeze()

        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device)
            for i in range(self.batch_size):
                if dones[i] < 0.5:
                    sample_actions = torch.randint(0, 25, (10, 3), device=self.device)
                    sample_q = self.agent.model.predict_q_values(next_states[i], sample_actions)
                    next_q_values[i] = sample_q.max()

            target_q_values = rewards - self.gamma * next_q_values * (1 - dones)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.agent.get_model_parameters().parameters(), max_norm=1.0)

        self.optimizer.step()

        self.agent.eval_mode()

        return float(loss.item())

    def _log_progress(self, episode: int, total_episodes: int, epsilon: float) -> None:
        recent_rewards = self.episode_rewards[-10:]
        recent_losses = [l for l in self.episode_losses[-10:] if l > 0]

        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0

        win_rate = self.win_counts["agent"] / episode if episode > 0 else 0

        logger.info(
            "training_progress",
            episode=episode,
            total_episodes=total_episodes,
            epsilon=epsilon,
            avg_reward=avg_reward,
            avg_loss=avg_loss,
            win_rate=win_rate,
            buffer_size=len(self.replay_buffer),
        )

    def _log_final_statistics(self) -> None:
        total_games = sum(self.win_counts.values())

        final_avg_loss = None
        if self.episode_losses:
            recent_losses = [l for l in self.episode_losses[-100:] if l > 0]
            if recent_losses:
                final_avg_loss = sum(recent_losses) / len(recent_losses)

        logger.info(
            "final_training_statistics",
            total_games=total_games,
            agent_wins=self.win_counts["agent"],
            agent_win_rate=self.win_counts["agent"] / total_games if total_games > 0 else 0,
            opponent_wins=self.win_counts["opponent"],
            opponent_win_rate=self.win_counts["opponent"] / total_games if total_games > 0 else 0,
            draws=self.win_counts["draw"],
            draw_rate=self.win_counts["draw"] / total_games if total_games > 0 else 0,
            total_training_steps=self.total_steps,
            replay_buffer_size=len(self.replay_buffer),
            final_avg_loss=final_avg_loss,
        )

    def _get_training_metrics(self) -> dict:
        """Get training metrics as dictionary."""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_losses": self.episode_losses,
            "win_counts": self.win_counts.copy(),
            "total_steps": self.total_steps,
            "buffer_size": len(self.replay_buffer),
        }

    def evaluate(self, num_games: int = 100, opponent: str = "random", verbose: bool = True) -> dict:
        """
        Evaluate the agent against an opponent.

        Args:
            num_games: Number of games to play
            opponent: Type of opponent
            verbose: Whether to print results

        Returns:
            Dictionary with evaluation metrics
        """
        self.agent.eval_mode()

        agent_player_instance = DQNPlayer(agent=self.agent)

        if opponent == "random":
            opponent_player_instance: RandomPlayer | DQNPlayer = RandomPlayer()
        elif opponent == "self":
            opponent_agent = DQNAgent(config=self.agent.config)
            opponent_player_instance = DQNPlayer(agent=opponent_agent)
        else:
            raise ValueError(f"Unsupported opponent type: {opponent}")

        wins = 0
        losses = 0
        draws = 0
        total_rewards = []

        for _ in range(num_games):
            board = jdm_ru.PyBoard()
            agent_color = random.choice([1, -1])
            total_reward = 0.0
            step = 0
            max_steps = 200

            while not board.is_terminal() and step < max_steps:
                current_player = board.current_player()
                prev_board = board
                legal_moves = board.legal_moves()

                if not legal_moves:
                    break

                if current_player == agent_color:
                    move, _ = agent_player_instance.select_move(board)
                    board = board.apply_move(move)
                    reward = self.reward_calculator.calculate_reward(board, prev_board, move, agent_color)
                    total_reward += reward
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

            total_rewards.append(total_reward)

        metrics = {
            "num_games": num_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / num_games,
            "loss_rate": losses / num_games,
            "draw_rate": draws / num_games,
            "avg_reward": sum(total_rewards) / len(total_rewards),
        }

        if verbose:
            logger.info(
                "evaluation_results",
                num_games=num_games,
                opponent=opponent,
                wins=wins,
                win_rate=metrics["win_rate"],
                losses=losses,
                loss_rate=metrics["loss_rate"],
                draws=draws,
                draw_rate=metrics["draw_rate"],
                avg_reward=metrics["avg_reward"],
            )

        return metrics

    def _save_checkpoint(self, save_folder: Path, episode: int) -> None:
        model_path = save_folder / "checkpoints" / f"model_ep{episode:04d}.safetensors"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        save_model(self.agent.model, str(model_path))
        with open(model_path.parent.parent / "config.json", "w") as f:
            json.dump(self.agent.config.model_dump(), f, indent=4)
        logger.info("checkpoint_saved", path=str(model_path), episode=episode)
