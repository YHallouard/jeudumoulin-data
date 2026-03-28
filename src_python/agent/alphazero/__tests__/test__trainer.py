import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from agent.alphazero._agent import AlphaZeroAgent, AlphaZeroAgentConfig
from agent.alphazero._backbone import MLPBackboneConfig
from agent.alphazero._conditional_policy import SemiConditionalPolicyHeadConfig
from agent.alphazero._models import MLPDualNetConfig
from agent.alphazero._position import PositionalEmbeddingConfig
from agent.alphazero._trainer import AlphaZeroTrainer, StepLRSchedulerConfig
from safetensors.torch import save as safetensors_save
from utils.checkpoints import CheckpointHandler

TOY_CONFIG = AlphaZeroAgentConfig(
    model=MLPDualNetConfig(
        backbone=MLPBackboneConfig(
            input_dim=77,
            num_layers=1,
            hidden_dim=8,
            output_dim=8,
        ),
        policy_head=SemiConditionalPolicyHeadConfig(
            state_embedding_dim=8,
            embedding=PositionalEmbeddingConfig(embedding_dim=4),
            from_head_hidden_dim=8,
            to_head_hidden_dim=8,
            remove_head_hidden_dim=8,
        ),
        value_head=MLPDualNetConfig.ValueHeadConfig(
            hidden_dim=8,
            output_dim=1,
        ),
    ),
    device="cpu",
)

DEFAULT_LR_SCHEDULER_CONFIG = StepLRSchedulerConfig(step_size=10, gamma=0.1)


class InMemoryCheckpointHandler(CheckpointHandler):
    def __init__(self) -> None:
        self.checkpoints: dict[str, dict[str, bytes]] = {}

    def save_checkpoint(self, trainer: AlphaZeroTrainer, prefix: str, iteration: int) -> None:
        model_bytes = safetensors_save({k: v.cpu() for k, v in trainer.agent.model.state_dict().items()})
        buffer_bytes = trainer.replay_buffer.to_bytes()
        opt_buf = io.BytesIO()
        torch.save(trainer.optimizer.state_dict(), opt_buf)
        sched_buf = io.BytesIO()
        torch.save(trainer.scheduler.state_dict(), sched_buf)
        config_bytes = json.dumps(trainer.agent.config.model_dump()).encode()
        meta = {"iteration": iteration, "agent_config": trainer.agent.config.model_dump()}

        self.checkpoints[prefix] = {
            "model.safetensors": model_bytes,
            "buffer.pkl": buffer_bytes,
            "optimizer.pt": opt_buf.getvalue(),
            "scheduler.pt": sched_buf.getvalue(),
            "config.json": config_bytes,
            "meta.json": json.dumps(meta).encode(),
        }

    def restore_checkpoint(
        self, trainer: AlphaZeroTrainer, prefix: str, load_buffer: bool, load_optimizer: bool
    ) -> None:
        raise NotImplementedError

    def download_agent_files(self, prefix: str, tmp_path: Path) -> None:
        raise NotImplementedError

    def upload_eval_weights(self, agent: AlphaZeroAgent, iteration: int, execution_id: str) -> str:
        raise NotImplementedError

    def download_eval_weights(self, agent: AlphaZeroAgent, key: str) -> None:
        raise NotImplementedError

    def cleanup_eval_weights(self, key: str) -> None:
        raise NotImplementedError


class TestAlphaZeroTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AlphaZeroAgent(config=TOY_CONFIG)
        self.mlflow_patcher = patch("agent.alphazero._trainer.MLflowLogger")
        self.mlflow_patcher.start()
        self.trainer = AlphaZeroTrainer(
            agent=self.agent,
            lr_scheduler_config=DEFAULT_LR_SCHEDULER_CONFIG,
            learning_rate=0.001,
            batch_size=4,
            buffer_size=100,
        )

    def tearDown(self) -> None:
        self.mlflow_patcher.stop()

    def test_initialization(self) -> None:
        self.assertIs(self.trainer.agent, self.agent)
        self.assertEqual(self.trainer.learning_rate, 0.001)
        self.assertEqual(self.trainer.batch_size, 4)
        self.assertEqual(len(self.trainer.replay_buffer), 0)
        self.assertEqual(self.trainer.replay_buffer.max_size, 100)

    def test_get_buffer_statistics(self) -> None:
        stats = self.trainer.get_buffer_statistics()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["capacity"], 100)

    @patch("agent.alphazero._trainer.generate_train_examples")
    def test_train_single_iteration(self, mock_generate: Mock) -> None:
        handler = InMemoryCheckpointHandler()
        mock_generate.return_value = (
            [[0.1] * 77 for _ in range(10)],
            [[[i, i + 1, i + 2]] for i in range(10)],
            [[1.0] for _ in range(10)],
            [0.1 * i for i in range(10)],
        )

        metrics = self.trainer.train(
            num_iterations=1,
            episodes_per_iteration=5,
            simulations_per_move=10,
            max_episode_steps=50,
            epochs_per_iteration=2,
            temperature=1.0,
            checkpoint_handler=handler,
            eval_frequency=0,
            verbose=False,
        )

        self.assertIn("buffer_size", metrics)
        self.assertIn("buffer_statistics", metrics)
        self.assertEqual(len(handler.checkpoints), 1)
        self.assertIn("iter_0001", handler.checkpoints)

    @patch("agent.alphazero._trainer.generate_train_examples")
    def test_train_multiple_iterations(self, mock_generate: Mock) -> None:
        handler = InMemoryCheckpointHandler()
        mock_generate.return_value = (
            [[0.1] * 77 for _ in range(10)],
            [[[i, i + 1, i + 2]] for i in range(10)],
            [[1.0] for _ in range(10)],
            [0.1 * i for i in range(10)],
        )

        metrics = self.trainer.train(
            num_iterations=3,
            episodes_per_iteration=5,
            simulations_per_move=10,
            max_episode_steps=50,
            epochs_per_iteration=2,
            temperature=1.0,
            checkpoint_handler=handler,
            eval_frequency=0,
            verbose=False,
        )

        self.assertIn("buffer_size", metrics)
        self.assertEqual(mock_generate.call_count, 3)
        self.assertEqual(len(handler.checkpoints), 3)

    def test_train_on_batch(self) -> None:
        state_embeddings = [[0.1] * 77 for _ in range(4)]
        legal_moves = [[[i, i + 1, i + 2]] for i in range(4)]
        policy_targets = [[1.0] for _ in range(4)]
        value_targets = [0.1 * i for i in range(4)]

        policy_loss, value_loss, total_loss = self.trainer._train_on_batch(
            state_embeddings, legal_moves, policy_targets, value_targets
        )

        self.assertIsInstance(policy_loss, float)
        self.assertIsInstance(value_loss, float)
        self.assertIsInstance(total_loss, float)
        self.assertGreaterEqual(policy_loss, 0)
        self.assertGreaterEqual(value_loss, 0)
        self.assertGreaterEqual(total_loss, 0)

    def test_get_training_metrics(self) -> None:
        metrics = self.trainer._get_training_metrics()
        self.assertIn("buffer_size", metrics)
        self.assertIn("buffer_statistics", metrics)
        self.assertEqual(metrics["buffer_size"], 0)
