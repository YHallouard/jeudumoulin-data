import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agent.alphazero._agent import AlphaZeroAgent, AlphaZeroAgentConfig
from agent.alphazero._backbone import MLPBackboneConfig
from agent.alphazero._conditional_policy import SemiConditionalPolicyHeadConfig
from agent.alphazero._models import MLPDualNetConfig
from agent.alphazero._position import PositionalEmbeddingConfig
from agent.alphazero._trainer import AlphaZeroTrainer, StepLRSchedulerConfig

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
        with tempfile.TemporaryDirectory() as tmp_dir:
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
                save_folder=Path(tmp_dir),
                eval_frequency=0,
                verbose=False,
            )

            self.assertIn("iteration_metrics", metrics)
            self.assertEqual(len(metrics["iteration_metrics"]), 1)

    @patch("agent.alphazero._trainer.generate_train_examples")
    def test_train_multiple_iterations(self, mock_generate: Mock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
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
                save_folder=Path(tmp_dir),
                eval_frequency=0,
                verbose=False,
            )

            self.assertEqual(len(metrics["iteration_metrics"]), 3)
            self.assertEqual(mock_generate.call_count, 3)

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

    def test_save_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.trainer.save_checkpoint(Path(tmp_dir), 0)

            checkpoint_path = Path(tmp_dir) / "checkpoints" / "model_iter0000.safetensors"
            config_path = Path(tmp_dir) / "config.json"

            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(config_path.exists())

    def test_get_training_metrics(self) -> None:
        self.trainer.iteration_metrics = [
            {"iteration": 1, "avg_policy_loss": 0.5},
            {"iteration": 2, "avg_policy_loss": 0.4},
        ]

        metrics = self.trainer._get_training_metrics()
        self.assertIn("iteration_metrics", metrics)
        self.assertEqual(len(metrics["iteration_metrics"]), 2)
        self.assertIn("buffer_size", metrics)
        self.assertIn("buffer_statistics", metrics)
