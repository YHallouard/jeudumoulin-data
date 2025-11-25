from unittest.mock import MagicMock, patch

import pytest
from agent.alphazero._agent import AlphaZeroAgent
from agent.alphazero._trainer import AlphaZeroTrainer


@pytest.fixture
def mock_agent():
    import torch

    agent = MagicMock(spec=AlphaZeroAgent)
    agent.model = MagicMock()
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    agent.model.parameters.return_value = [mock_param]
    return agent


@pytest.fixture
def trainer(mock_agent, tmp_path):
    return AlphaZeroTrainer(
        agent=mock_agent,
        learning_rate=0.001,
        batch_size=4,
        buffer_size=100,
    )


def test_trainer_initialization(mock_agent):
    trainer = AlphaZeroTrainer(
        agent=mock_agent,
        learning_rate=0.001,
        batch_size=64,
        buffer_size=1000,
    )
    assert trainer.agent == mock_agent
    assert trainer.learning_rate == 0.001
    assert trainer.batch_size == 64
    assert len(trainer.replay_buffer) == 0
    assert trainer.replay_buffer.max_size == 1000


def test_get_buffer_statistics(trainer):
    stats = trainer.get_buffer_statistics()
    assert stats["size"] == 0
    assert stats["capacity"] == 100


@patch("agent.alphazero._trainer.generate_train_examples")
def test_train_single_iteration(mock_generate, trainer, tmp_path):
    import torch

    save_folder = tmp_path / "models"

    mock_generate.return_value = (
        [[0.1] * 77 for _ in range(10)],
        [[[[i, i + 1, i + 2]]] for i in range(10)],
        [[0.5, 0.5] for _ in range(10)],
        [0.1 * i for i in range(10)],
    )

    trainer.agent.model.policy_value.return_value = (
        torch.tensor([0.5, 0.5], requires_grad=True),
        torch.tensor([0.5], requires_grad=True),
    )

    metrics = trainer.train(
        num_iterations=1,
        episodes_per_iteration=5,
        simulations_per_move=10,
        max_episode_steps=50,
        epochs_per_iteration=2,
        temperature=1.0,
        save_folder=save_folder,
        verbose=False,
    )

    assert "iteration_metrics" in metrics
    assert len(metrics["iteration_metrics"]) == 1
    assert save_folder.exists()


@patch("agent.alphazero._trainer.generate_train_examples")
def test_train_multiple_iterations(mock_generate, trainer, tmp_path):
    import torch

    save_folder = tmp_path / "models"

    mock_generate.return_value = (
        [[0.1] * 77 for _ in range(10)],
        [[[[i, i + 1, i + 2]]] for i in range(10)],
        [[0.5, 0.5] for _ in range(10)],
        [0.1 * i for i in range(10)],
    )

    trainer.agent.model.policy_value.return_value = (
        torch.tensor([0.5, 0.5], requires_grad=True),
        torch.tensor([0.5], requires_grad=True),
    )

    metrics = trainer.train(
        num_iterations=3,
        episodes_per_iteration=5,
        simulations_per_move=10,
        max_episode_steps=50,
        epochs_per_iteration=2,
        temperature=1.0,
        save_folder=save_folder,
        verbose=False,
    )

    assert len(metrics["iteration_metrics"]) == 3
    assert mock_generate.call_count == 3


def test_train_on_batch(trainer):
    state_embeddings = [[0.1] * 77 for _ in range(4)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(4)]
    policy_targets = [[0.5, 0.5] for _ in range(4)]
    value_targets = [0.1 * i for i in range(4)]

    import torch

    mock_policy = torch.tensor([0.5, 0.5], requires_grad=True)
    mock_value = torch.tensor([0.5], requires_grad=True)

    trainer.agent.model.policy_value.return_value = (mock_policy, mock_value)

    policy_loss, value_loss, total_loss = trainer._train_on_batch(
        state_embeddings, legal_moves, policy_targets, value_targets
    )

    assert isinstance(policy_loss, float)
    assert isinstance(value_loss, float)
    assert isinstance(total_loss, float)
    assert policy_loss >= 0
    assert value_loss >= 0
    assert total_loss >= 0


def test_save_checkpoint(trainer, tmp_path):
    save_folder = tmp_path / "models"
    save_folder.mkdir()

    trainer._save_checkpoint(save_folder, 0)

    checkpoint_path = save_folder / "model_0000.safetensors"
    assert checkpoint_path.exists()


def test_get_training_metrics(trainer):
    trainer.iteration_metrics = [
        {"iteration": 1, "avg_policy_loss": 0.5},
        {"iteration": 2, "avg_policy_loss": 0.4},
    ]

    metrics = trainer._get_training_metrics()
    assert "iteration_metrics" in metrics
    assert len(metrics["iteration_metrics"]) == 2
    assert "buffer_size" in metrics
    assert "buffer_statistics" in metrics
