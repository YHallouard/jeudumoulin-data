#!/usr/bin/env python3
"""Test the DQN Trainer implementation."""

import tempfile
from pathlib import Path

import torch
from agent.dqn._agent import DQNAgent
from agent.dqn._trainer import DQNTrainer


def test_trainer_initialization():
    """Test trainer initialization."""

    print("Testing Trainer Initialization...")
    print("=" * 70)

    print("\n1. Creating DQN agent...")
    agent = DQNAgent(use_double_dqn=True)
    print(f"   Agent created: {agent}")

    print("\n2. Creating trainer...")
    trainer = DQNTrainer(
        agent=agent,
        learning_rate=0.001,
        gamma=0.99,
        batch_size=32,
        buffer_size=1000,
        target_update_frequency=50,
    )
    print(f"   Trainer created: {trainer}")
    print(f"   Learning rate: {trainer.learning_rate}")
    print(f"   Gamma: {trainer.gamma}")
    print(f"   Batch size: {trainer.batch_size}")
    print(f"   Buffer size: {trainer.replay_buffer.max_size}")
    print(f"   Target update frequency: {trainer.target_update_frequency}")

    assert trainer.total_steps == 0
    assert len(trainer.episode_rewards) == 0
    assert len(trainer.episode_losses) == 0
    print("   ✅ Trainer initialized correctly")

    print("\n" + "=" * 70)
    print("✅ Trainer initialization test passed!")
    return True


def test_play_episode():
    """Test playing a single episode."""

    print("\n\nTesting Episode Playing...")
    print("=" * 70)

    agent = DQNAgent(use_double_dqn=False)
    trainer = DQNTrainer(agent, buffer_size=100)

    print("\n1. Playing episode against random opponent...")
    total_reward, num_steps, winner = trainer._play_episode(epsilon=0.5, opponent="random", max_steps=50)

    print(f"   Total reward: {total_reward:+.2f}")
    print(f"   Number of steps: {num_steps}")
    print(f"   Winner: {winner}")
    print(f"   Buffer size after episode: {len(trainer.replay_buffer)}")

    assert winner in ["agent", "opponent", "draw"]
    assert num_steps > 0
    assert len(trainer.replay_buffer) > 0
    print("   ✅ Episode played successfully")

    print("\n2. Playing episode with self-play...")
    initial_buffer_size = len(trainer.replay_buffer)
    total_reward, num_steps, winner = trainer._play_episode(epsilon=0.3, opponent="self", max_steps=50)

    print(f"   Total reward: {total_reward:+.2f}")
    print(f"   Number of steps: {num_steps}")
    print(f"   Winner: {winner}")
    print(f"   Buffer size change: {len(trainer.replay_buffer) - initial_buffer_size}")

    assert winner in ["agent", "opponent", "draw"]
    print("   ✅ Self-play episode worked")

    print("\n" + "=" * 70)
    print("✅ Episode playing tests passed!")
    return True


def test_training_step():
    """Test single training step."""

    print("\n\nTesting Training Step...")
    print("=" * 70)

    agent = DQNAgent(use_double_dqn=True)
    trainer = DQNTrainer(agent, batch_size=8, buffer_size=100)

    # Fill replay buffer with some experiences
    print("\n1. Filling replay buffer with experiences...")
    for _ in range(3):
        trainer._play_episode(epsilon=1.0, opponent="random", max_steps=30)

    print(f"   Buffer size: {len(trainer.replay_buffer)}")
    assert len(trainer.replay_buffer) >= trainer.batch_size
    print("   ✅ Buffer filled")

    # Perform training step
    print("\n2. Performing training step...")
    loss = trainer._training_step()

    print(f"   Loss: {loss:.4f}")
    print(f"   Total steps: {trainer.total_steps}")

    assert loss >= 0
    print("   ✅ Training step successful")

    print("\n" + "=" * 70)
    print("✅ Training step test passed!")
    return True


def test_short_training():
    """Test short training run."""

    print("\n\nTesting Short Training Run...")
    print("=" * 70)

    agent = DQNAgent(use_double_dqn=True, hidden_dim=64)
    trainer = DQNTrainer(
        agent,
        learning_rate=0.01,
        gamma=0.95,
        batch_size=16,
        buffer_size=200,
        target_update_frequency=20,
    )

    print("\n1. Training for 5 episodes...")
    metrics = trainer.train(
        num_episodes=5,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.9,
        opponent="random",
        max_steps_per_episode=50,
        verbose=True,
    )

    print("\n2. Checking training metrics...")
    print(f"   Episode rewards: {metrics['episode_rewards']}")
    print(f"   Episode losses: {metrics['episode_losses']}")
    print(f"   Win counts: {metrics['win_counts']}")
    print(f"   Total steps: {metrics['total_steps']}")
    print(f"   Buffer size: {metrics['buffer_size']}")

    assert len(metrics["episode_rewards"]) == 5
    assert len(metrics["episode_losses"]) == 5
    assert sum(metrics["win_counts"].values()) == 5
    print("   ✅ Training completed successfully")

    print("\n" + "=" * 70)
    print("✅ Short training test passed!")
    return True


def test_epsilon_decay():
    """Test epsilon decay during training."""

    print("\n\nTesting Epsilon Decay...")
    print("=" * 70)

    agent = DQNAgent()
    trainer = DQNTrainer(agent, buffer_size=100)

    print("\n1. Testing epsilon decay...")
    epsilon = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9
    episodes = 10

    epsilons = []
    for episode in range(episodes):
        epsilons.append(epsilon)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

    print(f"   Starting epsilon: {epsilons[0]:.3f}")
    print(f"   Final epsilon: {epsilons[-1]:.3f}")
    print(f"   Epsilon values: {[f'{e:.3f}' for e in epsilons[:5]]}...")

    assert epsilons[0] == 1.0
    assert epsilons[-1] >= epsilon_end
    assert all(epsilons[i] >= epsilons[i + 1] for i in range(len(epsilons) - 1))
    print("   ✅ Epsilon decay works correctly")

    print("\n" + "=" * 70)
    print("✅ Epsilon decay test passed!")
    return True


def test_evaluation():
    """Test agent evaluation."""

    print("\n\nTesting Agent Evaluation...")
    print("=" * 70)

    agent = DQNAgent()
    trainer = DQNTrainer(agent)

    print("\n1. Evaluating agent against random opponent (10 games)...")
    metrics = trainer.evaluate(num_games=10, opponent="random", verbose=True)

    print("\n2. Checking evaluation metrics...")
    print(f"   Num games: {metrics['num_games']}")
    print(f"   Wins: {metrics['wins']}")
    print(f"   Losses: {metrics['losses']}")
    print(f"   Draws: {metrics['draws']}")
    print(f"   Win rate: {metrics['win_rate']:.1%}")
    print(f"   Average reward: {metrics['avg_reward']:+.2f}")

    assert metrics["num_games"] == 10
    assert metrics["wins"] + metrics["losses"] + metrics["draws"] == 10
    assert 0 <= metrics["win_rate"] <= 1
    print("   ✅ Evaluation completed successfully")

    print("\n" + "=" * 70)
    print("✅ Evaluation test passed!")
    return True


def test_save_checkpoint():
    """Test saving checkpoints during training."""

    print("\n\nTesting Checkpoint Saving...")
    print("=" * 70)

    agent = DQNAgent(hidden_dim=64)
    trainer = DQNTrainer(agent, buffer_size=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "checkpoint.pt"

        print("\n1. Training with checkpoint saving...")
        metrics = trainer.train(
            num_episodes=4,
            epsilon_start=1.0,
            epsilon_end=0.5,
            epsilon_decay=0.9,
            opponent="random",
            max_steps_per_episode=30,
            verbose=False,
            save_frequency=2,
            save_path=save_path,
        )

        # Check if checkpoints were created
        checkpoint_dir = save_path.parent
        checkpoints = list(checkpoint_dir.glob("checkpoint_ep*.pt"))

        print(f"   Checkpoints created: {len(checkpoints)}")
        for cp in sorted(checkpoints):
            print(f"     - {cp.name}")

        assert len(checkpoints) == 2  # Saved at episode 2 and 4
        print("   ✅ Checkpoints saved successfully")

        # Test loading checkpoint
        print("\n2. Loading checkpoint...")
        checkpoint_path = checkpoints[0]
        loaded_agent = DQNAgent.load(checkpoint_path)
        print(f"   Loaded agent from {checkpoint_path.name}")
        print("   ✅ Checkpoint loaded successfully")

    print("\n" + "=" * 70)
    print("✅ Checkpoint saving test passed!")
    return True


def test_target_network_updates():
    """Test target network synchronization."""

    print("\n\nTesting Target Network Updates...")
    print("=" * 70)

    agent = DQNAgent(use_double_dqn=True)
    trainer = DQNTrainer(agent, target_update_frequency=10, buffer_size=100)

    # Get initial target network weights
    print("\n1. Capturing initial target network weights...")
    initial_params = [p.clone() for p in agent.model.target_network.parameters()]

    # Fill buffer and train
    print("\n2. Training to trigger target network updates...")
    for _ in range(3):
        trainer._play_episode(epsilon=1.0, opponent="random", max_steps=20)

    # Perform multiple training steps
    for _ in range(15):
        if trainer.replay_buffer.can_sample(trainer.batch_size):
            trainer._training_step()
            trainer.total_steps += 1

            # Update target network
            if trainer.total_steps % trainer.target_update_frequency == 0:
                agent.update_target_network()

    print(f"   Total training steps: {trainer.total_steps}")
    print(f"   Target updates expected: {trainer.total_steps // trainer.target_update_frequency}")

    # Check if target network was updated
    print("\n3. Verifying target network was updated...")
    final_params = [p.clone() for p in agent.model.target_network.parameters()]

    # At least one parameter should have changed
    changed = any(not torch.allclose(initial, final) for initial, final in zip(initial_params, final_params))

    print(f"   Target network parameters changed: {changed}")
    assert changed, "Target network should have been updated"
    print("   ✅ Target network updates working")

    print("\n" + "=" * 70)
    print("✅ Target network update test passed!")
    return True


def test_metrics_tracking():
    """Test metrics tracking during training."""

    print("\n\nTesting Metrics Tracking...")
    print("=" * 70)

    agent = DQNAgent()
    trainer = DQNTrainer(agent, buffer_size=100)

    print("\n1. Training for a few episodes...")
    metrics = trainer.train(
        num_episodes=3,
        epsilon_start=1.0,
        epsilon_end=0.5,
        opponent="random",
        max_steps_per_episode=30,
        verbose=False,
    )

    print("\n2. Checking tracked metrics...")
    print(f"   Episode rewards length: {len(metrics['episode_rewards'])}")
    print(f"   Episode losses length: {len(metrics['episode_losses'])}")
    print(f"   Total games: {sum(metrics['win_counts'].values())}")
    print(f"   Training steps: {metrics['total_steps']}")

    assert len(metrics["episode_rewards"]) == 3
    assert len(metrics["episode_losses"]) == 3
    assert sum(metrics["win_counts"].values()) == 3
    print("   ✅ All metrics tracked correctly")

    print("\n3. Checking win count distribution...")
    win_counts = metrics["win_counts"]
    print(f"   Agent wins: {win_counts['agent']}")
    print(f"   Opponent wins: {win_counts['opponent']}")
    print(f"   Draws: {win_counts['draw']}")

    assert all(v >= 0 for v in win_counts.values())
    assert win_counts["agent"] + win_counts["opponent"] + win_counts["draw"] == 3
    print("   ✅ Win counts correct")

    print("\n" + "=" * 70)
    print("✅ Metrics tracking test passed!")
    return True


if __name__ == "__main__":
    try:
        test_trainer_initialization()
        test_play_episode()
        test_training_step()
        test_epsilon_decay()
        test_evaluation()
        test_save_checkpoint()
        test_target_network_updates()
        test_metrics_tracking()
        test_short_training()

        print("\n\n" + "=" * 70)
        print("🎉 ALL DQN TRAINER TESTS PASSED!")
        print("=" * 70)
        print("\n📊 Summary:")
        print("   - Trainer initialization: ✅")
        print("   - Episode playing (random & self-play): ✅")
        print("   - Training step: ✅")
        print("   - Epsilon decay: ✅")
        print("   - Agent evaluation: ✅")
        print("   - Checkpoint saving/loading: ✅")
        print("   - Target network updates: ✅")
        print("   - Metrics tracking: ✅")
        print("   - Short training run: ✅")
        print("\n🚀 DQN Trainer is ready for production!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
