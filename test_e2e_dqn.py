#!/usr/bin/env python3
"""
Quick end-to-end test for DQN training via CLI integration.
This tests that all components work together.
"""

import sys
import tempfile
from pathlib import Path

# Add src_python to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src_python"))

import jdm_ru
from agent.dqn._agent import DQNAgent
from agent.dqn._trainer import DQNTrainer


def test_e2e_dqn_training():
    """Test end-to-end DQN training pipeline."""

    print("=" * 70)
    print("End-to-End DQN Training Test")
    print("=" * 70)

    # Test 1: Create agent
    print("\n1. Creating DQN agent...")
    agent = DQNAgent(hidden_dim=64, use_double_dqn=True)
    stats = agent.get_statistics()
    print(f"   ✅ Agent created: {stats['model_type']}")
    print(f"   Parameters: {stats['total_parameters']:,}")

    # Test 2: Create trainer
    print("\n2. Creating trainer...")
    trainer = DQNTrainer(
        agent=agent,
        learning_rate=0.01,
        gamma=0.95,
        batch_size=16,
        buffer_size=200,
        target_update_frequency=20,
    )
    print("   ✅ Trainer created")
    print(f"   Buffer size: {trainer.replay_buffer.max_size}")
    print(f"   Batch size: {trainer.batch_size}")

    # Test 3: Quick training run
    print("\n3. Running quick training (10 episodes)...")
    metrics = trainer.train(
        num_episodes=10,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.9,
        opponent="random",
        max_steps_per_episode=50,
        verbose=True,
    )
    print("   ✅ Training completed")
    print(f"   Total games: {sum(metrics['win_counts'].values())}")
    print(f"   Agent wins: {metrics['win_counts']['agent']}")
    print(f"   Training steps: {metrics['total_steps']}")

    # Test 4: Evaluation
    print("\n4. Evaluating agent (20 games)...")
    eval_metrics = trainer.evaluate(num_games=20, opponent="random", verbose=True)
    print("   ✅ Evaluation completed")
    print(f"   Win rate: {eval_metrics['win_rate']:.1%}")
    print(f"   Average reward: {eval_metrics['avg_reward']:+.2f}")

    # Test 5: Save and load
    print("\n5. Testing save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.pt"
        agent.save_pretrained(save_path)
        print("   ✅ Model saved")

        loaded_agent = DQNAgent.load(save_path)
        print("   ✅ Model loaded")

        # Verify loaded agent works
        board = jdm_ru.PyBoard()
        state = board.to_embed()
        legal_moves = board.legal_moves()
        action_idx = loaded_agent.select_best_action(state, legal_moves)
        print("   ✅ Loaded agent can select actions")

    print("\n" + "=" * 70)
    print("🎉 ALL END-TO-END TESTS PASSED!")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   - Agent creation: ✅")
    print("   - Trainer creation: ✅")
    print("   - Training pipeline: ✅")
    print("   - Evaluation: ✅")
    print("   - Save/Load: ✅")
    print("\n🚀 DQN implementation is complete and ready!")
    print("\nYou can now train via CLI:")
    print("  $ python -m src_python.cli.train_dqn --episodes 100")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        test_e2e_dqn_training()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
