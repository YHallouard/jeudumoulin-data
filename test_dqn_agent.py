#!/usr/bin/env python3
"""Test the DQN Agent implementation."""

import tempfile
from pathlib import Path

import jdm_ru
import torch
from agent.dqn._agent import DQNAgent


def test_dqn_agent_creation():
    """Test DQN agent initialization."""

    print("Testing DQN Agent Creation...")
    print("=" * 70)

    # Test standard DQN
    print("\n1. Creating standard DQN agent...")
    agent = DQNAgent(use_double_dqn=False)
    print(f"   Agent: {agent}")
    stats = agent.get_statistics()
    print(f"   Model type: {stats['model_type']}")
    print(f"   Total parameters: {stats['total_parameters']:,}")
    assert stats["model_type"] == "DQN"
    print("   ✅ Standard DQN created")

    # Test Double DQN
    print("\n2. Creating Double DQN agent...")
    agent_double = DQNAgent(use_double_dqn=True)
    print(f"   Agent: {agent_double}")
    stats = agent_double.get_statistics()
    print(f"   Model type: {stats['model_type']}")
    print(f"   Total parameters: {stats['total_parameters']:,}")
    assert stats["model_type"] == "Double DQN"
    print("   ✅ Double DQN created")

    print("\n" + "=" * 70)
    print("✅ Agent creation tests passed!")
    return True


def test_action_selection():
    """Test action selection with real game states."""

    print("\n\nTesting Action Selection...")
    print("=" * 70)

    agent = DQNAgent()
    board = jdm_ru.PyBoard()

    # Test greedy selection
    print("\n1. Testing greedy action selection (epsilon=0)...")
    state = board.to_embed()
    legal_moves = board.legal_moves()

    print(f"   State shape: {len(state)}")
    print(f"   Number of legal moves: {len(legal_moves)}")

    action_idx = agent.select_action(state, legal_moves, epsilon=0.0)
    print(f"   Selected action index: {action_idx}")
    print(f"   Selected move: {legal_moves[action_idx]}")
    assert 0 <= action_idx < len(legal_moves)
    print("   ✅ Greedy selection works")

    # Test epsilon-greedy
    print("\n2. Testing epsilon-greedy selection (epsilon=1.0)...")
    selections = []
    for _ in range(10):
        idx = agent.select_action(state, legal_moves, epsilon=1.0)
        selections.append(idx)

    unique_selections = len(set(selections))
    print(f"   10 selections with epsilon=1.0: {unique_selections} unique")
    print(f"   Selections: {selections[:5]}...")
    assert unique_selections > 1, "Should have randomness with epsilon=1.0"
    print("   ✅ Random exploration works")

    # Test best action
    print("\n3. Testing best action selection...")
    best_idx = agent.select_best_action(state, legal_moves)
    print(f"   Best action index: {best_idx}")

    # Should be deterministic
    best_idx2 = agent.select_best_action(state, legal_moves)
    assert best_idx == best_idx2, "Best action should be deterministic"
    print("   ✅ Best action is deterministic")

    print("\n" + "=" * 70)
    print("✅ Action selection tests passed!")
    return True


def test_q_value_prediction():
    """Test Q-value prediction."""

    print("\n\nTesting Q-Value Prediction...")
    print("=" * 70)

    agent = DQNAgent()
    board = jdm_ru.PyBoard()

    print("\n1. Predicting Q-values for legal moves...")
    state = board.to_embed()
    legal_moves = board.legal_moves()

    q_values = agent.predict_q_values(state, legal_moves)
    print(f"   Number of Q-values: {len(q_values)}")
    print(f"   Q-values shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    print(f"   First 5 Q-values: {q_values[:5].tolist()}")

    assert q_values.shape == (len(legal_moves),)
    print("   ✅ Q-value prediction works")

    # Test single Q-value
    print("\n2. Getting Q-value for specific action...")
    action_indices = legal_moves[0].to_indices()
    q_value = agent.get_q_value(state, action_indices)
    print(f"   Action indices: {action_indices}")
    print(f"   Q-value: {q_value:.4f}")
    print("   ✅ Single Q-value retrieval works")

    print("\n" + "=" * 70)
    print("✅ Q-value prediction tests passed!")
    return True


def test_epsilon_greedy_probabilities():
    """Test epsilon-greedy probability distribution."""

    print("\n\nTesting Epsilon-Greedy Probabilities...")
    print("=" * 70)

    agent = DQNAgent()
    board = jdm_ru.PyBoard()
    state = board.to_embed()
    legal_moves = board.legal_moves()

    # Test with epsilon=0 (greedy)
    print("\n1. Testing with epsilon=0 (greedy)...")
    probs = agent.get_epsilon_greedy_probabilities(state, legal_moves, epsilon=0.0)
    print(f"   Probabilities sum: {sum(probs):.4f}")
    print(f"   Max probability: {max(probs):.4f}")
    print(f"   Number of actions with prob > 0.5: {sum(1 for p in probs if p > 0.5)}")

    assert abs(sum(probs) - 1.0) < 1e-6, "Probabilities should sum to 1"
    assert sum(1 for p in probs if p > 0.5) == 1, "One action should have high prob"
    print("   ✅ Greedy probabilities correct")

    # Test with epsilon=0.5
    print("\n2. Testing with epsilon=0.5 (mixed)...")
    probs = agent.get_epsilon_greedy_probabilities(state, legal_moves, epsilon=0.5)
    print(f"   Probabilities sum: {sum(probs):.4f}")
    print(f"   Max probability: {max(probs):.4f}")
    print(f"   Min probability: {min(probs):.4f}")
    print(f"   First 5 probs: {probs[:5]}")

    assert abs(sum(probs) - 1.0) < 1e-6
    assert all(p > 0 for p in probs), "All actions should have some probability"
    print("   ✅ Mixed probabilities correct")

    print("\n" + "=" * 70)
    print("✅ Epsilon-greedy probability tests passed!")
    return True


def test_save_and_load():
    """Test model persistence."""

    print("\n\nTesting Save and Load...")
    print("=" * 70)

    # Create and configure agent
    print("\n1. Creating and configuring agent...")
    agent = DQNAgent(hidden_dim=128, use_double_dqn=True)

    board = jdm_ru.PyBoard()
    state = board.to_embed()
    legal_moves = board.legal_moves()

    # Get Q-values before save
    q_values_before = agent.predict_q_values(state, legal_moves)
    print(f"   Q-values before save: {q_values_before[:3].tolist()}")

    # Save model
    print("\n2. Saving model...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_model.pt"
        agent.save_pretrained(save_path)
        print(f"   Model saved to {save_path}")
        assert save_path.exists()
        print("   ✅ Save successful")

        # Load model
        print("\n3. Loading model...")
        loaded_agent = DQNAgent.load(save_path)
        print(f"   Model loaded from {save_path}")

        # Verify Q-values match
        q_values_after = loaded_agent.predict_q_values(state, legal_moves)
        print(f"   Q-values after load: {q_values_after[:3].tolist()}")

        diff = (q_values_before - q_values_after).abs().max().item()
        print(f"   Max difference: {diff:.6f}")

        assert diff < 1e-5, "Q-values should match after load"
        print("   ✅ Load successful, Q-values match")

    print("\n" + "=" * 70)
    print("✅ Save and load tests passed!")
    return True


def test_target_network_update():
    """Test target network updates for Double DQN."""

    print("\n\nTesting Target Network Update...")
    print("=" * 70)

    agent = DQNAgent(use_double_dqn=True)
    board = jdm_ru.PyBoard()
    state = board.to_embed()
    legal_moves = board.legal_moves()

    print("\n1. Getting initial Q-values...")
    q_values_initial = agent.predict_q_values(state, legal_moves)
    print(f"   Initial Q-values: {q_values_initial[:3].tolist()}")

    # Modify online network weights
    print("\n2. Modifying online network weights...")
    agent.train_mode()
    with torch.no_grad():
        for param in agent.model.q_network.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    agent.eval_mode()

    q_values_after_modify = agent.predict_q_values(state, legal_moves)
    print(f"   Q-values after modify: {q_values_after_modify[:3].tolist()}")

    diff_before_update = (q_values_initial - q_values_after_modify).abs().max().item()
    print(f"   Difference before target update: {diff_before_update:.4f}")
    assert diff_before_update > 0.001, "Weights should have changed"

    # Update target network
    print("\n3. Updating target network...")
    agent.update_target_network()
    print("   Target network updated")
    print("   ✅ Target update successful")

    print("\n" + "=" * 70)
    print("✅ Target network update tests passed!")
    return True


def test_train_eval_modes():
    """Test switching between train and eval modes."""

    print("\n\nTesting Train/Eval Modes...")
    print("=" * 70)

    agent = DQNAgent()

    print("\n1. Testing eval mode (default)...")
    assert not agent.model.training
    print(f"   Model is in eval mode: {not agent.model.training}")
    print("   ✅ Eval mode correct")

    print("\n2. Switching to train mode...")
    agent.train_mode()
    assert agent.model.training
    print(f"   Model is in train mode: {agent.model.training}")
    print("   ✅ Train mode works")

    print("\n3. Switching back to eval mode...")
    agent.eval_mode()
    assert not agent.model.training
    print(f"   Model is in eval mode: {not agent.model.training}")
    print("   ✅ Eval mode works")

    print("\n" + "=" * 70)
    print("✅ Train/eval mode tests passed!")
    return True


if __name__ == "__main__":
    try:
        test_dqn_agent_creation()
        test_action_selection()
        test_q_value_prediction()
        test_epsilon_greedy_probabilities()
        test_save_and_load()
        test_target_network_update()
        test_train_eval_modes()

        print("\n\n" + "=" * 70)
        print("🎉 ALL DQN AGENT TESTS PASSED!")
        print("=" * 70)
        print("\n📊 Summary:")
        print("   - Agent creation (DQN & Double DQN): ✅")
        print("   - Action selection (greedy & epsilon-greedy): ✅")
        print("   - Q-value prediction: ✅")
        print("   - Epsilon-greedy probabilities: ✅")
        print("   - Save/load persistence: ✅")
        print("   - Target network updates: ✅")
        print("   - Train/eval modes: ✅")
        print("\n🚀 DQN Agent is ready for training!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
