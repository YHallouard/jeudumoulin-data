#!/usr/bin/env python3
"""Test the DQN Network implementation."""

import torch
from agent.dqn._models import DQNNetwork, create_dqn_model


def test_dqn_network():
    """Test the DQN Network architecture."""

    print("Testing DQN Network Architecture...")
    print("=" * 70)

    # Create model
    print("\n1. Creating DQN Network...")
    model = DQNNetwork(
        state_dim=77,
        action_emb_dim=32,
        hidden_dim=256,
        state_emb_dim=128,
    )
    print(f"   ✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass with single sample
    print("\n2. Testing forward pass (single sample)...")
    state = torch.randn(1, 77)  # Single state
    action_indices = torch.tensor([[5, 10, 24]])  # Single action: from=5, to=10, remove=None(24)

    q_value = model(state, action_indices)
    print(f"   Input state shape: {state.shape}")
    print(f"   Input action indices: {action_indices}")
    print(f"   Output Q-value: {q_value.item():.4f}")
    print(f"   Output shape: {q_value.shape}")
    assert q_value.shape == (1, 1), f"Expected shape (1, 1), got {q_value.shape}"
    print("   ✅ Forward pass works")

    # Test forward pass with batch
    print("\n3. Testing forward pass (batch)...")
    batch_size = 32
    states = torch.randn(batch_size, 77)
    action_indices_batch = torch.randint(0, 25, (batch_size, 3))

    q_values = model(states, action_indices_batch)
    print(f"   Batch size: {batch_size}")
    print(f"   Input states shape: {states.shape}")
    print(f"   Input actions shape: {action_indices_batch.shape}")
    print(f"   Output Q-values shape: {q_values.shape}")
    print(f"   Q-values range: [{q_values.min().item():.4f}, {q_values.max().item():.4f}]")
    assert q_values.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {q_values.shape}"
    print("   ✅ Batch forward pass works")

    # Test predict_q_values for action selection
    print("\n4. Testing action Q-value prediction...")
    state = torch.randn(77)  # Single state without batch dim
    num_actions = 10
    actions = torch.randint(0, 25, (num_actions, 3))

    q_values = model.predict_q_values(state, actions)
    print(f"   Single state shape: {state.shape}")
    print(f"   Number of actions: {num_actions}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Output Q-values shape: {q_values.shape}")
    print(f"   Q-values: {q_values[:5].tolist()}")  # Show first 5
    best_action_idx = torch.argmax(q_values).item()
    print(f"   Best action index: {best_action_idx}")
    print(f"   Best action: {actions[best_action_idx].tolist()}")
    assert q_values.shape == (num_actions,), f"Expected shape ({num_actions},), got {q_values.shape}"
    print("   ✅ Action Q-value prediction works")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    state = torch.randn(1, 77, requires_grad=True)
    action_indices = torch.tensor([[5, 10, 24]])

    q_value = model(state, action_indices)
    loss = q_value.sum()
    loss.backward()

    # Check gradients exist
    has_grads = state.grad is not None
    print(f"   State has gradients: {has_grads}")
    print(f"   State gradient norm: {state.grad.norm().item():.4f}")

    # Check model parameters have gradients
    param_grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    all_have_grads = all(param_grads)
    print(f"   All parameters have gradients: {all_have_grads}")
    assert all_have_grads, "Not all parameters have gradients"
    print("   ✅ Gradient flow works")

    print("\n" + "=" * 70)
    print("✅ All DQN Network tests passed!")

    return True


def test_double_dqn():
    """Test the Double DQN implementation."""

    print("\n\nTesting Double DQN Architecture...")
    print("=" * 70)

    # Create model
    print("\n1. Creating Double DQN Agent...")
    agent = DoubleDQNAgent(
        state_dim=77,
        action_emb_dim=32,
        hidden_dim=256,
        state_emb_dim=128,
    )
    online_params = sum(p.numel() for p in agent.q_network.parameters())
    target_params = sum(p.numel() for p in agent.target_network.parameters())
    total_params = online_params + target_params
    print("   ✅ Agent created")
    print(f"   Online network: {online_params} parameters")
    print(f"   Target network: {target_params} parameters")
    print(f"   Total: {total_params} parameters")

    # Test forward pass uses online network
    print("\n2. Testing forward pass (uses online network)...")
    state = torch.randn(1, 77)
    action_indices = torch.tensor([[5, 10, 24]])

    q_value = agent(state, action_indices)
    print(f"   Q-value from online network: {q_value.item():.4f}")
    assert q_value.shape == (1, 1)
    print("   ✅ Forward pass works")

    # Test target network
    print("\n3. Testing target network...")
    with torch.no_grad():
        q_value_target = agent.target_network(state, action_indices)
    print(f"   Q-value from target network: {q_value_target.item():.4f}")
    print(f"   Difference: {abs(q_value.item() - q_value_target.item()):.6f}")
    # Should be very small since target was just initialized from online
    print("   ✅ Target network works")

    # Test target network update
    print("\n4. Testing target network update...")
    # Modify online network
    with torch.no_grad():
        for param in agent.q_network.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    # Get Q-values before update
    with torch.no_grad():
        q_before = agent.target_network(state, action_indices).item()

    # Update target network
    agent.update_target_network()

    # Get Q-values after update
    with torch.no_grad():
        q_after = agent.target_network(state, action_indices).item()
        q_online = agent.q_network(state, action_indices).item()

    print(f"   Target Q before update: {q_before:.4f}")
    print(f"   Target Q after update:  {q_after:.4f}")
    print(f"   Online Q:               {q_online:.4f}")
    print(f"   Difference (target vs online): {abs(q_after - q_online):.6f}")
    assert abs(q_after - q_online) < 1e-5, "Target should match online after update"
    print("   ✅ Target network update works")

    # Test get_target_q_values (Double DQN logic)
    print("\n5. Testing Double DQN target Q-value calculation...")
    batch_size = 4
    next_states = torch.randn(batch_size, 77)
    next_legal_actions = [
        torch.randint(0, 25, (5, 3)),  # 5 legal actions
        torch.randint(0, 25, (3, 3)),  # 3 legal actions
        torch.randint(0, 25, (7, 3)),  # 7 legal actions
        torch.empty(0, 3, dtype=torch.long),  # 0 legal actions (terminal)
    ]

    target_q_values = agent.get_target_q_values(next_states, next_legal_actions)
    print(f"   Batch size: {batch_size}")
    print(f"   Target Q-values: {target_q_values.tolist()}")
    print(f"   Shape: {target_q_values.shape}")
    assert target_q_values.shape == (batch_size,)
    assert target_q_values[3] == 0.0, "Terminal state should have Q-value 0"
    print("   ✅ Double DQN target calculation works")

    # Test target network requires_grad=False
    print("\n6. Testing target network is frozen...")
    target_requires_grad = [p.requires_grad for p in agent.target_network.parameters()]
    all_frozen = not any(target_requires_grad)
    print(f"   All target parameters frozen: {all_frozen}")
    assert all_frozen, "Target network parameters should not require gradients"
    print("   ✅ Target network is properly frozen")

    print("\n" + "=" * 70)
    print("✅ All Double DQN tests passed!")

    return True


def test_create_model_factory():
    """Test the model factory function."""

    print("\n\nTesting Model Factory...")
    print("=" * 70)

    # Test DQN creation
    print("\n1. Creating DQN via factory...")
    dqn = create_dqn_model("dqn", state_dim=77, action_emb_dim=32)
    print(f"   Model type: {type(dqn).__name__}")
    assert isinstance(dqn, DQNNetwork)
    print("   ✅ DQN creation works")

    # Test Double DQN creation
    print("\n2. Creating Double DQN via factory...")
    double_dqn = create_dqn_model("double_dqn", state_dim=77, action_emb_dim=32)
    print(f"   Model type: {type(double_dqn).__name__}")
    assert isinstance(double_dqn, DoubleDQNAgent)
    print("   ✅ Double DQN creation works")

    # Test invalid model type
    print("\n3. Testing invalid model type...")
    try:
        create_dqn_model("invalid_model")
        print("   ❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")

    print("\n" + "=" * 70)
    print("✅ All factory tests passed!")

    return True


if __name__ == "__main__":
    try:
        test_dqn_network()
        test_double_dqn()
        test_create_model_factory()

        print("\n\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\n📊 Summary:")
        print("   - DQN Network architecture: ✅")
        print("   - Forward pass (single and batch): ✅")
        print("   - Action Q-value prediction: ✅")
        print("   - Gradient flow: ✅")
        print("   - Double DQN with target network: ✅")
        print("   - Target network updates: ✅")
        print("   - Target Q-value calculation: ✅")
        print("   - Model factory: ✅")
        print("\n🚀 DQN models are ready for training!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
