#!/usr/bin/env python3
"""Test the Experience Replay Buffer implementation."""

import torch
from agent.dqn._replay_buffer import ExperienceReplayBuffer, PrioritizedExperienceReplayBuffer


def test_experience_replay_buffer():
    """Test the standard Experience Replay Buffer."""

    print("Testing Experience Replay Buffer...")
    print("=" * 70)

    # Create buffer
    print("\n1. Creating buffer with max_size=100...")
    buffer = ExperienceReplayBuffer(max_size=100, device="cpu")
    print(f"   Initial size: {len(buffer)}")
    print(f"   Max capacity: {buffer.max_size}")
    assert len(buffer) == 0
    print("   ✅ Buffer created")

    # Add single transition
    print("\n2. Adding single transition...")
    state = [0.1] * 77
    action = [5, 10, 24]
    reward = 1.0
    next_state = [0.2] * 77
    done = False

    buffer.add(state, action, reward, next_state, done)
    print(f"   Buffer size after add: {len(buffer)}")
    assert len(buffer) == 1
    print("   ✅ Transition added")

    # Add multiple transitions
    print("\n3. Adding 50 more transitions...")
    for i in range(50):
        state = [float(i) / 100] * 77
        action = [i % 24, (i + 1) % 24, 24]
        reward = float(i) / 50
        next_state = [float(i + 1) / 100] * 77
        done = i == 49
        buffer.add(state, action, reward, next_state, done)

    print(f"   Buffer size: {len(buffer)}")
    assert len(buffer) == 51
    print("   ✅ Multiple transitions added")

    # Test can_sample
    print("\n4. Testing can_sample...")
    print(f"   Can sample batch of 32? {buffer.can_sample(32)}")
    print(f"   Can sample batch of 100? {buffer.can_sample(100)}")
    assert buffer.can_sample(32) == True
    assert buffer.can_sample(100) == False
    print("   ✅ can_sample works")

    # Test sampling
    print("\n5. Testing batch sampling...")
    batch_size = 32
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)

    print(f"   Batch size: {batch_size}")
    print(f"   States shape: {states.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Rewards shape: {rewards.shape}")
    print(f"   Next states shape: {next_states.shape}")
    print(f"   Dones shape: {dones.shape}")

    assert states.shape == (batch_size, 77)
    assert actions.shape == (batch_size, 3)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, 77)
    assert dones.shape == (batch_size,)

    print(f"   Rewards range: [{rewards.min().item():.3f}, {rewards.max().item():.3f}]")
    print(f"   Number of terminal states: {dones.sum().item():.0f}")
    print("   ✅ Sampling works")

    # Test tensor types
    print("\n6. Testing tensor types...")
    print(f"   States dtype: {states.dtype}")
    print(f"   Actions dtype: {actions.dtype}")
    print(f"   Rewards dtype: {rewards.dtype}")
    print(f"   Dones dtype: {dones.dtype}")

    assert states.dtype == torch.float32
    assert actions.dtype == torch.long
    assert rewards.dtype == torch.float32
    assert dones.dtype == torch.float32
    print("   ✅ Tensor types correct")

    # Test buffer overflow (FIFO behavior)
    print("\n7. Testing buffer overflow (FIFO)...")
    buffer_small = ExperienceReplayBuffer(max_size=10)

    for i in range(15):
        buffer_small.add([float(i)] * 77, [i, i, 24], float(i), [float(i)] * 77, False)

    print("   Added 15 transitions to buffer with max_size=10")
    print(f"   Current buffer size: {len(buffer_small)}")
    assert len(buffer_small) == 10
    print("   ✅ FIFO overflow works")

    # Test statistics
    print("\n8. Testing buffer statistics...")
    stats = buffer.get_statistics()
    print("   Statistics:")
    for key, value in stats.items():
        print(f"     - {key}: {value}")

    assert stats["size"] == len(buffer)
    assert 0 <= stats["fill_ratio"] <= 1
    print("   ✅ Statistics work")

    # Test clear
    print("\n9. Testing buffer clear...")
    buffer.clear()
    print(f"   Buffer size after clear: {len(buffer)}")
    assert len(buffer) == 0
    print("   ✅ Clear works")

    # Test error handling
    print("\n10. Testing error handling...")
    try:
        buffer.sample(10)  # Buffer is empty
        print("   ❌ Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {str(e)[:50]}...")

    print("\n" + "=" * 70)
    print("✅ All Experience Replay Buffer tests passed!")

    return True


def test_prioritized_replay_buffer():
    """Test the Prioritized Experience Replay Buffer."""

    print("\n\nTesting Prioritized Experience Replay Buffer...")
    print("=" * 70)

    # Create buffer
    print("\n1. Creating prioritized buffer...")
    buffer = PrioritizedExperienceReplayBuffer(max_size=100, alpha=0.6, beta=0.4, beta_increment=0.001, device="cpu")
    print(f"   Alpha (priority exponent): {buffer.alpha}")
    print(f"   Beta (IS exponent): {buffer.beta}")
    print("   ✅ Prioritized buffer created")

    # Add transitions with different priorities
    print("\n2. Adding transitions with priorities...")
    for i in range(50):
        state = [float(i) / 100] * 77
        action = [i % 24, (i + 1) % 24, 24]
        reward = float(i) / 50
        next_state = [float(i + 1) / 100] * 77
        done = False
        priority = float(i + 1)  # Higher index = higher priority
        buffer.add(state, action, reward, next_state, done, priority=priority)

    print(f"   Buffer size: {len(buffer)}")
    print(f"   Max priority: {buffer.max_priority:.2f}")
    assert len(buffer) == 50
    print("   ✅ Transitions with priorities added")

    # Test prioritized sampling
    print("\n3. Testing prioritized sampling...")
    batch_size = 16
    states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size)

    print(f"   Batch size: {batch_size}")
    print(f"   States shape: {states.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Number of indices: {len(indices)}")
    print(f"   Weights range: [{weights.min().item():.3f}, {weights.max().item():.3f}]")
    print(f"   Beta after sampling: {buffer.beta:.4f}")

    assert len(indices) == batch_size
    assert weights.shape == (batch_size,)
    assert weights.min() >= 0 and weights.max() <= 1.0
    print("   ✅ Prioritized sampling works")

    # Test priority updates
    print("\n4. Testing priority updates...")
    new_priorities = torch.rand(batch_size) * 10
    buffer.update_priorities(indices, new_priorities)
    print(f"   Updated priorities for {len(indices)} transitions")
    print(f"   New max priority: {buffer.max_priority:.2f}")
    print("   ✅ Priority updates work")

    # Test beta increment
    print("\n5. Testing beta increment...")
    initial_beta = buffer.beta
    for _ in range(10):
        buffer.sample(8)
    final_beta = buffer.beta
    print(f"   Beta before 10 samples: {initial_beta:.4f}")
    print(f"   Beta after 10 samples: {final_beta:.4f}")
    print(f"   Beta increased: {final_beta > initial_beta}")
    assert final_beta > initial_beta
    assert final_beta <= 1.0
    print("   ✅ Beta increment works")

    print("\n" + "=" * 70)
    print("✅ All Prioritized Replay Buffer tests passed!")

    return True


def test_buffer_with_real_data():
    """Test buffer with realistic game data."""

    print("\n\nTesting Buffer with Realistic Data...")
    print("=" * 70)

    print("\n1. Creating buffer and adding game-like data...")
    buffer = ExperienceReplayBuffer(max_size=1000)

    # Simulate a game episode
    num_steps = 50
    for step in range(num_steps):
        # Realistic state (77 floats)
        state = [0.0] * 77
        state[0] = 1.0  # White player
        state[2] = 1.0  # Placing phase

        # Realistic action (from, to, remove)
        action = [24, step % 24, 24]  # Placement move

        # Small reward during game, larger at end
        reward = 0.1 if step < num_steps - 1 else 10.0

        # Next state
        next_state = [0.0] * 77
        next_state[1] = 1.0  # Black player (switched)
        next_state[2] = 1.0  # Still placing

        # Done at last step
        done = step == num_steps - 1

        buffer.add(state, action, reward, next_state, done)

    print(f"   Added {num_steps} game transitions")
    print(f"   Buffer size: {len(buffer)}")

    # Sample and verify
    print("\n2. Sampling batch and verifying data...")
    batch = buffer.sample(16)
    states, actions, rewards, next_states, dones = batch

    print("   Sampled batch of 16 transitions")
    print(f"   Action indices range: [{actions.min().item()}, {actions.max().item()}]")
    print(f"   Reward range: [{rewards.min().item():.2f}, {rewards.max().item():.2f}]")
    print(f"   Terminal states in batch: {dones.sum().item():.0f}")

    # Verify realistic constraints
    assert (actions >= 0).all() and (actions <= 24).all()
    assert (states[:, 0] + states[:, 1]).sum() == 16  # One player is always active
    print("   ✅ Data constraints verified")

    # Test statistics
    print("\n3. Checking buffer statistics...")
    stats = buffer.get_statistics()
    print(f"   Fill ratio: {stats['fill_ratio']:.1%}")
    print(f"   Average reward: {stats['avg_reward']:.3f}")
    print(f"   Terminal ratio: {stats['terminal_ratio']:.1%}")

    assert stats["terminal_ratio"] == 1.0 / num_steps  # Only last state is terminal
    print("   ✅ Statistics match expectations")

    print("\n" + "=" * 70)
    print("✅ All realistic data tests passed!")

    return True


if __name__ == "__main__":
    try:
        test_experience_replay_buffer()
        test_prioritized_replay_buffer()
        test_buffer_with_real_data()

        print("\n\n" + "=" * 70)
        print("🎉 ALL REPLAY BUFFER TESTS PASSED!")
        print("=" * 70)
        print("\n📊 Summary:")
        print("   - Basic replay buffer: ✅")
        print("   - Add/sample transitions: ✅")
        print("   - FIFO overflow: ✅")
        print("   - Tensor types and shapes: ✅")
        print("   - Buffer statistics: ✅")
        print("   - Prioritized replay: ✅")
        print("   - Priority updates: ✅")
        print("   - Beta annealing: ✅")
        print("   - Realistic game data: ✅")
        print("\n🚀 Replay buffers are ready for DQN training!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
