import pytest
from agent.alphazero._replay_buffer import AlphaZeroReplayBuffer


def test_replay_buffer_initialization():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    assert len(buffer) == 0
    assert buffer.max_size == 100


def test_add_examples():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    state_embeddings = [[0.1] * 77, [0.2] * 77]
    legal_moves = [[[[1, 2, 3]]], [[[4, 5, 6]]]]
    policy_labels = [[0.5, 0.5], [0.3, 0.7]]
    value_labels = [0.8, -0.2]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
    assert len(buffer) == 2


def test_sample_batch():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    state_embeddings = [[0.1] * 77 for _ in range(10)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(10)]
    policy_labels = [[0.5, 0.5] for _ in range(10)]
    value_labels = [0.1 * i for i in range(10)]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)

    states, moves, policies, values = buffer.sample_batch(5)
    assert len(states) == 5
    assert len(moves) == 5
    assert len(policies) == 5
    assert len(values) == 5


def test_sample_batch_all_available():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    state_embeddings = [[0.1] * 77 for _ in range(3)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(3)]
    policy_labels = [[0.5, 0.5] for _ in range(3)]
    value_labels = [0.1 * i for i in range(3)]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)

    states, moves, policies, values = buffer.sample_batch(10)
    assert len(states) == 3
    assert len(moves) == 3
    assert len(policies) == 3
    assert len(values) == 3


def test_can_sample():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    assert not buffer.can_sample(1)

    state_embeddings = [[0.1] * 77 for _ in range(5)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(5)]
    policy_labels = [[0.5, 0.5] for _ in range(5)]
    value_labels = [0.1 * i for i in range(5)]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
    assert buffer.can_sample(3)
    assert buffer.can_sample(5)
    assert not buffer.can_sample(10)


def test_clear():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    state_embeddings = [[0.1] * 77 for _ in range(5)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(5)]
    policy_labels = [[0.5, 0.5] for _ in range(5)]
    value_labels = [0.1 * i for i in range(5)]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
    assert len(buffer) == 5

    buffer.clear()
    assert len(buffer) == 0


def test_max_size_constraint():
    buffer = AlphaZeroReplayBuffer(max_size=5)
    state_embeddings = [[0.1] * 77 for _ in range(10)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(10)]
    policy_labels = [[0.5, 0.5] for _ in range(10)]
    value_labels = [0.1 * i for i in range(10)]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
    assert len(buffer) == 5


def test_get_statistics_empty():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    stats = buffer.get_statistics()
    assert stats["size"] == 0
    assert stats["capacity"] == 100
    assert stats["fill_ratio"] == 0.0
    assert stats["avg_value"] == 0.0


def test_get_statistics_with_data():
    buffer = AlphaZeroReplayBuffer(max_size=100)
    state_embeddings = [[0.1] * 77 for _ in range(10)]
    legal_moves = [[[[i, i + 1, i + 2]]] for i in range(10)]
    policy_labels = [[0.5, 0.5] for _ in range(10)]
    value_labels = [float(i) for i in range(10)]

    buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
    stats = buffer.get_statistics()
    assert stats["size"] == 10
    assert stats["capacity"] == 100
    assert stats["fill_ratio"] == 0.1
    assert stats["avg_value"] == pytest.approx(4.5)
