import unittest

from agent.alphazero._replay_buffer import AlphaZeroReplayBuffer


class TestAlphaZeroReplayBuffer(unittest.TestCase):
    def test_initialization(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(buffer.max_size, 100)

    def test_add_examples(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        state_embeddings = [[0.1] * 77, [0.2] * 77]
        legal_moves: list[list[list[int | None]]] = [[[1, 2, 3]], [[4, 5, 6]]]
        policy_labels = [[0.5, 0.5], [0.3, 0.7]]
        value_labels = [0.8, -0.2]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
        self.assertEqual(len(buffer), 2)

    def test_sample_batch(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        state_embeddings = [[0.1] * 77 for _ in range(10)]
        legal_moves: list[list[list[int | None]]] = [[[i, i + 1, i + 2]] for i in range(10)]
        policy_labels = [[0.5, 0.5] for _ in range(10)]
        value_labels = [0.1 * i for i in range(10)]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)

        states, moves, policies, values = buffer.sample_batch(5)
        self.assertEqual(len(states), 5)
        self.assertEqual(len(moves), 5)
        self.assertEqual(len(policies), 5)
        self.assertEqual(len(values), 5)

    def test_sample_batch_all_available(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        state_embeddings = [[0.1] * 77 for _ in range(3)]
        legal_moves: list[list[list[int | None]]] = [[[i, i + 1, i + 2]] for i in range(3)]
        policy_labels = [[0.5, 0.5] for _ in range(3)]
        value_labels = [0.1 * i for i in range(3)]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)

        states, moves, policies, values = buffer.sample_batch(10)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(moves), 3)
        self.assertEqual(len(policies), 3)
        self.assertEqual(len(values), 3)

    def test_can_sample(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        self.assertFalse(buffer.can_sample(1))

        state_embeddings = [[0.1] * 77 for _ in range(5)]
        legal_moves: list[list[list[int | None]]] = [[[i, i + 1, i + 2]] for i in range(5)]
        policy_labels = [[0.5, 0.5] for _ in range(5)]
        value_labels = [0.1 * i for i in range(5)]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
        self.assertTrue(buffer.can_sample(3))
        self.assertTrue(buffer.can_sample(5))
        self.assertFalse(buffer.can_sample(10))

    def test_clear(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        state_embeddings = [[0.1] * 77 for _ in range(5)]
        legal_moves: list[list[list[int | None]]] = [[[i, i + 1, i + 2]] for i in range(5)]
        policy_labels = [[0.5, 0.5] for _ in range(5)]
        value_labels = [0.1 * i for i in range(5)]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
        self.assertEqual(len(buffer), 5)

        buffer.clear()
        self.assertEqual(len(buffer), 0)

    def test_max_size_constraint(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=5)
        state_embeddings = [[0.1] * 77 for _ in range(10)]
        legal_moves: list[list[list[int | None]]] = [[[i, i + 1, i + 2]] for i in range(10)]
        policy_labels = [[0.5, 0.5] for _ in range(10)]
        value_labels = [0.1 * i for i in range(10)]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
        self.assertEqual(len(buffer), 5)

    def test_get_statistics_empty(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        stats = buffer.get_statistics()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["capacity"], 100)
        self.assertAlmostEqual(stats["fill_ratio"], 0.0)
        self.assertAlmostEqual(stats["avg_value"], 0.0)

    def test_get_statistics_with_data(self) -> None:
        buffer = AlphaZeroReplayBuffer(max_size=100)
        state_embeddings = [[0.1] * 77 for _ in range(10)]
        legal_moves: list[list[list[int | None]]] = [[[i, i + 1, i + 2]] for i in range(10)]
        policy_labels = [[0.5, 0.5] for _ in range(10)]
        value_labels = [float(i) for i in range(10)]

        buffer.add_examples(state_embeddings, legal_moves, policy_labels, value_labels)
        stats = buffer.get_statistics()
        self.assertEqual(stats["size"], 10)
        self.assertEqual(stats["capacity"], 100)
        self.assertAlmostEqual(stats["fill_ratio"], 0.1)
        self.assertAlmostEqual(stats["avg_value"], 4.5)
