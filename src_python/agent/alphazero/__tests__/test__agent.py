import unittest

import torch
from agent.alphazero._agent import AlphaZeroAgent, AlphaZeroAgentConfig
from agent.alphazero._backbone import MLPBackboneConfig
from agent.alphazero._conditional_policy import SemiConditionalPolicyHeadConfig
from agent.alphazero._models import MLPDualNetConfig
from agent.alphazero._position import PositionalEmbeddingConfig
from agent.alphazero.random_agent import RandomAgent

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

DUMMY_STATE = [0.0] * 77
DUMMY_LEGAL_MOVES: list[list[int | None]] = [[None, 0, None], [None, 1, None], [None, 2, None]]


class TestAlphaZeroAgentBatchPredict(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AlphaZeroAgent(config=TOY_CONFIG)

    def test_single_item_batch(self) -> None:
        results = self.agent.predict([DUMMY_STATE], [DUMMY_LEGAL_MOVES])
        self.assertEqual(len(results), 1)
        policy_dict, value = results[0]
        self.assertEqual(len(policy_dict), len(DUMMY_LEGAL_MOVES))
        self.assertAlmostEqual(sum(policy_dict.values()), 1.0, places=4)
        self.assertIsInstance(value, float)

    def test_batch_matches_individual(self) -> None:
        state_1 = [1.0] + [0.0] * 76
        state_2 = [0.0, 1.0] + [0.0] * 75
        moves_1: list[list[int | None]] = [[None, 0, None], [None, 3, None]]
        moves_2: list[list[int | None]] = [[None, 1, None], [None, 2, None], [None, 5, None]]

        individual_1 = self.agent.predict([state_1], [moves_1])[0]
        individual_2 = self.agent.predict([state_2], [moves_2])[0]

        batch_results = self.agent.predict([state_1, state_2], [moves_1, moves_2])

        for key in individual_1[0]:
            self.assertAlmostEqual(individual_1[0][key], batch_results[0][0][key], places=5)
        self.assertAlmostEqual(individual_1[1], batch_results[0][1], places=5)

        for key in individual_2[0]:
            self.assertAlmostEqual(individual_2[0][key], batch_results[1][0][key], places=5)
        self.assertAlmostEqual(individual_2[1], batch_results[1][1], places=5)


class TestRandomAgentBatchPredict(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RandomAgent()

    def test_batch_predict(self) -> None:
        moves_1: list[list[int | None]] = [[None, 0, None], [None, 1, None]]
        moves_2: list[list[int | None]] = [[None, 0, None], [None, 1, None], [None, 2, None]]

        results = self.agent.predict([DUMMY_STATE, DUMMY_STATE], [moves_1, moves_2])
        self.assertEqual(len(results), 2)

        self.assertEqual(len(results[0][0]), 2)
        self.assertAlmostEqual(results[0][0][0], 0.5)
        self.assertAlmostEqual(results[0][1], 0.0)

        self.assertEqual(len(results[1][0]), 3)
        self.assertAlmostEqual(results[1][0][0], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(results[1][1], 0.0)


class TestPolicyValueBatch(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = AlphaZeroAgent(config=TOY_CONFIG)
        self.model = self.agent.model

    def test_batch_matches_single(self) -> None:
        state_1 = [1.0] + [0.0] * 76
        state_2 = [0.0, 1.0] + [0.0] * 75
        moves_1: list[list[int | None]] = [[None, 0, None], [None, 3, None]]
        moves_2: list[list[int | None]] = [[None, 1, None], [None, 2, None], [None, 5, None]]

        self.model.eval()
        with torch.no_grad():
            single_p1, single_v1 = self.model.policy_value(state_1, moves_1)
            single_p2, single_v2 = self.model.policy_value(state_2, moves_2)

            batch_policies, batch_values = self.model.policy_value_batch([state_1, state_2], [moves_1, moves_2])

        torch.testing.assert_close(batch_policies[0], single_p1)
        torch.testing.assert_close(batch_policies[1], single_p2)
        torch.testing.assert_close(batch_values[0].squeeze(), single_v1.squeeze())
        torch.testing.assert_close(batch_values[1].squeeze(), single_v2.squeeze())


if __name__ == "__main__":
    unittest.main()
