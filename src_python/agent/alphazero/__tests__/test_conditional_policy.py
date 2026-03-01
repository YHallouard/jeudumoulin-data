import unittest

import torch
from agent.alphazero._conditional_policy import (
    FullyConditionalPolicyHead,
    FullyConditionalPolicyHeadConfig,
    SemiConditionalPolicyHead,
    SemiConditionalPolicyHeadConfig,
)
from agent.alphazero._position import PositionalEmbeddingConfig
from jdm_ru import PyBoard as Board


def _make_semi_config(state_dim: int = 128, emb_dim: int = 32) -> SemiConditionalPolicyHeadConfig:
    return SemiConditionalPolicyHeadConfig(
        state_embedding_dim=state_dim,
        embedding=PositionalEmbeddingConfig(embedding_dim=emb_dim),
    )


def _make_fully_config(state_dim: int = 128, emb_dim: int = 32) -> FullyConditionalPolicyHeadConfig:
    return FullyConditionalPolicyHeadConfig(
        state_embedding_dim=state_dim,
        embedding=PositionalEmbeddingConfig(embedding_dim=emb_dim),
    )


def _legal_moves_as_indices(board: Board) -> list[list[int]]:
    return [move.to_indices() for move in board.legal_moves()]


class TestSemiConditionalPolicyHead(unittest.TestCase):
    def setUp(self) -> None:
        self.policy_head = SemiConditionalPolicyHead(_make_semi_config())
        self.state = Board()
        self.legal_actions = _legal_moves_as_indices(self.state)

    def test_output_shape(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertEqual(probs.shape, (len(self.legal_actions),))

    def test_probabilities_sum_to_one(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_all_probabilities_positive(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_empty_actions(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, [])

        self.assertEqual(len(probs), 0)

    def test_gradient_flow(self) -> None:
        state_embedding = torch.randn(128, requires_grad=True)
        probs = self.policy_head(state_embedding, self.legal_actions)

        loss = -torch.log(probs[0])
        loss.backward()

        self.assertIsNotNone(state_embedding.grad)
        self.assertTrue(torch.any(state_embedding.grad != 0).item())  # type: ignore[arg-type]


class TestFullyConditionalPolicyHead(unittest.TestCase):
    def setUp(self) -> None:
        self.policy_head = FullyConditionalPolicyHead(_make_fully_config())
        self.state = Board()
        self.legal_actions = _legal_moves_as_indices(self.state)

    def test_output_shape(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertEqual(probs.shape, (len(self.legal_actions),))

    def test_probabilities_sum_to_one(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_all_probabilities_positive(self) -> None:
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))


class TestConditionalPolicyComparison(unittest.TestCase):
    def test_both_policies_produce_valid_distributions(self) -> None:
        semi_policy = SemiConditionalPolicyHead(_make_semi_config())
        full_policy = FullyConditionalPolicyHead(_make_fully_config())

        state = Board()
        legal_actions = _legal_moves_as_indices(state)
        state_embedding = torch.randn(128)

        semi_probs = semi_policy(state_embedding, legal_actions)
        full_probs = full_policy(state_embedding, legal_actions)

        self.assertAlmostEqual(semi_probs.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(full_probs.sum().item(), 1.0, places=5)

        self.assertEqual(semi_probs.shape, full_probs.shape)


if __name__ == "__main__":
    unittest.main()
