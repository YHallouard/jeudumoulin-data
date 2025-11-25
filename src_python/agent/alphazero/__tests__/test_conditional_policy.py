import unittest

import torch
from agent.alphazero._conditional_policy import (
    FullyConditionalPolicyHead,
    SemiConditionalPolicyHead,
)
from jdm_ru import PyBoard as Board


class TestSemiConditionalPolicyHead(unittest.TestCase):
    def setUp(self):
        self.policy_head = SemiConditionalPolicyHead(state_embedding_dim=128, position_embedding_dim=32)
        self.state = Board()
        self.legal_actions = self.state.legal_moves()

    def test_output_shape(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertEqual(probs.shape, (len(self.legal_actions),))

    def test_probabilities_sum_to_one(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_all_probabilities_positive(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))

    def test_empty_actions(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, [])

        self.assertEqual(len(probs), 0)

    def test_gradient_flow(self):
        state_embedding = torch.randn(128, requires_grad=True)
        probs = self.policy_head(state_embedding, self.legal_actions)

        loss = -torch.log(probs[0])
        loss.backward()

        self.assertIsNotNone(state_embedding.grad)
        self.assertTrue(torch.any(state_embedding.grad != 0))


class TestFullyConditionalPolicyHead(unittest.TestCase):
    def setUp(self):
        self.policy_head = FullyConditionalPolicyHead(state_embedding_dim=128, position_embedding_dim=32)
        self.state = Board()
        self.legal_actions = self.state.legal_moves()

    def test_output_shape(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertEqual(probs.shape, (len(self.legal_actions),))

    def test_probabilities_sum_to_one(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertAlmostEqual(probs.sum().item(), 1.0, places=5)

    def test_all_probabilities_positive(self):
        state_embedding = torch.randn(128)
        probs = self.policy_head(state_embedding, self.legal_actions)

        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.all(probs <= 1))


class TestConditionalPolicyComparison(unittest.TestCase):
    def test_both_policies_produce_valid_distributions(self):
        semi_policy = SemiConditionalPolicyHead(128, 32)
        full_policy = FullyConditionalPolicyHead(128, 32)

        state = Board()
        legal_actions = state.legal_moves()
        state_embedding = torch.randn(128)

        semi_probs = semi_policy(state_embedding, legal_actions)
        full_probs = full_policy(state_embedding, legal_actions)

        self.assertAlmostEqual(semi_probs.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(full_probs.sum().item(), 1.0, places=5)

        self.assertEqual(semi_probs.shape, full_probs.shape)


if __name__ == "__main__":
    unittest.main()
