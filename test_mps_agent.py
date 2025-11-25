import torch
from agent.alphazero._agent import AlphaZeroAgent, AlphaZeroAgentConfig
from agent.alphazero._backbone import GraphConvBackboneConfig
from agent.alphazero._conditional_policy import GatedConditionalPolicyHeadConfig
from agent.alphazero._models import MLPDualNetConfig
from agent.alphazero._position import PositionalEmbeddingConfig

print("=" * 70)
print("🧪 Test MPS Device Support")
print("=" * 70)

print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ MPS available: {torch.backends.mps.is_available()}")
print(f"✓ MPS built: {torch.backends.mps.is_built()}")

if not torch.backends.mps.is_available():
    print("\n❌ MPS not available on this system!")
    exit(1)

print("\n" + "=" * 70)
print("Creating agent with MPS device...")
print("=" * 70)

config = AlphaZeroAgentConfig(
    model=MLPDualNetConfig(
        backbone=GraphConvBackboneConfig(
            player_embedding_dim=8,
            phase_embedding_dim=8,
            board_embedding_dim=64,
            hidden_dim=128,
            output_dim=128,
            graph_layer_hidden_dim=32,
            graph_layer_output_dim=64,
            num_graph_layers=2,
            use_attention_pooling=False,
        ),
        policy_head=GatedConditionalPolicyHeadConfig(
            state_embedding_dim=128,
            embedding=PositionalEmbeddingConfig(embedding_dim=16),
            hidden_dim=128,
            dropout_rate=0.1,
        ),
        value_head=MLPDualNetConfig.ValueHeadConfig(
            hidden_dim=128,
            dropout_rate=0.2,
            output_dim=1,
        ),
    ),
    device="mps",
)

try:
    agent = AlphaZeroAgent(config=config)
    print("✓ Agent created successfully")
    print(f"✓ Model device: {next(agent.model.parameters()).device}")
except Exception as e:
    print(f"❌ Failed to create agent: {e}")
    exit(1)

print("\n" + "=" * 70)
print("Testing prediction (simulating Rust call)...")
print("=" * 70)

state_embedding = [1.0, 0.0] + [0.0, 1.0, 0.0] + [0.5] * 72
legal_moves = [
    [0, 5, None],
    [1, 6, None],
    [2, 7, 10],
]

try:
    policy_dict, value = agent.predict(state_embedding, legal_moves)

    print("✓ Prediction successful!")
    print(f"✓ Policy dict type: {type(policy_dict)}")
    print(f"✓ Policy dict length: {len(policy_dict)}")
    print(f"✓ Value type: {type(value)}")
    print(f"✓ Value: {value:.4f}")

    for idx, prob in policy_dict.items():
        print(f"  Action {idx}: {prob:.4f} (type: {type(prob).__name__})")
        assert isinstance(prob, float), f"Policy prob should be float, got {type(prob)}"

    assert isinstance(value, float), f"Value should be float, got {type(value)}"
    assert len(policy_dict) == len(legal_moves), "Policy dict should match legal moves"

    print("\n✅ All assertions passed!")

except Exception as e:
    print(f"❌ Prediction failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("Testing multiple predictions (stress test)...")
print("=" * 70)

import time

num_predictions = 100
start = time.time()

for i in range(num_predictions):
    policy_dict, value = agent.predict(state_embedding, legal_moves)

elapsed = time.time() - start
avg_time = (elapsed / num_predictions) * 1000

print(f"✓ {num_predictions} predictions in {elapsed:.2f}s")
print(f"✓ Average: {avg_time:.2f}ms per prediction")

print("\n" + "=" * 70)
print("🎉 All tests passed! MPS is working correctly with Rust interface")
print("=" * 70)
