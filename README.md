# Jeu du Moulin (Nine Men's Morris)

A hybrid Python/Rust implementation of the Nine Men's Morris game with AlphaZero AI agent.

## Project Structure

```
.
├── src_rust/           # Rust game engine
│   ├── lib.rs         # PyO3 Python bindings
│   └── game/          # Game logic (Board, Move, etc.)
├── src_python/        # Python AI agent and CLI
│   ├── agent/         # AI agents (AlphaZero, Random)
│   └── cli/           # Command-line interface
├── Cargo.toml         # Rust dependencies
└── pyproject.toml     # Python project config
```

## Architecture

The project implements a strict separation between Rust and Python:

- **Rust side**: Game engine (Board, Move) with high-performance logic
- **Python side**: AI agents using PyTorch for neural networks
- **Bridge**: Board and Move have `to_embed()` methods that return flat float vectors
- **Agent interface**: Agents accept embeddings (list of floats) instead of Board objects

## Building

### Prerequisites

- Rust 1.70+ with cargo
- Python 3.14
- maturin (`pip install maturin`)

### Build the Rust module

```bash
# Development build
maturin develop

# Release build
maturin build --release
```

### Install Python dependencies

```bash
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

#### Play a game

```bash
# Human vs Random AI
python -m jeudumoulin_py.cli.main play --player1 human --player2 random

# AI vs AI
python -m jeudumoulin_py.cli.main play --player1 random --player2 alphazero --model-path models/checkpoint.pt

# Human vs Human
python -m jeudumoulin_py.cli.main play --player1 human --player2 human
```

#### Train an AlphaZero agent

```bash
python -m jeudumoulin_py.cli.main train \
    --num-iterations 100 \
    --num-episodes 100 \
    --num-simulations 50 \
    --save-folder ./models
```

### Python API

```python
import jdm_ru

# Create a new board
board = jdm_ru.PyBoard()

# Get legal moves
moves = board.legal_moves()

# Convert board to embedding for neural network
embedding = board.to_embed()  # Returns list of 77 floats

# Apply a move
new_board = board.apply_move(moves[0])

# Check game status
if board.is_terminal():
    winner = board.winner()  # Returns 1 (White), -1 (Black), or None
```

### Using with AI Agent

```python
from jeudumoulin_py.agent.alphazero._agent import AlphaZeroAgent
import jdm_ru

# Create agent
agent = AlphaZeroAgent()

# Get board state
board = jdm_ru.PyBoard()
state_embedding = board.to_embed()
legal_moves = board.legal_moves()

# Get policy and value
policy, value = agent.predict(state_embedding, legal_moves)

# Select best move
best_move_idx = max(policy.keys(), key=lambda k: policy[k])
best_move = legal_moves[best_move_idx]
```

## Development

### Run tests

```bash
# Rust tests
cargo test

# Python tests
pytest
```

### Format code

```bash
# Rust
cargo fmt

# Python
ruff format src_python/
```

### Type checking

```bash
mypy src_python/
```

## Implementation Details

### Board Embedding (77 features)

- `[0-1]`: One-hot encoded current player (White, Black)
- `[2-4]`: One-hot encoded game phase (Placing, Moving, Flying)
- `[5-76]`: Board state (24 positions × 3 one-hot encoded: White, Black, None)

### Move Embedding (72 features)

- `[0-23]`: One-hot encoded from_position
- `[24-47]`: One-hot encoded to_position
- `[48-71]`: One-hot encoded removed_position

### Python-Rust Interface

All interactions between Python and Rust use native types (lists, ints, floats) except for the Board and Move wrapper classes (`PyBoard` and `PyMove`). The Agent classes work exclusively with embeddings:

```python
# ✅ Correct: Agent accepts embeddings
def predict(self, state_embedding: list[float], legal_moves: list) -> tuple[dict[int, float], float]:
    pass

# ❌ Wrong: Don't pass Board objects to Agent
# def predict(self, board: Board) -> ...:
```

## License

MIT
