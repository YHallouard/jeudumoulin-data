# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nine Men's Morris (Jeu du Moulin) game with an AlphaZero AI agent. The architecture enforces a strict separation: Rust handles the game engine and MCTS search, Python handles neural networks and training, connected via PyO3 bindings.

## Commands

```bash
make install      # Create venv and install pre-commit hooks
make check        # Run ruff linting + mypy type checking
make test         # Run pytest with coverage
make train        # Run training (pass ARGS="-c config/train_alphazero.yaml")
make play         # Play a game (pass ARGS="-c config/play.yaml")
make build        # Build wheel file
```

Running a single test:
```bash
uv run pytest tests/path/to/test.py::test_name -v
```

Rebuild the Rust extension after changes:
```bash
uv run maturin develop
```

## Architecture

### Rust (`src_rust/`)
- `game/` — Board state, move generation, game rules, Phase/Player enums
- `search/mcts.rs` — MCTS algorithm using neural network priors
- `training/self_play.rs` — Self-play game generation
- `lib.rs` — PyO3 bindings: `PyBoard`, `PyMove`, `PyMCTS`, `PyNode`

### Python (`src_python/`)
- `agent/alphazero/` — AlphaZero: `_agent.py` (inference), `_models.py` (MLPDualNet dual-head network), `_backbone.py`, `_replay_buffer.py`
- `agent/dqn/` — DQN alternative agent
- `player/` — `HumanPlayer`, `RandomPlayer`, `AlphaZeroPlayer`, `DQNPlayer`
- `cli/` — Click CLI with `play` and `train` subcommands
- `monitoring/` — MLflow integration + Cloudflare Access authentication
- `connectors/storage/minio.py` — MinIO S3-compatible storage backend
- `workflows/` — Prefect workflow orchestration
- `settings.py` — Pydantic settings (MinIO credentials from env)

### Data Flow
```
CLI → PlayerConfig → Python Agent (inference) ↔ Rust Board (game state)
                          ↓
              PyTorch Neural Net (policy + value)
                          ↓
              MCTS (Rust) with NN priors → selected move
```

### Key Embedding Formats
- **Board embedding** (77 features): [0-1] current player, [2-4] game phase, [5-76] board positions (24 × 3 for White/Black/None)
- **Move embedding** (72 features): [0-23] from_pos, [24-47] to_pos, [48-71] removed_pos (all one-hot)

Agents receive embeddings (lists of floats), never raw `Board` objects — this is the key interface contract.

## Code Quality

- **Line length:** 120 chars
- **Type checking:** mypy strict (`disallow_untyped_defs = true`, `disallow_any_unimported = true`)
- **Linter/formatter:** ruff
- Pre-commit hooks run ruff, trailing whitespace, YAML/JSON validation automatically

## Environment

Required env vars (copy `.env.example`):
```
CF_ACCESS_CLIENT_ID, CF_ACCESS_CLIENT_SECRET   # Cloudflare Access
MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET
```

MLflow server runs locally via:
```bash
docker compose up
```
(port 5001 external → 5000 internal, SQLite backend)

## CI/CD

- PRs: runs validate workflow (tests + type check + linting, both Python and Rust)
- Push to `main`: automatic semantic versioning via python-semantic-release
- Version tags (`v*`): builds multi-platform Docker image → `ghcr.io/YHallouard/jeudumoulin-worker`
- Skip CI: include `[skip ci]` in commit message
