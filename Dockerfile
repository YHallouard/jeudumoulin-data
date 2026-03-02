FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install --no-cache-dir uv

WORKDIR /app
# FIXME: Should keep readme for mathurin why ?
COPY pyproject.toml uv.lock Cargo.toml README.md Cargo.lock ./
COPY src_rust/ ./src_rust/
COPY src_python/ ./src_python/

RUN uv sync --frozen
RUN uv run maturin develop --release

FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src_python/ ./src_python/

ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONPATH="/app/src_python"
