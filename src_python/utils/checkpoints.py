import io
import json
import tempfile
from pathlib import Path

import structlog
import torch
from agent.alphazero._agent import AlphaZeroAgent
from agent.alphazero._trainer import AlphaZeroTrainer
from connectors.storage.minio import get_minio_client
from mypy_boto3_s3 import S3Client
from prefect import task
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from settings import settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Low-level S3 helpers
# ---------------------------------------------------------------------------


def s3_upload_bytes(s3: S3Client, data: bytes, key: str) -> None:
    s3.upload_fileobj(io.BytesIO(data), settings.MINIO_BUCKET, key)


def s3_download_bytes(s3: S3Client, key: str) -> bytes:
    buf = io.BytesIO()
    s3.download_fileobj(settings.MINIO_BUCKET, key, buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Eval weights (model-only, ephemeral)
# ---------------------------------------------------------------------------


def upload_weights(agent: AlphaZeroAgent, iteration: int, execution_id: str) -> str:
    s3 = get_minio_client()
    weights_bytes = safetensors_save({k: v.cpu() for k, v in agent.model.state_dict().items()})
    s3_key = f"eval/{execution_id}/iter_{iteration:04d}.safetensors"
    s3_upload_bytes(s3, weights_bytes, s3_key)
    return s3_key


def download_weights(agent: AlphaZeroAgent, s3_key: str) -> None:
    s3 = get_minio_client()
    agent.model.load_state_dict(safetensors_load(s3_download_bytes(s3, s3_key)))


def cleanup_weights(s3_key: str) -> None:
    try:
        get_minio_client().delete_object(Bucket=settings.MINIO_BUCKET, Key=s3_key)
    except Exception:
        logger.warning("failed_to_cleanup_weights", s3_key=s3_key)


# ---------------------------------------------------------------------------
# Agent files download (config + weights → tempdir for from_pretrained)
# ---------------------------------------------------------------------------


def download_agent_files(s3_prefix: str, tmp_path: Path) -> None:
    s3 = get_minio_client()
    (tmp_path / "config.json").write_bytes(s3_download_bytes(s3, f"{s3_prefix}/config.json"))
    (tmp_path / "model.safetensors").write_bytes(s3_download_bytes(s3, f"{s3_prefix}/model.safetensors"))


# ---------------------------------------------------------------------------
# Full checkpoint — save / restore
# ---------------------------------------------------------------------------


def save_full_checkpoint(trainer: AlphaZeroTrainer, s3_prefix: str, iteration: int) -> None:
    s3 = get_minio_client()

    model_bytes = safetensors_save({k: v.cpu() for k, v in trainer.agent.model.state_dict().items()})
    s3_upload_bytes(s3, model_bytes, f"{s3_prefix}/model.safetensors")

    s3_upload_bytes(s3, trainer.replay_buffer.to_bytes(), f"{s3_prefix}/buffer.pkl")

    opt_buf = io.BytesIO()
    torch.save(trainer.optimizer.state_dict(), opt_buf)
    s3_upload_bytes(s3, opt_buf.getvalue(), f"{s3_prefix}/optimizer.pt")

    sched_buf = io.BytesIO()
    torch.save(trainer.scheduler.state_dict(), sched_buf)
    s3_upload_bytes(s3, sched_buf.getvalue(), f"{s3_prefix}/scheduler.pt")

    config_bytes = json.dumps(trainer.agent.config.model_dump()).encode()
    s3_upload_bytes(s3, config_bytes, f"{s3_prefix}/config.json")

    meta = {"iteration": iteration, "agent_config": trainer.agent.config.model_dump()}
    s3_upload_bytes(s3, json.dumps(meta).encode(), f"{s3_prefix}/meta.json")

    logger.info("full_checkpoint_saved", s3_prefix=s3_prefix, iteration=iteration)


def restore_checkpoint_state(
    trainer: AlphaZeroTrainer,
    s3_prefix: str,
    load_buffer: bool,
    load_optimizer: bool,
) -> None:
    """Restore trainer state (buffer, optimizer, scheduler) from an S3 checkpoint.

    Model weights are NOT loaded here — they are already loaded by
    ``_create_agent_from_init`` via ``AlphaZeroAgent.from_pretrained``.
    """
    s3 = get_minio_client()

    if load_buffer:
        trainer.replay_buffer.load_from_bytes(s3_download_bytes(s3, f"{s3_prefix}/buffer.pkl"))
        logger.info("checkpoint_buffer_loaded", size=len(trainer.replay_buffer))

    if load_optimizer:
        opt_bytes = s3_download_bytes(s3, f"{s3_prefix}/optimizer.pt")
        trainer.optimizer.load_state_dict(torch.load(io.BytesIO(opt_bytes), weights_only=True))
        sched_bytes = s3_download_bytes(s3, f"{s3_prefix}/scheduler.pt")
        trainer.scheduler.load_state_dict(torch.load(io.BytesIO(sched_bytes), weights_only=False))
        logger.info("checkpoint_optimizer_loaded")


# ---------------------------------------------------------------------------
# Prefect task
# ---------------------------------------------------------------------------


@task(name="save-full-checkpoint", persist_result=False)
def save_full_checkpoint_task(trainer: AlphaZeroTrainer, s3_prefix: str, iteration: int) -> None:
    save_full_checkpoint(trainer, s3_prefix, iteration)
