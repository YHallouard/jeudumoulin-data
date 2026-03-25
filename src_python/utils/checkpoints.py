import io
import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Annotated, Literal

import structlog
import torch
from agent.alphazero._agent import AlphaZeroAgent
from agent.alphazero._trainer import AlphaZeroTrainer
from connectors.storage.minio import get_minio_client
from mypy_boto3_s3 import S3Client
from prefect import task
from pydantic import BaseModel, Field
from safetensors.torch import load as safetensors_load
from safetensors.torch import save as safetensors_save
from settings import settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Storage config (discriminated union)
# ---------------------------------------------------------------------------


class LocalCheckpointStorage(BaseModel):
    storage: Literal["local"] = "local"
    base_path: Path


class S3CheckpointStorage(BaseModel):
    storage: Literal["s3"] = "s3"


CheckpointStorageConfig = Annotated[
    LocalCheckpointStorage | S3CheckpointStorage,
    Field(discriminator="storage"),
]


# ---------------------------------------------------------------------------
# Handler ABC
# ---------------------------------------------------------------------------


class CheckpointHandler(ABC):
    @abstractmethod
    def save_checkpoint(self, trainer: AlphaZeroTrainer, prefix: str, iteration: int) -> None: ...

    @abstractmethod
    def restore_checkpoint(
        self, trainer: AlphaZeroTrainer, prefix: str, load_buffer: bool, load_optimizer: bool
    ) -> None: ...

    @abstractmethod
    def download_agent_files(self, prefix: str, tmp_path: Path) -> None: ...

    @abstractmethod
    def upload_eval_weights(self, agent: AlphaZeroAgent, iteration: int, execution_id: str) -> str: ...

    @abstractmethod
    def download_eval_weights(self, agent: AlphaZeroAgent, key: str) -> None: ...

    @abstractmethod
    def cleanup_eval_weights(self, key: str) -> None: ...


# ---------------------------------------------------------------------------
# S3 implementation
# ---------------------------------------------------------------------------


def _s3_upload_bytes(s3: S3Client, data: bytes, key: str) -> None:
    s3.upload_fileobj(io.BytesIO(data), settings.MINIO_BUCKET, key)


def _s3_download_bytes(s3: S3Client, key: str) -> bytes:
    buf = io.BytesIO()
    s3.download_fileobj(settings.MINIO_BUCKET, key, buf)
    return buf.getvalue()


class S3CheckpointHandler(CheckpointHandler):
    def __init__(self) -> None:
        self._s3: S3Client = get_minio_client()

    def save_checkpoint(self, trainer: AlphaZeroTrainer, prefix: str, iteration: int) -> None:
        s3 = self._s3

        model_bytes = safetensors_save({k: v.cpu() for k, v in trainer.agent.model.state_dict().items()})
        _s3_upload_bytes(s3, model_bytes, f"{prefix}/model.safetensors")

        _s3_upload_bytes(s3, trainer.replay_buffer.to_bytes(), f"{prefix}/buffer.pkl")

        opt_buf = io.BytesIO()
        torch.save(trainer.optimizer.state_dict(), opt_buf)
        _s3_upload_bytes(s3, opt_buf.getvalue(), f"{prefix}/optimizer.pt")

        sched_buf = io.BytesIO()
        torch.save(trainer.scheduler.state_dict(), sched_buf)
        _s3_upload_bytes(s3, sched_buf.getvalue(), f"{prefix}/scheduler.pt")

        config_bytes = json.dumps(trainer.agent.config.model_dump()).encode()
        _s3_upload_bytes(s3, config_bytes, f"{prefix}/config.json")

        meta = {"iteration": iteration, "agent_config": trainer.agent.config.model_dump()}
        _s3_upload_bytes(s3, json.dumps(meta).encode(), f"{prefix}/meta.json")

        logger.info("checkpoint_saved", storage="s3", prefix=prefix, iteration=iteration)

    def restore_checkpoint(
        self, trainer: AlphaZeroTrainer, prefix: str, load_buffer: bool, load_optimizer: bool
    ) -> None:
        s3 = self._s3

        if load_buffer:
            trainer.replay_buffer.load_from_bytes(_s3_download_bytes(s3, f"{prefix}/buffer.pkl"))
            logger.info("checkpoint_buffer_loaded", size=len(trainer.replay_buffer))

        if load_optimizer:
            opt_bytes = _s3_download_bytes(s3, f"{prefix}/optimizer.pt")
            trainer.optimizer.load_state_dict(torch.load(io.BytesIO(opt_bytes), weights_only=True))
            sched_bytes = _s3_download_bytes(s3, f"{prefix}/scheduler.pt")
            trainer.scheduler.load_state_dict(torch.load(io.BytesIO(sched_bytes), weights_only=False))
            logger.info("checkpoint_optimizer_loaded")

    def download_agent_files(self, prefix: str, tmp_path: Path) -> None:
        s3 = self._s3
        (tmp_path / "config.json").write_bytes(_s3_download_bytes(s3, f"{prefix}/config.json"))
        (tmp_path / "model.safetensors").write_bytes(_s3_download_bytes(s3, f"{prefix}/model.safetensors"))

    def upload_eval_weights(self, agent: AlphaZeroAgent, iteration: int, execution_id: str) -> str:
        weights_bytes = safetensors_save({k: v.cpu() for k, v in agent.model.state_dict().items()})
        key = f"eval/{execution_id}/iter_{iteration:04d}.safetensors"
        _s3_upload_bytes(self._s3, weights_bytes, key)
        return key

    def download_eval_weights(self, agent: AlphaZeroAgent, key: str) -> None:
        agent.model.load_state_dict(safetensors_load(_s3_download_bytes(self._s3, key)))

    def cleanup_eval_weights(self, key: str) -> None:
        try:
            self._s3.delete_object(Bucket=settings.MINIO_BUCKET, Key=key)
        except Exception:
            logger.warning("failed_to_cleanup_eval_weights", key=key)


# ---------------------------------------------------------------------------
# Local implementation
# ---------------------------------------------------------------------------


class LocalCheckpointHandler(CheckpointHandler):
    def __init__(self, base_path: Path) -> None:
        self._base = base_path

    def _resolve(self, prefix: str) -> Path:
        path = self._base / prefix
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(self, trainer: AlphaZeroTrainer, prefix: str, iteration: int) -> None:
        dest = self._resolve(prefix)

        model_bytes = safetensors_save({k: v.cpu() for k, v in trainer.agent.model.state_dict().items()})
        (dest / "model.safetensors").write_bytes(model_bytes)

        (dest / "buffer.pkl").write_bytes(trainer.replay_buffer.to_bytes())

        opt_buf = io.BytesIO()
        torch.save(trainer.optimizer.state_dict(), opt_buf)
        (dest / "optimizer.pt").write_bytes(opt_buf.getvalue())

        sched_buf = io.BytesIO()
        torch.save(trainer.scheduler.state_dict(), sched_buf)
        (dest / "scheduler.pt").write_bytes(sched_buf.getvalue())

        (dest / "config.json").write_text(json.dumps(trainer.agent.config.model_dump()))

        meta = {"iteration": iteration, "agent_config": trainer.agent.config.model_dump()}
        (dest / "meta.json").write_text(json.dumps(meta))

        logger.info("checkpoint_saved", storage="local", path=str(dest), iteration=iteration)

    def restore_checkpoint(
        self, trainer: AlphaZeroTrainer, prefix: str, load_buffer: bool, load_optimizer: bool
    ) -> None:
        src = self._base / prefix

        if load_buffer:
            trainer.replay_buffer.load_from_bytes((src / "buffer.pkl").read_bytes())
            logger.info("checkpoint_buffer_loaded", size=len(trainer.replay_buffer))

        if load_optimizer:
            trainer.optimizer.load_state_dict(torch.load(src / "optimizer.pt", weights_only=True))
            trainer.scheduler.load_state_dict(torch.load(src / "scheduler.pt", weights_only=False))
            logger.info("checkpoint_optimizer_loaded")

    def download_agent_files(self, prefix: str, tmp_path: Path) -> None:
        src = self._base / prefix
        shutil.copy2(src / "config.json", tmp_path / "config.json")
        shutil.copy2(src / "model.safetensors", tmp_path / "model.safetensors")

    def upload_eval_weights(self, agent: AlphaZeroAgent, iteration: int, execution_id: str) -> str:
        key = f"eval/{execution_id}/iter_{iteration:04d}.safetensors"
        dest = self._resolve(key).parent
        dest.mkdir(parents=True, exist_ok=True)
        weights_bytes = safetensors_save({k: v.cpu() for k, v in agent.model.state_dict().items()})
        (dest / f"iter_{iteration:04d}.safetensors").write_bytes(weights_bytes)
        return key

    def download_eval_weights(self, agent: AlphaZeroAgent, key: str) -> None:
        path = self._base / key
        agent.model.load_state_dict(safetensors_load(path.read_bytes()))

    def cleanup_eval_weights(self, key: str) -> None:
        path = self._base / key
        try:
            path.unlink(missing_ok=True)
        except Exception:
            logger.warning("failed_to_cleanup_eval_weights", key=key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_checkpoint_handler(config: LocalCheckpointStorage | S3CheckpointStorage) -> CheckpointHandler:
    if isinstance(config, LocalCheckpointStorage):
        return LocalCheckpointHandler(config.base_path)
    return S3CheckpointHandler()


# ---------------------------------------------------------------------------
# Prefect task
# ---------------------------------------------------------------------------


@task(name="save-checkpoint", persist_result=False)
def save_checkpoint_task(handler: CheckpointHandler, trainer: AlphaZeroTrainer, prefix: str, iteration: int) -> None:
    handler.save_checkpoint(trainer, prefix, iteration)
