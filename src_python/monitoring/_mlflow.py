import json
from inspect import Traceback
from pathlib import Path
from typing import Any

import mlflow
import structlog

logger = structlog.get_logger(__name__)


class MLflowLogger:
    def __init__(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tracking_uri: str = "http://localhost:5001",
        tags: dict[str, str] | None = None,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._active = False

    def start(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
            self._active = True
        except Exception:
            self._active = False
            logger.exception("Failed to start mlflow logger")

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._active:
            return
        try:
            mlflow.log_params(params)
        except Exception:
            logger.exception("Failed to log params")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._active:
            logger.warning("mlflow_not_active_skipping_metrics", metrics=metrics)
            return
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception:
            logger.exception("Failed to log metrics", metrics=metrics, step=step)

    def log_config_artifact(self, config: dict[str, Any], filename: str = "config.json") -> None:
        if not self._active:
            return
        try:
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / filename
                with open(temp_file, "w") as f:
                    json.dump(config, f, indent=4)
                mlflow.log_artifact(str(temp_file), artifact_path="config")
        except Exception:
            logger.exception("Failed to log config artifact")

    def set_tag(self, key: str, value: str) -> None:
        if not self._active:
            return
        try:
            mlflow.set_tag(key, value)
        except Exception:
            logger.exception("Failed to set tag")

    def register_model(self, model_dir: str | Path, registered_model_name: str) -> None:
        if not self._active:
            return
        try:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            model_uri = f"runs:/{self.run.info.run_id}/model"
            mlflow.register_model(model_uri, registered_model_name)
            logger.info(
                "model_registered",
                model_name=registered_model_name,
                run_id=self.run.info.run_id,
            )
        except Exception:
            logger.exception("Failed to register model")

    def finish(self, status: str = "FINISHED") -> None:
        if not self._active:
            return
        try:
            mlflow.end_run(status=status)
            self._active = False
        except Exception:
            logger.exception("Failed to finish")

    def __enter__(self) -> "MLflowLogger":
        self.start()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Traceback | None
    ) -> None:
        if exc_type is KeyboardInterrupt:
            self.finish(status="KILLED")
        elif exc_type is not None:
            self.finish(status="FAILED")
        else:
            self.finish(status="FINISHED")
