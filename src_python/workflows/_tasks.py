from __future__ import annotations

from typing import Any

import torch
import yaml
from jinja2 import Environment
from prefect import task


# NOTE: Not used anymore, but kept because i'm still thinking
@task(name="load-config")
def load_training_config(config_path: str) -> dict[str, Any]:
    with open(config_path) as f:
        yaml_content = f.read()
    jinja_env = Environment(autoescape=False)  # noqa: S701
    template = jinja_env.from_string(yaml_content)
    rendered = template.render({})
    return dict(yaml.safe_load(rendered)["config"])


@task(name="detect-device")
def detect_compute_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
