import uuid
from collections.abc import Callable
from typing import Any

from prefect.context import get_run_context
from prefect.deployments import run_deployment as _prefect_run_deployment
from prefect.exceptions import MissingContextError


def get_execution_id() -> str:
    """Return the Prefect flow run ID, or a random UUID when running outside Prefect."""
    try:
        return str(get_run_context().flow_run.id)
    except MissingContextError:
        return str(uuid.uuid4())


def run_deployment(
    flow_fn: Callable[..., Any],
    deployment_name: str,
    parameters: dict[str, Any],
    timeout: int = 0,
) -> Any:
    """Run a Prefect deployment if inside a Prefect context, otherwise call the flow directly.

    This allows flows to be triggered locally (CLI, tests) without a running Prefect server.

    Args:
        flow_fn: The flow function to call directly when outside a Prefect context.
        deployment_name: The deployment name used when inside a Prefect context.
        parameters: Parameters dict passed to the deployment or unpacked as kwargs.
        timeout: Passed to ``run_deployment`` (0 = fire-and-forget).
    """
    try:
        get_run_context()
        return _prefect_run_deployment(name=deployment_name, parameters=parameters, timeout=timeout)
    except MissingContextError:
        return flow_fn(**parameters)
