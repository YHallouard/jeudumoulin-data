from typing import Any, cast

import click
import yaml
from jinja2 import Environment


class ArgumentTypeError(click.ClickException):
    """Custom exception for argument type errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


def render_yaml_with_jinja(yaml_content: str) -> str:
    """Render YAML content with Jinja templating to resolve environment variables."""
    jinja_env = Environment(autoescape=False)  # noqa: S701 # nosec B701 this is required to prevent expression like this > to be converted to gt;
    template = jinja_env.from_string(yaml_content)
    return template.render({})


def yaml_arg(value: str) -> dict[str, Any]:
    """Parse a YAML file with Jinja templating and return a dictionary."""
    try:
        with open(value) as file:
            yaml_content = file.read()
            rendered_yaml = render_yaml_with_jinja(yaml_content)
            yaml_data = yaml.safe_load(rendered_yaml)
            return cast(dict[str, Any], yaml_data)
    except Exception as e:
        raise ArgumentTypeError(message=f"Invalid YAML file: {e}") from e
