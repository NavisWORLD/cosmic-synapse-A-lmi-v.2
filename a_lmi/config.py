"""Canonical configuration loading for the active A-LMI runtime.

Configuration may be supplied as an already parsed mapping or as a YAML path.
Environment placeholders use ``${NAME}`` or ``${NAME:-default}`` syntax so
active configuration files do not need to contain credentials.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Union

import yaml

ConfigSource = Union[str, Path, MutableMapping[str, Any], Mapping[str, Any]]
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        return match.group(0)

    return _ENV_PATTERN.sub(replace, value)


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _resolve_environment(item)
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = _resolve_environment(item)
        return value
    if isinstance(value, str):
        return _expand_env_string(value)
    return value


def load_config(source: ConfigSource = "infrastructure/config.yaml") -> Mapping[str, Any]:
    """Load the canonical configuration from a mapping or YAML file.

    Existing mutable mappings are returned by identity. This is important for
    callers such as the orchestrator that already parsed and validated a
    configuration before constructing downstream services.
    """

    if isinstance(source, Mapping):
        config = source
    else:
        path = Path(source)
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

    if not isinstance(config, Mapping):
        raise ValueError("A-LMI configuration root must be a mapping")

    # Standard dictionaries/lists are mutable and can be resolved in place.
    _resolve_environment(config)
    return config


def require_config_value(config: Mapping[str, Any], *path: str) -> Any:
    """Return a nested required value with a useful error if it is missing."""

    current: Any = config
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            dotted = ".".join(path)
            raise KeyError(f"Missing required configuration value: {dotted}")
        current = current[part]
    return current
