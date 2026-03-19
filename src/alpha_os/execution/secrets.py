"""Shared secrets loading for all execution venues."""
from __future__ import annotations

import os
from pathlib import Path


def load_secrets(name: str) -> dict[str, str]:
    """Load key-value pairs from ~/.secrets/{name}.

    Supports formats:
        export KEY='value'
        KEY=value
    """
    secret_path = Path.home() / ".secrets" / name
    if not secret_path.exists():
        return {}
    result: dict[str, str] = {}
    for line in secret_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, val = line.split("=", 1)
            result[key.strip()] = val.strip().strip("'\"")
    return result


def get_env_or_secret(
    env_keys: list[str],
    secrets_file: str,
) -> dict[str, str]:
    """Get credentials from env vars first, then fall back to secrets file.

    Returns a dict mapping env_keys to their resolved values.
    """
    result: dict[str, str] = {}
    all_from_env = True
    for key in env_keys:
        val = os.environ.get(key, "")
        if val:
            result[key] = val
        else:
            all_from_env = False

    if all_from_env:
        return result

    secrets = load_secrets(secrets_file)
    for key in env_keys:
        if key not in result or not result[key]:
            result[key] = secrets.get(key, "")
    return result
