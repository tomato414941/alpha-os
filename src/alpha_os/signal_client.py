from __future__ import annotations

import os
from pathlib import Path

try:
    from signal_noise.client import SignalClient
except ModuleNotFoundError:
    SignalClient = None  # type: ignore[assignment]

from .config import DEFAULT_SIGNAL_NOISE_BASE_URL

SIGNAL_NOISE_API_KEY_ENV = "ALPHA_OS_SIGNAL_NOISE_API_KEY"
SIGNAL_NOISE_TIMEOUT_ENV = "ALPHA_OS_SIGNAL_NOISE_TIMEOUT"
SIGNAL_NOISE_RETRY_COUNT_ENV = "ALPHA_OS_SIGNAL_NOISE_RETRY_COUNT"
SIGNAL_NOISE_RETRY_BACKOFF_ENV = "ALPHA_OS_SIGNAL_NOISE_RETRY_BACKOFF"
_SECRETS_FILE = Path.home() / ".secrets" / "alpha-os-env"


def signal_noise_api_key() -> str | None:
    api_key = os.getenv(SIGNAL_NOISE_API_KEY_ENV, "").strip()
    if api_key:
        return api_key
    if _SECRETS_FILE.exists():
        for line in _SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if item.startswith(f"{SIGNAL_NOISE_API_KEY_ENV}="):
                value = item.split("=", 1)[1].strip()
                if value:
                    return value
    return None


def _env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _env_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _default_signal_noise_timeout(base_url: str) -> int:
    normalized = str(base_url).strip().lower()
    if (
        "localhost" in normalized
        or "127.0.0.1" in normalized
        or normalized.startswith("http://0.0.0.0")
    ):
        return 30
    return 90


def build_signal_client(
    *,
    base_url: str = DEFAULT_SIGNAL_NOISE_BASE_URL,
    timeout: int | None = None,
    retry_count: int | None = None,
    retry_backoff: float | None = None,
) -> SignalClient:
    if SignalClient is None:
        raise RuntimeError(
            "signal-noise is required for data service access. "
            'Install alpha-os with the "data" extra to enable SignalClient integration.'
        )
    resolved_timeout = (
        timeout
        if timeout is not None
        else _env_int(SIGNAL_NOISE_TIMEOUT_ENV) or _default_signal_noise_timeout(base_url)
    )
    resolved_retry_count = (
        retry_count
        if retry_count is not None
        else _env_int(SIGNAL_NOISE_RETRY_COUNT_ENV) or 2
    )
    resolved_retry_backoff = (
        retry_backoff
        if retry_backoff is not None
        else _env_float(SIGNAL_NOISE_RETRY_BACKOFF_ENV) or 1.0
    )
    return SignalClient(
        base_url=base_url,
        timeout=resolved_timeout,
        retry_count=resolved_retry_count,
        retry_backoff=resolved_retry_backoff,
        api_key=signal_noise_api_key(),
    )
