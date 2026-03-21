from __future__ import annotations

import os
from pathlib import Path

from signal_noise.client import SignalClient

from ..config import APIConfig

SIGNAL_NOISE_API_KEY_ENV = "ALPHA_OS_SIGNAL_NOISE_API_KEY"
_SECRETS_FILE = Path.home() / ".secrets" / "alpha-os-env"


def signal_noise_api_key() -> str | None:
    # 1. Environment variable (systemd EnvironmentFile sets this)
    api_key = os.getenv(SIGNAL_NOISE_API_KEY_ENV, "").strip()
    if api_key:
        return api_key
    # 2. Read directly from secrets file (CLI / manual execution)
    if _SECRETS_FILE.exists():
        for line in _SECRETS_FILE.read_text().strip().splitlines():
            line = line.strip()
            if line.startswith(f"{SIGNAL_NOISE_API_KEY_ENV}="):
                api_key = line.split("=", 1)[1].strip()
                if api_key:
                    return api_key
    return None


def build_signal_client(*, base_url: str, timeout: int) -> SignalClient:
    return SignalClient(
        base_url=base_url,
        timeout=timeout,
        api_key=signal_noise_api_key(),
    )


def build_signal_client_from_config(api: APIConfig) -> SignalClient:
    return build_signal_client(base_url=api.base_url, timeout=api.timeout)
