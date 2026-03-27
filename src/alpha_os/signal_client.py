from __future__ import annotations

import os
from pathlib import Path

from signal_noise.client import SignalClient

from .config import DEFAULT_SIGNAL_NOISE_BASE_URL

SIGNAL_NOISE_API_KEY_ENV = "ALPHA_OS_SIGNAL_NOISE_API_KEY"
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


def build_signal_client(*, base_url: str = DEFAULT_SIGNAL_NOISE_BASE_URL, timeout: int = 30) -> SignalClient:
    return SignalClient(
        base_url=base_url,
        timeout=timeout,
        api_key=signal_noise_api_key(),
    )
