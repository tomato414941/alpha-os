from __future__ import annotations

import os

from signal_noise.client import SignalClient

from ..config import APIConfig

SIGNAL_NOISE_API_KEY_ENV = "ALPHA_OS_SIGNAL_NOISE_API_KEY"


def signal_noise_api_key() -> str | None:
    api_key = os.getenv(SIGNAL_NOISE_API_KEY_ENV, "").strip()
    return api_key or None


def build_signal_client(*, base_url: str, timeout: int) -> SignalClient:
    return SignalClient(
        base_url=base_url,
        timeout=timeout,
        api_key=signal_noise_api_key(),
    )


def build_signal_client_from_config(api: APIConfig) -> SignalClient:
    return build_signal_client(base_url=api.base_url, timeout=api.timeout)
