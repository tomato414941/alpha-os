from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import pandas as pd

from .signal_client import build_signal_client

if TYPE_CHECKING:
    from signal_noise.client import SignalClient

    from .portfolio_decision import ObservationSpec


def observation_contract_key(
    observation_spec: "ObservationSpec",
    *,
    asset: str,
) -> str:
    return (
        f"{observation_spec.source_id}:"
        f"{observation_spec.adapter_kind}:"
        f"{str(asset).strip().upper()}:"
        f"{observation_spec.observable_id}:"
        f"{observation_spec.resolution}"
    )


def resolve_observation_metadata(
    observation_spec: "ObservationSpec",
    *,
    asset: str,
    base_url: str,
    client: "SignalClient | None" = None,
) -> dict[str, str | bool | None]:
    signal_client = client or build_signal_client(base_url=base_url)
    return signal_client.resolve_observation(
        asset=asset,
        observable_id=observation_spec.observable_id,
        resolution=observation_spec.resolution,
        source_id=observation_spec.source_id,
    )


def load_observation_frame(
    observation_spec: "ObservationSpec",
    *,
    asset: str,
    base_url: str,
    client: "SignalClient | None" = None,
) -> pd.DataFrame:
    if observation_spec.adapter_kind == "fixture_csv":
        source_path = observation_spec.source_id.format(
            asset=str(asset).strip().upper()
        )
        path = Path(source_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[2] / path
        if not path.exists():
            raise ValueError(f"fixture observation CSV does not exist: {path}")
        return pd.read_csv(path)
    signal_client = client or build_signal_client(base_url=base_url)
    return signal_client.get_observation_data(
        asset=asset,
        observable_id=observation_spec.observable_id,
        resolution=observation_spec.resolution,
        source_id=observation_spec.source_id,
    )
