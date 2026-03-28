from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    DEFAULT_ASSET,
    DEFAULT_HORIZON_DAYS,
    DEFAULT_PRICE_SIGNAL,
    DEFAULT_TARGET,
    target_name_for_horizon,
)


@dataclass(frozen=True)
class HypothesisDefinition:
    hypothesis_id: str
    kind: str
    signal_name: str
    lookback: int
    horizon_days: int = DEFAULT_HORIZON_DAYS
    asset: str = DEFAULT_ASSET
    target: str = DEFAULT_TARGET

    def to_document(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "signal_name": self.signal_name,
            "params": {
                "lookback": self.lookback,
                "horizon_days": self.horizon_days,
            },
        }

    @classmethod
    def from_document(
        cls,
        *,
        hypothesis_id: str,
        document: dict[str, Any],
        asset: str = DEFAULT_ASSET,
        target: str = DEFAULT_TARGET,
    ) -> "HypothesisDefinition":
        kind = document.get("kind")
        signal_name = document.get("signal_name")
        params = document.get("params")
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"hypothesis document is missing kind: {hypothesis_id}")
        if not isinstance(signal_name, str) or not signal_name:
            raise ValueError(f"hypothesis document is missing signal_name: {hypothesis_id}")
        if not isinstance(params, dict):
            raise ValueError(f"hypothesis document is missing params: {hypothesis_id}")
        lookback = params.get("lookback")
        if not isinstance(lookback, int):
            raise ValueError(f"hypothesis document is missing integer lookback: {hypothesis_id}")
        horizon_days = params.get("horizon_days")
        if not isinstance(horizon_days, int):
            raise ValueError(
                f"hypothesis document is missing integer horizon_days: {hypothesis_id}"
            )
        return cls(
            hypothesis_id=hypothesis_id,
            kind=kind,
            signal_name=signal_name,
            lookback=lookback,
            horizon_days=horizon_days,
            asset=asset,
            target=target_name_for_horizon(horizon_days),
        )


_DEFINITIONS = {
    "momentum_1d": HypothesisDefinition(
        hypothesis_id="momentum_1d",
        kind="momentum",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=1,
    ),
    "momentum_3d": HypothesisDefinition(
        hypothesis_id="momentum_3d",
        kind="momentum",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=3,
    ),
    "momentum_5d": HypothesisDefinition(
        hypothesis_id="momentum_5d",
        kind="momentum",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=5,
    ),
    "reversal_1d": HypothesisDefinition(
        hypothesis_id="reversal_1d",
        kind="reversal",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=1,
    ),
    "reversal_3d": HypothesisDefinition(
        hypothesis_id="reversal_3d",
        kind="reversal",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=3,
    ),
    "reversal_5d": HypothesisDefinition(
        hypothesis_id="reversal_5d",
        kind="reversal",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=5,
    ),
    "average_gap_3d": HypothesisDefinition(
        hypothesis_id="average_gap_3d",
        kind="average_gap",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=3,
    ),
    "average_gap_5d": HypothesisDefinition(
        hypothesis_id="average_gap_5d",
        kind="average_gap",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=5,
    ),
    "range_position_5d": HypothesisDefinition(
        hypothesis_id="range_position_5d",
        kind="range_position",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=5,
    ),
}


def get_hypothesis_definition(hypothesis_id: str) -> HypothesisDefinition:
    try:
        return _DEFINITIONS[hypothesis_id]
    except KeyError as exc:
        available = ", ".join(sorted(_DEFINITIONS))
        raise ValueError(
            f"unknown hypothesis definition: {hypothesis_id} "
            f"(available: {available})"
        ) from exc


def find_hypothesis_definition(hypothesis_id: str) -> HypothesisDefinition | None:
    return _DEFINITIONS.get(hypothesis_id)
