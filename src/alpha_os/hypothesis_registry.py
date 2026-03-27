from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_ASSET, DEFAULT_PRICE_SIGNAL, DEFAULT_TARGET


@dataclass(frozen=True)
class HypothesisDefinition:
    hypothesis_id: str
    kind: str
    signal_name: str
    lookback: int
    asset: str = DEFAULT_ASSET
    target: str = DEFAULT_TARGET


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
    "reversal_1d": HypothesisDefinition(
        hypothesis_id="reversal_1d",
        kind="reversal",
        signal_name=DEFAULT_PRICE_SIGNAL,
        lookback=1,
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
