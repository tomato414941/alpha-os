from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping


StrategyExecutionKind = Literal["trainless", "trained", "frozen"]


@dataclass(frozen=True)
class StrategyExecutionSpec:
    kind: StrategyExecutionKind
    signal_discovery_id: str | None
    requires_signal_train: bool
    retrains_per_fold: bool
    reuses_frozen_state: bool


def resolve_strategy_execution_spec(
    axis_map: Mapping[str, str],
) -> StrategyExecutionSpec:
    signal_discovery_id = axis_map.get("signal_discovery") or None
    explicit = axis_map.get("execution_kind")
    if explicit:
        if explicit not in {"trainless", "trained", "frozen"}:
            raise ValueError(f"unsupported strategy execution_kind: {explicit}")
        kind: StrategyExecutionKind = explicit
    elif signal_discovery_id is not None:
        kind = "trained"
    elif axis_map.get("signal") in {
        "signal_discovery",
        "neural_model",
        "trained_model",
    }:
        kind = "trained"
    else:
        kind = "trainless"
    return StrategyExecutionSpec(
        kind=kind,
        signal_discovery_id=signal_discovery_id,
        requires_signal_train=(kind == "trained"),
        retrains_per_fold=(kind == "trained"),
        reuses_frozen_state=(kind == "frozen"),
    )
