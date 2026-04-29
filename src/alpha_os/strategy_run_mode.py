from __future__ import annotations

from typing import Literal


StrategyRunMode = Literal[
    "backtest_oos",
    "fixed_state_replay",
    "paper",
    "live",
]


def normalize_strategy_run_mode(mode: str | None) -> StrategyRunMode:
    if mode is None:
        return "backtest_oos"
    if mode not in {"backtest_oos", "fixed_state_replay", "paper", "live"}:
        raise ValueError(f"unsupported strategy run mode: {mode}")
    return mode


def run_mode_requires_fixed_initial_strategy_state(mode: StrategyRunMode) -> bool:
    return mode == "fixed_state_replay"
