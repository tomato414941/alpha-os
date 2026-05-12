from __future__ import annotations

from typing import Literal


StrategyRunMode = Literal[
    "backtest_oos",
    "paper",
    "live",
]


def normalize_strategy_run_mode(mode: str | None) -> StrategyRunMode:
    if mode is None:
        return "backtest_oos"
    if mode not in {"backtest_oos", "paper", "live"}:
        raise ValueError(f"unsupported strategy run mode: {mode}")
    return mode
