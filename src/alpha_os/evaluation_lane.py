from __future__ import annotations

from typing import Literal, cast


EvaluationLane = Literal[
    "exploratory",
    "diagnostic",
    "candidate",
    "backtest_oos",
    "operational",
]

EVALUATION_LANES: tuple[EvaluationLane, ...] = (
    "exploratory",
    "diagnostic",
    "candidate",
    "backtest_oos",
    "operational",
)


def normalize_evaluation_lane(lane: str | None) -> EvaluationLane:
    if lane is None:
        return "backtest_oos"
    normalized = str(lane)
    if normalized not in EVALUATION_LANES:
        allowed = ", ".join(EVALUATION_LANES)
        raise ValueError(f"evaluation_lane must be one of: {allowed}")
    return cast(EvaluationLane, normalized)
