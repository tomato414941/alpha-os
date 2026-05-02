"""Small allocation rules that should not depend on rich portfolio construction.

This module exists to keep simple strategy-side portfolio choices explicit:
given current position directions, produce target weights. It intentionally does
not own rebalance cadence, execution costs, risk budgets, target volatility,
group caps, or legacy `PortfolioConstructionSpec` compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


_PositionDirection = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class EqualWeightLongOnlyAllocator:
    gross_exposure_cap: float = 1.0

    def __post_init__(self) -> None:
        if self.gross_exposure_cap < 0.0:
            raise ValueError("gross_exposure_cap must be non-negative")

    def allocate(
        self,
        directions_by_subject: dict[str, _PositionDirection],
    ) -> dict[str, float]:
        long_subject_ids = tuple(
            subject_id
            for subject_id, direction in directions_by_subject.items()
            if direction == "long"
        )
        if not long_subject_ids:
            return {subject_id: 0.0 for subject_id in directions_by_subject}

        weight = self.gross_exposure_cap / float(len(long_subject_ids))
        return {
            subject_id: weight if subject_id in long_subject_ids else 0.0
            for subject_id in directions_by_subject
        }


__all__ = ("EqualWeightLongOnlyAllocator",)
