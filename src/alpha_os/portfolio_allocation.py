from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PositionDirection = Literal["long", "short", "flat"]


@dataclass(frozen=True)
class PositionCandidate:
    subject_id: str
    direction: PositionDirection


@dataclass(frozen=True)
class PortfolioAllocation:
    target_weights: dict[str, float]


@dataclass(frozen=True)
class EqualWeightLongOnlyAllocator:
    gross_exposure_cap: float = 1.0

    def allocate(
        self,
        candidates: tuple[PositionCandidate, ...],
    ) -> PortfolioAllocation:
        target_weights = {item.subject_id: 0.0 for item in candidates}
        long_candidates = tuple(item for item in candidates if item.direction == "long")
        if not long_candidates:
            return PortfolioAllocation(target_weights=target_weights)

        gross_cap = max(float(self.gross_exposure_cap), 0.0)
        weight = gross_cap / float(len(long_candidates))
        for item in long_candidates:
            target_weights[item.subject_id] = weight
        return PortfolioAllocation(target_weights=target_weights)

