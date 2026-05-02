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

    def __post_init__(self) -> None:
        if self.gross_exposure_cap < 0.0:
            raise ValueError("gross_exposure_cap must be non-negative")

    def allocate(
        self,
        candidates: tuple[PositionCandidate, ...],
    ) -> PortfolioAllocation:
        long_subject_ids = tuple(
            candidate.subject_id
            for candidate in candidates
            if candidate.direction == "long"
        )
        if not long_subject_ids:
            return PortfolioAllocation(
                target_weights={
                    candidate.subject_id: 0.0
                    for candidate in candidates
                }
            )

        weight = self.gross_exposure_cap / float(len(long_subject_ids))
        return PortfolioAllocation(
            target_weights={
                candidate.subject_id: (
                    weight if candidate.subject_id in long_subject_ids else 0.0
                )
                for candidate in candidates
            }
        )

