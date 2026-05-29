from __future__ import annotations

from typing import Literal, cast


PortfolioDirectionMode = Literal["long_short", "long_only", "short_only"]

PORTFOLIO_DIRECTION_MODES: tuple[PortfolioDirectionMode, ...] = (
    "long_short",
    "long_only",
    "short_only",
)


def normalize_portfolio_direction_mode(
    direction_mode: str | None,
) -> PortfolioDirectionMode:
    if direction_mode is None:
        return "long_short"
    normalized = str(direction_mode)
    if normalized not in PORTFOLIO_DIRECTION_MODES:
        allowed = ", ".join(PORTFOLIO_DIRECTION_MODES)
        raise ValueError(f"portfolio direction_mode must be one of: {allowed}")
    return cast(PortfolioDirectionMode, normalized)
