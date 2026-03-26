from __future__ import annotations

from dataclasses import dataclass

from ..config import HYPOTHESES_DB
from .serious_templates import serious_seed_specs, serious_template_gap_scores
from .store import HypothesisStore


@dataclass(frozen=True)
class SleeveSearchBudget:
    asset: str
    requested_limit: int
    effective_limit: int
    missing_template_count: int


def build_template_gap_search_budget(*, asset: str, base_limit: int) -> SleeveSearchBudget:
    asset = str(asset).upper()
    try:
        requested_limit = int(base_limit)
    except (TypeError, ValueError):
        requested_limit = 0
    if requested_limit <= 0:
        return SleeveSearchBudget(
            asset=asset,
            requested_limit=0,
            effective_limit=0,
            missing_template_count=0,
        )

    if not serious_seed_specs(asset):
        return SleeveSearchBudget(
            asset=asset,
            requested_limit=requested_limit,
            effective_limit=requested_limit,
            missing_template_count=0,
        )

    store = HypothesisStore(HYPOTHESES_DB)
    try:
        gaps = serious_template_gap_scores(
            asset,
            store.list_observation_active(asset=asset),
        )
    finally:
        store.close()

    missing_template_count = sum(1 for gap in gaps.values() if float(gap) > 0.0)
    if missing_template_count <= 0:
        effective_limit = 0
    else:
        effective_limit = min(requested_limit, max(4, missing_template_count * 2))

    return SleeveSearchBudget(
        asset=asset,
        requested_limit=requested_limit,
        effective_limit=effective_limit,
        missing_template_count=missing_template_count,
    )
