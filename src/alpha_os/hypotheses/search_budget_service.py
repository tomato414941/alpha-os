from __future__ import annotations

from dataclasses import dataclass

from .serious_templates import serious_seed_specs
from .sleeve_control_metrics import load_sleeve_control_metrics


@dataclass(frozen=True)
class SleeveSearchBudget:
    asset: str
    requested_limit: int
    effective_limit: int
    missing_template_count: int
    closed_template_count: int
    new_template_count: int
    coverage_retention: float
    capital_conversion: float
    breadth_trend: float


def build_template_gap_search_budget(
    *,
    asset: str,
    base_limit: int,
    previous_template_gaps: list[str] | None = None,
) -> SleeveSearchBudget:
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
            closed_template_count=0,
            new_template_count=0,
            coverage_retention=1.0,
            capital_conversion=0.0,
            breadth_trend=0.0,
        )

    if not serious_seed_specs(asset):
        return SleeveSearchBudget(
            asset=asset,
            requested_limit=requested_limit,
            effective_limit=requested_limit,
            missing_template_count=0,
            closed_template_count=0,
            new_template_count=0,
            coverage_retention=1.0,
            capital_conversion=0.0,
            breadth_trend=0.0,
        )

    metrics = load_sleeve_control_metrics(asset=asset)
    missing_template_count = int(metrics.template_gap_count)
    current_template_gaps = {
        str(item) for item in metrics.template_gaps if str(item)
    }
    previous_gap_set = {
        str(item) for item in (previous_template_gaps or []) if str(item)
    }
    closed_template_count = len(previous_gap_set - current_template_gaps)
    new_template_count = len(current_template_gaps - previous_gap_set)
    if (
        missing_template_count <= 0
        and metrics.coverage_retention >= 1.0
        and metrics.capital_conversion >= 0.5
        and metrics.breadth_trend >= 0.0
    ):
        effective_limit = 0
    else:
        effective_limit = max(0, missing_template_count * 2)
        if missing_template_count > 0:
            effective_limit = max(4, effective_limit)

        if metrics.coverage_retention < 0.5:
            effective_limit += 3
        elif metrics.coverage_retention < 0.85:
            effective_limit += 2
        elif metrics.coverage_retention < 1.0:
            effective_limit += 1

        if metrics.capital_conversion < 0.2:
            effective_limit += 2
        elif metrics.capital_conversion < 0.5:
            effective_limit += 1

        if metrics.breadth_trend < -0.5:
            effective_limit += 2
        elif metrics.breadth_trend < 0.0:
            effective_limit += 1

        if previous_gap_set and closed_template_count == 0 and missing_template_count > 0:
            effective_limit += 1

        if missing_template_count <= 0 and effective_limit > 0:
            effective_limit = max(2, effective_limit)

        effective_limit = min(requested_limit, effective_limit)

    return SleeveSearchBudget(
        asset=asset,
        requested_limit=requested_limit,
        effective_limit=effective_limit,
        missing_template_count=missing_template_count,
        closed_template_count=closed_template_count,
        new_template_count=new_template_count,
        coverage_retention=metrics.coverage_retention,
        capital_conversion=metrics.capital_conversion,
        breadth_trend=metrics.breadth_trend,
    )
