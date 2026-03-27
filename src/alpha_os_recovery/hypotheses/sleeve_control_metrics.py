from __future__ import annotations

from dataclasses import dataclass

from ..config import DATA_DIR, HYPOTHESES_DB
from .sleeve_compare_service import load_recent_sleeve_compare_history
from .sleeve_status import build_asset_sleeve_summary
from .store import HypothesisStore

_SLEEVE_COMPARE_SNAPSHOT_PATH = DATA_DIR / "metrics" / "sleeve_compare_reports.jsonl"


@dataclass(frozen=True)
class SleeveControlMetrics:
    asset: str
    template_gap_count: int
    template_gaps: list[str]
    serious_template_backed_count: int
    serious_template_target_count: int
    coverage_retention: float
    capital_conversion: float
    breadth_trend: float


def _clamp_unit(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def load_sleeve_control_metrics(*, asset: str) -> SleeveControlMetrics:
    asset = str(asset).upper()

    store = HypothesisStore(HYPOTHESES_DB)
    try:
        records = store.list_observation_active(asset=asset)
    finally:
        store.close()

    summary = build_asset_sleeve_summary(records)
    history = load_recent_sleeve_compare_history(
        _SLEEVE_COMPARE_SNAPSHOT_PATH,
        limit=2,
    ).get(asset, [])
    previous_row = history[0] if history else {}
    prior_row = history[1] if len(history) > 1 else {}

    previous_template_backed = int(previous_row.get("serious_template_backed", 0))
    if previous_template_backed > 0:
        coverage_retention = _clamp_unit(
            float(summary.serious_template_backed_count) / previous_template_backed
        )
    elif summary.serious_template_target_count > 0:
        coverage_retention = _clamp_unit(
            float(summary.serious_template_backed_count)
            / float(summary.serious_template_target_count)
        )
    else:
        coverage_retention = 1.0

    if summary.actionable_live > 0:
        capital_conversion = _clamp_unit(
            float(summary.capital_backed) / float(summary.actionable_live)
        )
    elif summary.capital_backed > 0:
        capital_conversion = 1.0
    else:
        capital_conversion = 0.0

    previous_breadth = float(previous_row.get("breadth", 0.0))
    prior_breadth = float(prior_row.get("breadth", previous_breadth))

    return SleeveControlMetrics(
        asset=asset,
        template_gap_count=len(summary.serious_template_gaps),
        template_gaps=list(summary.serious_template_gaps),
        serious_template_backed_count=summary.serious_template_backed_count,
        serious_template_target_count=summary.serious_template_target_count,
        coverage_retention=coverage_retention,
        capital_conversion=capital_conversion,
        breadth_trend=previous_breadth - prior_breadth,
    )
