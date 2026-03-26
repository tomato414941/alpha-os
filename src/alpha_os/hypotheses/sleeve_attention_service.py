from __future__ import annotations

from dataclasses import dataclass

from .sleeve_control_metrics import load_sleeve_control_metrics


@dataclass(frozen=True)
class SleeveAttentionPlan:
    asset: str
    level: str
    maintenance_lookback_days: int
    rebalance_required: bool


def build_sleeve_attention_plan(*, asset: str, config) -> SleeveAttentionPlan:
    asset = str(asset).upper()
    metrics = load_sleeve_control_metrics(asset=asset)
    is_reference = asset == str(config.cross_sectional.registry_asset).upper()

    if (
        metrics.template_gap_count > 0
        or metrics.coverage_retention < 0.5
        or metrics.capital_conversion < 0.2
        or metrics.breadth_trend < -0.5
    ):
        level = "high"
    elif (
        metrics.coverage_retention < 1.0
        or metrics.capital_conversion < 0.5
        or metrics.breadth_trend < 0.0
    ):
        level = "normal"
    else:
        level = "light"

    if level == "high":
        maintenance_lookback_days = 30
    elif level == "normal":
        maintenance_lookback_days = 21
    else:
        maintenance_lookback_days = 14

    return SleeveAttentionPlan(
        asset=asset,
        level=level,
        maintenance_lookback_days=maintenance_lookback_days,
        rebalance_required=is_reference or level != "light",
    )
