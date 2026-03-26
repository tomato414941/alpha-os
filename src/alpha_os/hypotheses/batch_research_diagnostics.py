from __future__ import annotations

from collections import Counter

import numpy as np

from .allocation_policy import normalized_research_quality
from .identity import expression_feature_families


def is_batch_research_record(record) -> bool:
    metadata = getattr(record, "metadata", {}) or {}
    source = str(
        metadata.get("research_quality_source")
        or metadata.get("lifecycle_research_quality_source")
        or ""
    )
    status = str(metadata.get("research_quality_status", ""))
    return record.source == "random_dsl" and (
        source == "batch_research_score" or status == "scored"
    )


def batch_research_drop_reason(record) -> str:
    metadata = getattr(record, "metadata", {}) or {}
    try:
        stake = float(record.stake)
    except (TypeError, ValueError):
        stake = 0.0
    capital_backed = bool(
        metadata.get(
            "lifecycle_capital_backed",
            bool(metadata.get("lifecycle_capital_eligible", stake > 0)) and stake > 0,
        )
    )
    if capital_backed:
        return "backed"
    if str(metadata.get("lifecycle_redundancy_capped_by") or ""):
        return "redundancy"
    if bool(metadata.get("lifecycle_research_candidate_capped", False)):
        return "candidate_cap"
    if not bool(metadata.get("lifecycle_research_backed", False)):
        return "research_quality"
    blocker = str(metadata.get("lifecycle_live_promotion_blocker", ""))
    if blocker == "insufficient_observations":
        return "observation"
    if blocker == "weak_signal_activity":
        return "signal"
    if blocker in {"weak_marginal_contribution", "weak_live_quality_and_contribution"}:
        return "contribution"
    if blocker == "weak_live_quality":
        return "live_quality"
    return "other"


def distribution_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def format_distribution_line(label: str, summary: dict[str, float]) -> str:
    return (
        f"{label}:n={int(summary['count'])}"
        f"/p50={summary['p50']:.2f}"
        f"/p90={summary['p90']:.2f}"
        f"/max={summary['max']:.2f}"
    )


def build_batch_research_summary(
    records,
    *,
    top: int = 5,
    quality_min: float = 0.10,
    families: tuple[str, ...] | None = None,
) -> dict[str, object]:
    family_filter = set(families or ())
    filtered = [
        record
        for record in records
        if is_batch_research_record(record)
        and (
            not family_filter
            or bool(set(expression_feature_families(record.expression)) & family_filter)
        )
    ]

    reason_counts: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {}
    ranked: list[tuple[float, object, str, str]] = []
    near_miss: list[tuple[float, object, str]] = []
    retained = 0
    actionable = 0
    backed = 0
    quality_dropped_norms: list[float] = []
    quality_dropped_sharpes: list[float] = []
    quality_dropped_log_growths: list[float] = []
    quality_dropped_folds: list[float] = []
    backed_norms: list[float] = []
    family_norms: dict[str, list[float]] = {}
    quality_family_norms: dict[str, list[float]] = {}

    for record in filtered:
        metadata = getattr(record, "metadata", {}) or {}
        if bool(metadata.get("lifecycle_research_retained", False)):
            retained += 1
        if bool(metadata.get("lifecycle_actionable_live", False)):
            actionable += 1
        try:
            stake = float(record.stake)
        except (TypeError, ValueError):
            stake = 0.0
        if bool(
            metadata.get(
                "lifecycle_capital_backed",
                bool(metadata.get("lifecycle_capital_eligible", stake > 0)) and stake > 0,
            )
        ):
            backed += 1
        reason = batch_research_drop_reason(record)
        reason_counts[reason] += 1
        record_families = set(expression_feature_families(record.expression)) or {"unknown"}
        counter = family_counts.setdefault(reason, Counter())
        for family in record_families:
            counter[family] += 1
        norm_quality = normalized_research_quality(record.oos_fitness("sharpe"), metric="sharpe")
        for family in record_families:
            family_norms.setdefault(family, []).append(norm_quality)
        family_label = ",".join(sorted(record_families))
        ranked.append((norm_quality, record, reason, family_label))
        if reason == "backed":
            backed_norms.append(norm_quality)
        if reason in {"research_quality", "live_quality"}:
            quality_dropped_norms.append(norm_quality)
            quality_dropped_sharpes.append(record.oos_fitness("sharpe"))
            quality_dropped_log_growths.append(record.oos_fitness("log_growth"))
            for family in record_families:
                quality_family_norms.setdefault(family, []).append(norm_quality)
            try:
                quality_dropped_folds.append(float(metadata.get("research_score_n_folds", 0)))
            except (TypeError, ValueError):
                pass
            if reason == "research_quality" and norm_quality >= max(float(quality_min) - 0.02, 0.0):
                near_miss.append((norm_quality, record, family_label))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top_entries = [
        f"{record.hypothesis_id} reason={reason} q={quality:.2f} fam={family_label}"
        for quality, record, reason, family_label in ranked[: max(top, 0)]
    ]
    near_miss.sort(key=lambda item: item[0], reverse=True)
    near_miss_entries = [
        f"{record.hypothesis_id} q={quality:.2f} fam={family_label}"
        for quality, record, family_label in near_miss[: max(top, 0)]
    ]
    family_summary = {
        reason: [f"{family}:{count}" for family, count in counts.most_common(3)]
        for reason, counts in family_counts.items()
    }
    family_quality_summary = sorted(
        (
            distribution_summary(values)["p90"],
            family,
            distribution_summary(values),
            distribution_summary(quality_family_norms.get(family, [])),
        )
        for family, values in family_norms.items()
    )
    family_quality_lines = [
        f"{family}({format_distribution_line('all', all_summary)};"
        f"{format_distribution_line('drop', drop_summary)})"
        for _, family, all_summary, drop_summary in reversed(family_quality_summary[: max(top, 0)])
    ]
    return {
        "total": len(filtered),
        "retained": retained,
        "actionable": actionable,
        "backed": backed,
        "reasons": reason_counts,
        "families": family_summary,
        "top_entries": top_entries,
        "quality_threshold": float(quality_min),
        "quality_drop_norm": distribution_summary(quality_dropped_norms),
        "quality_drop_sharpe": distribution_summary(quality_dropped_sharpes),
        "quality_drop_log_growth": distribution_summary(quality_dropped_log_growths),
        "quality_drop_folds": distribution_summary(quality_dropped_folds),
        "backed_norm": distribution_summary(backed_norms),
        "near_miss_entries": near_miss_entries,
        "family_quality_lines": family_quality_lines,
    }
