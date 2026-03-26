from __future__ import annotations

from collections import Counter

import numpy as np

from .identity import expression_feature_families


def actionable_live_drop_reason(
    record,
    *,
    signal_nonzero_ratio_min: float,
    signal_mean_abs_min: float,
) -> str:
    metadata = getattr(record, "metadata", {}) or {}
    if not bool(metadata.get("lifecycle_live_proven", False)):
        return "not_live_proven"
    try:
        stake = float(record.stake)
    except (TypeError, ValueError):
        stake = 0.0
    if bool(metadata.get("lifecycle_actionable_live", False)):
        if stake > 0 and bool(metadata.get("lifecycle_capital_eligible", stake > 0)):
            return "backed"
        if str(metadata.get("lifecycle_redundancy_capped_by") or ""):
            return "redundancy"
        return "actionable_unbacked"

    signal_ratio = float(metadata.get("lifecycle_signal_nonzero_ratio", 0.0) or 0.0)
    signal_mean_abs = float(metadata.get("lifecycle_signal_mean_abs", 0.0) or 0.0)
    weak_ratio = signal_ratio < float(signal_nonzero_ratio_min)
    weak_mean_abs = signal_mean_abs < float(signal_mean_abs_min)
    if weak_ratio and weak_mean_abs:
        return "signal_both"
    if weak_ratio:
        return "signal_ratio"
    if weak_mean_abs:
        return "signal_mean_abs"
    return "other"


def _distribution_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def build_actionable_live_summary(
    records,
    *,
    top: int = 5,
    signal_nonzero_ratio_min: float,
    signal_mean_abs_min: float,
    families: tuple[str, ...] | None = None,
) -> dict[str, object]:
    family_filter = set(families or ())
    filtered = []
    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        if not bool(metadata.get("lifecycle_live_proven", False)):
            continue
        record_families = set(expression_feature_families(record.expression))
        if family_filter and not (record_families & family_filter):
            continue
        filtered.append(record)

    reasons: Counter[str] = Counter()
    family_counts: dict[str, Counter[str]] = {}
    ranked: list[tuple[float, float, object, str, str]] = []
    ratio_drop_values: list[float] = []
    mean_abs_drop_values: list[float] = []
    backed_ratio_values: list[float] = []
    backed_mean_abs_values: list[float] = []

    actionable = 0
    backed = 0

    for record in filtered:
        metadata = getattr(record, "metadata", {}) or {}
        record_families = set(expression_feature_families(record.expression)) or {"unknown"}
        family_label = ",".join(sorted(record_families))
        reason = actionable_live_drop_reason(
            record,
            signal_nonzero_ratio_min=signal_nonzero_ratio_min,
            signal_mean_abs_min=signal_mean_abs_min,
        )
        reasons[reason] += 1
        counter = family_counts.setdefault(reason, Counter())
        for family in record_families:
            counter[family] += 1

        signal_ratio = float(metadata.get("lifecycle_signal_nonzero_ratio", 0.0) or 0.0)
        signal_mean_abs = float(metadata.get("lifecycle_signal_mean_abs", 0.0) or 0.0)
        ranked.append((signal_ratio, signal_mean_abs, record, reason, family_label))

        if bool(metadata.get("lifecycle_actionable_live", False)):
            actionable += 1
            try:
                stake = float(record.stake)
            except (TypeError, ValueError):
                stake = 0.0
            if stake > 0:
                backed += 1
                backed_ratio_values.append(signal_ratio)
                backed_mean_abs_values.append(signal_mean_abs)
        else:
            ratio_drop_values.append(signal_ratio)
            mean_abs_drop_values.append(signal_mean_abs)

    ranked.sort(key=lambda item: (item[0], item[1], item[2].hypothesis_id), reverse=True)
    top_entries = [
        (
            f"{record.hypothesis_id} reason={reason} "
            f"sig={signal_ratio:.2f}/{signal_mean_abs:.2f} fam={family_label}"
        )
        for signal_ratio, signal_mean_abs, record, reason, family_label in ranked[: max(top, 0)]
    ]
    family_summary = {
        reason: [f"{family}:{count}" for family, count in counts.most_common(3)]
        for reason, counts in family_counts.items()
    }

    return {
        "total": len(filtered),
        "actionable": actionable,
        "backed": backed,
        "reasons": reasons,
        "families": family_summary,
        "ratio_drop": _distribution_summary(ratio_drop_values),
        "mean_abs_drop": _distribution_summary(mean_abs_drop_values),
        "ratio_backed": _distribution_summary(backed_ratio_values),
        "mean_abs_backed": _distribution_summary(backed_mean_abs_values),
        "signal_nonzero_ratio_min": float(signal_nonzero_ratio_min),
        "signal_mean_abs_min": float(signal_mean_abs_min),
        "top_entries": top_entries,
    }
