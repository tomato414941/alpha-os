from __future__ import annotations

from collections import Counter

import numpy as np

from .identity import expression_feature_families, representative_feature_family


def _representative_family(record) -> str:
    metadata = getattr(record, "metadata", {}) or {}
    family = str(metadata.get("lifecycle_representative_family", "")).strip()
    if family:
        return family
    families = set(expression_feature_families(record.expression)) or {"unknown"}
    return representative_feature_family(families)


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
    capital_backed = bool(
        metadata.get(
            "lifecycle_capital_backed",
            bool(metadata.get("lifecycle_capital_eligible", stake > 0)) and stake > 0,
        )
    )
    if bool(metadata.get("lifecycle_actionable_live", False)):
        if capital_backed:
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
    redundancy_family_pairs: Counter[str] = Counter()
    family_signal_values: dict[str, dict[str, list[float]]] = {}
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
            family_signal_values.setdefault(
                family,
                {
                    "ratio": [],
                    "mean_abs": [],
                    "actionable": [],
                    "backed": [],
                },
            )
        if reason == "redundancy":
            blocker_id = str(metadata.get("lifecycle_redundancy_capped_by") or "")
            blocker_label = "unknown"
            if blocker_id:
                blocker_record = next(
                    (candidate for candidate in filtered if candidate.hypothesis_id == blocker_id),
                    None,
                )
                if blocker_record is not None:
                    blocker_label = _representative_family(blocker_record)
            redundancy_family_pairs[
                f"{_representative_family(record)}->{blocker_label}"
            ] += 1

        signal_ratio = float(metadata.get("lifecycle_signal_nonzero_ratio", 0.0) or 0.0)
        signal_mean_abs = float(metadata.get("lifecycle_signal_mean_abs", 0.0) or 0.0)
        ranked.append((signal_ratio, signal_mean_abs, record, reason, family_label))
        for family in record_families:
            stats = family_signal_values[family]
            stats["ratio"].append(signal_ratio)
            stats["mean_abs"].append(signal_mean_abs)

        if bool(metadata.get("lifecycle_actionable_live", False)):
            actionable += 1
            for family in record_families:
                family_signal_values[family]["actionable"].append(1.0)
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
                backed += 1
                backed_ratio_values.append(signal_ratio)
                backed_mean_abs_values.append(signal_mean_abs)
                for family in record_families:
                    family_signal_values[family]["backed"].append(1.0)
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
    family_signal_lines = []
    for family, stats in sorted(
        family_signal_values.items(),
        key=lambda item: (
            len(item[1]["backed"]),
            len(item[1]["actionable"]),
            len(item[1]["ratio"]),
            item[0],
        ),
        reverse=True,
    ):
        ratio_summary = _distribution_summary(stats["ratio"])
        mean_summary = _distribution_summary(stats["mean_abs"])
        family_signal_lines.append(
            f"{family}:n={int(ratio_summary['count'])}"
            f"/a={len(stats['actionable'])}"
            f"/b={len(stats['backed'])}"
            f" ratio_p50={ratio_summary['p50']:.2f}"
            f" ratio_p90={ratio_summary['p90']:.2f}"
            f" mean_p50={mean_summary['p50']:.2f}"
            f" mean_p90={mean_summary['p90']:.2f}"
        )

    return {
        "total": len(filtered),
        "actionable": actionable,
        "backed": backed,
        "reasons": reasons,
        "families": family_summary,
        "family_signal_lines": family_signal_lines,
        "redundancy_family_lines": [
            f"{label}:{count}"
            for label, count in redundancy_family_pairs.most_common(5)
        ],
        "ratio_drop": _distribution_summary(ratio_drop_values),
        "mean_abs_drop": _distribution_summary(mean_abs_drop_values),
        "ratio_backed": _distribution_summary(backed_ratio_values),
        "mean_abs_backed": _distribution_summary(backed_mean_abs_values),
        "signal_nonzero_ratio_min": float(signal_nonzero_ratio_min),
        "signal_mean_abs_min": float(signal_mean_abs_min),
        "top_entries": top_entries,
    }
