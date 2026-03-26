from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .identity import expression_feature_families


def _format_runtime_hypothesis_entry(record, value: float) -> str:
    return (
        f"{record.hypothesis_id}={value:.3f}"
        f"({record.kind}/{record.source or 'unknown'})"
    )


def runtime_cohort(record) -> str:
    metadata = getattr(record, "metadata", {}) or {}
    if bool(metadata.get("lifecycle_live_proven", False)):
        return "live"
    if bool(metadata.get("lifecycle_research_retained", False)):
        if str(getattr(record, "source", "")) == "bootstrap_serious":
            return "serious"
        if str(metadata.get("lifecycle_research_quality_source", "")) == "batch_research_score":
            return "batch"
        return "bootstrap"
    return "other"


def combine_cohort(record) -> str:
    if str(getattr(record, "source", "")) == "bootstrap_serious":
        return "serious"
    return runtime_cohort(record)


def _top_runtime_hypotheses(records, metric_key: str, *, n: int = 3) -> list[str]:
    ranked: list[tuple[float, object]] = []
    for record in records:
        if metric_key == "stake":
            raw = record.stake
        else:
            raw = record.metadata.get(metric_key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        ranked.append((value, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        _format_runtime_hypothesis_entry(record, value)
        for value, record in ranked[:n]
    ]


def _top_actionable_redundancy_caps(records, *, n: int = 3) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        if not bool(metadata.get("lifecycle_actionable_live", False)):
            continue
        blocker = str(metadata.get("lifecycle_redundancy_capped_by") or "")
        if not blocker:
            continue
        try:
            live_quality = float(metadata.get("lifecycle_live_quality", 0.0))
        except (TypeError, ValueError):
            live_quality = 0.0
        try:
            corr = float(metadata.get("lifecycle_redundancy_correlation", 0.0))
        except (TypeError, ValueError):
            corr = 0.0
        ranked.append(
            (
                live_quality,
                f"{record.hypothesis_id}->{blocker}(corr={corr:.2f})",
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [entry for _, entry in ranked[:n]]


def _top_batch_family_counts(records, *, backed_only: bool, n: int = 3) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if runtime_cohort(record) != "batch":
            continue
        if backed_only and float(record.stake) <= 0:
            continue
        for family in set(expression_feature_families(record.expression)):
            counts[family] += 1
    return [f"{family}:{count}" for family, count in counts.most_common(n)]


def _top_serious_template_counts(records, *, backed_only: bool, n: int = 3) -> list[str]:
    counts: Counter[str] = Counter()
    for record in records:
        if str(getattr(record, "source", "") or "") != "bootstrap_serious":
            continue
        metadata = getattr(record, "metadata", {}) or {}
        retained = bool(metadata.get("lifecycle_research_retained", False)) or bool(
            metadata.get("lifecycle_live_proven", False)
        )
        if not retained:
            continue
        if backed_only and not bool(metadata.get("lifecycle_capital_backed", False)):
            continue
        template_id = str(metadata.get("serious_template", "") or "")
        if not template_id:
            continue
        counts[template_id] += 1
    return [f"{template}:{count}" for template, count in counts.most_common(n)]


@dataclass(frozen=True)
class HypothesisStatusCounts:
    active: int
    paused: int
    archived: int
    live: int


@dataclass(frozen=True)
class ActionableWindowSummary:
    lookback: int
    tracked: int
    expressing: int
    mean_ratio: float
    mean_action: float
    breadth: float


@dataclass(frozen=True)
class CombineCohortStats:
    n: int = 0
    nonzero: int = 0
    weight: float = 0.0
    weighted_signal: float = 0.0


@dataclass(frozen=True)
class CombineTopEntry:
    hypothesis_id: str
    cohort: str
    weight: float
    signal: float
    contribution: float


@dataclass(frozen=True)
class LatestCombineSummary:
    current_combined: float
    current_backed: int
    dropped_current: int
    missing_current: int
    nonzero_current: int
    zero_current: int
    dropped_reasons: dict[str, int] = field(default_factory=dict)
    cohorts: dict[str, CombineCohortStats] = field(default_factory=dict)
    top_entries: list[CombineTopEntry] = field(default_factory=list)


@dataclass(frozen=True)
class AssetSleeveSummary:
    observed: int
    bootstrap_backed: int
    capital_backed: int
    research_retained: int
    bootstrap_research_retained: int
    serious_research_retained: int
    batch_research_retained: int
    live_proven: int
    actionable_live: int
    promoted_live: int
    research_demoted: int
    research_candidate_capped: int
    bootstrap_capital_backed: int
    serious_capital_backed: int
    batch_research_capital_backed: int
    actionable_live_capital_backed: int
    actionable_redundancy_capped: int
    actionable_other_dropped: int
    promotion_blockers: dict[str, int] = field(default_factory=dict)
    top_allocation: list[str] = field(default_factory=list)
    top_effective_live: list[str] = field(default_factory=list)
    top_raw_live: list[str] = field(default_factory=list)
    top_bootstrap: list[str] = field(default_factory=list)
    top_actionable_capped: list[str] = field(default_factory=list)
    batch_retained_families: list[str] = field(default_factory=list)
    batch_backed_families: list[str] = field(default_factory=list)
    serious_retained_templates: list[str] = field(default_factory=list)
    serious_backed_templates: list[str] = field(default_factory=list)


def build_hypothesis_status_counts(store, *, asset: str | None = None) -> HypothesisStatusCounts:
    from .store import HypothesisStatus

    live = len(store.list_live(asset=asset))
    return HypothesisStatusCounts(
        active=len(store.list_by_status(HypothesisStatus.ACTIVE, asset=asset)),
        paused=len(store.list_by_status(HypothesisStatus.PAUSED, asset=asset)),
        archived=len(store.list_by_status(HypothesisStatus.ARCHIVED, asset=asset)),
        live=live,
    )


def live_hypothesis_ids(store, *, asset: str | None = None) -> list[str]:
    return [record.hypothesis_id for record in store.list_live(asset=asset)]


def build_asset_sleeve_summary(records) -> AssetSleeveSummary:
    bootstrap_backed = 0
    observed = 0
    capital_backed = 0
    research_retained = 0
    bootstrap_research_retained = 0
    serious_research_retained = 0
    batch_research_retained = 0
    live_proven = 0
    actionable_live = 0
    promoted_live = 0
    research_demoted = 0
    research_candidate_capped = 0
    bootstrap_capital_backed = 0
    serious_capital_backed = 0
    batch_research_capital_backed = 0
    actionable_live_capital_backed = 0
    actionable_redundancy_capped = 0
    actionable_other_dropped = 0
    blocker_counts = {
        "insufficient_observations": 0,
        "weak_live_quality": 0,
        "weak_marginal_contribution": 0,
        "weak_live_quality_and_contribution": 0,
        "weak_signal_activity": 0,
    }

    for record in records:
        metadata = getattr(record, "metadata", {}) or {}
        cohort = runtime_cohort(record)
        source = str(getattr(record, "source", "") or "")
        try:
            stake = float(record.stake)
        except (TypeError, ValueError):
            stake = 0.0
        capital_backed_now = bool(
            metadata.get("lifecycle_capital_backed", stake > 0)
        )
        try:
            bootstrap_trust = float(metadata.get("lifecycle_bootstrap_trust", 0.0))
        except (TypeError, ValueError):
            bootstrap_trust = 0.0
        if capital_backed_now:
            capital_backed += 1
        if bootstrap_trust > 0:
            bootstrap_backed += 1
        observed += 1
        if bool(metadata.get("lifecycle_research_retained", False)):
            research_retained += 1
            if cohort == "batch":
                batch_research_retained += 1
            elif cohort != "serious":
                bootstrap_research_retained += 1
        if bool(metadata.get("lifecycle_live_proven", False)):
            live_proven += 1
            if bootstrap_trust <= 0:
                promoted_live += 1
        if source == "bootstrap_serious" and (
            bool(metadata.get("lifecycle_research_retained", False))
            or bool(metadata.get("lifecycle_live_proven", False))
        ):
            serious_research_retained += 1
        if bool(metadata.get("lifecycle_actionable_live", False)):
            actionable_live += 1
            if not capital_backed_now:
                if metadata.get("lifecycle_redundancy_capped_by"):
                    actionable_redundancy_capped += 1
                else:
                    actionable_other_dropped += 1
        if capital_backed_now:
            if source == "bootstrap_serious":
                serious_capital_backed += 1
            elif cohort == "live":
                actionable_live_capital_backed += 1
            elif cohort == "batch":
                batch_research_capital_backed += 1
            elif cohort == "serious":
                serious_capital_backed += 1
            else:
                bootstrap_capital_backed += 1
        if bootstrap_trust > 0 and not bool(metadata.get("lifecycle_capital_eligible", capital_backed_now)):
            research_demoted += 1
        if bool(metadata.get("lifecycle_research_candidate_capped", False)):
            research_candidate_capped += 1
        if bootstrap_trust <= 0 and not bool(metadata.get("lifecycle_actionable_live", False)):
            blocker = str(metadata.get("lifecycle_live_promotion_blocker", ""))
            if blocker in blocker_counts:
                blocker_counts[blocker] += 1

    return AssetSleeveSummary(
        observed=observed,
        bootstrap_backed=bootstrap_backed,
        capital_backed=capital_backed,
        research_retained=research_retained,
        bootstrap_research_retained=bootstrap_research_retained,
        serious_research_retained=serious_research_retained,
        batch_research_retained=batch_research_retained,
        live_proven=live_proven,
        actionable_live=actionable_live,
        promoted_live=promoted_live,
        research_demoted=research_demoted,
        research_candidate_capped=research_candidate_capped,
        bootstrap_capital_backed=bootstrap_capital_backed,
        serious_capital_backed=serious_capital_backed,
        batch_research_capital_backed=batch_research_capital_backed,
        actionable_live_capital_backed=actionable_live_capital_backed,
        actionable_redundancy_capped=actionable_redundancy_capped,
        actionable_other_dropped=actionable_other_dropped,
        promotion_blockers=blocker_counts,
        top_allocation=_top_runtime_hypotheses(records, "stake"),
        top_effective_live=_top_runtime_hypotheses(records, "lifecycle_live_quality"),
        top_raw_live=_top_runtime_hypotheses(records, "lifecycle_raw_live_quality"),
        top_bootstrap=_top_runtime_hypotheses(records, "lifecycle_bootstrap_trust"),
        top_actionable_capped=_top_actionable_redundancy_caps(records),
        batch_retained_families=_top_batch_family_counts(records, backed_only=False),
        batch_backed_families=_top_batch_family_counts(records, backed_only=True),
        serious_retained_templates=_top_serious_template_counts(records, backed_only=False),
        serious_backed_templates=_top_serious_template_counts(records, backed_only=True),
    )


def build_actionable_window_summary(
    records,
    *,
    tracker,
    lookback: int,
    supports_short: bool,
) -> ActionableWindowSummary | None:
    if not records:
        return None

    ratios: list[float] = []
    mean_actions: list[float] = []
    rows: list[np.ndarray] = []
    expressing = 0

    for record in records:
        history = tracker.get_hypothesis_signal_history(record.hypothesis_id, limit=lookback)
        if not history:
            continue
        values = np.asarray([float(v) for v in history], dtype=np.float64)
        if supports_short:
            actionable = np.abs(values)
        else:
            actionable = np.maximum(values, 0.0)
        actionable = np.nan_to_num(actionable, nan=0.0, posinf=0.0, neginf=0.0)
        if actionable.size == 0:
            continue
        ratio = float(np.count_nonzero(actionable > 1e-12) / actionable.size)
        mean_action = float(actionable.mean())
        ratios.append(ratio)
        mean_actions.append(mean_action)
        if np.any(actionable > 1e-12):
            expressing += 1
        if actionable.size >= 2 and float(np.nanstd(actionable)) > 1e-12:
            rows.append(actionable)

    tracked = len(ratios)
    if tracked == 0:
        return None

    breadth = 0.0
    if len(rows) == 1:
        breadth = 1.0
    elif len(rows) >= 2:
        min_len = min(len(row) for row in rows)
        if min_len >= 2:
            aligned = np.vstack([row[-min_len:] for row in rows])
            non_constant = aligned[np.nanstd(aligned, axis=1) > 1e-12]
            if non_constant.shape[0] == 1:
                breadth = 1.0
            elif non_constant.shape[0] >= 2:
                corr = np.corrcoef(non_constant)
                corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
                eigvals = np.linalg.eigvalsh(corr)
                eigvals = np.clip(np.real(eigvals), 0.0, None)
                denom = float(np.square(eigvals).sum())
                if denom > 1e-12:
                    breadth = float(np.square(eigvals.sum()) / denom)

    return ActionableWindowSummary(
        lookback=int(lookback),
        tracked=tracked,
        expressing=expressing,
        mean_ratio=float(sum(ratios) / tracked),
        mean_action=float(sum(mean_actions) / tracked),
        breadth=breadth,
    )


def build_latest_combine_summary(
    *,
    record_map: dict[str, object],
    signals: dict[str, float],
    weights: dict[str, float],
    top_n: int,
) -> LatestCombineSummary:
    cohort_data = {
        "bootstrap": {"n": 0, "nonzero": 0, "weight": 0.0, "weighted_signal": 0.0},
        "serious": {"n": 0, "nonzero": 0, "weight": 0.0, "weighted_signal": 0.0},
        "batch": {"n": 0, "nonzero": 0, "weight": 0.0, "weighted_signal": 0.0},
        "live": {"n": 0, "nonzero": 0, "weight": 0.0, "weighted_signal": 0.0},
        "other": {"n": 0, "nonzero": 0, "weight": 0.0, "weighted_signal": 0.0},
    }
    ranked: list[tuple[float, CombineTopEntry]] = []
    dropped_reasons: dict[str, int] = {}
    missing_current = 0
    dropped_current = 0
    nonzero_current = 0

    for hypothesis_id, raw_signal in signals.items():
        record = record_map.get(hypothesis_id)
        if record is None:
            missing_current += 1
            continue
        cohort = combine_cohort(record)
        weight = float(weights.get(hypothesis_id, 0.0))
        if weight <= 0.0:
            dropped_current += 1
            metadata = getattr(record, "metadata", {}) or {}
            reason = str(metadata.get("lifecycle_capital_reason") or "other")
            if metadata.get("lifecycle_redundancy_capped_by"):
                reason = "redundancy_capped"
            dropped_reasons[reason] = dropped_reasons.get(reason, 0) + 1
            continue
        signal = float(raw_signal)
        if abs(signal) > 1e-12:
            nonzero_current += 1
            cohort_data[cohort]["nonzero"] += 1
        contribution = weight * signal
        cohort_data[cohort]["n"] += 1
        cohort_data[cohort]["weight"] += weight
        cohort_data[cohort]["weighted_signal"] += contribution
        ranked.append(
            (
                abs(contribution),
                CombineTopEntry(
                    hypothesis_id=hypothesis_id,
                    cohort=cohort,
                    weight=weight,
                    signal=signal,
                    contribution=contribution,
                ),
            )
        )

    cohorts = {
        name: CombineCohortStats(**values)
        for name, values in cohort_data.items()
    }
    current_backed = sum(item.n for item in cohorts.values())
    ranked.sort(key=lambda item: item[0], reverse=True)
    top_entries = [entry for _, entry in ranked[: max(int(top_n), 1)]]

    return LatestCombineSummary(
        current_combined=sum(item.weighted_signal for item in cohorts.values()),
        current_backed=current_backed,
        dropped_current=dropped_current,
        missing_current=missing_current,
        nonzero_current=nonzero_current,
        zero_current=current_backed - nonzero_current,
        dropped_reasons=dropped_reasons,
        cohorts=cohorts,
        top_entries=top_entries,
    )
