from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

from .allocation_policy import (
    DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN,
    DEFAULT_BATCH_RESEARCH_WEIGHT,
    DEFAULT_BOOTSTRAP_RETENTION_MARGINAL_CONTRIBUTION_MIN,
    DEFAULT_BOOTSTRAP_RETENTION_QUALITY_MIN,
    DEFAULT_BOOTSTRAP_WEIGHT,
    DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN,
    DEFAULT_LIVE_PROVEN_QUALITY_MIN,
    DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN,
    DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN,
    DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
    DEFAULT_QUALITY_WEIGHT,
    bootstrap_trust,
    capital_eligibility_breakdown,
    live_promotion_blocker,
    target_stake,
)
from .identity import expression_feature_families
from .quality import blend_quality
from .store import HypothesisStatus, HypothesisStore

DEFAULT_LOOKBACK = 20
DEFAULT_MIN_OBSERVATIONS = 5
DEFAULT_STAKE_UPDATE_RATE = 0.10
DEFAULT_BATCH_RESEARCH_CAPITAL_CANDIDATES_MAX = 12


@dataclass(frozen=True)
class AllocationRebalanceEntry:
    hypothesis_id: str
    current_stake: float
    target_stake: float
    proposed_stake: float
    research_backed: bool
    research_retained: bool
    live_proven: bool
    actionable_live: bool
    capital_eligible: bool
    capital_reason: str
    live_promotion_blocker: str
    n_observations: int
    bootstrap_trust_value: float
    blended_quality: float
    live_quality: float
    raw_live_quality: float
    confidence: float
    marginal_contribution: float
    signal_nonzero_ratio: float = 0.0
    signal_mean_abs: float = 0.0
    research_quality_source: str = ""
    representative_family: str = ""
    research_candidate_capped: bool = False
    redundancy_capped_by: str = ""
    redundancy_correlation: float = 0.0


def _apply_batch_research_candidate_gate(
    plan: list[AllocationRebalanceEntry],
    *,
    max_candidates: int,
    floor: float,
) -> list[AllocationRebalanceEntry]:
    if max_candidates <= 0:
        return plan

    ranked = sorted(
        (
            entry
            for entry in plan
            if entry.research_quality_source == "batch_research_score"
            and entry.research_retained
            and not entry.live_proven
            and entry.proposed_stake > floor
        ),
        key=lambda entry: (
            entry.bootstrap_trust_value,
            entry.blended_quality,
            entry.n_observations,
            entry.hypothesis_id,
        ),
        reverse=True,
    )
    family_counts: dict[str, int] = {}
    for entry in ranked:
        family = entry.representative_family or "other"
        family_counts[family] = family_counts.get(family, 0) + 1

    diversified: list[AllocationRebalanceEntry] = []
    seen_families: set[str] = set()
    for entry in ranked:
        family = entry.representative_family or "other"
        if family in seen_families:
            continue
        diversified.append(entry)
        seen_families.add(family)
        if len(diversified) >= max_candidates:
            break

    if len(diversified) < max_candidates:
        selected_ids = {entry.hypothesis_id for entry in diversified}
        for entry in ranked:
            if entry.hypothesis_id in selected_ids:
                continue
            diversified.append(entry)
            selected_ids.add(entry.hypothesis_id)
            if len(diversified) >= max_candidates:
                break

    allowed_ids = {entry.hypothesis_id for entry in diversified}
    if len(allowed_ids) == len(ranked):
        return plan

    gated: list[AllocationRebalanceEntry] = []
    for entry in plan:
        if (
            entry.research_quality_source == "batch_research_score"
            and entry.research_retained
            and not entry.live_proven
            and entry.proposed_stake > floor
            and entry.hypothesis_id not in allowed_ids
        ):
            gated.append(
                replace(
                    entry,
                    proposed_stake=float(floor),
                    capital_eligible=False,
                    capital_reason="research_candidate_capped",
                    research_candidate_capped=True,
                )
            )
            continue
        gated.append(entry)
    return gated


def weighted_prediction(
    predictions: dict[str, float],
    stakes: dict[str, float],
) -> float:
    weighted_sum = 0.0
    total_stake = 0.0
    for hypothesis_id, prediction in predictions.items():
        stake = stakes.get(hypothesis_id, 0.0)
        if stake <= 0 or not math.isfinite(prediction):
            continue
        weighted_sum += stake * prediction
        total_stake += stake
    if total_stake <= 0:
        return 0.0
    return weighted_sum / total_stake


def compute_daily_contributions(
    predictions: dict[str, float],
    realized_return: float,
    stakes: dict[str, float],
) -> dict[str, float]:
    live_ids = [
        hypothesis_id
        for hypothesis_id, prediction in predictions.items()
        if stakes.get(hypothesis_id, 0.0) > 0 and math.isfinite(prediction)
    ]
    if not live_ids:
        return {}

    portfolio_prediction = weighted_prediction(predictions, stakes)
    full_score = portfolio_prediction * realized_return
    contributions: dict[str, float] = {}

    for hypothesis_id in live_ids:
        reduced_predictions = {
            other_id: prediction
            for other_id, prediction in predictions.items()
            if other_id != hypothesis_id
        }
        reduced_stakes = {
            other_id: stake
            for other_id, stake in stakes.items()
            if other_id != hypothesis_id
        }
        score_without = weighted_prediction(reduced_predictions, reduced_stakes) * realized_return
        contributions[hypothesis_id] = full_score - score_without

    return contributions


def rolling_stake(
    contributions: list[float],
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    prior_stake: float = 0.0,
    floor: float = 0.0,
) -> float:
    if len(contributions) < min_observations:
        return max(prior_stake, floor)
    recent = contributions[:lookback]
    mean_contribution = sum(recent) / len(recent)
    return max(mean_contribution, floor)


def updated_stake(
    current_stake: float,
    target_stake_value: float,
    *,
    stake_update_rate: float = DEFAULT_STAKE_UPDATE_RATE,
    floor: float = 0.0,
) -> float:
    rate = min(max(float(stake_update_rate), 0.0), 1.0)
    next_value = (1.0 - rate) * float(current_stake) + rate * float(target_stake_value)
    return max(next_value, floor)


def record_daily_contributions(
    store: HypothesisStore,
    *,
    date: str,
    predictions: dict[str, float],
    realized_return: float,
) -> dict[str, float]:
    stakes = {
        record.hypothesis_id: record.stake
        for record in store.list_active()
    }
    contributions = compute_daily_contributions(predictions, realized_return, stakes)
    for hypothesis_id, contribution in contributions.items():
        store.record_contribution(
            hypothesis_id,
            date=date,
            contribution=contribution,
        )
    return contributions


def update_stakes_from_history(
    store: HypothesisStore,
    *,
    metric: str = "sharpe",
    lookback: int = DEFAULT_LOOKBACK,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    full_weight_observations: int = 63,
    early_stage_full_weight_observations: int | None = None,
    sharpe_clip_abs: float = 3.0,
    log_growth_clip_abs: float = 0.20,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    marginal_contribution_weight: float = DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    batch_research_normalized_quality_min: float = (
        DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN
    ),
    batch_research_capital_candidates_max: int = (
        DEFAULT_BATCH_RESEARCH_CAPITAL_CANDIDATES_MAX
    ),
    stake_update_rate: float = DEFAULT_STAKE_UPDATE_RATE,
    live_proven_quality_min: float = DEFAULT_LIVE_PROVEN_QUALITY_MIN,
    live_proven_marginal_contribution_min: float = DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN,
    live_proven_signal_nonzero_ratio_min: float = (
        DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN
    ),
    live_proven_signal_mean_abs_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN,
    bootstrap_retention_quality_min: float = DEFAULT_BOOTSTRAP_RETENTION_QUALITY_MIN,
    bootstrap_retention_marginal_contribution_min: float = (
        DEFAULT_BOOTSTRAP_RETENTION_MARGINAL_CONTRIBUTION_MIN
    ),
    floor: float = 0.0,
    archive_on_zero: bool = False,
    live_returns_for: Callable[[str], list[float]] | None = None,
    signal_activity_for: Callable[[str], tuple[float, float]] | None = None,
) -> dict[str, float]:
    plan = build_allocation_rebalance_plan(
        store,
        metric=metric,
        lookback=lookback,
        min_observations=min_observations,
        full_weight_observations=full_weight_observations,
        early_stage_full_weight_observations=early_stage_full_weight_observations,
        sharpe_clip_abs=sharpe_clip_abs,
        log_growth_clip_abs=log_growth_clip_abs,
        quality_weight=quality_weight,
        marginal_contribution_weight=marginal_contribution_weight,
        bootstrap_weight=bootstrap_weight,
        batch_research_weight=batch_research_weight,
        batch_research_normalized_quality_min=batch_research_normalized_quality_min,
        batch_research_capital_candidates_max=batch_research_capital_candidates_max,
        live_proven_quality_min=live_proven_quality_min,
        live_proven_marginal_contribution_min=live_proven_marginal_contribution_min,
        live_proven_signal_nonzero_ratio_min=live_proven_signal_nonzero_ratio_min,
        live_proven_signal_mean_abs_min=live_proven_signal_mean_abs_min,
        bootstrap_retention_quality_min=bootstrap_retention_quality_min,
        bootstrap_retention_marginal_contribution_min=bootstrap_retention_marginal_contribution_min,
        floor=floor,
        live_returns_for=live_returns_for,
        signal_activity_for=signal_activity_for,
    )
    smoothed_plan = [
        replace(
            entry,
            proposed_stake=updated_stake(
                entry.current_stake,
                entry.proposed_stake,
                stake_update_rate=stake_update_rate,
                floor=floor,
            ),
        )
        for entry in plan
    ]
    updates = apply_allocation_rebalance_plan(store, smoothed_plan)
    if archive_on_zero:
        for entry in smoothed_plan:
            if entry.proposed_stake <= floor:
                record = store.get(entry.hypothesis_id)
                if record is not None and record.status == HypothesisStatus.ACTIVE:
                    store.update_status(entry.hypothesis_id, HypothesisStatus.ARCHIVED)
    return updates


def build_allocation_rebalance_plan(
    store: HypothesisStore,
    *,
    metric: str = "sharpe",
    lookback: int = DEFAULT_LOOKBACK,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
    full_weight_observations: int = 63,
    early_stage_full_weight_observations: int | None = None,
    sharpe_clip_abs: float = 3.0,
    log_growth_clip_abs: float = 0.20,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    marginal_contribution_weight: float = DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    batch_research_weight: float = DEFAULT_BATCH_RESEARCH_WEIGHT,
    batch_research_normalized_quality_min: float = (
        DEFAULT_BATCH_RESEARCH_NORMALIZED_QUALITY_MIN
    ),
    batch_research_capital_candidates_max: int = (
        DEFAULT_BATCH_RESEARCH_CAPITAL_CANDIDATES_MAX
    ),
    live_proven_quality_min: float = DEFAULT_LIVE_PROVEN_QUALITY_MIN,
    live_proven_marginal_contribution_min: float = DEFAULT_LIVE_PROVEN_MARGINAL_CONTRIBUTION_MIN,
    live_proven_signal_nonzero_ratio_min: float = (
        DEFAULT_LIVE_PROVEN_SIGNAL_NONZERO_RATIO_MIN
    ),
    live_proven_signal_mean_abs_min: float = DEFAULT_LIVE_PROVEN_SIGNAL_MEAN_ABS_MIN,
    bootstrap_retention_quality_min: float = DEFAULT_BOOTSTRAP_RETENTION_QUALITY_MIN,
    bootstrap_retention_marginal_contribution_min: float = (
        DEFAULT_BOOTSTRAP_RETENTION_MARGINAL_CONTRIBUTION_MIN
    ),
    floor: float = 0.0,
    live_returns_for: Callable[[str], list[float]] | None = None,
    signal_activity_for: Callable[[str], tuple[float, float]] | None = None,
) -> list[AllocationRebalanceEntry]:
    plan: list[AllocationRebalanceEntry] = []
    for record in store.list_observation_active():
        research_quality_source = str(
            record.metadata.get("prior_quality_source")
            or record.metadata.get("research_quality_source")
            or ""
        )
        history = store.contribution_history(record.hypothesis_id, limit=lookback)
        recent = history[:lookback]
        marginal = sum(recent) / len(recent) if recent else 0.0
        live_returns = (
            live_returns_for(record.hypothesis_id)
            if live_returns_for is not None
            else list(reversed(recent))
        )
        research_quality = record.oos_fitness(metric)
        signal_nonzero_ratio, signal_mean_abs = (
            signal_activity_for(record.hypothesis_id)
            if signal_activity_for is not None
            else (0.0, 0.0)
        )
        families = expression_feature_families(record.expression)
        representative_family = (
            min(
                families,
                key=lambda family: (families.count(family), family),
            )
            if families
            else "other"
        )
        estimate = blend_quality(
            research_quality,
            live_returns,
            metric=metric,
            rolling_window=lookback,
            min_observations=min_observations,
            full_weight_observations=full_weight_observations,
            early_stage_full_weight_observations=early_stage_full_weight_observations,
            sharpe_clip_abs=sharpe_clip_abs,
            log_growth_clip_abs=log_growth_clip_abs,
        )
        bootstrap_value = bootstrap_trust(
            research_quality,
            metric=metric,
            bootstrap_weight=bootstrap_weight,
            batch_research_weight=batch_research_weight,
            research_quality_source=research_quality_source,
        )
        target = target_stake(
            estimate.blended_quality,
            estimate.confidence,
            marginal,
            research_quality=research_quality,
            research_quality_source=research_quality_source,
            metric=metric,
            bootstrap_weight=bootstrap_weight,
            batch_research_weight=batch_research_weight,
            quality_weight=quality_weight,
            marginal_contribution_weight=marginal_contribution_weight,
            floor=floor,
        )
        research_backed, research_retained, live_proven, actionable_live, capital_reason = (
            capital_eligibility_breakdown(
                research_quality=research_quality,
                research_quality_source=research_quality_source,
                metric=metric,
                bootstrap_weight=bootstrap_weight,
                batch_research_weight=batch_research_weight,
                batch_research_normalized_quality_min=batch_research_normalized_quality_min,
                has_min_observations=estimate.has_min_observations,
                live_quality=estimate.live_quality,
                marginal_contribution=marginal,
                signal_nonzero_ratio=signal_nonzero_ratio,
                signal_mean_abs=signal_mean_abs,
                live_proven_quality_min=live_proven_quality_min,
                live_proven_marginal_contribution_min=live_proven_marginal_contribution_min,
                live_proven_signal_nonzero_ratio_min=live_proven_signal_nonzero_ratio_min,
                live_proven_signal_mean_abs_min=live_proven_signal_mean_abs_min,
                bootstrap_retention_quality_min=bootstrap_retention_quality_min,
                bootstrap_retention_marginal_contribution_min=(
                    bootstrap_retention_marginal_contribution_min
                ),
                floor=floor,
            )
        )
        capital_eligible = research_retained or actionable_live
        proposed = target if capital_eligible else floor
        plan.append(
            AllocationRebalanceEntry(
                hypothesis_id=record.hypothesis_id,
                research_quality_source=research_quality_source,
                current_stake=float(record.stake),
                target_stake=float(target),
                proposed_stake=float(proposed),
                research_backed=research_backed,
                research_retained=research_retained,
                live_proven=live_proven,
                actionable_live=actionable_live,
                capital_eligible=capital_eligible,
                capital_reason=capital_reason,
                live_promotion_blocker=live_promotion_blocker(
                    has_min_observations=estimate.has_min_observations,
                    live_quality=estimate.live_quality,
                    marginal_contribution=marginal,
                    signal_nonzero_ratio=signal_nonzero_ratio,
                    signal_mean_abs=signal_mean_abs,
                    live_proven_quality_min=live_proven_quality_min,
                    live_proven_marginal_contribution_min=live_proven_marginal_contribution_min,
                    live_proven_signal_nonzero_ratio_min=live_proven_signal_nonzero_ratio_min,
                    live_proven_signal_mean_abs_min=live_proven_signal_mean_abs_min,
                ),
                n_observations=estimate.n_observations,
                bootstrap_trust_value=float(bootstrap_value),
                blended_quality=float(estimate.blended_quality),
                live_quality=float(estimate.live_quality),
                raw_live_quality=float(estimate.raw_live_quality),
                confidence=float(estimate.confidence),
                marginal_contribution=float(marginal),
                signal_nonzero_ratio=float(signal_nonzero_ratio),
                signal_mean_abs=float(signal_mean_abs),
                representative_family=representative_family,
            )
        )
    return _apply_batch_research_candidate_gate(
        plan,
        max_candidates=batch_research_capital_candidates_max,
        floor=floor,
    )


def apply_allocation_rebalance_plan(
    store: HypothesisStore,
    plan: list[AllocationRebalanceEntry],
) -> dict[str, float]:
    updates: dict[str, float] = {}
    for entry in plan:
        store.update_metadata(
            entry.hypothesis_id,
            {
                "lifecycle_live_quality": entry.live_quality,
                "lifecycle_raw_live_quality": entry.raw_live_quality,
                "lifecycle_blended_quality": entry.blended_quality,
                "lifecycle_quality_confidence": entry.confidence,
                "lifecycle_marginal_contribution": entry.marginal_contribution,
                "lifecycle_signal_nonzero_ratio": entry.signal_nonzero_ratio,
                "lifecycle_signal_mean_abs": entry.signal_mean_abs,
                "lifecycle_bootstrap_trust": entry.bootstrap_trust_value,
                "lifecycle_research_quality_source": entry.research_quality_source,
                "lifecycle_n_observations": entry.n_observations,
                "lifecycle_research_backed": entry.research_backed,
                "lifecycle_research_retained": entry.research_retained,
                "lifecycle_live_proven": entry.live_proven,
                "lifecycle_actionable_live": entry.actionable_live,
                "lifecycle_capital_eligible": entry.capital_eligible,
                "lifecycle_capital_reason": entry.capital_reason,
                "lifecycle_research_candidate_capped": entry.research_candidate_capped,
                "lifecycle_live_promotion_blocker": entry.live_promotion_blocker,
                "lifecycle_target_stake": entry.target_stake,
                "lifecycle_rebalance_proposed_stake": entry.proposed_stake,
                "lifecycle_redundancy_capped_by": entry.redundancy_capped_by,
                "lifecycle_redundancy_correlation": entry.redundancy_correlation,
            },
        )
        if abs(entry.proposed_stake - entry.current_stake) > 1e-12:
            store.update_stake(entry.hypothesis_id, entry.proposed_stake)
            updates[entry.hypothesis_id] = entry.proposed_stake
    return updates
