from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..alpha.quality import blend_quality
from .store import HypothesisStatus, HypothesisStore

DEFAULT_LOOKBACK = 20
DEFAULT_MIN_OBSERVATIONS = 5
DEFAULT_BOOTSTRAP_WEIGHT = 0.25
DEFAULT_QUALITY_WEIGHT = 1.0
DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT = 0.25
DEFAULT_STAKE_UPDATE_RATE = 0.10


@dataclass(frozen=True)
class AllocationRebalanceEntry:
    hypothesis_id: str
    current_stake: float
    target_stake: float
    proposed_stake: float
    research_backed: bool
    live_proven: bool
    n_observations: int
    bootstrap_trust_value: float
    blended_quality: float
    live_quality: float
    raw_live_quality: float
    confidence: float
    marginal_contribution: float
    redundancy_capped_by: str = ""
    redundancy_correlation: float = 0.0


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


def trust_score(
    blended_quality: float,
    marginal_contribution: float,
    *,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    marginal_contribution_weight: float = DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
) -> float:
    score = (
        quality_weight * float(blended_quality)
        + marginal_contribution_weight * float(marginal_contribution)
    )
    return max(score, 0.0)


def normalized_research_quality(
    research_quality: float,
    *,
    metric: str = "sharpe",
) -> float:
    if metric == "sharpe":
        scale = 2.0
    elif metric == "log_growth":
        scale = 0.20
    else:
        raise ValueError(f"Unsupported fitness metric: {metric}")
    return min(max(float(research_quality) / scale, 0.0), 1.0)


def bootstrap_trust(
    research_quality: float,
    *,
    metric: str = "sharpe",
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
) -> float:
    weight = max(float(bootstrap_weight), 0.0)
    return weight * normalized_research_quality(research_quality, metric=metric)


def target_stake(
    blended_quality: float,
    quality_confidence: float,
    marginal_contribution: float,
    *,
    research_quality: float = 0.0,
    metric: str = "sharpe",
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    quality_weight: float = DEFAULT_QUALITY_WEIGHT,
    marginal_contribution_weight: float = DEFAULT_MARGINAL_CONTRIBUTION_WEIGHT,
    floor: float = 0.0,
) -> float:
    live_trust = trust_score(
        blended_quality,
        marginal_contribution,
        quality_weight=quality_weight,
        marginal_contribution_weight=marginal_contribution_weight,
    )
    confidence = min(max(float(quality_confidence), 0.0), 1.0)
    initial_trust = bootstrap_trust(
        research_quality,
        metric=metric,
        bootstrap_weight=bootstrap_weight,
    )
    trust = (1.0 - confidence) * initial_trust + confidence * live_trust
    return max(trust, floor)


def is_capital_eligible(
    *,
    research_quality: float,
    metric: str,
    bootstrap_weight: float,
    has_min_observations: bool,
    floor: float = 0.0,
) -> bool:
    return (
        bootstrap_trust(
            research_quality,
            metric=metric,
            bootstrap_weight=bootstrap_weight,
        ) > floor
        or has_min_observations
    )


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
    stake_update_rate: float = DEFAULT_STAKE_UPDATE_RATE,
    floor: float = 0.0,
    archive_on_zero: bool = False,
    live_returns_for: Callable[[str], list[float]] | None = None,
) -> dict[str, float]:
    updates: dict[str, float] = {}
    for record in store.list_all():
        history = store.contribution_history(record.hypothesis_id, limit=lookback)
        recent = history[:lookback]
        marginal = sum(recent) / len(recent) if recent else 0.0
        live_returns = (
            live_returns_for(record.hypothesis_id)
            if live_returns_for is not None
            else list(reversed(recent))
        )
        estimate = blend_quality(
            record.oos_fitness(metric),
            live_returns,
            metric=metric,
            rolling_window=lookback,
            min_observations=min_observations,
            full_weight_observations=full_weight_observations,
            early_stage_full_weight_observations=early_stage_full_weight_observations,
            sharpe_clip_abs=sharpe_clip_abs,
            log_growth_clip_abs=log_growth_clip_abs,
        )
        target = target_stake(
            estimate.blended_quality,
            estimate.confidence,
            marginal,
            research_quality=record.oos_fitness(metric),
            metric=metric,
            bootstrap_weight=bootstrap_weight,
            quality_weight=quality_weight,
            marginal_contribution_weight=marginal_contribution_weight,
            floor=floor,
        )
        if not is_capital_eligible(
            research_quality=record.oos_fitness(metric),
            metric=metric,
            bootstrap_weight=bootstrap_weight,
            has_min_observations=estimate.has_min_observations,
            floor=floor,
        ):
            target = floor
        new_stake = updated_stake(
            record.stake,
            target,
            stake_update_rate=stake_update_rate,
            floor=floor,
        )
        store.update_metadata(
            record.hypothesis_id,
            {
                "lifecycle_live_quality": estimate.live_quality,
                "lifecycle_raw_live_quality": estimate.raw_live_quality,
                "lifecycle_blended_quality": estimate.blended_quality,
                "lifecycle_quality_confidence": estimate.confidence,
                "lifecycle_marginal_contribution": marginal,
                "lifecycle_bootstrap_trust": bootstrap_trust(
                    record.oos_fitness(metric),
                    metric=metric,
                    bootstrap_weight=bootstrap_weight,
                ),
                "lifecycle_target_stake": target,
            },
        )
        if abs(new_stake - record.stake) > 1e-12:
            store.update_stake(record.hypothesis_id, new_stake)
            updates[record.hypothesis_id] = new_stake
        if archive_on_zero and new_stake <= floor and record.status == HypothesisStatus.ACTIVE:
            store.update_status(record.hypothesis_id, HypothesisStatus.ARCHIVED)
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
    floor: float = 0.0,
    live_returns_for: Callable[[str], list[float]] | None = None,
) -> list[AllocationRebalanceEntry]:
    plan: list[AllocationRebalanceEntry] = []
    for record in store.list_observation_active():
        history = store.contribution_history(record.hypothesis_id, limit=lookback)
        recent = history[:lookback]
        marginal = sum(recent) / len(recent) if recent else 0.0
        live_returns = (
            live_returns_for(record.hypothesis_id)
            if live_returns_for is not None
            else list(reversed(recent))
        )
        research_quality = record.oos_fitness(metric)
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
        )
        target = target_stake(
            estimate.blended_quality,
            estimate.confidence,
            marginal,
            research_quality=research_quality,
            metric=metric,
            bootstrap_weight=bootstrap_weight,
            quality_weight=quality_weight,
            marginal_contribution_weight=marginal_contribution_weight,
            floor=floor,
        )
        research_backed = bootstrap_value > floor
        live_proven = estimate.has_min_observations
        proposed = target if (research_backed or live_proven) else floor
        plan.append(
            AllocationRebalanceEntry(
                hypothesis_id=record.hypothesis_id,
                current_stake=float(record.stake),
                target_stake=float(target),
                proposed_stake=float(proposed),
                research_backed=research_backed,
                live_proven=live_proven,
                n_observations=estimate.n_observations,
                bootstrap_trust_value=float(bootstrap_value),
                blended_quality=float(estimate.blended_quality),
                live_quality=float(estimate.live_quality),
                raw_live_quality=float(estimate.raw_live_quality),
                confidence=float(estimate.confidence),
                marginal_contribution=float(marginal),
            )
        )
    return plan


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
                "lifecycle_bootstrap_trust": entry.bootstrap_trust_value,
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
