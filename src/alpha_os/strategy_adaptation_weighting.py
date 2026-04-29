from __future__ import annotations

from dataclasses import dataclass

from .strategy_adaptation import StrategyAdaptationState


@dataclass(frozen=True)
class StrategyAdaptationFamilyWeight:
    family_id: str
    baseline_weight: float
    adaptation_weight: float
    blended_weight: float


@dataclass(frozen=True)
class StrategyAdaptationSignalWeight:
    signal_id: str
    baseline_multiplier: float
    adaptation_multiplier: float
    blended_multiplier: float


def build_strategy_adaptation_family_weights(
    *,
    family_ids: tuple[str, ...],
    strategy_adaptation_state: StrategyAdaptationState | None,
    blend: float = 0.2,
) -> dict[str, StrategyAdaptationFamilyWeight]:
    if not family_ids:
        return {}
    normalized_blend = min(max(float(blend), 0.0), 1.0)
    baseline_weight = 1.0 / float(len(family_ids))
    if strategy_adaptation_state is None:
        return {
            family_id: StrategyAdaptationFamilyWeight(
                family_id=family_id,
                baseline_weight=baseline_weight,
                adaptation_weight=baseline_weight,
                blended_weight=baseline_weight,
            )
            for family_id in family_ids
        }

    reputation_by_family_id = {
        item.family_id: item for item in strategy_adaptation_state.family_reputations
    }
    adaptation_scores = {
        family_id: _family_adaptation_score(reputation_by_family_id.get(family_id))
        for family_id in family_ids
    }
    total_adaptation_score = sum(adaptation_scores.values())
    if total_adaptation_score <= 0.0:
        total_adaptation_score = float(len(family_ids))
        adaptation_scores = {family_id: 1.0 for family_id in family_ids}

    weights: dict[str, StrategyAdaptationFamilyWeight] = {}
    for family_id in family_ids:
        adaptation_weight = adaptation_scores[family_id] / total_adaptation_score
        effective_blend = normalized_blend * _reputation_readiness_blend(
            reputation_by_family_id.get(family_id)
        )
        blended_weight = ((1.0 - effective_blend) * baseline_weight) + (
            effective_blend * adaptation_weight
        )
        weights[family_id] = StrategyAdaptationFamilyWeight(
            family_id=family_id,
            baseline_weight=baseline_weight,
            adaptation_weight=adaptation_weight,
            blended_weight=blended_weight,
        )
    return weights


def build_strategy_adaptation_signal_weights(
    *,
    signal_ids: tuple[str, ...],
    strategy_adaptation_state: StrategyAdaptationState | None,
    blend: float = 0.2,
) -> dict[str, StrategyAdaptationSignalWeight]:
    if not signal_ids:
        return {}
    normalized_blend = min(max(float(blend), 0.0), 1.0)
    if strategy_adaptation_state is None:
        return {
            signal_id: StrategyAdaptationSignalWeight(
                signal_id=signal_id,
                baseline_multiplier=1.0,
                adaptation_multiplier=1.0,
                blended_multiplier=1.0,
            )
            for signal_id in signal_ids
        }
    reputation_by_signal_id = {
        item.signal_id: item
        for item in strategy_adaptation_state.signal_reputations
    }
    raw_scores = {
        signal_id: _signal_adaptation_score(
            reputation_by_signal_id.get(signal_id)
        )
        for signal_id in signal_ids
    }
    mean_score = sum(raw_scores.values()) / float(len(raw_scores))
    if mean_score <= 0.0:
        mean_score = 1.0
    weights: dict[str, StrategyAdaptationSignalWeight] = {}
    for signal_id in signal_ids:
        adaptation_multiplier = raw_scores[signal_id] / mean_score
        effective_blend = normalized_blend * _reputation_readiness_blend(
            reputation_by_signal_id.get(signal_id)
        )
        blended_multiplier = ((1.0 - effective_blend) * 1.0) + (
            effective_blend * adaptation_multiplier
        )
        weights[signal_id] = StrategyAdaptationSignalWeight(
            signal_id=signal_id,
            baseline_multiplier=1.0,
            adaptation_multiplier=adaptation_multiplier,
            blended_multiplier=blended_multiplier,
        )
    return weights

def _family_adaptation_score(reputation) -> float:
    if reputation is None:
        return 1.0
    return max(
        (
            float(reputation.mean_edge_score)
            * (0.5 + (0.5 * float(reputation.mean_confidence)))
            * (0.5 + (0.1 * min(float(reputation.update_count), 5.0)))
        ),
        1e-6,
    )


def _signal_adaptation_score(reputation) -> float:
    if reputation is None:
        return 1.0
    return max(
        (
            float(reputation.contribution_score)
            * (0.5 + (0.5 * float(reputation.confidence)))
            * (0.5 + (0.1 * min(float(reputation.update_count), 5.0)))
        ),
        1e-6,
    )


def _reputation_readiness_blend(reputation) -> float:
    if reputation is None:
        return 0.0
    update_count = float(getattr(reputation, "update_count", 0.0))
    # Keep the adaptation weights close to baseline until they have seen
    # multiple evaluation refreshes. One report is too noisy.
    return min(max((update_count - 1.0) / 4.0, 0.0), 1.0)
