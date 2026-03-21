"""Alpha lifecycle state machine: candidate → active ↔ dormant."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .quality import QualityEstimate
from .managed_alphas import AlphaRecord, AlphaState


@dataclass
class LifecycleConfig:
    candidate_quality_min: float = 0.5
    active_quality_min: float = 0.3
    dormant_revival_quality: float = 0.3
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0
    correlation_max: float = 0.5


def passes_candidate_gate(
    record: AlphaRecord,
    config: LifecycleConfig,
    *,
    metric: str = "sharpe",
) -> bool:
    """Return True when a candidate clears the admission gate."""
    return (
        record.oos_fitness(metric) >= config.candidate_quality_min
        and record.pbo <= config.pbo_max
        and record.dsr_pvalue <= config.dsr_pvalue_max
        and record.correlation_avg <= config.correlation_max
    )


def compute_transition(state: str, quality: float, config: LifecycleConfig) -> str:
    """Pure function: compute state transition without DB side effects."""
    state = AlphaState.canonical(state)
    if state == AlphaState.ACTIVE:
        if quality < config.active_quality_min:
            return AlphaState.DORMANT
        return AlphaState.ACTIVE
    if state == AlphaState.DORMANT:
        if quality >= config.dormant_revival_quality:
            return AlphaState.ACTIVE
        return AlphaState.DORMANT
    return state


def next_live_state(
    state: str,
    estimate: QualityEstimate,
    config: LifecycleConfig,
    *,
    dormant_revival_min_observations: int = 20,
) -> str:
    """Compute the next live state from a quality estimate."""
    state = AlphaState.canonical(state)
    if (
        state == AlphaState.DORMANT
        and estimate.n_observations < dormant_revival_min_observations
    ):
        return AlphaState.DORMANT
    return compute_transition(state, estimate.blended_quality, config)


# Vectorized state codes for batch transitions (used by simulator)
ST_CANDIDATE, ST_ACTIVE, ST_DORMANT = 0, 1, 2

def batch_transitions(
    states: np.ndarray,
    quality_scores: np.ndarray,
    config: LifecycleConfig,
) -> np.ndarray:
    """Vectorized state transitions for batch simulation."""
    new = states.copy()

    active = states == ST_ACTIVE
    new[active & (quality_scores < config.active_quality_min)] = ST_DORMANT

    dorm = states == ST_DORMANT
    new[dorm & (quality_scores >= config.dormant_revival_quality)] = ST_ACTIVE

    return new


def batch_live_transitions(
    states: np.ndarray,
    quality_scores: np.ndarray,
    observation_counts: np.ndarray,
    config: LifecycleConfig,
    *,
    dormant_revival_min_observations: int = 20,
) -> np.ndarray:
    """Vectorized live transitions with dormant revival gating."""
    new = batch_transitions(states, quality_scores, config)
    gated = (
        (states == ST_DORMANT)
        & (observation_counts < dormant_revival_min_observations)
    )
    new[gated] = ST_DORMANT
    return new


