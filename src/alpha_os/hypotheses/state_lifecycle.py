from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .quality import QualityEstimate

CANDIDATE_STATE = "candidate"
ACTIVE_STATE = "active"
DORMANT_STATE = "dormant"
REJECTED_STATE = "rejected"

_CANONICAL_STATES = {
    CANDIDATE_STATE: CANDIDATE_STATE,
    ACTIVE_STATE: ACTIVE_STATE,
    DORMANT_STATE: DORMANT_STATE,
    REJECTED_STATE: REJECTED_STATE,
}


class CandidateGateRecord(Protocol):
    pbo: float
    dsr_pvalue: float
    correlation_avg: float

    def oos_fitness(self, metric: str = "sharpe") -> float: ...


@dataclass
class LifecycleConfig:
    candidate_quality_min: float = 0.5
    active_quality_min: float = 0.3
    dormant_revival_quality: float = 0.3
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0
    correlation_max: float = 0.5


def canonical_state(state: str) -> str:
    return _CANONICAL_STATES.get(state, state)


def passes_candidate_gate(
    record: CandidateGateRecord,
    config: LifecycleConfig,
    *,
    metric: str = "sharpe",
) -> bool:
    return (
        record.oos_fitness(metric) >= config.candidate_quality_min
        and record.pbo <= config.pbo_max
        and record.dsr_pvalue <= config.dsr_pvalue_max
        and record.correlation_avg <= config.correlation_max
    )


def compute_transition(state: str, quality: float, config: LifecycleConfig) -> str:
    state = canonical_state(state)
    if state == ACTIVE_STATE:
        if quality < config.active_quality_min:
            return DORMANT_STATE
        return ACTIVE_STATE
    if state == DORMANT_STATE:
        if quality >= config.dormant_revival_quality:
            return ACTIVE_STATE
        return DORMANT_STATE
    return state


def next_live_state(
    state: str,
    estimate: QualityEstimate,
    config: LifecycleConfig,
    *,
    dormant_revival_min_observations: int = 20,
) -> str:
    state = canonical_state(state)
    if state == DORMANT_STATE and estimate.n_observations < dormant_revival_min_observations:
        return DORMANT_STATE
    return compute_transition(state, estimate.blended_quality, config)


ST_CANDIDATE, ST_ACTIVE, ST_DORMANT = 0, 1, 2


def batch_transitions(
    states: np.ndarray,
    quality_scores: np.ndarray,
    config: LifecycleConfig,
) -> np.ndarray:
    new = states.copy()

    active = states == ST_ACTIVE
    new[active & (quality_scores < config.active_quality_min)] = ST_DORMANT

    dormant = states == ST_DORMANT
    new[dormant & (quality_scores >= config.dormant_revival_quality)] = ST_ACTIVE

    return new


def batch_live_transitions(
    states: np.ndarray,
    quality_scores: np.ndarray,
    observation_counts: np.ndarray,
    config: LifecycleConfig,
    *,
    dormant_revival_min_observations: int = 20,
) -> np.ndarray:
    new = batch_transitions(states, quality_scores, config)
    gated = (
        (states == ST_DORMANT)
        & (observation_counts < dormant_revival_min_observations)
    )
    new[gated] = ST_DORMANT
    return new
