"""Alpha lifecycle state machine: candidate → active ↔ dormant."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .quality import QualityEstimate
from .managed_alphas import AlphaRecord, ManagedAlphaStore, AlphaState


@dataclass
class LifecycleConfig:
    candidate_quality_min: float = 0.5
    active_quality_min: float = 0.3
    dormant_revival_quality: float = 0.3
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0
    correlation_max: float = 0.5


def passes_candidate_gate(record: AlphaRecord, config: LifecycleConfig) -> bool:
    """Return True when a candidate clears the admission gate."""
    return (
        record.oos_sharpe >= config.candidate_quality_min
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


class AlphaLifecycle:
    """Manage state transitions for alphas based on validation metrics."""

    def __init__(
        self,
        registry: ManagedAlphaStore,
        config: LifecycleConfig | None = None,
    ):
        self.registry = registry
        self.config = config or LifecycleConfig()

    def evaluate_candidate(self, alpha_id: str) -> str:
        """Evaluate a candidate alpha for promotion to active or rejection."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.CANDIDATE:
            return record.state

        if self._passes_gate(record):
            self.registry.update_state(alpha_id, AlphaState.ACTIVE)
            return AlphaState.ACTIVE

        self.registry.update_state(alpha_id, AlphaState.REJECTED)
        return AlphaState.REJECTED

    def evaluate_active(self, alpha_id: str, live_quality: float) -> str:
        """Check if an active alpha should become dormant."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.ACTIVE:
            return record.state

        if live_quality < self.config.active_quality_min:
            self.registry.update_state(alpha_id, AlphaState.DORMANT)
            return AlphaState.DORMANT
        return AlphaState.ACTIVE

    def evaluate_dormant(self, alpha_id: str, live_quality: float) -> str:
        """Check if a dormant alpha should revive to active."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.DORMANT:
            return record.state

        if live_quality >= self.config.dormant_revival_quality:
            self.registry.update_state(alpha_id, AlphaState.ACTIVE)
            return AlphaState.ACTIVE
        return AlphaState.DORMANT

    def evaluate(self, alpha_id: str, live_quality: float) -> str:
        """Evaluate any alpha regardless of current state."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        new_state = compute_transition(record.state, live_quality, self.config)
        if new_state != record.state:
            self.registry.update_state(alpha_id, new_state)
        return new_state

    def evaluate_live(
        self,
        alpha_id: str,
        estimate: QualityEstimate,
        *,
        dormant_revival_min_observations: int = 20,
    ) -> str:
        """Evaluate live state from a blended quality estimate."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        new_state = next_live_state(
            record.state,
            estimate,
            self.config,
            dormant_revival_min_observations=dormant_revival_min_observations,
        )
        if new_state != record.state:
            self.registry.update_state(alpha_id, new_state)
        return new_state

    def _passes_gate(self, record: AlphaRecord) -> bool:
        return passes_candidate_gate(record, self.config)


# ---------------------------------------------------------------------------
# Path A: Tenure bonus — reward long-lived alphas to stabilize top-30
# ---------------------------------------------------------------------------


def tenure_days(record: AlphaRecord, now: float | None = None) -> float:
    """Days since alpha creation."""
    now = now or time.time()
    return max(0.0, (now - record.created_at) / 86400.0)


def tenure_bonus(
    record: AlphaRecord,
    max_bonus: float = 0.2,
    half_life_days: float = 7.0,
    now: float | None = None,
) -> float:
    """Quality bonus that grows with alpha age (exponential saturation)."""
    age = tenure_days(record, now)
    if age <= 0 or half_life_days <= 0:
        return 0.0
    return max_bonus * (1.0 - 0.5 ** (age / half_life_days))


def apply_tenure_bonus(
    quality: float,
    record: AlphaRecord,
    max_bonus: float = 0.2,
    half_life_days: float = 7.0,
    now: float | None = None,
) -> float:
    """Add tenure bonus to a quality score."""
    return quality + tenure_bonus(record, max_bonus, half_life_days, now)
