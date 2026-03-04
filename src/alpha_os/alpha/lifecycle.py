"""Alpha lifecycle state machine: born → active → probation → dormant (with revival)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import AlphaRecord, AlphaRegistry, AlphaState


@dataclass
class LifecycleConfig:
    oos_quality_min: float = 0.5
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0
    probation_quality_min: float = 0.3
    dormant_quality_max: float = 0.0
    correlation_max: float = 0.5
    dormant_revival_quality: float = 0.3


def compute_transition(state: str, quality: float, config: LifecycleConfig) -> str:
    """Pure function: compute state transition without DB side effects."""
    if state == AlphaState.ACTIVE:
        if quality < config.probation_quality_min:
            return AlphaState.PROBATION
        return AlphaState.ACTIVE
    elif state == AlphaState.PROBATION:
        if quality >= config.oos_quality_min:
            return AlphaState.ACTIVE
        elif quality < config.dormant_quality_max:
            return AlphaState.DORMANT
        return AlphaState.PROBATION
    elif state == AlphaState.DORMANT:
        if quality >= config.dormant_revival_quality:
            return AlphaState.PROBATION
        return AlphaState.DORMANT
    return state


# Vectorized state codes for batch transitions (used by simulator)
ST_ACTIVE, ST_PROBATION, ST_DORMANT = 0, 1, 2


def batch_transitions(
    states: np.ndarray,
    quality_scores: np.ndarray,
    config: LifecycleConfig,
) -> np.ndarray:
    """Vectorized state transitions for batch simulation.

    Args:
        states: int8 array (0=ACTIVE, 1=PROBATION, 2=DORMANT)
        quality_scores: float array of rolling quality metrics
        config: lifecycle thresholds

    Returns:
        new states array (same dtype as input)
    """
    new = states.copy()

    active = states == ST_ACTIVE
    new[active & (quality_scores < config.probation_quality_min)] = ST_PROBATION

    prob = states == ST_PROBATION
    new[prob & (quality_scores >= config.oos_quality_min)] = ST_ACTIVE
    new[prob & (quality_scores < config.dormant_quality_max)] = ST_DORMANT

    dorm = states == ST_DORMANT
    new[dorm & (quality_scores >= config.dormant_revival_quality)] = ST_PROBATION

    return new


class AlphaLifecycle:
    """Manage state transitions for alphas based on validation metrics."""

    def __init__(
        self,
        registry: AlphaRegistry,
        config: LifecycleConfig | None = None,
    ):
        self.registry = registry
        self.config = config or LifecycleConfig()

    def evaluate_born(self, alpha_id: str) -> str:
        """Evaluate a born alpha for promotion to active or rejection."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.BORN:
            return record.state

        if self._passes_gate(record):
            self.registry.update_state(alpha_id, AlphaState.ACTIVE)
            return AlphaState.ACTIVE
        else:
            self.registry.update_state(alpha_id, AlphaState.REJECTED)
            return AlphaState.REJECTED

    def evaluate_active(self, alpha_id: str, live_quality: float) -> str:
        """Check if an active alpha should enter probation."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.ACTIVE:
            return record.state

        if live_quality < self.config.probation_quality_min:
            self.registry.update_state(alpha_id, AlphaState.PROBATION)
            return AlphaState.PROBATION
        return AlphaState.ACTIVE

    def evaluate_probation(self, alpha_id: str, live_quality: float) -> str:
        """Check if a probation alpha should go dormant or be restored."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.PROBATION:
            return record.state

        if live_quality >= self.config.oos_quality_min:
            self.registry.update_state(alpha_id, AlphaState.ACTIVE)
            return AlphaState.ACTIVE
        elif live_quality < self.config.dormant_quality_max:
            self.registry.update_state(alpha_id, AlphaState.DORMANT)
            return AlphaState.DORMANT
        return AlphaState.PROBATION

    def evaluate_dormant(self, alpha_id: str, live_quality: float) -> str:
        """Check if a dormant alpha should revive to probation."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.DORMANT:
            return record.state

        if live_quality >= self.config.dormant_revival_quality:
            self.registry.update_state(alpha_id, AlphaState.PROBATION)
            return AlphaState.PROBATION
        return AlphaState.DORMANT

    def evaluate(self, alpha_id: str, live_quality: float) -> str:
        """Evaluate any alpha regardless of current state. Dispatches to the
        appropriate state-specific method and updates the registry."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        new_state = compute_transition(record.state, live_quality, self.config)
        if new_state != record.state:
            self.registry.update_state(alpha_id, new_state)
        return new_state

    def _passes_gate(self, record: AlphaRecord) -> bool:
        cfg = self.config
        return (
            record.oos_sharpe >= cfg.oos_quality_min
            and record.pbo <= cfg.pbo_max
            and record.dsr_pvalue <= cfg.dsr_pvalue_max
            and record.correlation_avg <= cfg.correlation_max
        )
