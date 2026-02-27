"""Alpha lifecycle state machine: born → active → probation → dormant (with revival)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .registry import AlphaRecord, AlphaRegistry, AlphaState


@dataclass
class LifecycleConfig:
    oos_sharpe_min: float = 0.5
    pbo_max: float = 1.0
    dsr_pvalue_max: float = 1.0
    probation_sharpe_min: float = 0.3
    dormant_sharpe_max: float = 0.0
    correlation_max: float = 0.5
    dormant_revival_sharpe: float = 0.3


def compute_transition(state: str, sharpe: float, config: LifecycleConfig) -> str:
    """Pure function: compute state transition without DB side effects."""
    if state == AlphaState.ACTIVE:
        if sharpe < config.probation_sharpe_min:
            return AlphaState.PROBATION
        return AlphaState.ACTIVE
    elif state == AlphaState.PROBATION:
        if sharpe >= config.oos_sharpe_min:
            return AlphaState.ACTIVE
        elif sharpe < config.dormant_sharpe_max:
            return AlphaState.DORMANT
        return AlphaState.PROBATION
    elif state == AlphaState.DORMANT:
        if sharpe >= config.dormant_revival_sharpe:
            return AlphaState.PROBATION
        return AlphaState.DORMANT
    return state


# Vectorized state codes for batch transitions (used by simulator)
ST_ACTIVE, ST_PROBATION, ST_DORMANT = 0, 1, 2


def batch_transitions(
    states: np.ndarray,
    sharpes: np.ndarray,
    config: LifecycleConfig,
) -> np.ndarray:
    """Vectorized state transitions for batch simulation.

    Args:
        states: int8 array (0=ACTIVE, 1=PROBATION, 2=DORMANT)
        sharpes: float array of rolling Sharpe ratios
        config: lifecycle thresholds

    Returns:
        new states array (same dtype as input)
    """
    new = states.copy()

    active = states == ST_ACTIVE
    new[active & (sharpes < config.probation_sharpe_min)] = ST_PROBATION

    prob = states == ST_PROBATION
    new[prob & (sharpes >= config.oos_sharpe_min)] = ST_ACTIVE
    new[prob & (sharpes < config.dormant_sharpe_max)] = ST_DORMANT

    dorm = states == ST_DORMANT
    new[dorm & (sharpes >= config.dormant_revival_sharpe)] = ST_PROBATION

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

    def evaluate_active(self, alpha_id: str, live_sharpe: float) -> str:
        """Check if an active alpha should enter probation."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.ACTIVE:
            return record.state

        if live_sharpe < self.config.probation_sharpe_min:
            self.registry.update_state(alpha_id, AlphaState.PROBATION)
            return AlphaState.PROBATION
        return AlphaState.ACTIVE

    def evaluate_probation(self, alpha_id: str, live_sharpe: float) -> str:
        """Check if a probation alpha should go dormant or be restored."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.PROBATION:
            return record.state

        if live_sharpe >= self.config.oos_sharpe_min:
            self.registry.update_state(alpha_id, AlphaState.ACTIVE)
            return AlphaState.ACTIVE
        elif live_sharpe < self.config.dormant_sharpe_max:
            self.registry.update_state(alpha_id, AlphaState.DORMANT)
            return AlphaState.DORMANT
        return AlphaState.PROBATION

    def evaluate_dormant(self, alpha_id: str, live_sharpe: float) -> str:
        """Check if a dormant alpha should revive to probation."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.DORMANT:
            return record.state

        if live_sharpe >= self.config.dormant_revival_sharpe:
            self.registry.update_state(alpha_id, AlphaState.PROBATION)
            return AlphaState.PROBATION
        return AlphaState.DORMANT

    def _passes_gate(self, record: AlphaRecord) -> bool:
        cfg = self.config
        return (
            record.oos_sharpe >= cfg.oos_sharpe_min
            and record.pbo <= cfg.pbo_max
            and record.dsr_pvalue <= cfg.dsr_pvalue_max
            and record.correlation_avg <= cfg.correlation_max
        )
