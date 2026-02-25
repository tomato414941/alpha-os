"""Alpha lifecycle state machine: born → active → probation → dormant (with revival)."""
from __future__ import annotations

from dataclasses import dataclass

from .registry import AlphaRecord, AlphaRegistry, AlphaState


@dataclass
class LifecycleConfig:
    oos_sharpe_min: float = 0.5
    pbo_max: float = 0.50
    dsr_pvalue_max: float = 0.05
    probation_sharpe_min: float = 0.3
    correlation_max: float = 0.5
    dormant_revival_sharpe: float = 0.3


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
        """Evaluate a born alpha for promotion to active or retirement."""
        record = self.registry.get(alpha_id)
        if record is None:
            raise ValueError(f"Alpha {alpha_id} not found")
        if record.state != AlphaState.BORN:
            return record.state

        if self._passes_gate(record):
            self.registry.update_state(alpha_id, AlphaState.ACTIVE)
            return AlphaState.ACTIVE
        else:
            self.registry.update_state(alpha_id, AlphaState.RETIRED)
            return AlphaState.RETIRED

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
        elif live_sharpe < 0:
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
