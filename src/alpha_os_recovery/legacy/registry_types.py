"""Legacy registry data types."""
from __future__ import annotations

from dataclasses import dataclass, field


class AlphaState:
    CANDIDATE = "candidate"
    ACTIVE = "active"
    DORMANT = "dormant"
    REJECTED = "rejected"

    _CANONICAL = {
        CANDIDATE: CANDIDATE,
        ACTIVE: ACTIVE,
        DORMANT: DORMANT,
        REJECTED: REJECTED,
    }

    @classmethod
    def canonical(cls, state: str) -> str:
        return cls._CANONICAL.get(state, state)


@dataclass
class AlphaRecord:
    alpha_id: str
    expression: str
    state: str = AlphaState.CANDIDATE
    fitness: float = 0.0
    oos_sharpe: float = 0.0
    oos_log_growth: float = 0.0
    pbo: float = 1.0
    dsr_pvalue: float = 1.0
    turnover: float = 0.0
    correlation_avg: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: dict = field(default_factory=dict)
    stake: float = 0.0

    _OOS_FITNESS_MAP = {"sharpe": "oos_sharpe", "log_growth": "oos_log_growth"}

    def oos_fitness(self, metric: str = "sharpe") -> float:
        return getattr(self, self._OOS_FITNESS_MAP[metric])


@dataclass(frozen=True)
class DeployedAlphaEntry:
    alpha_id: str
    slot: int
    deployed_at: float
    deployment_score: float = 0.0
    metadata: dict = field(default_factory=dict)
