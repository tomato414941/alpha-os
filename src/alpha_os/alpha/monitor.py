"""Alpha monitor — rolling performance tracking and degradation detection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MonitorConfig:
    rolling_window: int = 63
    sharpe_threshold: float = 0.0
    drawdown_threshold: float = 0.10
    min_observations: int = 20


@dataclass
class MonitorStatus:
    alpha_id: str
    rolling_sharpe: float
    rolling_max_dd: float
    is_degraded: bool
    degradation_reasons: list[str]


class AlphaMonitor:
    """Track live alpha performance and flag degradation."""

    def __init__(self, config: MonitorConfig | None = None):
        self.config = config or MonitorConfig()
        self._returns: dict[str, list[float]] = {}

    def record(self, alpha_id: str, daily_return: float) -> None:
        if alpha_id not in self._returns:
            self._returns[alpha_id] = []
        self._returns[alpha_id].append(daily_return)

    def record_batch(self, alpha_id: str, returns: list[float] | np.ndarray) -> None:
        if alpha_id not in self._returns:
            self._returns[alpha_id] = []
        self._returns[alpha_id].extend(float(r) for r in returns)

    def check(self, alpha_id: str) -> MonitorStatus:
        """Check current health of an alpha."""
        cfg = self.config
        rets = self._returns.get(alpha_id, [])

        if len(rets) < cfg.min_observations:
            return MonitorStatus(
                alpha_id=alpha_id,
                rolling_sharpe=0.0,
                rolling_max_dd=0.0,
                is_degraded=False,
                degradation_reasons=[],
            )

        recent = np.array(rets[-cfg.rolling_window:])
        std = recent.std()
        rolling_sharpe = float(recent.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        cum = np.cumprod(1 + recent)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        rolling_max_dd = float(dd.max())

        reasons: list[str] = []
        if rolling_sharpe < cfg.sharpe_threshold:
            reasons.append(
                f"Rolling Sharpe {rolling_sharpe:.3f} < {cfg.sharpe_threshold}"
            )
        if rolling_max_dd > cfg.drawdown_threshold:
            reasons.append(
                f"Rolling MaxDD {rolling_max_dd:.1%} > {cfg.drawdown_threshold:.1%}"
            )

        return MonitorStatus(
            alpha_id=alpha_id,
            rolling_sharpe=rolling_sharpe,
            rolling_max_dd=rolling_max_dd,
            is_degraded=len(reasons) > 0,
            degradation_reasons=reasons,
        )

    def check_all(self) -> list[MonitorStatus]:
        return [self.check(aid) for aid in self._returns]

    def clear(self, alpha_id: str) -> None:
        self._returns.pop(alpha_id, None)
