"""Risk manager — drawdown staged response + volatility targeting."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskConfig:
    target_vol: float = 0.15
    dd_stage1_pct: float = 0.05
    dd_stage1_scale: float = 0.75
    dd_stage2_pct: float = 0.10
    dd_stage2_scale: float = 0.50
    dd_stage3_pct: float = 0.15
    dd_stage3_scale: float = 0.25
    max_position: float = 1.0
    lookback_vol: int = 63


class RiskManager:
    """Position-level risk management with DD staged response and vol-targeting."""

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()
        self._equity_curve: list[float] = []
        self._peak: float = 0.0

    def reset(self, initial_equity: float = 1.0) -> None:
        self._equity_curve = [initial_equity]
        self._peak = initial_equity

    def update_equity(self, equity: float) -> None:
        self._equity_curve.append(equity)
        self._peak = max(self._peak, equity)

    @property
    def current_drawdown(self) -> float:
        if self._peak <= 0:
            return 0.0
        current = self._equity_curve[-1] if self._equity_curve else 0.0
        return (self._peak - current) / self._peak

    @property
    def dd_scale(self) -> float:
        """Position scale factor based on current drawdown stage."""
        dd = self.current_drawdown
        cfg = self.config
        if dd >= cfg.dd_stage3_pct:
            return cfg.dd_stage3_scale
        if dd >= cfg.dd_stage2_pct:
            return cfg.dd_stage2_scale
        if dd >= cfg.dd_stage1_pct:
            return cfg.dd_stage1_scale
        return 1.0

    def vol_scale(self, recent_returns: np.ndarray) -> float:
        """Vol-targeting scale: target_vol / realized_vol (annualized)."""
        if len(recent_returns) < 10:
            return 1.0
        realized = float(np.std(recent_returns) * np.sqrt(252))
        if realized < 1e-8:
            return 1.0
        return min(self.config.target_vol / realized, 2.0)

    def adjust_position(
        self, raw_position: float, recent_returns: np.ndarray
    ) -> float:
        """Apply DD scaling + vol-targeting to a raw position."""
        dd_s = self.dd_scale
        vol_s = self.vol_scale(recent_returns)
        adjusted = raw_position * dd_s * vol_s
        return float(np.clip(adjusted, -self.config.max_position, self.config.max_position))

    def adjust_positions(
        self, raw_positions: np.ndarray, returns_history: np.ndarray
    ) -> np.ndarray:
        """Vectorized position adjustment for a time series.

        Parameters
        ----------
        raw_positions : (n_days,) raw position signal
        returns_history : (n_days,) daily return series for vol estimation
        """
        cfg = self.config
        n = len(raw_positions)
        adjusted = np.empty(n)

        equity = 1.0
        peak = 1.0

        for t in range(n):
            # DD scale
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd >= cfg.dd_stage3_pct:
                dd_s = cfg.dd_stage3_scale
            elif dd >= cfg.dd_stage2_pct:
                dd_s = cfg.dd_stage2_scale
            elif dd >= cfg.dd_stage1_pct:
                dd_s = cfg.dd_stage1_scale
            else:
                dd_s = 1.0

            # Vol scale
            lookback_start = max(0, t - cfg.lookback_vol)
            recent = returns_history[lookback_start:t]
            if len(recent) >= 10:
                realized = float(np.std(recent) * np.sqrt(252))
                vol_s = min(cfg.target_vol / realized, 2.0) if realized > 1e-8 else 1.0
            else:
                vol_s = 1.0

            adjusted[t] = np.clip(
                raw_positions[t] * dd_s * vol_s,
                -cfg.max_position, cfg.max_position,
            )

            # Update equity (simplified: assume returns_history drives equity)
            if t > 0:
                equity *= (1 + returns_history[t - 1] * adjusted[t - 1])
                peak = max(peak, equity)

        return adjusted
