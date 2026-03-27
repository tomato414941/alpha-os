"""Risk manager — drawdown staged response + volatility targeting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


@dataclass
class RiskManagerConfig:
    target_vol: float = 0.15
    dd_stage1_pct: float = 0.05
    dd_stage1_scale: float = 0.75
    dd_stage2_pct: float = 0.10
    dd_stage2_scale: float = 0.50
    dd_stage3_pct: float = 0.15
    dd_stage3_scale: float = 0.25
    max_position: float = 1.0
    lookback_vol: int = 63
    max_vol_scale: float = 1.5


class RiskManager:
    """Position-level risk management with DD staged response and vol-targeting."""

    def __init__(self, config: RiskManagerConfig | None = None):
        self.config = config or RiskManagerConfig()
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
        return min(self.config.target_vol / realized, self.config.max_vol_scale)

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
                vol_s = min(cfg.target_vol / realized, cfg.max_vol_scale) if realized > 1e-8 else 1.0
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


class KellySizing(NamedTuple):
    """Kelly criterion result for a binary outcome market."""
    fraction: float
    edge: float
    expected_return: float


@dataclass
class BinaryOutcomeRiskConfig:
    max_fraction: float = 0.25
    min_edge: float = 0.02
    kelly_fraction: float = 0.5


class BinaryOutcomeRiskManager:
    """Risk manager for binary outcome markets (e.g. Polymarket).

    Uses Kelly criterion for position sizing:
        f* = (p - market_price) / (1 - market_price)
    where p is the model's estimated probability and market_price is the
    current market price (which represents the market's implied probability).

    Positive edge means buy YES, negative edge means sell YES (or buy NO).
    """

    def __init__(self, config: BinaryOutcomeRiskConfig | None = None):
        self.config = config or BinaryOutcomeRiskConfig()

    def kelly_size(self, model_prob: float, market_price: float) -> KellySizing:
        """Compute Kelly fraction for a binary outcome.

        Parameters
        ----------
        model_prob : Model's estimated probability of YES outcome [0, 1].
        market_price : Current market price of YES token [0, 1].

        Returns
        -------
        KellySizing with fraction (of bankroll), edge, and expected return.
        """
        model_prob = float(np.clip(model_prob, 0.01, 0.99))
        market_price = float(np.clip(market_price, 0.01, 0.99))

        edge = model_prob - market_price

        if abs(edge) < self.config.min_edge:
            return KellySizing(fraction=0.0, edge=edge, expected_return=0.0)

        if edge > 0:
            # Buy YES: max loss = price, max gain = 1 - price
            kelly_f = edge / (1.0 - market_price)
        else:
            # Buy NO (sell YES): max loss = 1 - price, max gain = price
            kelly_f = -edge / market_price

        fraction = kelly_f * self.config.kelly_fraction
        fraction = float(np.clip(fraction, -self.config.max_fraction, self.config.max_fraction))

        if edge > 0:
            expected_return = model_prob * (1.0 - market_price) - (1.0 - model_prob) * market_price
        else:
            expected_return = (1.0 - model_prob) * market_price - model_prob * (1.0 - market_price)
            fraction = -fraction

        return KellySizing(
            fraction=fraction,
            edge=edge,
            expected_return=expected_return,
        )

    def position_usd(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        max_position_usd: float = 100.0,
    ) -> float:
        """Compute dollar position size for a binary outcome market.

        Returns positive value for YES, negative for NO.
        """
        sizing = self.kelly_size(model_prob, market_price)
        if abs(sizing.fraction) < 1e-8:
            return 0.0
        raw_usd = sizing.fraction * bankroll
        return float(np.clip(raw_usd, -max_position_usd, max_position_usd))
