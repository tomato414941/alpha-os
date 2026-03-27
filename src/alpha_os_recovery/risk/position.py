"""Position sizing — convert alpha signals to dollar positions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PositionConfig:
    capital: float = 10000.0
    max_position_pct: float = 1.0
    min_trade_usd: float = 10.0


def signal_to_positions(
    signal: np.ndarray,
    prices: np.ndarray,
    config: PositionConfig | None = None,
) -> np.ndarray:
    """Convert normalized signal [-1, 1] to share quantities.

    Returns array of share counts (fractional allowed).
    """
    cfg = config or PositionConfig()
    n = min(len(signal), len(prices))
    sig = signal[:n].copy()
    p = prices[:n]

    # Clip to [-1, 1]
    sig = np.clip(sig, -1.0, 1.0)

    # Dollar allocation
    dollar_pos = sig * cfg.capital * cfg.max_position_pct

    # Convert to shares
    shares = np.where(p > 0, dollar_pos / p, 0.0)

    # Filter out trades below minimum
    dollar_value = np.abs(shares * p)
    shares[dollar_value < cfg.min_trade_usd] = 0.0

    return shares


def compute_pnl(
    shares: np.ndarray,
    prices: np.ndarray,
) -> np.ndarray:
    """Compute daily PnL from share positions and prices.

    PnL[t] = shares[t-1] * (prices[t] - prices[t-1])
    """
    n = min(len(shares), len(prices))
    if n < 2:
        return np.array([])

    price_changes = np.diff(prices[:n])
    pnl = shares[:n - 1] * price_changes
    return pnl
