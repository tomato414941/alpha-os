"""Execution optimizer: microstructure-aware order timing."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from signal_noise.client import SignalClient

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    imbalance_threshold: float = 0.1
    vpin_threshold: float = 0.5
    spread_threshold_bps: float = 5.0
    max_slices: int = 5


class ExecutionOptimizer:
    """Layer 1: decide WHEN to execute based on microstructure signals."""

    def __init__(
        self,
        client: SignalClient,
        config: ExecutionConfig | None = None,
    ):
        self._client = client
        self._config = config or ExecutionConfig()

    def get_signal(self, name: str) -> float | None:
        try:
            latest = self._client.get_latest(name)
            if latest is None:
                return None
            return latest.get("value")
        except Exception:
            return None

    def optimal_execution_window(self, side: str) -> bool:
        """Check if microstructure conditions favor execution now."""
        imbalance = self.get_signal("book_imbalance_btc")
        vpin = self.get_signal("vpin_btc")
        spread = self.get_signal("spread_bps_btc")

        # If signals unavailable, don't block execution
        if any(v is None for v in [imbalance, vpin, spread]):
            return True

        cfg = self._config

        if vpin > cfg.vpin_threshold:
            logger.info("Execution delayed: VPIN %.3f > %.3f", vpin, cfg.vpin_threshold)
            return False

        if spread > cfg.spread_threshold_bps:
            logger.info("Execution delayed: spread %.1f > %.1f bps", spread, cfg.spread_threshold_bps)
            return False

        # Unfavorable imbalance for our side
        if side == "buy" and imbalance < -cfg.imbalance_threshold:
            logger.info("Execution delayed: ask-heavy imbalance %.3f for buy", imbalance)
            return False
        if side == "sell" and imbalance > cfg.imbalance_threshold:
            logger.info("Execution delayed: bid-heavy imbalance %.3f for sell", imbalance)
            return False

        return True

    def split_order(self, total_qty: float) -> list[float]:
        """TWAP-style order splitting based on current conditions."""
        cfg = self._config
        vpin = self.get_signal("vpin_btc")
        spread = self.get_signal("spread_bps_btc")

        if vpin is not None and vpin > cfg.vpin_threshold:
            n_slices = cfg.max_slices
        elif spread is not None and spread > cfg.spread_threshold_bps:
            n_slices = max(3, cfg.max_slices - 1)
        else:
            n_slices = 1

        slice_qty = total_qty / n_slices
        return [slice_qty] * n_slices
