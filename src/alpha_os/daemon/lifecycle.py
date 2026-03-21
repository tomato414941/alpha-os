"""Lifecycle daemon — daily stake update from forward returns."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..alpha.managed_alphas import ManagedAlphaStore, AlphaState
from ..config import Config, asset_data_dir
from ..forward.tracker import ForwardTracker

logger = logging.getLogger(__name__)

STAKE_LOOKBACK_DAYS = 60
MIN_STAKE_OBSERVATIONS = 10


def _compute_rolling_stake(
    returns: np.ndarray,
    lookback: int = STAKE_LOOKBACK_DAYS,
    prior_stake: float = 0.0,
) -> float:
    """Compute stake from rolling mean of recent returns.

    Stake = mean of last N daily returns. Positive returns → positive stake.
    If insufficient observations, returns prior_stake unchanged.
    """
    if len(returns) < MIN_STAKE_OBSERVATIONS:
        return prior_stake
    recent = returns[-lookback:] if len(returns) > lookback else returns
    mean_ret = float(np.mean(recent))
    return max(mean_ret, 0.0)


class LifecycleDaemon:
    """Daily stake update from forward returns.

    Designed to run as a daily oneshot (via systemd timer).
    Reads forward_returns, computes rolling stake, updates registry.
    No state transitions — stake is the sole selection mechanism.
    """

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config

    def run(self) -> None:
        t0 = time.perf_counter()
        adir = asset_data_dir(self.asset)

        registry = ManagedAlphaStore(db_path=adir / "alpha_registry.db")
        forward_tracker = ForwardTracker(db_path=adir / "forward_returns.db")

        active = registry.list_by_state(AlphaState.ACTIVE)
        dormant = registry.list_by_state(AlphaState.DORMANT)
        all_alphas = active + dormant

        logger.info(
            "Stake update: %d active, %d dormant (%d total)",
            len(active), len(dormant), len(all_alphas),
        )

        n_stake_updated = 0
        stake_updates: dict[str, float] = {}

        for record in all_alphas:
            returns = forward_tracker.get_returns(record.alpha_id)
            new_stake = _compute_rolling_stake(np.array(returns), prior_stake=record.stake)
            if abs(new_stake - record.stake) > 1e-6:
                stake_updates[record.alpha_id] = new_stake
                n_stake_updated += 1

        if stake_updates:
            registry.bulk_update_stakes(stake_updates)

        registry.close()
        forward_tracker.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Stake update complete: %d evaluated, %d updated, %.1fs",
            len(all_alphas), n_stake_updated, elapsed,
        )
