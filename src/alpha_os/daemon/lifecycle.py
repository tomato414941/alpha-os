"""Lifecycle daemon — daily stake update via marginal contribution."""
from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np

from ..legacy.managed_alphas import ManagedAlphaStore
from ..config import Config, HYPOTHESIS_OBSERVATIONS_DB_NAME, asset_data_dir
from ..forward.tracker import ForwardTracker

logger = logging.getLogger(__name__)

STAKE_LOOKBACK_DAYS = 60
MIN_STAKE_OBSERVATIONS = 10


def _compute_daily_marginal_contributions(
    forward_db_path: str,
    hypothesis_ids: list[str],
    stakes: dict[str, float],
    date: str,
) -> dict[str, float]:
    """Compute leave-one-out marginal contribution for each alpha on one day.

    portfolio = Σ(stake_i × signal_i) / Σ(stake_i)
    portfolio_pnl = portfolio × price_return
    marginal_j = portfolio_pnl - portfolio_without_j_pnl

    Returns {hypothesis_id: marginal_contribution}.
    """
    conn = sqlite3.connect(forward_db_path)
    rows = conn.execute(
        "SELECT hypothesis_id, signal_value, daily_return FROM forward_returns WHERE date = ?",
        (date,),
    ).fetchall()
    conn.close()

    if not rows:
        return {}

    # Build signal/return arrays for alphas with stake > 0
    signals: dict[str, float] = {}
    for hypothesis_id, sig_val, daily_ret in rows:
        if (
            hypothesis_id in hypothesis_ids
            and hypothesis_id in stakes
            and stakes[hypothesis_id] > 0
            and np.isfinite(sig_val)
        ):
            signals[hypothesis_id] = sig_val

    if len(signals) < 2:
        return {}

    # Infer price return from any alpha: daily_return = signal_value × price_return
    price_return = None
    for hypothesis_id, sig_val, daily_ret in rows:
        if abs(sig_val) > 1e-8 and np.isfinite(daily_ret):
            price_return = daily_ret / sig_val
            break
    if price_return is None:
        return {}

    # Compute full portfolio
    ids = list(signals.keys())
    stake_arr = np.array([stakes.get(aid, 0.0) for aid in ids])
    sig_arr = np.array([signals[aid] for aid in ids])
    total_stake = stake_arr.sum()
    if total_stake <= 0:
        return {}

    portfolio = float(np.dot(stake_arr, sig_arr) / total_stake)
    full_pnl = portfolio * price_return

    # Leave-one-out marginal for each alpha
    marginals: dict[str, float] = {}
    for i, hypothesis_id in enumerate(ids):
        remaining_stake = total_stake - stake_arr[i]
        if remaining_stake <= 0:
            marginals[hypothesis_id] = full_pnl
            continue
        portfolio_without = float(
            (np.dot(stake_arr, sig_arr) - stake_arr[i] * sig_arr[i]) / remaining_stake
        )
        pnl_without = portfolio_without * price_return
        marginals[hypothesis_id] = full_pnl - pnl_without

    return marginals


def _compute_rolling_marginal_stake(
    marginal_history: list[float],
    lookback: int = STAKE_LOOKBACK_DAYS,
    prior_stake: float = 0.0,
) -> float:
    """Compute stake from rolling mean of marginal contributions."""
    if len(marginal_history) < MIN_STAKE_OBSERVATIONS:
        return prior_stake
    recent = marginal_history[-lookback:]
    mean_marginal = float(np.mean(recent))
    return max(mean_marginal, 0.0)


class LifecycleDaemon:
    """Daily stake update via leave-one-out marginal contribution.

    Designed to run as a daily oneshot (via systemd timer).
    For each recent day, computes each alpha's marginal contribution
    to portfolio P&L. Stake = rolling mean of marginal contributions.
    """

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config

    def run(self) -> None:
        t0 = time.perf_counter()
        adir = asset_data_dir(self.asset)

        registry = ManagedAlphaStore(db_path=adir / "alpha_registry.db")
        forward_tracker = ForwardTracker(db_path=adir / HYPOTHESIS_OBSERVATIONS_DB_NAME)
        forward_db = str(adir / HYPOTHESIS_OBSERVATIONS_DB_NAME)

        all_alphas = [r for r in registry.list_all() if r.stake > 0]
        stakes = {r.alpha_id: r.stake for r in all_alphas}

        logger.info("Stake update: %d alphas with stake > 0", len(all_alphas))

        # Get available dates
        conn = sqlite3.connect(forward_db)
        dates = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT date FROM forward_returns ORDER BY date DESC LIMIT ?",
                (STAKE_LOOKBACK_DAYS,),
            ).fetchall()
        ]
        conn.close()

        if not dates:
            logger.info("No forward return dates available")
            registry.close()
            forward_tracker.close()
            return

        # Compute marginal contributions for each date
        # marginal_history[alpha_id] = [marginal_day1, marginal_day2, ...]
        marginal_history: dict[str, list[float]] = {r.alpha_id: [] for r in all_alphas}

        for d in reversed(dates):  # oldest first
            marginals = _compute_daily_marginal_contributions(
                forward_db, [r.alpha_id for r in all_alphas], stakes, d,
            )
            for aid in marginal_history:
                marginal_history[aid].append(marginals.get(aid, 0.0))

        # Update stakes from rolling marginal
        n_stake_updated = 0
        stake_updates: dict[str, float] = {}

        for record in all_alphas:
            history = marginal_history.get(record.alpha_id, [])
            new_stake = _compute_rolling_marginal_stake(
                history, prior_stake=record.stake,
            )
            if abs(new_stake - record.stake) > 1e-6:
                stake_updates[record.alpha_id] = new_stake
                n_stake_updated += 1

        if stake_updates:
            registry.bulk_update_stakes(stake_updates)

        registry.close()
        forward_tracker.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Stake update complete: %d evaluated, %d dates, %d updated, %.1fs",
            len(all_alphas), len(dates), n_stake_updated, elapsed,
        )
