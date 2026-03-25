"""Legacy daily stake update via marginal contribution."""
from __future__ import annotations

import logging
import sqlite3
import time

import numpy as np

from ..config import Config, HYPOTHESIS_OBSERVATIONS_DB_NAME, asset_data_dir
from ..forward.tracker import HypothesisObservationTracker
from .managed_alphas import ManagedAlphaStore

logger = logging.getLogger(__name__)

STAKE_LOOKBACK_DAYS = 60
MIN_STAKE_OBSERVATIONS = 10


def _compute_daily_marginal_contributions(
    forward_db_path: str,
    hypothesis_ids: list[str],
    stakes: dict[str, float],
    date: str,
) -> dict[str, float]:
    """Compute leave-one-out marginal contribution for each legacy stake on one day."""
    conn = sqlite3.connect(forward_db_path)
    rows = conn.execute(
        "SELECT hypothesis_id, signal_value, daily_return "
        "FROM hypothesis_observations WHERE date = ?",
        (date,),
    ).fetchall()
    conn.close()

    if not rows:
        return {}

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

    price_return = None
    for _hypothesis_id, sig_val, daily_ret in rows:
        if abs(sig_val) > 1e-8 and np.isfinite(daily_ret):
            price_return = daily_ret / sig_val
            break
    if price_return is None:
        return {}

    ids = list(signals.keys())
    stake_arr = np.array([stakes.get(hypothesis_id, 0.0) for hypothesis_id in ids])
    sig_arr = np.array([signals[hypothesis_id] for hypothesis_id in ids])
    total_stake = stake_arr.sum()
    if total_stake <= 0:
        return {}

    portfolio = float(np.dot(stake_arr, sig_arr) / total_stake)
    full_pnl = portfolio * price_return

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
    """Legacy daily stake update against the registry substrate."""

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config

    def run(self) -> None:
        t0 = time.perf_counter()
        adir = asset_data_dir(self.asset)

        registry = ManagedAlphaStore(db_path=adir / "alpha_registry.db")
        observation_tracker = HypothesisObservationTracker(
            db_path=adir / HYPOTHESIS_OBSERVATIONS_DB_NAME
        )
        observation_db = str(adir / HYPOTHESIS_OBSERVATIONS_DB_NAME)

        legacy_records = [record for record in registry.list_all() if record.stake > 0]
        stakes = {record.alpha_id: record.stake for record in legacy_records}

        logger.info("Legacy stake update: %d records with stake > 0", len(legacy_records))

        conn = sqlite3.connect(observation_db)
        dates = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT date FROM hypothesis_observations ORDER BY date DESC LIMIT ?",
                (STAKE_LOOKBACK_DAYS,),
            ).fetchall()
        ]
        conn.close()

        if not dates:
            logger.info("No observation dates available for legacy stake update")
            registry.close()
            observation_tracker.close()
            return

        marginal_history: dict[str, list[float]] = {
            record.alpha_id: [] for record in legacy_records
        }

        for date in reversed(dates):
            marginals = _compute_daily_marginal_contributions(
                observation_db,
                [record.alpha_id for record in legacy_records],
                stakes,
                date,
            )
            for hypothesis_id in marginal_history:
                marginal_history[hypothesis_id].append(marginals.get(hypothesis_id, 0.0))

        n_stake_updated = 0
        stake_updates: dict[str, float] = {}

        for record in legacy_records:
            history = marginal_history.get(record.alpha_id, [])
            new_stake = _compute_rolling_marginal_stake(
                history,
                prior_stake=record.stake,
            )
            if abs(new_stake - record.stake) > 1e-6:
                stake_updates[record.alpha_id] = new_stake
                n_stake_updated += 1

        if stake_updates:
            registry.bulk_update_stakes(stake_updates)

        registry.close()
        observation_tracker.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Legacy stake update complete: %d evaluated, %d dates, %d updated, %.1fs",
            len(legacy_records),
            len(dates),
            n_stake_updated,
            elapsed,
        )
