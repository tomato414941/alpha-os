"""Legacy daily stake update via marginal contribution."""
from __future__ import annotations

import logging
import sqlite3
import time

from ..config import Config, HYPOTHESIS_OBSERVATIONS_DB_NAME, asset_data_dir
from ..forward.tracker import HypothesisObservationTracker
from .managed_alphas import ManagedAlphaStore
from .stake_update import (
    STAKE_LOOKBACK_DAYS,
    compute_daily_marginal_contributions,
    compute_rolling_marginal_stake,
)

logger = logging.getLogger(__name__)


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
            marginals = compute_daily_marginal_contributions(
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
            new_stake = compute_rolling_marginal_stake(
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
