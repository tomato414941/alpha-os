"""Lifecycle daemon — daily batch evaluation of alpha state transitions."""
from __future__ import annotations

import logging
import time

import numpy as np

from ..alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from ..alpha.monitor import AlphaMonitor, MonitorConfig
from ..alpha.quality import blend_quality
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, asset_data_dir
from ..forward.tracker import ForwardTracker
from ..governance.audit_log import AuditLog

logger = logging.getLogger(__name__)


class LifecycleDaemon:
    """Evaluate all alphas' forward performance and apply state transitions.

    Designed to run as a daily oneshot (via systemd timer).
    Reads forward_returns, computes rolling Sharpe, transitions states.
    """

    def __init__(self, asset: str, config: Config):
        self.asset = asset
        self.config = config

    def run(self) -> None:
        t0 = time.perf_counter()
        adir = asset_data_dir(self.asset)

        registry = AlphaRegistry(db_path=adir / "alpha_registry.db")
        forward_tracker = ForwardTracker(db_path=adir / "forward_returns.db")
        audit_log = AuditLog(log_path=adir / "audit.jsonl")

        mon_cfg = MonitorConfig(
            rolling_window=self.config.forward.degradation_window,
            min_observations=self.config.live_quality.min_observations,
        )
        monitor = AlphaMonitor(config=mon_cfg)

        lifecycle = AlphaLifecycle(
            registry,
            config=LifecycleConfig(
                oos_quality_min=self.config.lifecycle.oos_quality_min,
                probation_quality_min=self.config.lifecycle.probation_quality_min,
                dormant_quality_max=self.config.lifecycle.dormant_quality_max,
                correlation_max=self.config.lifecycle.correlation_max,
                dormant_revival_quality=self.config.lifecycle.dormant_revival_quality,
            ),
        )

        # Gather all alphas that need lifecycle evaluation
        active = registry.list_by_state(AlphaState.ACTIVE)
        probation = registry.list_by_state(AlphaState.PROBATION)
        dormant = registry.list_by_state(AlphaState.DORMANT)
        all_alphas = active + probation + dormant

        logger.info(
            "Lifecycle evaluation: %d active, %d probation, %d dormant (%d total)",
            len(active), len(probation), len(dormant), len(all_alphas),
        )

        n_transitions = 0
        n_promoted = 0
        n_demoted = 0
        n_skipped = 0
        degradation_window = self.config.forward.degradation_window

        for record in all_alphas:
            returns = forward_tracker.get_returns(record.alpha_id)
            estimate = blend_quality(
                record.oos_fitness(self.config.fitness_metric),
                returns,
                metric=self.config.fitness_metric,
                rolling_window=self.config.forward.degradation_window,
                min_observations=self.config.live_quality.min_observations,
                full_weight_observations=self.config.live_quality.full_weight_observations,
            )
            if (
                record.state == AlphaState.DORMANT
                and estimate.n_observations
                < self.config.live_quality.dormant_revival_min_observations
            ):
                n_skipped += 1
                continue

            recent = returns[-degradation_window:]
            monitor.clear(record.alpha_id)
            monitor.record_batch(record.alpha_id, recent)
            status = monitor.check(record.alpha_id)

            old_state = record.state
            new_state = lifecycle.evaluate(
                record.alpha_id,
                estimate.blended_quality,
            )

            if new_state != old_state:
                n_transitions += 1
                audit_log.log_state_change(
                    record.alpha_id, old_state, new_state,
                    reason=(
                        "lifecycle_daemon: blended="
                        f"{estimate.blended_quality:.3f} "
                        f"prior={estimate.prior_quality:.3f} "
                        f"live={estimate.live_quality:.3f} "
                        f"n={estimate.n_observations} "
                        f"max_dd={status.rolling_max_dd:.3%}"
                    ),
                )

                # Track direction
                state_rank = {
                    AlphaState.DORMANT: 0,
                    AlphaState.PROBATION: 1,
                    AlphaState.ACTIVE: 2,
                }
                if state_rank.get(new_state, 0) > state_rank.get(old_state, 0):
                    n_promoted += 1
                else:
                    n_demoted += 1

        registry.close()
        forward_tracker.close()

        elapsed = time.perf_counter() - t0
        logger.info(
            "Lifecycle complete: %d evaluated, %d skipped, "
            "%d transitions (%d promoted, %d demoted), %.1fs",
            len(all_alphas) - n_skipped, n_skipped,
            n_transitions, n_promoted, n_demoted, elapsed,
        )
