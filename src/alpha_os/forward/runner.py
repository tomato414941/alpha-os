"""Forward test runner — daily evaluation loop for active and dormant alphas."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.lifecycle import AlphaLifecycle
from ..alpha.monitor import AlphaMonitor
from ..alpha.managed_alphas import ManagedAlphaStore, AlphaState
from ..config import Config, DATA_DIR, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..data.store import DataStore
from ..data.universe import price_signal, MACRO_SIGNALS, build_hourly_feature_list
from ..dsl import parse
from ..governance.audit_log import AuditLog
from .tracker import ForwardTracker

logger = logging.getLogger(__name__)


@dataclass
class ForwardConfig:
    check_interval: int = 14400
    min_forward_days: int = 30
    degradation_window: int = 63


@dataclass
class ForwardCycleResult:
    n_evaluated: int
    n_degraded: int
    n_rejected: int
    n_restored: int
    n_dormant: int
    n_revived: int
    elapsed: float


class ForwardRunner:
    """Run one cycle of forward testing on all ACTIVE and DORMANT alphas."""

    def __init__(
        self,
        asset: str,
        config: Config,
        forward_config: ForwardConfig | None = None,
        registry: ManagedAlphaStore | None = None,
        tracker: ForwardTracker | None = None,
        monitor: AlphaMonitor | None = None,
        lifecycle: AlphaLifecycle | None = None,
        audit_log: AuditLog | None = None,
        store: DataStore | None = None,
        resolution: str = "1d",
    ):
        self.asset = asset
        self.config = config
        self.fwd_config = forward_config or ForwardConfig()
        self.resolution = resolution

        try:
            price_sig = price_signal(asset)
        except KeyError:
            price_sig = asset.lower()
        self.price_signal = price_sig

        if resolution == "1h":
            self.features = build_hourly_feature_list(asset)
        else:
            self.features = [price_sig] + MACRO_SIGNALS

        adir = asset_data_dir(asset)
        if resolution == "1h":
            self.registry = registry or ManagedAlphaStore(db_path=adir / "alpha_registry_l2.db")
            self.tracker = tracker or ForwardTracker(db_path=adir / "forward_returns_l2.db")
        else:
            self.registry = registry or ManagedAlphaStore(db_path=adir / "alpha_registry.db")
            self.tracker = tracker or ForwardTracker(db_path=adir / "forward_returns.db")
        self.audit_log = audit_log or AuditLog(log_path=adir / "audit.jsonl")

        mon_cfg = config.to_monitor_config()
        self.monitor = monitor or AlphaMonitor(config=mon_cfg)

        self.lifecycle = lifecycle or AlphaLifecycle(
            self.registry,
            config=config.to_lifecycle_config(),
        )

        if store is not None:
            self.store = store
        else:
            client = build_signal_client_from_config(config.api)
            db_name = "alpha_cache_l2.db" if resolution == "1h" else "alpha_cache.db"
            self.store = DataStore(DATA_DIR / db_name, client)

    def run_cycle(self) -> ForwardCycleResult:
        t0 = time.perf_counter()

        logger.info("Forward: syncing %d signals...", len(self.features))
        self.store.sync(self.features, resolution=self.resolution)

        active = self.registry.list_by_state(AlphaState.ACTIVE)
        dormant = self.registry.list_by_state(AlphaState.DORMANT)
        all_alphas = active + dormant

        n_evaluated = 0
        n_degraded = 0
        n_rejected = 0
        n_restored = 0
        n_dormant = 0
        n_revived = 0
        n_failed = 0

        today = date.today().isoformat()

        for record in all_alphas:
            alpha_id = record.alpha_id

            start_date = self.tracker.get_start_date(alpha_id)
            if start_date is None:
                self.tracker.register_alpha(alpha_id, today)
                start_date = today

            last_date = self.tracker.get_last_date(alpha_id)
            if last_date and last_date >= today:
                logger.debug("%s already recorded for %s", alpha_id, today)
                continue

            try:
                expr = parse(record.expression)
                matrix = self.store.get_matrix(self.features, start=start_date,
                                                resolution=self.resolution)
                if len(matrix) < 2:
                    logger.debug("Insufficient data for %s (%d rows)", alpha_id, len(matrix))
                    continue

                data = {col: matrix[col].values for col in matrix.columns}
                signal = evaluate_expression(expr, data, len(matrix))

                prices = data[self.price_signal]
                if len(prices) < 2:
                    continue
                price_return = (prices[-1] - prices[-2]) / prices[-2]

                signal_norm = normalize_signal(signal)
                signal_yesterday = float(signal_norm[-2])
                daily_return = signal_yesterday * price_return

                self.tracker.record(alpha_id, today, signal_yesterday, daily_return)
                n_evaluated += 1

                all_returns = self.tracker.get_returns(alpha_id)
                self.monitor.clear(alpha_id)
                self.monitor.record_batch(alpha_id, all_returns)
                status = self.monitor.check(alpha_id)
                estimate = self.config.estimate_alpha_quality(
                    record.oos_fitness(self.config.fitness_metric),
                    all_returns,
                )

                old_state = record.state
                new_state = self.lifecycle.evaluate_live(
                    alpha_id,
                    estimate,
                    dormant_revival_min_observations=(
                        self.config.live_quality.dormant_revival_min_observations
                    ),
                )

                if new_state != old_state:
                    self.audit_log.log_state_change(
                        alpha_id, old_state, new_state,
                        reason=(
                            "forward_test: blended="
                            f"{estimate.blended_quality:.3f} "
                            f"prior={estimate.prior_quality:.3f} "
                            f"live={estimate.live_quality:.3f} "
                            f"n={estimate.n_observations} "
                            f"max_dd={status.rolling_max_dd:.3%}"
                        ),
                    )
                    if new_state == AlphaState.DORMANT and old_state == AlphaState.ACTIVE:
                        n_degraded += 1
                        n_dormant += 1
                    elif new_state == AlphaState.REJECTED:
                        n_rejected += 1
                    elif new_state == AlphaState.ACTIVE and old_state == AlphaState.DORMANT:
                        n_revived += 1
                        n_restored += 1

                logger.info(
                    "  %s: ret=%.4f blended=%.3f prior=%.3f live=%.3f dd=%.3f state=%s%s",
                    alpha_id, daily_return,
                    estimate.blended_quality,
                    estimate.prior_quality,
                    estimate.live_quality,
                    status.rolling_max_dd, new_state,
                    " [DEGRADED]" if status.is_degraded else "",
                )

            except EvaluationError as exc:
                logger.warning("Failed to evaluate %s: %s", alpha_id, exc)
                n_failed += 1

        elapsed = time.perf_counter() - t0

        if n_failed:
            logger.info("Forward cycle: %d/%d alphas failed evaluation", n_failed, len(all_alphas))

        self.audit_log.log(
            "forward_cycle",
            details={
                "n_evaluated": n_evaluated,
                "n_degraded": n_degraded,
                "n_rejected": n_rejected,
                "n_restored": n_restored,
                "n_dormant": n_dormant,
                "n_revived": n_revived,
                "n_failed": n_failed,
                "elapsed": round(elapsed, 2),
                "date": today,
            },
        )

        return ForwardCycleResult(
            n_evaluated=n_evaluated,
            n_degraded=n_degraded,
            n_rejected=n_rejected,
            n_restored=n_restored,
            n_dormant=n_dormant,
            n_revived=n_revived,
            elapsed=elapsed,
        )

    def print_summary(self) -> None:
        alpha_ids = self.tracker.tracked_alpha_ids()
        if not alpha_ids:
            print("No alphas in forward testing.")
            return

        print(
            f"\n{'Alpha':>20}  {'Days':>5}  {'Return':>8}  {'Sharpe':>8}  "
            f"{'MaxDD':>8}  {'State':>10}  Since"
        )
        print("-" * 85)

        for alpha_id in alpha_ids:
            summary = self.tracker.summary(alpha_id)
            record = self.registry.get(alpha_id)
            state = record.state if record else "?"
            if summary:
                print(
                    f"{alpha_id:>20}  {summary.n_days:>5}  "
                    f"{summary.total_return:>7.2%}  {summary.sharpe:>8.3f}  "
                    f"{summary.max_dd:>7.2%}  {state:>10}  "
                    f"{summary.forward_start_date}"
                )

    def close(self) -> None:
        self.tracker.close()
        self.store.close()
        self.registry.close()
