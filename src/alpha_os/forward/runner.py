"""Forward test runner — daily evaluation loop for adopted alphas."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from ..alpha.monitor import AlphaMonitor, MonitorConfig
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, DATA_DIR
from ..data.client import SignalClient
from ..data.store import DataStore
from ..data.universe import price_signal, MACRO_SIGNALS
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
    """Run one cycle of forward testing on all ACTIVE, PROBATION, and DORMANT alphas."""

    def __init__(
        self,
        asset: str,
        config: Config,
        forward_config: ForwardConfig | None = None,
        registry: AlphaRegistry | None = None,
        tracker: ForwardTracker | None = None,
        monitor: AlphaMonitor | None = None,
        lifecycle: AlphaLifecycle | None = None,
        audit_log: AuditLog | None = None,
        store: DataStore | None = None,
    ):
        self.asset = asset
        self.config = config
        self.fwd_config = forward_config or ForwardConfig()

        try:
            price_sig = price_signal(asset)
        except KeyError:
            price_sig = asset.lower()
        self.price_signal = price_sig
        self.features = [price_sig] + MACRO_SIGNALS

        self.registry = registry or AlphaRegistry()
        self.tracker = tracker or ForwardTracker()
        self.audit_log = audit_log or AuditLog()

        mon_cfg = MonitorConfig(rolling_window=self.fwd_config.degradation_window)
        self.monitor = monitor or AlphaMonitor(config=mon_cfg)

        self.lifecycle = lifecycle or AlphaLifecycle(
            self.registry,
            config=LifecycleConfig(
                oos_sharpe_min=config.validation.oos_sharpe_min,
            ),
        )

        if store is not None:
            self.store = store
        else:
            client = SignalClient(
                base_url=config.api.base_url,
                timeout=config.api.timeout,
            )
            self.store = DataStore(DATA_DIR / "alpha_cache.db", client)

    def run_cycle(self) -> ForwardCycleResult:
        t0 = time.perf_counter()

        logger.info("Forward: syncing %d signals...", len(self.features))
        self.store.sync(self.features)

        active = self.registry.list_by_state(AlphaState.ACTIVE)
        probation = self.registry.list_by_state(AlphaState.PROBATION)
        dormant = self.registry.list_by_state(AlphaState.DORMANT)
        all_alphas = active + probation + dormant

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
                matrix = self.store.get_matrix(self.features, start=start_date)
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

                old_state = record.state
                new_state = self.lifecycle.evaluate(
                    alpha_id, status.rolling_sharpe,
                )

                if new_state != old_state:
                    self.audit_log.log_state_change(
                        alpha_id, old_state, new_state,
                        reason=f"forward_test: sharpe={status.rolling_sharpe:.3f}, "
                               f"max_dd={status.rolling_max_dd:.3%}",
                    )
                    if new_state == AlphaState.PROBATION and old_state == AlphaState.ACTIVE:
                        n_degraded += 1
                    elif new_state == AlphaState.DORMANT:
                        n_dormant += 1
                    elif new_state == AlphaState.REJECTED:
                        n_rejected += 1
                    elif (
                        new_state == AlphaState.PROBATION
                        and old_state == AlphaState.DORMANT
                    ):
                        n_revived += 1
                    elif (
                        new_state == AlphaState.ACTIVE
                        and old_state == AlphaState.PROBATION
                    ):
                        n_restored += 1

                logger.info(
                    "  %s: ret=%.4f sharpe=%.3f dd=%.3f state=%s%s",
                    alpha_id, daily_return, status.rolling_sharpe,
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
