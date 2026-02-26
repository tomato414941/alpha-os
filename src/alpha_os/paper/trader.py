"""Paper trader — connects existing components into a daily paper trading cycle."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import date

import numpy as np

from ..alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from ..alpha.monitor import AlphaMonitor, MonitorConfig
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, DATA_DIR
from ..data.client import SignalClient
from ..data.store import DataStore
from ..data.universe import price_signal, MACRO_SIGNALS
from ..dsl import parse
from ..execution.executor import Fill
from ..execution.paper import PaperExecutor
from ..forward.tracker import ForwardTracker
from ..governance.audit_log import AuditLog
from ..risk.manager import RiskManager, RiskConfig
from .tracker import PaperPortfolioTracker, PortfolioSnapshot

logger = logging.getLogger(__name__)


@dataclass
class PaperCycleResult:
    date: str
    combined_signal: float
    fills: list[Fill]
    portfolio_value: float
    daily_pnl: float
    daily_return: float
    dd_scale: float
    vol_scale: float
    n_alphas_active: int
    n_alphas_evaluated: int


class PaperTrader:
    """Orchestrates daily paper trading by connecting existing components."""

    def __init__(
        self,
        asset: str,
        config: Config,
        registry: AlphaRegistry | None = None,
        portfolio_tracker: PaperPortfolioTracker | None = None,
        forward_tracker: ForwardTracker | None = None,
        monitor: AlphaMonitor | None = None,
        lifecycle: AlphaLifecycle | None = None,
        executor: PaperExecutor | None = None,
        risk_manager: RiskManager | None = None,
        audit_log: AuditLog | None = None,
        store: DataStore | None = None,
    ):
        self.asset = asset
        self.config = config

        try:
            self.price_signal = price_signal(asset)
        except KeyError:
            self.price_signal = asset.lower()
        self.features = [self.price_signal] + MACRO_SIGNALS

        self.registry = registry or AlphaRegistry()
        self.portfolio_tracker = portfolio_tracker or PaperPortfolioTracker()
        self.forward_tracker = forward_tracker or ForwardTracker()
        self.audit_log = audit_log or AuditLog()

        mon_cfg = MonitorConfig(rolling_window=config.forward.degradation_window)
        self.monitor = monitor or AlphaMonitor(config=mon_cfg)

        self.lifecycle = lifecycle or AlphaLifecycle(
            self.registry,
            config=LifecycleConfig(
                oos_sharpe_min=config.validation.oos_sharpe_min,
            ),
        )

        risk_cfg = RiskConfig(
            target_vol=config.risk.target_vol_pct / 100.0,
            dd_stage1_pct=config.risk.dd_stage1_pct / 100.0,
            dd_stage2_pct=config.risk.dd_stage2_pct / 100.0,
            dd_stage3_pct=config.risk.dd_stage3_pct / 100.0,
        )
        self.risk_manager = risk_manager or RiskManager(risk_cfg)

        self.initial_capital = config.trading.initial_capital
        self.max_position_pct = config.paper.max_position_pct
        self.min_trade_usd = config.paper.min_trade_usd

        self.executor = executor or PaperExecutor(initial_cash=self.initial_capital)

        if store is not None:
            self.store = store
        else:
            client = SignalClient(
                base_url=config.api.base_url,
                timeout=config.api.timeout,
            )
            self.store = DataStore(DATA_DIR / "alpha_cache.db", client)

        self._restore_state()

    def _restore_state(self) -> None:
        """Reconstruct executor and risk manager state from last snapshot."""
        snapshot = self.portfolio_tracker.get_last_snapshot()
        if snapshot is None:
            self.risk_manager.reset(self.initial_capital)
            return

        self.executor._cash = snapshot.cash
        self.executor._positions = dict(snapshot.positions)

        equity_curve = self.portfolio_tracker.get_equity_curve()
        if equity_curve:
            self.risk_manager.reset(equity_curve[0][1])
            for _date, equity in equity_curve[1:]:
                self.risk_manager.update_equity(equity)
        else:
            self.risk_manager.reset(self.initial_capital)

        logger.info(
            "Restored state: $%.2f portfolio, %d days history",
            snapshot.portfolio_value, len(equity_curve),
        )

    def run_cycle(self, simulation_date: str | None = None) -> PaperCycleResult:
        """Execute one daily paper trading cycle.

        Args:
            simulation_date: ISO date string for historical simulation.
                When set, skips API sync and limits data to end=date.
        """
        t0 = time.perf_counter()
        today = simulation_date or date.today().isoformat()

        existing = self.portfolio_tracker.get_snapshot(today)
        if existing is not None:
            logger.info("Already traded for %s", today)
            return PaperCycleResult(
                date=today,
                combined_signal=existing.combined_signal,
                fills=[],
                portfolio_value=existing.portfolio_value,
                daily_pnl=existing.daily_pnl,
                daily_return=existing.daily_return,
                dd_scale=existing.dd_scale,
                vol_scale=existing.vol_scale,
                n_alphas_active=0,
                n_alphas_evaluated=0,
            )

        # 1. Sync data (skip in simulation mode — use cached data)
        if simulation_date is None:
            logger.info("Syncing %d signals...", len(self.features))
            self.store.sync(self.features)

        # 2. Get active alphas (dormant included for monitoring, not trading)
        active = self.registry.list_by_state(AlphaState.ACTIVE)
        probation = self.registry.list_by_state(AlphaState.PROBATION)
        dormant = self.registry.list_by_state(AlphaState.DORMANT)
        all_alphas = active + probation + dormant
        dormant_ids = {r.alpha_id for r in dormant}

        # 3. Evaluate each alpha's signal
        matrix = self.store.get_matrix(self.features, end=today)
        if len(matrix) < 2:
            logger.warning("Insufficient data for %s (%d rows)", today, len(matrix))
            return PaperCycleResult(
                date=today, combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=self.executor.portfolio_value,
                n_alphas_active=len(active),
            )

        data = {col: matrix[col].values for col in matrix.columns}
        prices_arr = data[self.price_signal]
        if len(prices_arr) < 2:
            return PaperCycleResult(
                date=today, combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=self.executor.portfolio_value,
                n_alphas_active=len(active),
            )
        price_return = (prices_arr[-1] - prices_arr[-2]) / prices_arr[-2]

        alpha_signals: dict[str, float] = {}
        n_evaluated = 0

        for record in all_alphas:
            try:
                expr = parse(record.expression)
                signal = expr.evaluate(data)
                signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0)
                if signal.ndim == 0:
                    signal = np.full(len(matrix), float(signal))

                std = signal.std()
                if std > 0:
                    signal_norm = np.clip(signal / std, -1, 1)
                else:
                    signal_norm = np.clip(np.sign(signal), -1, 1)
                signal_yesterday = float(signal_norm[-2])
                daily_return = signal_yesterday * price_return

                if record.alpha_id not in dormant_ids:
                    alpha_signals[record.alpha_id] = signal_yesterday
                n_evaluated += 1

                # Record per-alpha forward return
                self.forward_tracker.record(
                    record.alpha_id, today, signal_yesterday, daily_return,
                )

                # Monitor + lifecycle
                all_returns = self.forward_tracker.get_returns(record.alpha_id)
                self.monitor.clear(record.alpha_id)
                self.monitor.record_batch(record.alpha_id, all_returns)
                status = self.monitor.check(record.alpha_id)

                old_state = record.state
                if old_state == AlphaState.ACTIVE:
                    new_state = self.lifecycle.evaluate_active(
                        record.alpha_id, status.rolling_sharpe,
                    )
                elif old_state == AlphaState.PROBATION:
                    new_state = self.lifecycle.evaluate_probation(
                        record.alpha_id, status.rolling_sharpe,
                    )
                elif old_state == AlphaState.DORMANT:
                    new_state = self.lifecycle.evaluate_dormant(
                        record.alpha_id, status.rolling_sharpe,
                    )
                else:
                    new_state = old_state

                if new_state != old_state:
                    self.audit_log.log_state_change(
                        record.alpha_id, old_state, new_state,
                        reason=f"paper: sharpe={status.rolling_sharpe:.3f}",
                    )

            except Exception:
                logger.warning("Failed to evaluate %s", record.alpha_id, exc_info=True)
                continue

        # 4. Combine signals
        if alpha_signals:
            values = list(alpha_signals.values())
            combined = float(np.clip(np.mean(values), -1, 1))
        else:
            combined = 0.0

        # 5. Risk adjustment
        prev_value = self.executor.portfolio_value
        self.risk_manager.update_equity(prev_value)
        recent_returns = np.array(self.portfolio_tracker.get_returns())
        dd_s = self.risk_manager.dd_scale
        vol_s = self.risk_manager.vol_scale(recent_returns)
        adjusted = float(np.clip(combined * dd_s * vol_s, -1, 1))

        # 6. Compute target position and execute
        matrix = self.store.get_matrix([self.price_signal], end=today)
        current_price = float(matrix[self.price_signal].iloc[-1])

        dollar_pos = adjusted * self.initial_capital * self.max_position_pct
        target_shares_value = dollar_pos / current_price if current_price > 0 else 0.0

        if abs(dollar_pos) < self.min_trade_usd:
            target_shares_value = 0.0

        target = {self.price_signal: target_shares_value}
        self.executor.set_price(self.price_signal, current_price)
        fills = self.executor.rebalance(target)

        # 7. Record snapshot
        portfolio_value = self.executor.portfolio_value
        daily_pnl = portfolio_value - prev_value
        daily_return = daily_pnl / prev_value if prev_value > 0 else 0.0

        snapshot = PortfolioSnapshot(
            date=today,
            cash=self.executor.get_cash(),
            positions=self.executor.all_positions,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            combined_signal=combined,
            dd_scale=dd_s,
            vol_scale=vol_s,
        )
        self.portfolio_tracker.save_snapshot(snapshot)
        self.portfolio_tracker.save_fills(today, fills)
        self.portfolio_tracker.save_alpha_signals(today, alpha_signals)

        # 8. Audit log
        for fill in fills:
            self.audit_log.log_trade(
                "portfolio", fill.symbol, fill.side, fill.qty, fill.price,
            )

        elapsed = time.perf_counter() - t0
        logger.info("Paper cycle %s: %.2fs, $%.2f", today, elapsed, portfolio_value)

        return PaperCycleResult(
            date=today,
            combined_signal=combined,
            fills=fills,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            dd_scale=dd_s,
            vol_scale=vol_s,
            n_alphas_active=len(active),
            n_alphas_evaluated=n_evaluated,
        )

    def print_status(self) -> None:
        summary = self.portfolio_tracker.summary()
        if summary is None:
            print("No paper trading history.")
            return

        print(f"\nPaper Trading Summary")
        print("=" * 60)
        print(f"  Period:     {summary.start_date} to {summary.end_date} ({summary.n_days} days)")
        print(f"  Capital:    ${summary.initial_value:,.2f} -> ${summary.final_value:,.2f}")
        print(f"  Return:     {summary.total_return:+.2%}")
        print(f"  Sharpe:     {summary.sharpe:.3f}")
        print(f"  Max DD:     {summary.max_drawdown:.2%}")
        print(f"  Trades:     {summary.total_trades}")
        print(f"  Cash:       ${summary.current_cash:,.2f}")
        if summary.current_positions:
            for sym, qty in summary.current_positions.items():
                print(f"  Position:   {sym}: {qty:.6f}")

    def close(self) -> None:
        self.portfolio_tracker.close()
        self.forward_tracker.close()
        self.store.close()
        self.registry.close()
