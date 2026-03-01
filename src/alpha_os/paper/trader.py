"""Trader — connects existing components into a trading cycle.

Supports both daily and sub-daily execution. When run multiple times per day,
uses real-time price from exchange for position sizing and records snapshots
with datetime keys.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from ..alpha.monitor import AlphaMonitor, MonitorConfig
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, DATA_DIR, asset_data_dir
from signal_noise.client import SignalClient
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl import parse
from ..execution.executor import Executor, Fill
from ..execution.paper import PaperExecutor
from ..forward.tracker import ForwardTracker
from ..governance.audit_log import AuditLog
from ..alpha.combiner import (
    WeightedCombinerConfig,
    compute_diversity_scores,
    compute_weights,
    weighted_combine_scalar,
)
from ..risk.circuit_breaker import CircuitBreaker
from ..risk.manager import RiskManager
from .tactical import TacticalTrader
from .tracker import PaperPortfolioTracker, PortfolioSnapshot

logger = logging.getLogger(__name__)

# Minimum seconds between cycles to prevent duplicate execution
_MIN_CYCLE_COOLDOWN = 60


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
    order_failures: int = 0


class Trader:
    """Trading orchestrator. Executor determines paper vs live.

    Supports sub-daily execution: signal evaluation uses daily data,
    but position sizing uses real-time price from the executor when available.
    """

    def __init__(
        self,
        asset: str,
        config: Config,
        registry: AlphaRegistry | None = None,
        portfolio_tracker: PaperPortfolioTracker | None = None,
        forward_tracker: ForwardTracker | None = None,
        monitor: AlphaMonitor | None = None,
        lifecycle: AlphaLifecycle | None = None,
        executor: Executor | None = None,
        risk_manager: RiskManager | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        audit_log: AuditLog | None = None,
        store: DataStore | None = None,
        tactical: TacticalTrader | None = None,
    ):
        self.asset = asset
        self.config = config

        self.features = build_feature_list(asset)
        self.price_signal = self.features[0]

        adir = asset_data_dir(asset)
        self.registry = registry or AlphaRegistry(db_path=adir / "alpha_registry.db")
        self.portfolio_tracker = portfolio_tracker or PaperPortfolioTracker(
            db_path=adir / "paper_trading.db"
        )
        self.forward_tracker = forward_tracker or ForwardTracker(
            db_path=adir / "forward_returns.db"
        )
        self.audit_log = audit_log or AuditLog(log_path=adir / "audit.jsonl")

        mon_cfg = MonitorConfig(rolling_window=config.forward.degradation_window)
        self.monitor = monitor or AlphaMonitor(config=mon_cfg)

        self.lifecycle = lifecycle or AlphaLifecycle(
            self.registry,
            config=LifecycleConfig(
                oos_sharpe_min=config.validation.oos_sharpe_min,
            ),
        )

        self.risk_manager = risk_manager or RiskManager(config.risk.to_manager_config())
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

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

        self.tactical = tactical
        self._wcfg = WeightedCombinerConfig()
        self._diversity_cache: dict[str, float] = {}
        self._diversity_computed = False

        self._restore_state()

    def _restore_state(self) -> None:
        """Reconstruct executor and risk manager state from last snapshot."""
        snapshot = self.portfolio_tracker.get_last_snapshot()
        if snapshot is None:
            self.risk_manager.reset(self.initial_capital)
            return

        # PaperExecutor tracks state in memory; restore from last snapshot.
        # Live executors query the exchange directly, so skip.
        if isinstance(self.executor, PaperExecutor):
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
            "Restored state: $%.2f portfolio, %d snapshots",
            snapshot.portfolio_value, len(equity_curve),
        )

    def _recompute_diversity(
        self, data: dict[str, np.ndarray], all_records: list,
    ) -> None:
        """Recompute diversity scores from signal matrix."""
        n_days = len(next(iter(data.values())))
        signals = []
        alpha_ids = []

        for record in all_records:
            try:
                expr = parse(record.expression)
                sig = evaluate_expression(expr, data, n_days)
                sig = normalize_signal(sig)
                lookback = min(self._wcfg.corr_lookback, n_days)
                signals.append(sig[-lookback:])
                alpha_ids.append(record.alpha_id)
            except EvaluationError:
                continue

        if len(signals) < 2:
            self._diversity_cache = {aid: 1.0 for aid in alpha_ids}
            self._diversity_computed = True
            return

        sig_matrix = np.array(signals)
        diversity = compute_diversity_scores(sig_matrix, chunk_size=self._wcfg.chunk_size)
        self._diversity_cache = {
            aid: float(d) for aid, d in zip(alpha_ids, diversity)
        }
        self._diversity_computed = True
        logger.info("Diversity scores computed for %d alphas", len(alpha_ids))

    def run_cycle(self, simulation_date: str | None = None) -> PaperCycleResult:
        """Execute one trading cycle.

        Args:
            simulation_date: ISO date string for historical simulation.
                When set, skips API sync and limits data to end=date.
                Uses date key (daily mode) for backward compatibility.
        """
        t0 = time.perf_counter()
        now = datetime.now(timezone.utc)
        today_date = simulation_date or date.today().isoformat()

        # Snapshot key: datetime for live cycles, date for simulation
        if simulation_date:
            cycle_key = simulation_date
        else:
            cycle_key = now.strftime("%Y-%m-%dT%H:%M:%S")

        # Cooldown: skip if last snapshot was too recent
        last_snapshot = self.portfolio_tracker.get_last_snapshot()
        if last_snapshot is not None and simulation_date is None:
            last_recorded = self.portfolio_tracker._conn.execute(
                "SELECT recorded_at FROM portfolio_snapshots WHERE date = ?",
                (last_snapshot.date,),
            ).fetchone()
            if last_recorded and (time.time() - last_recorded["recorded_at"]) < _MIN_CYCLE_COOLDOWN:
                logger.info("Cooldown: last cycle was < %ds ago", _MIN_CYCLE_COOLDOWN)
                return PaperCycleResult(
                    date=cycle_key,
                    combined_signal=last_snapshot.combined_signal,
                    fills=[],
                    portfolio_value=last_snapshot.portfolio_value,
                    daily_pnl=last_snapshot.daily_pnl,
                    daily_return=last_snapshot.daily_return,
                    dd_scale=last_snapshot.dd_scale,
                    vol_scale=last_snapshot.vol_scale,
                    n_alphas_active=0,
                    n_alphas_evaluated=0,
                )

        # 0. Circuit breaker check
        prev_equity = self.executor.portfolio_value
        self.circuit_breaker.reset_daily(prev_equity)
        safe, reason = self.circuit_breaker.is_safe_to_trade(prev_equity)
        if not safe:
            logger.warning("Circuit breaker tripped: %s", reason)
            return PaperCycleResult(
                date=cycle_key, combined_signal=0.0, fills=[],
                portfolio_value=prev_equity, daily_pnl=0.0, daily_return=0.0,
                dd_scale=1.0, vol_scale=1.0, n_alphas_active=0, n_alphas_evaluated=0,
            )

        # 1. Sync data (skip in simulation mode — use cached data)
        if simulation_date is None:
            try:
                logger.info("Syncing %d signals...", len(self.features))
                self.store.sync(self.features)
            except Exception:
                logger.warning("API sync failed — using cached data")

        # 2. Get active alphas (dormant included for monitoring, not trading)
        active = self.registry.list_by_state(AlphaState.ACTIVE)
        probation = self.registry.list_by_state(AlphaState.PROBATION)
        dormant = self.registry.list_by_state(AlphaState.DORMANT)
        all_alphas = active + probation + dormant
        dormant_ids = {r.alpha_id for r in dormant}

        # 3. Evaluate each alpha's signal (uses daily data)
        matrix = self.store.get_matrix(self.features, end=today_date)
        if len(matrix) < 2:
            logger.warning("Insufficient data for %s (%d rows)", today_date, len(matrix))
            return PaperCycleResult(
                date=cycle_key, combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=self.executor.portfolio_value,
                n_alphas_active=len(active),
            )

        data = {col: matrix[col].values for col in matrix.columns}
        prices_arr = data[self.price_signal]
        if len(prices_arr) < 2:
            return PaperCycleResult(
                date=cycle_key, combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=self.executor.portfolio_value,
                n_alphas_active=len(active),
            )
        price_return = (prices_arr[-1] - prices_arr[-2]) / prices_arr[-2]

        alpha_signals: dict[str, float] = {}
        n_evaluated = 0
        n_failed = 0

        for record in all_alphas:
            try:
                expr = parse(record.expression)
                signal = evaluate_expression(expr, data, len(matrix))
                signal_norm = normalize_signal(signal)
                signal_yesterday = float(signal_norm[-2])
                daily_return = signal_yesterday * price_return

                if record.alpha_id not in dormant_ids:
                    alpha_signals[record.alpha_id] = signal_yesterday
                n_evaluated += 1

                # Record per-alpha forward return (daily granularity)
                self.forward_tracker.record(
                    record.alpha_id, today_date, signal_yesterday, daily_return,
                )

                # Monitor + lifecycle
                all_returns = self.forward_tracker.get_returns(record.alpha_id)
                self.monitor.clear(record.alpha_id)
                self.monitor.record_batch(record.alpha_id, all_returns)
                status = self.monitor.check(record.alpha_id)

                old_state = record.state
                new_state = self.lifecycle.evaluate(
                    record.alpha_id, status.rolling_sharpe,
                )

                if new_state != old_state:
                    self.audit_log.log_state_change(
                        record.alpha_id, old_state, new_state,
                        reason=f"paper: sharpe={status.rolling_sharpe:.3f}",
                    )

            except EvaluationError as exc:
                logger.warning("Failed to evaluate %s: %s", record.alpha_id, exc)
                n_failed += 1

        if n_failed:
            logger.info("Cycle: %d/%d alphas failed evaluation", n_failed, len(all_alphas))

        # 4. Combine signals with quality × diversity weighting
        if not self._diversity_computed:
            self._recompute_diversity(data, all_alphas)

        if alpha_signals:
            alpha_ids = list(alpha_signals.keys())
            sharpes_list = []
            diversity_list = []
            for aid in alpha_ids:
                status = self.monitor.check(aid)
                sharpes_list.append(max(status.rolling_sharpe, 0.0))
                diversity_list.append(self._diversity_cache.get(aid, 1.0))

            sharpes_np = np.array(sharpes_list)
            diversity_np = np.array(diversity_list)
            w = compute_weights(sharpes_np, diversity_np, min_weight=self._wcfg.min_weight)
            weights_dict = {aid: float(w[i]) for i, aid in enumerate(alpha_ids)}
            combined = weighted_combine_scalar(alpha_signals, weights_dict)
        else:
            combined = 0.0

        # 5. Risk adjustment
        prev_value = self.executor.portfolio_value
        self.risk_manager.update_equity(prev_value)
        recent_returns = np.array(self.portfolio_tracker.get_returns())
        dd_s = self.risk_manager.dd_scale
        vol_s = self.risk_manager.vol_scale(recent_returns)
        adjusted = float(np.clip(combined * dd_s * vol_s, -1, 1))

        # 5b. Tactical modulation (Layer 2)
        if self.tactical is not None:
            try:
                tac = self.tactical.run_cycle(strategic_bias=adjusted)
                adjusted = tac.combined_signal
                logger.info(
                    "Tactical: score=%.3f, combined=%.3f (was %.3f)",
                    tac.tactical_score, tac.combined_signal, adjusted,
                )
            except Exception:
                logger.warning("Tactical cycle failed — using strategic signal only")

        # 6. Get execution price: prefer real-time from exchange, fall back to daily close
        live_price = self.executor.fetch_ticker_price(self.price_signal)
        if live_price is not None and live_price > 0:
            current_price = live_price
            logger.info("Using live price: $%.2f", current_price)
        else:
            matrix = self.store.get_matrix([self.price_signal], end=today_date)
            current_price = float(matrix[self.price_signal].iloc[-1])
            logger.info("Using daily close: $%.2f", current_price)

        # 7. Compute target position and execute
        dollar_pos = adjusted * prev_value * self.max_position_pct
        target_shares_value = dollar_pos / current_price if current_price > 0 else 0.0

        if abs(dollar_pos) < self.min_trade_usd:
            target_shares_value = 0.0

        target = {self.price_signal: target_shares_value}
        self.executor.set_price(self.price_signal, current_price)

        # Record pre-rebalance positions to detect order failures
        pre_positions = {sym: self.executor.get_position(sym) for sym in target}
        fills = self.executor.rebalance(target)

        # Count order failures: non-trivial delta expected but no fill produced
        filled_symbols = {f.symbol for f in fills}
        order_failures = sum(
            1 for sym, tgt in target.items()
            if abs(tgt - pre_positions.get(sym, 0.0)) > 1e-6
            and sym not in filled_symbols
        )

        # 8. Record snapshot — PnL relative to previous snapshot
        portfolio_value = self.executor.portfolio_value
        daily_pnl = portfolio_value - prev_value
        daily_return = daily_pnl / prev_value if prev_value > 0 else 0.0

        snapshot = PortfolioSnapshot(
            date=cycle_key,
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
        self.portfolio_tracker.save_fills(cycle_key, fills)
        self.portfolio_tracker.save_alpha_signals(cycle_key, alpha_signals)

        # 9. Record trade P&L in circuit breaker
        self.circuit_breaker.record_trade(daily_pnl)

        # 10. Audit log
        for fill in fills:
            self.audit_log.log_trade(
                "portfolio", fill.symbol, fill.side, fill.qty, fill.price,
            )

        elapsed = time.perf_counter() - t0
        logger.info("Cycle %s: %.2fs, $%.2f", cycle_key, elapsed, portfolio_value)

        return PaperCycleResult(
            date=cycle_key,
            combined_signal=combined,
            fills=fills,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            dd_scale=dd_s,
            vol_scale=vol_s,
            n_alphas_active=len(active),
            n_alphas_evaluated=n_evaluated,
            order_failures=order_failures,
        )

    def print_status(self) -> None:
        summary = self.portfolio_tracker.summary()
        if summary is None:
            print("No paper trading history.")
            return

        print("\nPaper Trading Summary")
        print("=" * 60)
        print(f"  Period:     {summary.start_date} to {summary.end_date} ({summary.n_days} days)")
        print(f"  Capital:    ${summary.initial_value:,.2f} -> ${summary.final_value:,.2f}")
        print(f"  Return:     {summary.total_return:+.2%}")
        print(f"  Sharpe:     {summary.sharpe:.3f}")
        print(f"  Max DD:     {summary.max_drawdown:.2%}")
        print(f"  Trades:     {summary.total_trades}")
        print(f"  Cash:       ${summary.current_cash:,.2f}")
        if summary.current_positions:
            # Resolve traded asset ticker (btc_ohlcv → BTC)
            from ..data.universe import CRYPTO, STOCKS
            signal_to_asset = {v: k for k, v in {**CRYPTO, **STOCKS}.items()}
            traded = signal_to_asset.get(self.price_signal, "").upper()

            shown = 0
            for sym, qty in summary.current_positions.items():
                if sym.upper() == traded:
                    print(f"  Position:   {sym}: {qty:.6f}")
                    shown += 1
            hidden = len(summary.current_positions) - shown
            if hidden > 0:
                print(f"  ({hidden} other positions hidden)")

    def reconcile(self) -> dict:
        """Compare internal DB position vs exchange for traded asset."""
        snapshot = self.portfolio_tracker.get_last_snapshot()
        if snapshot is None:
            return {"status": "no_data"}

        from ..data.universe import CRYPTO, STOCKS
        signal_to_asset = {v: k for k, v in {**CRYPTO, **STOCKS}.items()}
        asset_ticker = signal_to_asset.get(self.price_signal, self.price_signal)

        internal_qty = snapshot.positions.get(asset_ticker, 0.0)
        internal_cash = snapshot.cash

        exchange_qty = self.executor.get_position(asset_ticker)
        exchange_cash = self.executor.get_cash()

        qty_diff = abs(exchange_qty - internal_qty)
        cash_diff = abs(exchange_cash - internal_cash)

        result = {
            "date": snapshot.date,
            "asset": asset_ticker,
            "internal_qty": internal_qty,
            "exchange_qty": exchange_qty,
            "qty_diff": qty_diff,
            "internal_cash": internal_cash,
            "exchange_cash": exchange_cash,
            "cash_diff": cash_diff,
            "match": qty_diff < 1e-6 and cash_diff < 1.0,
        }
        logger.info("Reconciliation: %s", result)
        return result

    def close(self) -> None:
        self.portfolio_tracker.close()
        self.forward_tracker.close()
        self.store.close()
        self.registry.close()


# Backward compatibility
PaperTrader = Trader
