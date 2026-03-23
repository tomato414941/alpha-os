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

from ..config import Config, HYPOTHESES_DB, SIGNAL_CACHE_DB, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl import parse, collect_feature_names, temporal_expression_issues
from ..dsl.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..execution.binance import BinanceExecutor
from ..execution.executor import Executor, Fill
from ..execution.planning import (
    ExecutionIntent,
    TargetPosition,
    build_target_position,
    plan_execution_intent,
)
from ..execution.constraints import ExecutableOrder
from ..execution.paper import PaperExecutor
from ..forward.tracker import ForwardTracker
from ..governance.audit_log import AuditLog
from ..hypotheses.combiner import (
    CombinerConfig,
    compute_stake_weights,
    compute_tc_scores,
    compute_tc_weights,
    select_low_correlation,
    signal_consensus,
    weighted_combine_scalar,
)
from ..hypotheses.monitor import AlphaMonitor, RegimeDetector
from ..hypotheses.producer import _quick_healthcheck
from ..hypotheses.quality import QualityEstimate
from ..hypotheses.runtime_policy import rank_trading_records
from ..risk.circuit_breaker import CircuitBreaker
from ..risk.manager import RiskManager
from ..runtime_profile import RuntimeProfile, build_runtime_profile
from ..hypotheses.store import HypothesisStore
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
    n_registry_active: int
    n_live_hypotheses: int
    n_shortlist_candidates: int
    n_selected_alphas: int
    n_signals_evaluated: int
    profile_id: str = ""
    profile_commit: str = ""
    profile_config_id: str = ""
    profile_live_set_id: str = ""
    order_failures: int = 0
    n_skipped_deadband: int = 0
    n_skipped_no_delta: int = 0
    n_skipped_min_notional: int = 0
    n_skipped_rounded_to_zero: int = 0
    strategic_signal: float | None = None
    regime_adjusted_signal: float | None = None
    tactical_adjusted_signal: float | None = None
    final_signal: float | None = None


@dataclass
class PredictionOutput:
    combined_signal: float
    strategic_signal: float
    regime_adjusted_signal: float
    tactical_adjusted_signal: float
    final_signal: float
    dd_scale: float
    tc_scores: dict[str, float] | None = None


@dataclass
class AllocationPlan:
    current_price: float
    target_position: TargetPosition

    @property
    def target_positions(self) -> dict[str, float]:
        return {self.target_position.symbol: self.target_position.qty}


@dataclass
class ExecutionOutcome:
    intents: list[ExecutionIntent]
    orders: list[ExecutableOrder]
    fills: list[Fill]
    order_failures: int
    skipped_deadband: int = 0
    skipped_no_delta: int = 0
    skipped_min_notional: int = 0
    skipped_rounded_to_zero: int = 0


# TODO: Split runtime input evaluation and trade recording out of Trader once
# the hypotheses-first runtime contract is fully stabilized.
# TODO: Move/rename this module out of paper/ once the runtime entrypoints are
# fully hypotheses-first; Trader is the single-asset runtime engine, not a
# paper-only helper.
class Trader:
    """Trading orchestrator. Executor determines paper vs live.

    Supports sub-daily execution: signal evaluation uses daily data,
    but position sizing uses real-time price from the executor when available.
    """

    def __init__(
        self,
        asset: str,
        config: Config,
        registry: HypothesisStore | None = None,
        portfolio_tracker: PaperPortfolioTracker | None = None,
        forward_tracker: ForwardTracker | None = None,
        monitor: AlphaMonitor | None = None,
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
        self.registry = registry or self._build_default_registry()
        self.portfolio_tracker = portfolio_tracker or PaperPortfolioTracker(
            db_path=adir / "paper_trading.db"
        )
        self.forward_tracker = forward_tracker or ForwardTracker(
            db_path=adir / "forward_returns.db"
        )
        self.audit_log = audit_log or AuditLog(log_path=adir / "audit.jsonl")

        mon_cfg = config.to_monitor_config()
        self.monitor = monitor or AlphaMonitor(config=mon_cfg)

        self.risk_manager = risk_manager or RiskManager(config.risk.to_manager_config())
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        self.initial_capital = config.trading.initial_capital
        self.max_position_pct = config.paper.max_position_pct
        self.min_trade_usd = config.paper.min_trade_usd
        self.rebalance_deadband_usd = config.paper.rebalance_deadband_usd

        self.executor = executor or PaperExecutor(
            initial_cash=self.initial_capital,
            supports_short=config.trading.supports_short,
            cost_model=config.execution.to_cost_model(),
        )

        if store is not None:
            self.store = store
        else:
            client = build_signal_client_from_config(config.api)
            self.store = DataStore(SIGNAL_CACHE_DB, client)

        self.tactical = tactical
        self._executor_state_date = ""
        self._last_raw_signal: float = float("nan")

        self._restore_state()

    def _build_default_registry(self):
        from ..hypotheses.store import HypothesisStore

        return HypothesisStore(HYPOTHESES_DB)

    @property
    def last_raw_signal(self) -> float:
        """Last raw (pre-neutralization) final signal from prediction layer."""
        return self._last_raw_signal

    def _restore_state(self) -> None:
        """Reconstruct executor and risk manager state from last snapshot."""
        self._sync_state_from_latest_snapshot(force=True)

    def _sync_state_from_latest_snapshot(self, *, force: bool = False) -> None:
        """Refresh in-memory state when the persisted snapshot has advanced."""
        snapshot = self.portfolio_tracker.get_last_snapshot()
        snapshot_date = snapshot.date if snapshot is not None else ""
        if not force and snapshot_date == self._executor_state_date:
            return

        if snapshot is None:
            self.risk_manager.reset(self.initial_capital)
            self._executor_state_date = ""
            return

        # Only restore positions for the asset's own price signal.
        # Snapshots may contain stale keys from exchange-level balance queries.
        own_positions = {
            k: v for k, v in snapshot.positions.items()
            if k == self.price_signal
        }
        if isinstance(self.executor, PaperExecutor):
            self.executor._cash = snapshot.cash
            self.executor._positions = dict(own_positions)
        elif isinstance(self.executor, BinanceExecutor):
            self.executor._managed_cash = snapshot.cash
            self.executor._managed_positions = dict(own_positions)
            tracked_symbols = list({self.price_signal})
            self.executor.sync_reconciliation_baseline(tracked_symbols)
        elif hasattr(self.executor, "_managed_cash"):
            self.executor._managed_cash = snapshot.cash
            self.executor._managed_positions = dict(own_positions)

        equity_curve = self.portfolio_tracker.get_equity_curve()
        if equity_curve:
            self.risk_manager.reset(equity_curve[0][1])
            for _date, equity in equity_curve[1:]:
                self.risk_manager.update_equity(equity)
        else:
            self.risk_manager.reset(self.initial_capital)

        previous_date = self._executor_state_date
        self._executor_state_date = snapshot.date
        if previous_date and previous_date != snapshot.date:
            logger.info(
                "Detected newer snapshot state: %s -> %s",
                previous_date,
                snapshot.date,
            )
        logger.info(
            "Restored state: $%.2f portfolio, %d snapshots",
            snapshot.portfolio_value, len(equity_curve),
        )

    def _snapshot_position_qty(
        self, positions: dict[str, float], asset_ticker: str,
    ) -> float:
        for key in (self.price_signal, asset_ticker, asset_ticker.upper(), asset_ticker.lower()):
            if key in positions:
                return positions[key]
        return 0.0

    def _estimate_quality(
        self,
        record,
        returns: list[float] | np.ndarray,
    ) -> QualityEstimate:
        return self.config.estimate_alpha_quality(
            record.oos_fitness(self.config.portfolio.objective),
            returns,
        )

    def _baseline_portfolio_value(self, snapshot=None) -> float:
        """Use the persisted snapshot value as the paper baseline when available."""
        if isinstance(self.executor, PaperExecutor) and snapshot is not None:
            return float(snapshot.portfolio_value)
        return float(self.executor.portfolio_value)

    def _predict_portfolio_signal(
        self,
        alpha_signals: dict[str, float],
        alpha_signal_arrays: dict[str, np.ndarray],
        data: dict[str, np.ndarray],
        alpha_stakes: dict[str, float] | None = None,
        prev_value: float | None = None,
    ) -> PredictionOutput:
        """Prediction layer: combine alpha signals via stake or TC weighting."""
        strategic_signal = 0.0
        regime_adjusted_signal = 0.0
        tactical_adjusted_signal = 0.0
        final_signal = 0.0

        if prev_value is None:
            prev_value = self.executor.portfolio_value
        self.risk_manager.update_equity(prev_value)
        recent_returns = np.array(self.portfolio_tracker.get_returns())
        dd_s = self.risk_manager.dd_scale

        tc_scores: dict[str, float] | None = None
        if alpha_signals:
            # Compute asset returns for TC (monitoring only)
            prices_arr = data.get(self.price_signal)
            if prices_arr is not None and len(prices_arr) >= 2:
                asset_returns = np.diff(prices_arr) / prices_arr[:-1]
            else:
                asset_returns = np.array([])

            tc_scores = compute_tc_scores(alpha_signal_arrays, asset_returns)

            # Stake-based weights with TC fallback
            positive_stakes = {
                aid: s for aid, s in (alpha_stakes or {}).items()
                if s > 0 and aid in alpha_signals
            }
            if positive_stakes:
                weights_dict = compute_stake_weights(positive_stakes)
            else:
                weights_dict = compute_tc_weights(tc_scores)

            combined = weighted_combine_scalar(alpha_signals, weights_dict)
            sig_mean, sig_std, consensus = signal_consensus(alpha_signals, weights_dict)
            strategic_signal = float(np.sign(sig_mean)) * consensus * dd_s
            strategic_signal = float(np.clip(strategic_signal, -1, 1))
            n_positive_tc = sum(1 for v in tc_scores.values() if v > 0)
            logger.info(
                "Sizing: dd=%.2f cons=%.3f sig=%.4f±%.4f (%d alphas, %d TC>0) combined=%.4f",
                dd_s, consensus, sig_mean, sig_std, len(alpha_signals), n_positive_tc, combined,
            )
        else:
            combined = 0.0
            strategic_signal = 0.0

        regime_adjusted_signal = strategic_signal

        if self.config.regime.enabled and len(recent_returns) >= self.config.regime.long_window:
            regime = RegimeDetector(
                short_window=self.config.regime.short_window,
                long_window=self.config.regime.long_window,
            ).detect(recent_returns)
            if regime.drift_score > self.config.regime.drift_threshold:
                scale = max(
                    self.config.regime.drift_position_scale_min,
                    1.0 - regime.drift_score,
                )
                regime_adjusted_signal *= scale
                logger.info(
                    "Regime drift detected: score=%.3f vol=%s trend=%s, scale=%.2f",
                    regime.drift_score, regime.current_vol_regime,
                    regime.trend_regime, scale,
                )

        tactical_adjusted_signal = regime_adjusted_signal
        final_signal = regime_adjusted_signal
        if self.tactical is not None:
            try:
                tac = self.tactical.run_cycle(strategic_bias=regime_adjusted_signal)
                tactical_adjusted_signal = tac.combined_signal
                final_signal = tac.combined_signal
                logger.info(
                    "Tactical: score=%.3f, combined=%.3f (was %.3f)",
                    tac.tactical_score, tac.combined_signal, regime_adjusted_signal,
                )
            except Exception:
                logger.warning("Tactical cycle failed — using strategic signal only")

        logger.info(
            "Signal stages: raw=%.4f strategic=%.4f regime=%.4f tactical=%.4f final=%.4f",
            combined,
            strategic_signal,
            regime_adjusted_signal,
            tactical_adjusted_signal,
            final_signal,
        )

        return PredictionOutput(
            combined_signal=float(combined),
            strategic_signal=float(strategic_signal),
            regime_adjusted_signal=float(regime_adjusted_signal),
            tactical_adjusted_signal=float(tactical_adjusted_signal),
            final_signal=float(final_signal),
            dd_scale=float(dd_s),
            tc_scores=tc_scores,
        )

    def _build_allocation_plan(
        self,
        final_signal: float,
        prev_value: float,
        today_date: str,
    ) -> AllocationPlan:
        """Allocation layer: convert the final signal into a target portfolio."""
        live_price = self.executor.fetch_ticker_price(self.price_signal)
        if live_price is not None and live_price > 0:
            current_price = float(live_price)
            logger.info("Using live price: $%.2f", current_price)
        else:
            matrix = self.store.get_matrix([self.price_signal], end=today_date)
            current_price = float(matrix[self.price_signal].iloc[-1])
            logger.info("Using daily close: $%.2f", current_price)

        target_position = build_target_position(
            symbol=self.price_signal,
            final_signal=final_signal,
            portfolio_value=prev_value,
            current_price=current_price,
            max_position_pct=self.max_position_pct,
            min_trade_usd=self.min_trade_usd,
            supports_short=self.executor.supports_short,
        )
        return AllocationPlan(current_price=current_price, target_position=target_position)

    def _execute_allocation(self, plan: AllocationPlan) -> ExecutionOutcome:
        """Execution layer: rebalance to target and report fill failures."""
        self.executor.set_price(self.price_signal, plan.current_price)
        current_qty = self.executor.get_position(plan.target_position.symbol)
        decision = plan_execution_intent(
            plan.target_position,
            current_qty=current_qty,
            rebalance_deadband_usd=self.rebalance_deadband_usd,
        )
        intent = decision.intent
        if intent is None:
            return ExecutionOutcome(
                intents=[],
                orders=[],
                fills=[],
                order_failures=0,
                skipped_deadband=1 if decision.skip_reason == "deadband" else 0,
                skipped_no_delta=1 if decision.skip_reason == "no_delta" else 0,
            )

        constrained = self.executor.constrain_intent(intent)
        if constrained.order is None:
            return ExecutionOutcome(
                intents=[intent],
                orders=[],
                fills=[],
                order_failures=0,
                skipped_min_notional=1 if constrained.rejection_reason == "below_min_notional" else 0,
                skipped_rounded_to_zero=1 if constrained.rejection_reason == "rounded_to_zero" else 0,
            )

        fill = self.executor.execute_intent(intent)
        fills = [fill] if fill is not None else []
        order_failures = 0 if fill is not None else 1
        return ExecutionOutcome(
            intents=[intent],
            orders=[constrained.order],
            fills=fills,
            order_failures=order_failures,
        )

    def _runtime_profile(self, live_records=None) -> RuntimeProfile:
        records = live_records
        if records is None:
            records = self.registry.list_live()
        return build_runtime_profile(
            asset=self.asset,
            config=self.config,
            live_hypothesis_ids=[record.alpha_id for record in records],
        )

    @staticmethod
    def _prediction_history_array(
        pred_store,
        signal_id: str,
        asset: str,
        *,
        n_days: int,
        fallback_value: float,
    ) -> np.ndarray:
        rows = pred_store.read_signal_history(signal_id, asset, n_days=n_days)
        values = [float(value) for _date, value in reversed(rows)]
        if not values:
            values = [fallback_value]
        if len(values) < n_days:
            values = [values[0]] * (n_days - len(values)) + values
        return np.asarray(values[-n_days:], dtype=np.float64)

    @staticmethod
    def _prepare_runtime_inputs(
        records,
        price_signal: str,
        store_signals: dict[str, float],
    ) -> tuple[list[str], list[tuple], int]:
        runtime_signals = {price_signal}
        parsed_records: list[tuple] = []
        n_failed = 0

        for record in records:
            if record.alpha_id in store_signals:
                parsed_records.append((record, None))
                continue
            if not record.expression:
                n_failed += 1
                continue
            try:
                expr = parse(record.expression)
            except SyntaxError as exc:
                logger.warning("Failed to parse %s: %s", record.alpha_id, exc)
                n_failed += 1
                continue
            issues = temporal_expression_issues(expr)
            if issues:
                logger.warning(
                    "Skipping structurally invalid %s: %s",
                    record.alpha_id,
                    issues[0],
                )
                n_failed += 1
                continue
            runtime_signals.update(collect_feature_names(expr))
            parsed_records.append((record, expr))

        return sorted(runtime_signals), parsed_records, n_failed

    def _strategy_epoch(self, universe_records) -> str:
        return self._runtime_profile(universe_records).profile_id

    def run_cycle(
        self,
        simulation_date: str | None = None,
        skip_lifecycle: bool = False,
        signal_override: float | None = None,
    ) -> PaperCycleResult:
        """Execute one trading cycle.

        Args:
            simulation_date: ISO date string for historical simulation.
                When set, skips API sync and limits data to end=date.
                Uses date key (daily mode) for backward compatibility.
            skip_lifecycle: When True, skip monitor/lifecycle evaluation
                (handled by separate lifecycle daemon in Pipeline v2).
        """
        t0 = time.perf_counter()
        self._sync_state_from_latest_snapshot()
        self.circuit_breaker.reload()
        now = datetime.now(timezone.utc)
        today_date = simulation_date or date.today().isoformat()

        # Snapshot key: datetime for live cycles, date for simulation
        if simulation_date:
            cycle_key = simulation_date
        else:
            cycle_key = now.strftime("%Y-%m-%dT%H:%M:%S")

        # Cooldown: skip if last snapshot was too recent
        last_snapshot = self.portfolio_tracker.get_last_snapshot()
        current_profile = self._runtime_profile()
        if last_snapshot is not None and simulation_date is None:
            last_recorded = self.portfolio_tracker._conn.execute(
                "SELECT recorded_at FROM portfolio_snapshots WHERE date = ?",
                (last_snapshot.date,),
            ).fetchone()
            if last_recorded and (time.time() - last_recorded["recorded_at"]) < _MIN_CYCLE_COOLDOWN:
                logger.info("Cooldown: last cycle was < %ds ago", _MIN_CYCLE_COOLDOWN)
                return PaperCycleResult(
                    date=cycle_key,
                    profile_id=current_profile.profile_id,
                    profile_commit=current_profile.git_commit,
                    profile_config_id=current_profile.config_id,
                    profile_live_set_id=current_profile.live_set_id,
                    combined_signal=last_snapshot.combined_signal,
                    fills=[],
                    portfolio_value=last_snapshot.portfolio_value,
                    daily_pnl=last_snapshot.daily_pnl,
                    daily_return=last_snapshot.daily_return,
                    dd_scale=last_snapshot.dd_scale,
                    vol_scale=last_snapshot.vol_scale,
                    n_registry_active=0,
                    n_live_hypotheses=0,
                    n_shortlist_candidates=0,
                    n_selected_alphas=0,
                    n_signals_evaluated=0,
                )

        # 0. Circuit breaker check
        prev_equity = self._baseline_portfolio_value(last_snapshot)
        live_hypotheses = self.registry.top_by_stake(n=200)
        if not live_hypotheses:
            raise RuntimeError(
                "No live hypotheses with stake > 0. Run hypothesis-seeder or lifecycle first."
            )
        current_profile = self._runtime_profile(live_hypotheses)
        self.circuit_breaker.sync_strategy_epoch(
            current_profile.profile_id,
            prev_equity,
        )
        self.circuit_breaker.reset_daily(prev_equity)
        safe, reason = self.circuit_breaker.is_safe_to_trade(prev_equity)
        if not safe:
            logger.warning("Circuit breaker tripped: %s", reason)
            return PaperCycleResult(
                date=cycle_key, profile_id=current_profile.profile_id,
                profile_commit=current_profile.git_commit,
                profile_config_id=current_profile.config_id,
                profile_live_set_id=current_profile.live_set_id,
                combined_signal=0.0, fills=[],
                portfolio_value=prev_equity, daily_pnl=0.0, daily_return=0.0,
                dd_scale=1.0, vol_scale=1.0,
                n_registry_active=0, n_live_hypotheses=0,
                n_shortlist_candidates=0,
                n_selected_alphas=0, n_signals_evaluated=0,
            )

        # 1. Build two sets:
        #    - observation candidates: active hypotheses that should be evaluated
        #    - live hypotheses: capital-eligible hypotheses with stake > 0
        max_trading = self.config.paper.max_trading_alphas
        observation_candidates = self.registry.list_observation_active()
        universe_records = live_hypotheses
        n_live_hypotheses = len(live_hypotheses)
        trading_candidates = rank_trading_records(
            universe_records,
            lambda record: self._estimate_quality(
                record,
                self.forward_tracker.get_returns(record.alpha_id),
            ),
            max_trading=max_trading,
            shortlist_preselect_factor=self.config.live_quality.shortlist_preselect_factor,
            metric=self.config.portfolio.objective,
        )
        all_alphas = observation_candidates

        n_total_active = len(trading_candidates)
        if n_live_hypotheses > max_trading or n_total_active > max_trading:
            logger.info(
                "Selection pool: %d shortlist candidates from %d live hypotheses (%d capital-backed, %d active)",
                len(trading_candidates),
                n_live_hypotheses,
                n_total_active,
                len(observation_candidates),
            )
        selected_records = trading_candidates

        store_signals: dict[str, float] = {}
        try:
            from alpha_os.predictions.store import PredictionStore
            pred_store = PredictionStore()
            try:
                store_preds = pred_store.read_latest(today_date, assets=[self.asset])
            finally:
                pred_store.close()
            for signal_id, assets in store_preds.items():
                if self.asset in assets:
                    store_signals[signal_id] = float(assets[self.asset])
        except Exception:
            pass

        runtime_signals, parsed_records, n_failed = self._prepare_runtime_inputs(
            all_alphas,
            self.price_signal,
            store_signals,
        )

        # 2. Sync data (skip in simulation mode — use cached data)
        if simulation_date is None:
            if _quick_healthcheck(self.config.api.base_url):
                try:
                    logger.info(
                        "Syncing %d runtime signals for %d candidates...",
                        len(runtime_signals),
                        len(all_alphas),
                    )
                    self.store.sync(runtime_signals)
                except Exception as exc:
                    logger.warning("API sync failed — using cached data: %s", exc)
            else:
                logger.warning("Skipping runtime signal sync: signal-noise health probe failed")

        # 3. Evaluate each alpha's signal (uses daily data)
        matrix = self.store.get_matrix(runtime_signals, end=today_date)
        if len(matrix) < 2:
            logger.warning("Insufficient data for %s (%d rows)", today_date, len(matrix))
            return PaperCycleResult(
                date=cycle_key, profile_id=current_profile.profile_id,
                profile_commit=current_profile.git_commit,
                profile_config_id=current_profile.config_id,
                profile_live_set_id=current_profile.live_set_id,
                combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=prev_equity,
                n_registry_active=n_total_active,
                n_live_hypotheses=n_live_hypotheses,
                n_shortlist_candidates=len(trading_candidates),
                n_selected_alphas=0,
                n_signals_evaluated=0,
            )

        data = {col: matrix[col].values for col in matrix.columns}
        prices_arr = data[self.price_signal]
        if len(prices_arr) < 2:
            return PaperCycleResult(
                date=cycle_key, profile_id=current_profile.profile_id,
                profile_commit=current_profile.git_commit,
                profile_config_id=current_profile.config_id,
                profile_live_set_id=current_profile.live_set_id,
                combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=prev_equity,
                n_registry_active=n_total_active,
                n_live_hypotheses=n_live_hypotheses,
                n_shortlist_candidates=len(trading_candidates),
                n_selected_alphas=0,
                n_signals_evaluated=0,
            )
        price_return = (prices_arr[-1] - prices_arr[-2]) / prices_arr[-2]

        alpha_signals: dict[str, float] = {}
        alpha_signal_arrays: dict[str, np.ndarray] = {}
        quality_estimates: dict[str, QualityEstimate] = {}
        alpha_exprs: dict[str, object] = {}  # aid → parsed Expr
        n_evaluated = 0
        n_feature_filtered = 0
        n_from_store = 0
        store_signal_arrays: dict[str, np.ndarray] = {}
        available_features = {
            col for col in matrix.columns
            if col == self.price_signal or not matrix[col].isna().all()
        }

        if store_signals:
            try:
                from alpha_os.predictions.store import PredictionStore
                pred_store = PredictionStore()
                try:
                    for record, _expr in parsed_records:
                        if record.alpha_id not in store_signals:
                            continue
                        value = store_signals[record.alpha_id]
                        store_signal_arrays[record.alpha_id] = self._prediction_history_array(
                            pred_store,
                            record.alpha_id,
                            self.asset,
                            n_days=len(matrix),
                            fallback_value=value,
                        )
                finally:
                    pred_store.close()
            except Exception:
                store_signal_arrays = {}

        filtered_records: list[tuple] = []
        for record, expr in parsed_records:
            if record.alpha_id in store_signals:
                filtered_records.append((record, expr))
                continue
            required = collect_feature_names(expr)
            if not required.issubset(available_features):
                n_feature_filtered += 1
                continue
            filtered_records.append((record, expr))

        for record, expr in filtered_records:
            try:
                # Use prediction store if available, else compute directly
                if record.alpha_id in store_signals:
                    signal_yesterday = store_signals[record.alpha_id]
                    signal_norm = store_signal_arrays.get(record.alpha_id)
                    if signal_norm is None:
                        signal_norm = np.full(len(matrix), signal_yesterday, dtype=np.float64)
                    n_from_store += 1
                else:
                    if expr is None:
                        n_failed += 1
                        continue
                    signal = evaluate_expression(expr, data, len(matrix))
                    signal_norm = normalize_signal(signal)
                    signal_yesterday = float(signal_norm[-2])
                    if not np.isfinite(signal_yesterday):
                        for offset in range(3, min(10, len(signal_norm))):
                            fallback = float(signal_norm[-offset])
                            if np.isfinite(fallback):
                                signal_yesterday = fallback
                                break
                daily_return = signal_yesterday * price_return if np.isfinite(signal_yesterday) else 0.0

                if np.isfinite(signal_yesterday):
                    alpha_signals[record.alpha_id] = signal_yesterday
                    alpha_signal_arrays[record.alpha_id] = signal_norm
                    if expr is not None:
                        alpha_exprs[record.alpha_id] = expr
                n_evaluated += 1

                # Record per-alpha forward return (daily granularity)
                self.forward_tracker.record(
                    record.alpha_id, today_date, signal_yesterday, daily_return,
                )

                # Monitor and quality estimate
                all_returns = self.forward_tracker.get_returns(record.alpha_id)
                recent_returns = all_returns[-self.config.forward.degradation_window:]
                self.monitor.clear(record.alpha_id)
                self.monitor.record_batch(record.alpha_id, recent_returns)
                estimate = self._estimate_quality(record, all_returns)
                quality_estimates[record.alpha_id] = estimate

            except EvaluationError as exc:
                logger.warning("Failed to evaluate %s: %s", record.alpha_id, exc)
                n_failed += 1

        if n_from_store:
            logger.info("Cycle: %d/%d signals from prediction store", n_from_store, n_evaluated)
        if n_feature_filtered:
            logger.info(
                "Cycle: %d/%d alphas skipped (missing features in current data)",
                n_feature_filtered, len(all_alphas),
            )
        if n_failed:
            logger.info("Cycle: %d/%d alphas failed evaluation", n_failed, len(all_alphas))
        total_skipped = n_feature_filtered + n_failed
        if all_alphas and total_skipped / len(all_alphas) > 0.7:
            logger.warning(
                "High alpha skip ratio: %.1f%% (%d/%d)",
                100.0 * total_skipped / len(all_alphas),
                total_skipped,
                len(all_alphas),
            )

        # TC computation removed — stake-based weights handle selection

        # 3d. Correlation filter: select top-N decorrelated capital-backed alphas.
        selection_signal_ids = {
            record.alpha_id for record in trading_candidates
            if record.alpha_id in alpha_signals
        }
        if len(selection_signal_ids) > max_trading:
            from alpha_os.hypotheses.breadth import hypothesis_signal_series

            candidate_ids = [
                aid for aid in alpha_signals.keys()
                if aid in selection_signal_ids
            ]
            lookback = min(252, len(matrix))
            signal_histories: list[np.ndarray] = []
            filtered_candidate_ids: list[str] = []
            record_map = {record.alpha_id: record for record in trading_candidates}
            for aid in candidate_ids:
                record = record_map.get(aid)
                history = None
                if record is not None:
                    history = hypothesis_signal_series(record, data=data, asset=self.asset)
                if history is None:
                    history = alpha_signal_arrays.get(aid)
                if history is None:
                    continue
                signal_histories.append(np.asarray(history[-lookback:], dtype=np.float64))
                filtered_candidate_ids.append(aid)
            candidate_ids = filtered_candidate_ids
            sig_matrix = np.array(signal_histories)
            quality_for_sel = np.array([
                quality_estimates[aid].blended_quality
                for aid in candidate_ids
            ])
            selected_idx = select_low_correlation(
                sig_matrix, quality_for_sel,
                CombinerConfig(max_alphas=max_trading),
            )
            selected_ids = {candidate_ids[i] for i in selected_idx}
            n_before = len(selection_signal_ids)
            alpha_signals = {k: v for k, v in alpha_signals.items() if k in selected_ids}
            logger.info(
                "Correlation filter: %d shortlist candidates → %d selected alphas (max_corr=%.2f)",
                n_before, len(alpha_signals), CombinerConfig().max_correlation,
            )
        else:
            alpha_signals = {
                aid: alpha_signals[aid]
                for aid in selection_signal_ids
            }
        selected_records = [
            r for r in trading_candidates if r.alpha_id in alpha_signals
        ]

        # 4. Prediction layer: alpha signals -> final portfolio signal
        alpha_stakes = {
            r.alpha_id: r.stake
            for r in selected_records
            if r.alpha_id in alpha_signals
        }
        prev_value = prev_equity
        prediction = self._predict_portfolio_signal(
            alpha_signals=alpha_signals,
            alpha_signal_arrays=alpha_signal_arrays,
            data=data,
            alpha_stakes=alpha_stakes,
            prev_value=prev_value,
        )
        self._last_raw_signal = prediction.final_signal

        # 4b. Cross-asset neutralization override
        effective_signal = prediction.final_signal
        if signal_override is not None:
            logger.info(
                "Signal override: %.4f -> %.4f (cross-asset neutralized)",
                prediction.final_signal, signal_override,
            )
            effective_signal = signal_override

        # 5. Allocation layer: final signal -> target portfolio
        plan = self._build_allocation_plan(
            final_signal=effective_signal,
            prev_value=prev_value,
            today_date=today_date,
        )

        # 6. Execution layer: target portfolio -> fills
        execution = self._execute_allocation(plan)
        fills = execution.fills
        order_failures = execution.order_failures

        # 7. Record snapshot — PnL relative to previous snapshot
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
            combined_signal=prediction.combined_signal,
            strategic_signal=prediction.strategic_signal,
            regime_adjusted_signal=prediction.regime_adjusted_signal,
            tactical_adjusted_signal=prediction.tactical_adjusted_signal,
            final_signal=effective_signal,
            dd_scale=prediction.dd_scale,
            vol_scale=1.0,
        )
        self.portfolio_tracker.save_snapshot(snapshot)
        self.portfolio_tracker.save_fills(cycle_key, fills)
        self.portfolio_tracker.save_alpha_signals(cycle_key, alpha_signals)
        self._executor_state_date = snapshot.date
        no_fill_streak = self.portfolio_tracker.count_consecutive_no_fill_cycles()
        if no_fill_streak >= 6:
            logger.warning("No fills for %d consecutive cycles", no_fill_streak)

        # 8. Record trade P&L in circuit breaker
        self.circuit_breaker.record_trade(daily_pnl)

        # 9. Audit log
        for fill in fills:
            self.audit_log.log_trade(
                "portfolio", fill.symbol, fill.side, fill.qty, fill.price,
            )

        elapsed = time.perf_counter() - t0
        logger.info("Cycle %s: %.2fs, $%.2f", cycle_key, elapsed, portfolio_value)

        return PaperCycleResult(
            date=cycle_key,
            profile_id=current_profile.profile_id,
            profile_commit=current_profile.git_commit,
            profile_config_id=current_profile.config_id,
            profile_live_set_id=current_profile.live_set_id,
            combined_signal=prediction.combined_signal,
            fills=fills,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            dd_scale=prediction.dd_scale,
            vol_scale=1.0,
            n_registry_active=n_total_active,
            n_live_hypotheses=n_live_hypotheses,
            n_shortlist_candidates=len(trading_candidates),
            n_selected_alphas=len(selected_records),
            n_signals_evaluated=n_evaluated,
            order_failures=order_failures,
            n_skipped_deadband=execution.skipped_deadband,
            n_skipped_no_delta=execution.skipped_no_delta,
            n_skipped_min_notional=execution.skipped_min_notional,
            n_skipped_rounded_to_zero=execution.skipped_rounded_to_zero,
            strategic_signal=prediction.strategic_signal,
            regime_adjusted_signal=prediction.regime_adjusted_signal,
            tactical_adjusted_signal=prediction.tactical_adjusted_signal,
            final_signal=effective_signal,
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
            from ..data.universe import asset_for_price_signal

            traded = (asset_for_price_signal(self.price_signal) or "").upper()

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

        from ..data.universe import asset_for_price_signal

        asset_ticker = asset_for_price_signal(self.price_signal) or self.price_signal

        internal_qty = self._snapshot_position_qty(snapshot.positions, asset_ticker)
        internal_cash = snapshot.cash

        exchange_qty = self.executor.get_reconciled_position(self.price_signal)
        exchange_cash = self.executor.get_reconciled_cash()

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
        if self.tactical is not None:
            self.tactical.close()
