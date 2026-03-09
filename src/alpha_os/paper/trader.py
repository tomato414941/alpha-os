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
from ..alpha.lifecycle import AlphaLifecycle
from ..alpha.monitor import AlphaMonitor, RegimeDetector
from ..alpha.quality import (
    QualityEstimate,
    confidence_weight_scale,
    shrink_weight_quality,
)
from ..alpha.runtime_policy import rank_trading_records
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, DATA_DIR, asset_data_dir
from signal_noise.client import SignalClient
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl import parse, collect_feature_names
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
from ..alpha.combiner import (
    CombinerConfig,
    WeightedCombinerConfig,
    compute_diversity_scores,
    compute_weights,
    select_low_correlation,
    signal_consensus,
    weighted_combine_scalar,
)
from ..risk.circuit_breaker import CircuitBreaker
from ..risk.manager import RiskManager
from ..runtime_profile import RuntimeProfile, build_runtime_profile
from ..voting.combiner import vote_combine
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
    n_deployed_alphas: int
    n_shortlist_candidates: int
    n_selected_alphas: int
    n_signals_evaluated: int
    profile_id: str = ""
    profile_commit: str = ""
    order_failures: int = 0
    n_skipped_deadband: int = 0
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

    @property
    def adjusted_signal(self) -> float:
        """Backward-compatible alias for the final post-tactical signal."""
        return self.final_signal


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
    skipped_min_notional: int = 0
    skipped_rounded_to_zero: int = 0


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

        mon_cfg = config.to_monitor_config()
        self.monitor = monitor or AlphaMonitor(config=mon_cfg)

        self.lifecycle = lifecycle or AlphaLifecycle(
            self.registry,
            config=config.to_lifecycle_config(),
        )

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
            client = SignalClient(
                base_url=config.api.base_url,
                timeout=config.api.timeout,
            )
            self.store = DataStore(DATA_DIR / "alpha_cache.db", client)

        self.tactical = tactical
        self._wcfg = WeightedCombinerConfig()
        self._diversity_cache: dict[str, float] = {}
        self._diversity_computed = False
        self._diversity_last_computed: float = 0.0
        self._executor_state_date = ""

        self._restore_state()

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

        if isinstance(self.executor, PaperExecutor):
            self.executor._cash = snapshot.cash
            self.executor._positions = dict(snapshot.positions)
        elif isinstance(self.executor, BinanceExecutor):
            self.executor._managed_cash = snapshot.cash
            self.executor._managed_positions = dict(snapshot.positions)
            tracked_symbols = list({self.price_signal, *snapshot.positions.keys()})
            self.executor.sync_reconciliation_baseline(tracked_symbols)

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
            record.oos_fitness(self.config.fitness_metric),
            returns,
        )

    def _recompute_diversity(
        self, data: dict[str, np.ndarray], parsed_records: list[tuple],
    ) -> None:
        """Recompute diversity scores from signal matrix."""
        n_days = len(next(iter(data.values())))
        signals = []
        alpha_ids = []

        for record, expr in parsed_records:
            try:
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

    def _load_diversity_from_db(self) -> None:
        """Load pre-computed diversity scores from diversity_cache table."""
        import sqlite3

        db_path = asset_data_dir(self.asset) / "alpha_registry.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA busy_timeout=30000")
        rows = conn.execute(
            "SELECT alpha_id, diversity_score FROM diversity_cache"
        ).fetchall()
        conn.close()

        if rows:
            self._diversity_cache = {r[0]: r[1] for r in rows}
            self._diversity_computed = True
            logger.info("Loaded %d diversity scores from cache", len(rows))

    def _compute_cycle_diversity(
        self,
        alpha_ids: list[str],
        alpha_signal_arrays: dict[str, np.ndarray],
    ) -> dict[str, float]:
        if not alpha_ids:
            return {}
        if len(alpha_ids) == 1:
            return {alpha_ids[0]: 1.0}

        available = [aid for aid in alpha_ids if aid in alpha_signal_arrays]
        if len(available) < 2:
            return {aid: 1.0 for aid in alpha_ids}

        lookback = min(
            self._wcfg.corr_lookback,
            *(len(alpha_signal_arrays[aid]) for aid in available),
        )
        if lookback < 2:
            return {aid: 1.0 for aid in alpha_ids}

        sig_matrix = np.array(
            [alpha_signal_arrays[aid][-lookback:] for aid in available],
            dtype=np.float64,
        )
        diversity = compute_diversity_scores(
            sig_matrix, chunk_size=self._wcfg.chunk_size
        )
        scores = {aid: 1.0 for aid in alpha_ids}
        for aid, score in zip(available, diversity):
            scores[aid] = float(score)
        return scores

    def _resolve_diversity_scores(
        self,
        alpha_ids: list[str],
        alpha_signal_arrays: dict[str, np.ndarray],
    ) -> dict[str, float]:
        cached = np.array(
            [self._diversity_cache.get(aid, np.nan) for aid in alpha_ids],
            dtype=np.float64,
        )
        if (
            alpha_ids
            and np.all(np.isfinite(cached))
            and np.ptp(cached) > 1e-9
        ):
            return {aid: float(cached[i]) for i, aid in enumerate(alpha_ids)}

        live_scores = self._compute_cycle_diversity(alpha_ids, alpha_signal_arrays)
        if live_scores:
            logger.info(
                "Using live cycle diversity for %d alphas (cache missing or degenerate)",
                len(alpha_ids),
            )
            return live_scores
        return {aid: float(self._diversity_cache.get(aid, 1.0)) for aid in alpha_ids}

    def _predict_portfolio_signal(
        self,
        alpha_signals: dict[str, float],
        alpha_signal_arrays: dict[str, np.ndarray],
        quality_estimates: dict[str, QualityEstimate],
        alpha_exprs: dict[str, object],
        all_alphas: list,
        data: dict[str, np.ndarray],
        parsed_records: list[tuple],
        skip_lifecycle: bool,
    ) -> PredictionOutput:
        """Prediction layer: combine alpha signals into an adjusted portfolio signal."""
        strategic_signal = 0.0
        regime_adjusted_signal = 0.0
        tactical_adjusted_signal = 0.0
        final_signal = 0.0
        use_map_elites = self.config.paper.combine_mode == "map_elites"
        if use_map_elites:
            pass  # diversity not needed — cell structure handles it
        elif skip_lifecycle and self.config.admission.enabled:
            self._load_diversity_from_db()
        else:
            diversity_stale = (
                not self._diversity_computed
                or (time.time() - self._diversity_last_computed)
                > self._wcfg.diversity_recompute_days * 86400
            )
            if diversity_stale:
                self._recompute_diversity(data, parsed_records)
                self._diversity_last_computed = time.time()

        prev_value = self.executor.portfolio_value
        self.risk_manager.update_equity(prev_value)
        recent_returns = np.array(self.portfolio_tracker.get_returns())
        dd_s = self.risk_manager.dd_scale

        combine_mode = self.config.paper.combine_mode

        if alpha_signals and combine_mode == "map_elites":
            from ..evolution.archive import AlphaArchive
            from ..evolution.behavior import compute_behavior
            from ..voting.ensemble import compute_cell_long_pcts, ensemble_sizing

            archive = AlphaArchive()
            cell_signals: dict[tuple[int, ...], list[float]] = {}
            for aid, sig_val in alpha_signals.items():
                sig_arr = alpha_signal_arrays[aid]
                behavior = compute_behavior(sig_arr, alpha_exprs[aid])
                cell = archive._to_cell(behavior)
                cell_signals.setdefault(cell, []).append(sig_val)

            cell_long_pcts = compute_cell_long_pcts(None, cell_signals)
            ens = ensemble_sizing(cell_long_pcts)
            strategic_signal = ens.direction * ens.confidence * ens.skew_adj * dd_s
            strategic_signal = float(np.clip(strategic_signal, -1, 1))
            combined = ens.direction * ens.confidence * ens.skew_adj
            logger.info(
                "MAP-Elites: dir=%.0f conf=%.3f skew=%.3f cells=%d/%d μ=%.3f σ=%.3f dd=%.2f",
                ens.direction, ens.confidence, ens.skew_adj,
                ens.n_cells, len(cell_signals), ens.mu_cells, ens.sigma_cells, dd_s,
            )
        elif alpha_signals and combine_mode == "voting":
            registry_records = {r.alpha_id: r for r in all_alphas}
            vote_result = vote_combine(
                alpha_signals, self.forward_tracker, registry_records,
            )
            strategic_signal = vote_result.direction * vote_result.confidence * dd_s
            strategic_signal = float(np.clip(strategic_signal, -1, 1))
            combined = vote_result.direction * vote_result.confidence
            logger.info(
                "Voting: dir=%.0f conf=%.3f voters=%d long=%.0f%% short=%.0f%% dd=%.2f",
                vote_result.direction, vote_result.confidence,
                vote_result.n_voters, vote_result.long_pct * 100,
                vote_result.short_pct * 100, dd_s,
            )
        elif alpha_signals:
            alpha_ids = list(alpha_signals.keys())
            quality_list = []
            confidence_list = []
            diversity_scores = self._resolve_diversity_scores(
                alpha_ids, alpha_signal_arrays
            )
            for aid in alpha_ids:
                estimate = quality_estimates[aid]
                quality_list.append(estimate.blended_quality)
                confidence_list.append(estimate.confidence)
            diversity_list = [diversity_scores.get(aid, 1.0) for aid in alpha_ids]

            quality_np = np.array(quality_list)
            confidence_np = np.array(confidence_list)
            diversity_np = np.array(diversity_list)
            shrunk_quality_np = shrink_weight_quality(
                quality_np,
                confidence_np,
                floor=self.config.live_quality.weight_confidence_floor,
                power=self.config.live_quality.weight_confidence_power,
            )
            w = compute_weights(
                shrunk_quality_np,
                diversity_np,
                min_weight=self._wcfg.min_weight,
            )
            weights_dict = {aid: float(w[i]) for i, aid in enumerate(alpha_ids)}

            combined = weighted_combine_scalar(alpha_signals, weights_dict)
            sig_mean, sig_std, consensus = signal_consensus(alpha_signals, weights_dict)
            strategic_signal = float(np.sign(sig_mean)) * consensus * dd_s
            strategic_signal = float(np.clip(strategic_signal, -1, 1))
            if (
                self.config.live_quality.weight_confidence_floor < 1.0
                or self.config.live_quality.weight_confidence_power != 1.0
            ):
                shrink_np = confidence_weight_scale(
                    confidence_np,
                    floor=self.config.live_quality.weight_confidence_floor,
                    power=self.config.live_quality.weight_confidence_power,
                )
                logger.info(
                    "Confidence shrink: floor=%.2f power=%.2f mean=%.3f min=%.3f",
                    self.config.live_quality.weight_confidence_floor,
                    self.config.live_quality.weight_confidence_power,
                    float(np.mean(shrink_np)),
                    float(np.min(shrink_np)),
                )
            logger.info(
                "Sizing: dd=%.2f cons=%.3f sig=%.4f±%.4f (%d alphas)",
                dd_s, consensus, sig_mean, sig_std, len(alpha_signals),
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
        )

    def _build_allocation_plan(
        self,
        adjusted_signal: float,
        prev_value: float,
        today_date: str,
    ) -> AllocationPlan:
        """Allocation layer: convert adjusted signal into target portfolio."""
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
            adjusted_signal=adjusted_signal,
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

    def _runtime_profile(self, deployed_records=None) -> RuntimeProfile:
        records = deployed_records
        if records is None:
            records = self.registry.list_deployed_alphas()
        return build_runtime_profile(
            asset=self.asset,
            config=self.config,
            deployed_alpha_ids=[record.alpha_id for record in records],
        )

    def _strategy_epoch(self, universe_records) -> str:
        return self._runtime_profile(universe_records).profile_id

    def run_cycle(
        self,
        simulation_date: str | None = None,
        skip_lifecycle: bool = False,
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
                    combined_signal=last_snapshot.combined_signal,
                    fills=[],
                    portfolio_value=last_snapshot.portfolio_value,
                    daily_pnl=last_snapshot.daily_pnl,
                    daily_return=last_snapshot.daily_return,
                    dd_scale=last_snapshot.dd_scale,
                    vol_scale=last_snapshot.vol_scale,
                    n_registry_active=0,
                    n_deployed_alphas=0,
                    n_shortlist_candidates=0,
                    n_selected_alphas=0,
                    n_signals_evaluated=0,
                )

        # 0. Circuit breaker check
        prev_equity = self.executor.portfolio_value
        deployed_universe = self.registry.list_deployed_alphas()
        if not deployed_universe:
            raise RuntimeError(
                "No deployed alphas. Run `refresh-deployed-alphas` before trading."
            )
        current_profile = self._runtime_profile(deployed_universe)
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
                profile_commit=current_profile.git_commit, combined_signal=0.0, fills=[],
                portfolio_value=prev_equity, daily_pnl=0.0, daily_return=0.0,
                dd_scale=1.0, vol_scale=1.0,
                n_registry_active=0, n_deployed_alphas=0,
                n_shortlist_candidates=0,
                n_selected_alphas=0, n_signals_evaluated=0,
            )

        # 1. Sync data (skip in simulation mode — use cached data)
        if simulation_date is None:
            try:
                logger.info("Syncing %d signals...", len(self.features))
                self.store.sync(self.features)
            except Exception as exc:
                logger.warning("API sync failed — using cached data: %s", exc)

        # 2. Get the deployed universe, then shortlist tradable candidates from it.
        max_trading = self.config.paper.max_trading_alphas
        universe_records = deployed_universe
        n_deployed_alphas = len(deployed_universe)
        trading_candidates = rank_trading_records(
            universe_records,
            lambda record: self._estimate_quality(
                record,
                self.forward_tracker.get_returns(record.alpha_id),
            ),
            max_trading=max_trading,
            shortlist_preselect_factor=self.config.live_quality.shortlist_preselect_factor,
            metric=self.config.fitness_metric,
        )
        dormant = [
            record for record in universe_records
            if AlphaState.canonical(record.state) == AlphaState.DORMANT
        ]
        all_alphas = trading_candidates + dormant
        dormant_ids = {r.alpha_id for r in dormant}

        n_total_active = self.registry.count(AlphaState.ACTIVE)
        if n_deployed_alphas > max_trading or n_total_active > max_trading:
            logger.info(
                "Selection pool: %d shortlist candidates from %d deployed alphas (%d registry-active)",
                len(trading_candidates),
                n_deployed_alphas,
                n_total_active,
            )
        selected_records = trading_candidates

        # 3. Evaluate each alpha's signal (uses daily data)
        matrix = self.store.get_matrix(self.features, end=today_date)
        if len(matrix) < 2:
            logger.warning("Insufficient data for %s (%d rows)", today_date, len(matrix))
            return PaperCycleResult(
                date=cycle_key, profile_id=current_profile.profile_id,
                profile_commit=current_profile.git_commit, combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=self.executor.portfolio_value,
                n_registry_active=n_total_active,
                n_deployed_alphas=n_deployed_alphas,
                n_shortlist_candidates=len(trading_candidates),
                n_selected_alphas=0,
                n_signals_evaluated=0,
            )

        data = {col: matrix[col].values for col in matrix.columns}
        prices_arr = data[self.price_signal]
        if len(prices_arr) < 2:
            return PaperCycleResult(
                date=cycle_key, profile_id=current_profile.profile_id,
                profile_commit=current_profile.git_commit, combined_signal=0.0, dd_scale=1.0,
                vol_scale=1.0, fills=[], portfolio_value=self.executor.portfolio_value,
                n_registry_active=n_total_active,
                n_deployed_alphas=n_deployed_alphas,
                n_shortlist_candidates=len(trading_candidates),
                n_selected_alphas=0,
                n_signals_evaluated=0,
            )
        price_return = (prices_arr[-1] - prices_arr[-2]) / prices_arr[-2]

        alpha_signals: dict[str, float] = {}
        alpha_signal_arrays: dict[str, np.ndarray] = {}
        quality_estimates: dict[str, QualityEstimate] = {}
        alpha_exprs: dict[str, object] = {}  # aid → parsed Expr (for map_elites)
        n_evaluated = 0
        n_failed = 0
        n_feature_filtered = 0
        parsed_records: list[tuple] = []
        available_features = set(data.keys())

        for record in all_alphas:
            try:
                expr = parse(record.expression)
            except SyntaxError as exc:
                logger.warning("Failed to parse %s: %s", record.alpha_id, exc)
                n_failed += 1
                continue
            required = collect_feature_names(expr)
            if not required.issubset(available_features):
                n_feature_filtered += 1
                continue
            parsed_records.append((record, expr))

        for record, expr in parsed_records:
            try:
                signal = evaluate_expression(expr, data, len(matrix))
                signal_norm = normalize_signal(signal)
                signal_yesterday = float(signal_norm[-2])
                daily_return = signal_yesterday * price_return

                if record.alpha_id not in dormant_ids:
                    alpha_signals[record.alpha_id] = signal_yesterday
                    alpha_signal_arrays[record.alpha_id] = signal_norm
                    alpha_exprs[record.alpha_id] = expr
                n_evaluated += 1

                # Record per-alpha forward return (daily granularity)
                self.forward_tracker.record(
                    record.alpha_id, today_date, signal_yesterday, daily_return,
                )

                # Monitor (always needed for weight calculation)
                all_returns = self.forward_tracker.get_returns(record.alpha_id)
                recent_returns = all_returns[-self.config.forward.degradation_window:]
                self.monitor.clear(record.alpha_id)
                self.monitor.record_batch(record.alpha_id, recent_returns)
                estimate = self._estimate_quality(record, all_returns)
                quality_estimates[record.alpha_id] = estimate

                # Lifecycle transitions (skip when lifecycle daemon handles it)
                if not skip_lifecycle:
                    old_state = record.state
                    new_state = self.lifecycle.evaluate_live(
                        record.alpha_id,
                        estimate,
                        dormant_revival_min_observations=(
                            self.config.live_quality.dormant_revival_min_observations
                        ),
                    )
                    if new_state != old_state:
                        self.audit_log.log_state_change(
                            record.alpha_id, old_state, new_state,
                            reason=(
                                "paper: blended="
                                f"{estimate.blended_quality:.3f} "
                                f"prior={estimate.prior_quality:.3f} "
                                f"live={estimate.live_quality:.3f} "
                                f"n={estimate.n_observations}"
                            ),
                        )

            except EvaluationError as exc:
                logger.warning("Failed to evaluate %s: %s", record.alpha_id, exc)
                n_failed += 1

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

        # 3b. Correlation filter: select top-N decorrelated alphas
        # (skipped in map_elites mode — diversity handled by cell structure)
        use_map_elites = self.config.paper.combine_mode == "map_elites"
        if len(alpha_signals) > max_trading and not use_map_elites:
            candidate_ids = list(alpha_signals.keys())
            lookback = min(self._wcfg.corr_lookback, len(matrix))
            sig_matrix = np.array([
                alpha_signal_arrays[aid][-lookback:] for aid in candidate_ids
            ])
            quality_for_sel = np.array([
                quality_estimates[aid].blended_quality
                for aid in candidate_ids
            ])
            selected_idx = select_low_correlation(
                sig_matrix, quality_for_sel,
                CombinerConfig(max_alphas=max_trading),
            )
            selected_ids = {candidate_ids[i] for i in selected_idx}
            n_before = len(alpha_signals)
            alpha_signals = {k: v for k, v in alpha_signals.items() if k in selected_ids}
            logger.info(
                "Correlation filter: %d shortlist candidates → %d selected alphas (max_corr=%.2f)",
                n_before, len(alpha_signals), CombinerConfig().max_correlation,
            )
        selected_records = [
            r for r in trading_candidates if r.alpha_id in alpha_signals
        ]

        # 4. Prediction layer: alpha signals -> adjusted portfolio signal
        prev_value = self.executor.portfolio_value
        prediction = self._predict_portfolio_signal(
            alpha_signals=alpha_signals,
            alpha_signal_arrays=alpha_signal_arrays,
            quality_estimates=quality_estimates,
            alpha_exprs=alpha_exprs,
            all_alphas=all_alphas,
            data=data,
            parsed_records=parsed_records,
            skip_lifecycle=skip_lifecycle,
        )

        # 5. Allocation layer: adjusted signal -> target portfolio
        plan = self._build_allocation_plan(
            adjusted_signal=prediction.final_signal,
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
            final_signal=prediction.final_signal,
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
            combined_signal=prediction.combined_signal,
            fills=fills,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            dd_scale=prediction.dd_scale,
            vol_scale=1.0,
            n_registry_active=n_total_active,
            n_deployed_alphas=n_deployed_alphas,
            n_shortlist_candidates=len(trading_candidates),
            n_selected_alphas=len(selected_records),
            n_signals_evaluated=n_evaluated,
            order_failures=order_failures,
            n_skipped_deadband=execution.skipped_deadband,
            n_skipped_min_notional=execution.skipped_min_notional,
            n_skipped_rounded_to_zero=execution.skipped_rounded_to_zero,
            strategic_signal=prediction.strategic_signal,
            regime_adjusted_signal=prediction.regime_adjusted_signal,
            tactical_adjusted_signal=prediction.tactical_adjusted_signal,
            final_signal=prediction.final_signal,
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


# Backward compatibility
PaperTrader = Trader
