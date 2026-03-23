"""Replay simulator — vectorized paper trading over historical dates."""
# TODO: This simulator still depends on legacy managed/deployed registry state.
# Keep it outside the hypotheses-first runtime mainline until replay can run
# directly from hypothesis snapshots or live-hypothesis selections.
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.lifecycle import batch_live_transitions, ST_ACTIVE, ST_DORMANT
from ..alpha.runtime_policy import dormant_indices, rank_trading_indices
from ..alpha.managed_alphas import ManagedAlphaStore, AlphaState
from ..alpha.deployed_alphas import refresh_deployed_alphas
from ..config import Config, SIGNAL_CACHE_DB, asset_data_dir
from ..data.signal_client import build_signal_client_from_config
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl import parse
from ..execution.paper import PaperExecutor
from ..execution.planning import build_target_position, plan_execution_intent
from ..alpha.combiner import (
    CombinerConfig,
    compute_tc_scores,
    compute_tc_weights,
    select_low_correlation,
)
from ..alpha.monitor import RegimeDetector
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)
ST_EXCLUDED = -1


@dataclass
class SimulationResult:
    n_days: int
    initial_capital: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    n_skipped_deadband: int
    n_skipped_min_notional: int
    n_skipped_rounded_to_zero: int
    win_rate: float
    best_day: tuple[str, float]   # (date, return)
    worst_day: tuple[str, float]  # (date, return)


def _initial_simulation_state(state: str) -> int:
    """Map registry state to simulator lifecycle state."""
    state = AlphaState.canonical(state)
    if state == AlphaState.ACTIVE:
        return ST_ACTIVE
    if state == AlphaState.DORMANT:
        return ST_DORMANT
    return ST_EXCLUDED


def _live_like_eval_indices(
    records: list,
    state_codes: np.ndarray,
    *,
    prior_quality: np.ndarray,
    blended_quality: np.ndarray,
    confidence: np.ndarray,
    max_trading: int,
    metric: str,
    shortlist_preselect_factor: int = 20,
) -> tuple[list[int], list[int], list[int]]:
    """Return (trading_candidates, dormant, eval_set) using live trader rules."""
    trading_candidates = rank_trading_indices(
        records,
        state_codes,
        prior_quality=prior_quality,
        blended_quality=blended_quality,
        confidence=confidence,
        max_trading=max_trading,
        metric=metric,
        shortlist_preselect_factor=shortlist_preselect_factor,
    )
    dormant = dormant_indices(state_codes)
    eval_set = trading_candidates + dormant
    return trading_candidates, dormant, eval_set


def _replay_signals_to_position_intent(
    signals: np.ndarray,
    weights: np.ndarray,
    *,
    dd_scale: float,
    vol_scale: float,
    sizing_mode: str = "runtime",
) -> tuple[float, float]:
    """Return (raw_combined, final_signal) for replay simulation."""
    raw_combined = float(np.clip(np.dot(weights, signals), -1.0, 1.0))
    if sizing_mode == "raw_mean":
        adjusted = raw_combined * dd_scale
        return raw_combined, float(np.clip(adjusted, -1.0, 1.0))

    # TC-weighted consensus sizing
    mean = float(np.dot(weights, signals))
    centered = signals - mean
    std = float(np.sqrt(np.dot(weights, centered * centered)))
    abs_mean = abs(mean)
    consensus = abs_mean / (abs_mean + std) if (abs_mean + std) > 1e-12 else 0.0
    adjusted = float(np.sign(mean)) * consensus * dd_scale
    return raw_combined, float(np.clip(adjusted, -1.0, 1.0))


def _apply_regime_adjustment(
    signal: float,
    portfolio_returns: np.ndarray,
    config: Config,
) -> float:
    """Apply the runtime trader's regime scaling to a replay signal."""
    if not config.regime.enabled or len(portfolio_returns) < config.regime.long_window:
        return signal

    regime = RegimeDetector(
        short_window=config.regime.short_window,
        long_window=config.regime.long_window,
    ).detect(portfolio_returns)
    if regime.drift_score <= config.regime.drift_threshold:
        return signal

    scale = max(
        config.regime.drift_position_scale_min,
        1.0 - regime.drift_score,
    )
    return float(np.clip(signal * scale, -1.0, 1.0))


def run_replay(
    asset: str,
    config: Config,
    start_date: str,
    end_date: str,
    registry_db: Path | None = None,
    sizing_mode: str = "runtime",
    refresh_universe_before_run: bool = False,
) -> SimulationResult:
    """Run vectorized paper trading replay over a historical date range.

    Pre-computes all alpha signals once on the full dataset, then iterates
    days using array indexing. This is orders of magnitude faster than
    calling run_cycle() per day.
    """
    # 1. Load full data range
    client = build_signal_client_from_config(config.api)
    store = DataStore(SIGNAL_CACHE_DB, client)
    features = build_feature_list(asset)
    price_sig = features[0]

    # Sync from REST API
    try:
        store.sync(features)
    except Exception as exc:
        logger.warning("API sync failed — using cached data: %s", exc)

    matrix = store.get_matrix(features, start=start_date, end=end_date)
    # Only require price signal; fill NaN for other signals
    if price_sig in matrix.columns:
        matrix = matrix[matrix[price_sig].notna()]
    matrix = matrix.bfill().fillna(0)
    store.close()

    if len(matrix) < 2:
        raise RuntimeError(
            f"Insufficient cached data for {start_date} to {end_date} "
            f"({len(matrix)} rows). Run `evolve` or `paper --once` first."
        )

    dates = [str(d) for d in matrix.index]
    data = {col: matrix[col].values for col in matrix.columns}
    prices = data[price_sig]
    n_days = len(prices)
    logger.info("Loaded %d days (%s to %s)", n_days, dates[0], dates[-1])

    # 2. Load current registry state
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        reg_path = registry_db or asset_data_dir(asset) / "alpha_registry.db"
        sim_reg_path = tmp_path / "registry.db"
        shutil.copy2(reg_path, sim_reg_path)
        if refresh_universe_before_run:
            refresh_deployed_alphas(
                sim_reg_path,
                config,
                asset=asset,
                forward_db_path=asset_data_dir(asset) / "forward_returns.db",
                dry_run=False,
                backup=False,
            )
        registry = ManagedAlphaStore(sim_reg_path)

        all_records = registry.list_deployed_alphas()
        if not all_records:
            raise RuntimeError(
                "No deployed alphas in replay. Run `refresh-deployed-alphas` first "
                "or use `replay-experiment --deployment-mode refresh`."
            )
        logger.info("%d deployed alphas loaded", len(all_records))
        n_alphas = len(all_records)
        logger.info("%d alphas loaded from registry", n_alphas)
        objective = config.portfolio.objective
        prior_quality_full = np.array(
            [record.oos_fitness(objective) for record in all_records],
            dtype=np.float64,
        )

        # 3. Pre-compute all signals (the key optimization)
        # signal_matrix[i, d] = normalized signal for alpha i at day d
        signal_matrix = np.zeros((n_alphas, n_days), dtype=np.float64)
        valid_mask = np.ones(n_alphas, dtype=bool)

        for i, record in enumerate(all_records):
            try:
                expr = parse(record.expression)
                sig = evaluate_expression(expr, data, n_days)
                sig = normalize_signal(sig)
                signal_matrix[i] = sig
            except EvaluationError:
                valid_mask[i] = False

        n_valid = int(valid_mask.sum())
        logger.info("Pre-computed signals: %d/%d valid", n_valid, n_alphas)

        # TC weights — recomputed periodically during simulation
        tc_recompute_interval = 63
        tc_weights_cache: dict[str, float] = {}
        last_tc_day = -tc_recompute_interval  # force initial compute

        # Pre-compute daily price returns
        price_returns = np.zeros(n_days)
        price_returns[1:] = np.diff(prices) / prices[:-1]

        # 4. Simulate day by day — fully vectorized inner loop
        risk_manager = RiskManager(config.risk.to_manager_config())
        executor = PaperExecutor(
            initial_cash=config.trading.initial_capital,
            supports_short=config.trading.supports_short,
            cost_model=config.execution.to_cost_model(),
        )
        initial_capital = config.trading.initial_capital
        max_position_pct = config.paper.max_position_pct
        min_trade_usd = config.paper.min_trade_usd
        rebalance_deadband_usd = config.paper.rebalance_deadband_usd

        # Lifecycle thresholds from config (single source of truth)
        lc_cfg = config.to_lifecycle_config()
        rolling_window = config.forward.degradation_window
        full_weight_obs = max(config.live_quality.full_weight_observations, 1)

        alpha_state_vec = np.array(
            [_initial_simulation_state(record.state) for record in all_records],
            dtype=np.int8,
        )

        # Pre-compute forward returns matrix
        returns_mat = np.zeros((n_alphas, n_days), dtype=np.float64)
        returns_mat[:, 1:] = signal_matrix[:, :-1] * price_returns[np.newaxis, 1:]

        daily_returns_out: list[float] = []
        daily_dates: list[str] = []
        total_trades = 0
        skipped_deadband = 0
        skipped_min_notional = 0
        skipped_rounded_to_zero = 0

        for d in range(1, n_days):
            today = dates[d]
            current_price = prices[d]

            # Rolling sharpe for all valid alphas (vectorized)
            window_start = max(0, d - rolling_window)
            window_len = d - window_start

            if window_len > 0:
                recent = returns_mat[valid_mask, window_start + 1:d + 1]
                if objective == "log_growth":
                    r_clipped = np.clip(recent, -0.999999, None)
                    rolling_quality = np.mean(np.log1p(r_clipped), axis=1) * 252
                else:
                    m = recent.mean(axis=1)
                    s = recent.std(axis=1)
                    rolling_quality = np.zeros(recent.shape[0], dtype=np.float64)
                    nonzero = s > 1e-12
                    rolling_quality[nonzero] = (
                        m[nonzero] / s[nonzero] * np.sqrt(252)
                    )
            else:
                rolling_quality = np.zeros(int(valid_mask.sum()))

            rolling_quality_full = np.zeros(n_alphas, dtype=np.float64)
            rolling_quality_full[valid_mask] = rolling_quality
            confidence = np.full(n_alphas, min(d / full_weight_obs, 1.0), dtype=np.float64)
            blended_quality_full = (
                (1.0 - confidence) * prior_quality_full
                + confidence * rolling_quality_full
            )

            trading_candidates, _, eval_indices = _live_like_eval_indices(
                all_records,
                alpha_state_vec,
                prior_quality=prior_quality_full,
                blended_quality=blended_quality_full,
                confidence=confidence,
                max_trading=config.paper.max_trading_alphas,
                metric=objective,
                shortlist_preselect_factor=config.live_quality.shortlist_preselect_factor,
            )

            # Live trader only evaluates top trading candidates plus dormant alphas.
            eval_indices = [i for i in eval_indices if valid_mask[i]]
            if d < config.live_quality.dormant_revival_min_observations:
                eval_indices = [
                    i for i in eval_indices if alpha_state_vec[i] != ST_DORMANT
                ]
            if eval_indices:
                eval_state_vec = alpha_state_vec[eval_indices]
                eval_quality = blended_quality_full[eval_indices]
                eval_observations = np.full(len(eval_indices), d, dtype=np.int32)
                new_states = batch_live_transitions(
                    eval_state_vec,
                    eval_quality,
                    eval_observations,
                    lc_cfg,
                    dormant_revival_min_observations=(
                        config.live_quality.dormant_revival_min_observations
                    ),
                )
                alpha_state_vec[eval_indices] = new_states

            # Recompute TC weights periodically
            if d - last_tc_day >= tc_recompute_interval:
                lookback_tc = min(252, d)
                if lookback_tc >= 20:
                    valid_indices = [i for i in range(n_alphas) if valid_mask[i]]
                    sig_dict = {}
                    for idx in valid_indices:
                        aid = all_records[idx].alpha_id
                        sig_dict[aid] = signal_matrix[idx, d - lookback_tc:d]
                    tc_returns = price_returns[d - lookback_tc:d]
                    tc_scores = compute_tc_scores(sig_dict, tc_returns)
                    tc_weights_cache = compute_tc_weights(tc_scores)
                    last_tc_day = d

            # Combined signal from live-like candidate set with correlation filter
            trading_indices = [i for i in trading_candidates if valid_mask[i]]
            if len(trading_indices) > config.paper.max_trading_alphas:
                lookback = min(252, d)
                corr_matrix = np.array([
                    signal_matrix[i, d - lookback:d] for i in trading_indices
                ])
                quality_for_sel = np.array([
                    blended_quality_full[i] for i in trading_indices
                ])
                selected_idx = select_low_correlation(
                    corr_matrix,
                    quality_for_sel,
                    CombinerConfig(max_alphas=config.paper.max_trading_alphas),
                )
                trading_indices = [trading_indices[i] for i in selected_idx]

            if trading_indices:
                # TC weights for selected trading alphas
                w_list = []
                for idx in trading_indices:
                    aid = all_records[idx].alpha_id
                    w_list.append(tc_weights_cache.get(aid, 1e-4))
                weights = np.array(w_list, dtype=np.float64)
                w_sum = weights.sum()
                if w_sum > 0:
                    weights = weights / w_sum
                else:
                    weights = np.full(len(trading_indices), 1.0 / len(trading_indices))
                selected_signals = np.array([signal_matrix[i, d - 1] for i in trading_indices])
            else:
                selected_signals = np.array([], dtype=np.float64)
                weights = np.array([], dtype=np.float64)

            # Risk adjustment
            prev_value = executor.portfolio_value
            risk_manager.update_equity(prev_value)
            recent_rets = np.array(daily_returns_out[-252:]) if daily_returns_out else np.array([])
            portfolio_rets = np.array(daily_returns_out) if daily_returns_out else np.array([])
            dd_s = risk_manager.dd_scale
            vol_s = risk_manager.vol_scale(recent_rets)
            if len(selected_signals):
                _, final_signal = _replay_signals_to_position_intent(
                    selected_signals,
                    weights,
                    dd_scale=dd_s,
                    vol_scale=vol_s,
                    sizing_mode=sizing_mode,
                )
                final_signal = _apply_regime_adjustment(final_signal, portfolio_rets, config)
            else:
                final_signal = 0.0

            # Execute trade
            executor.set_price(price_sig, current_price)
            target_position = build_target_position(
                symbol=price_sig,
                final_signal=final_signal,
                portfolio_value=prev_value,
                current_price=current_price,
                max_position_pct=max_position_pct,
                min_trade_usd=min_trade_usd,
                supports_short=executor.supports_short,
            )
            decision = plan_execution_intent(
                target_position,
                current_qty=executor.get_position(price_sig),
                rebalance_deadband_usd=rebalance_deadband_usd,
            )
            intent = decision.intent
            if intent is None:
                if decision.skip_reason == "deadband":
                    skipped_deadband += 1
                fills = []
            else:
                constrained = executor.constrain_intent(intent)
                if constrained.order is None and constrained.rejection_reason == "below_min_notional":
                    skipped_min_notional += 1
                if constrained.order is None and constrained.rejection_reason == "rounded_to_zero":
                    skipped_rounded_to_zero += 1
                fill = executor.execute_intent(intent) if constrained.order is not None else None
                fills = [fill] if fill is not None else []
            total_trades += len(fills)

            # Record daily return
            portfolio_value = executor.portfolio_value
            daily_pnl = portfolio_value - prev_value
            daily_ret = daily_pnl / prev_value if prev_value > 0 else 0.0
            daily_returns_out.append(daily_ret)
            daily_dates.append(today)

            if (d % 200 == 0) or d == n_days - 1:
                n_active = int((alpha_state_vec == ST_ACTIVE).sum())
                n_dormant = int((alpha_state_vec == ST_DORMANT).sum())
                logger.info(
                    "  Day %d/%d (%s): $%.2f  active=%d dormant=%d",
                    d, n_days - 1, today, portfolio_value, n_active, n_dormant,
                )

        registry.close()

    # 5. Compute summary statistics
    rets = np.array(daily_returns_out)
    n_sim_days = len(rets)
    if n_sim_days == 0:
        raise RuntimeError("No simulation days produced")

    final_value = executor.portfolio_value
    total_return = (final_value - initial_capital) / initial_capital
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(252))

    cum = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / running_max
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    wins = int(np.sum(rets > 0))
    win_rate = wins / n_sim_days

    best_idx = int(np.argmax(rets))
    worst_idx = int(np.argmin(rets))

    return SimulationResult(
        n_days=n_sim_days,
        initial_capital=initial_capital,
        final_value=final_value,
        total_return=total_return,
        sharpe=sharpe,
        max_drawdown=max_dd,
        total_trades=total_trades,
        n_skipped_deadband=skipped_deadband,
        n_skipped_min_notional=skipped_min_notional,
        n_skipped_rounded_to_zero=skipped_rounded_to_zero,
        win_rate=win_rate,
        best_day=(daily_dates[best_idx], float(rets[best_idx])),
        worst_day=(daily_dates[worst_idx], float(rets[worst_idx])),
    )
