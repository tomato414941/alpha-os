"""Backfill simulator — vectorized paper trading over historical dates."""
from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..alpha.evaluator import EvaluationError, evaluate_expression, normalize_signal
from ..alpha.lifecycle import LifecycleConfig, batch_transitions, ST_ACTIVE, ST_PROBATION, ST_DORMANT
from ..alpha.registry import AlphaRegistry, AlphaState
from ..config import Config, DATA_DIR, asset_data_dir
from ..data.store import DataStore
from ..data.universe import build_feature_list
from ..dsl import parse
from ..execution.paper import PaperExecutor
from ..alpha.combiner import (
    WeightedCombinerConfig,
    compute_diversity_scores,
    compute_weights,
)
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    n_days: int
    initial_capital: float
    final_value: float
    total_return: float
    sharpe: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    best_day: tuple[str, float]   # (date, return)
    worst_day: tuple[str, float]  # (date, return)


def run_backfill(
    asset: str,
    config: Config,
    start_date: str,
    end_date: str,
    registry_db: Path | None = None,
) -> SimulationResult:
    """Run vectorized paper trading simulation over a historical date range.

    Pre-computes all alpha signals once on the full dataset, then iterates
    days using array indexing. This is orders of magnitude faster than
    calling run_cycle() per day.
    """
    # 1. Load full data range
    from signal_noise.client import SignalClient
    client = SignalClient(
        base_url=config.api.base_url,
        timeout=config.api.timeout,
    )
    store = DataStore(DATA_DIR / "alpha_cache.db", client)
    features = build_feature_list(asset)
    price_sig = features[0]

    # Sync from REST API
    try:
        store.sync(features)
    except Exception:
        logger.warning("API sync failed — using cached data")

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

    # 2. Load and reset alphas
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        reg_path = registry_db or asset_data_dir(asset) / "alpha_registry.db"
        sim_reg_path = tmp_path / "registry.db"
        shutil.copy2(reg_path, sim_reg_path)
        registry = AlphaRegistry(sim_reg_path)

        all_states = [AlphaState.ACTIVE, AlphaState.PROBATION,
                      AlphaState.REJECTED, AlphaState.DORMANT]
        n_activated = 0
        for state in all_states:
            for record in registry.list_by_state(state):
                if record.state != AlphaState.ACTIVE:
                    registry.update_state(record.alpha_id, AlphaState.ACTIVE)
                    n_activated += 1
        all_records = registry.list_by_state(AlphaState.ACTIVE)
        n_alphas = len(all_records)
        logger.info("%d alphas (%d reactivated)", n_alphas, n_activated)
        if not all_records:
            raise RuntimeError("No alphas in registry. Run `evolve` first.")

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

        # Diversity scores for weighted combination
        wcfg = WeightedCombinerConfig()
        valid_signals = np.nan_to_num(signal_matrix[valid_mask])
        diversity_scores = compute_diversity_scores(valid_signals, chunk_size=wcfg.chunk_size)
        last_diversity_day = 0
        logger.info("Initial diversity scores computed for %d alphas", n_valid)

        # Pre-compute daily price returns
        price_returns = np.zeros(n_days)
        price_returns[1:] = np.diff(prices) / prices[:-1]

        # 4. Simulate day by day — fully vectorized inner loop
        risk_manager = RiskManager(config.risk.to_manager_config())
        executor = PaperExecutor(initial_cash=config.trading.initial_capital)
        initial_capital = config.trading.initial_capital
        max_position_pct = config.paper.max_position_pct
        min_trade_usd = config.paper.min_trade_usd

        # Lifecycle thresholds from config (single source of truth)
        lc_cfg = LifecycleConfig(
            oos_sharpe_min=config.lifecycle.oos_sharpe_min,
            probation_sharpe_min=config.lifecycle.probation_sharpe_min,
            dormant_sharpe_max=config.lifecycle.dormant_sharpe_max,
            correlation_max=config.lifecycle.correlation_max,
            dormant_revival_sharpe=config.lifecycle.dormant_revival_sharpe,
        )
        rolling_window = 63
        min_obs = 20

        alpha_state_vec = np.zeros(n_alphas, dtype=np.int8)

        # Pre-compute forward returns matrix
        returns_mat = np.zeros((n_alphas, n_days), dtype=np.float64)
        returns_mat[:, 1:] = signal_matrix[:, :-1] * price_returns[np.newaxis, 1:]

        daily_returns_out: list[float] = []
        daily_dates: list[str] = []
        total_trades = 0

        for d in range(1, n_days):
            today = dates[d]
            current_price = prices[d]

            # Rolling sharpe for all valid alphas (vectorized)
            window_start = max(0, d - rolling_window)
            window_len = d - window_start

            if window_len >= min_obs:
                recent = returns_mat[valid_mask, window_start + 1:d + 1]
                m = recent.mean(axis=1)
                s = recent.std(axis=1) + 1e-12
                rolling_sharpe = m / s * np.sqrt(252)
            else:
                rolling_sharpe = np.zeros(int(valid_mask.sum()))

            # State transitions (vectorized via shared lifecycle logic)
            valid_states = alpha_state_vec[valid_mask]
            new_states = batch_transitions(valid_states, rolling_sharpe, lc_cfg)
            alpha_state_vec[valid_mask] = new_states

            # Recompute diversity periodically (rolling window)
            if d - last_diversity_day >= wcfg.diversity_recompute_days:
                lookback = min(wcfg.corr_lookback, d)
                window_sigs = np.nan_to_num(signal_matrix[valid_mask, d - lookback:d])
                if window_sigs.shape[1] >= 20:
                    diversity_scores = compute_diversity_scores(
                        window_sigs, chunk_size=wcfg.chunk_size,
                    )
                    last_diversity_day = d

            # Combined signal from ACTIVE + PROBATION with quality × diversity weights
            trading_within_valid = (new_states == ST_ACTIVE) | (new_states == ST_PROBATION)
            if trading_within_valid.any():
                t_sharpes = rolling_sharpe[trading_within_valid]
                t_diversity = diversity_scores[trading_within_valid]
                weights = compute_weights(t_sharpes, t_diversity, min_weight=wcfg.min_weight)
                trading_mask = valid_mask.copy()
                trading_mask[valid_mask] &= trading_within_valid
                combined = float(np.clip(
                    np.dot(weights, signal_matrix[trading_mask, d - 1]), -1, 1,
                ))
            else:
                combined = 0.0

            # Risk adjustment
            prev_value = executor.portfolio_value
            risk_manager.update_equity(prev_value)
            recent_rets = np.array(daily_returns_out[-252:]) if daily_returns_out else np.array([])
            dd_s = risk_manager.dd_scale
            vol_s = risk_manager.vol_scale(recent_rets)
            adjusted = float(np.clip(combined * dd_s * vol_s, -1, 1))

            # Execute trade
            dollar_pos = adjusted * prev_value * max_position_pct
            target_shares = dollar_pos / current_price if current_price > 0 else 0.0
            if abs(dollar_pos) < min_trade_usd:
                target_shares = 0.0

            executor.set_price(price_sig, current_price)
            fills = executor.rebalance({price_sig: target_shares})
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
        win_rate=win_rate,
        best_day=(daily_dates[best_idx], float(rets[best_idx])),
        worst_day=(daily_dates[worst_idx], float(rets[worst_idx])),
    )
