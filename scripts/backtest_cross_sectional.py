"""Historical backtest of the cross-sectional trading strategy.

Simulates the CrossSectionalTrader logic over 2 years of data:
1. Load deployed alphas from BTC registry
2. For each day: evaluate alphas → per-asset TC signal → neutralize → allocate
3. Track portfolio P&L across BTC, ETH, SOL
"""
import numpy as np
from pathlib import Path
from datetime import date

from alpha_os.config import Config, DATA_DIR, asset_data_dir
from alpha_os.data.store import DataStore
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.universe import build_feature_list, price_signal
from alpha_os.alpha.managed_alphas import ManagedAlphaStore
from alpha_os.alpha.combiner import (
    compute_tc_scores,
    compute_tc_weights,
    cross_asset_neutralize,
    weighted_combine_scalar,
)
from alpha_os.alpha.evaluator import evaluate_expression, normalize_signal, sanitize_signal
from alpha_os.backtest.benchmark import build_benchmark_returns
from alpha_os.backtest import metrics
from alpha_os.dsl import parse


def run_backtest():
    cfg = Config.load(Path("/home/dev/.config/alpha-os/prod.toml"))
    client = build_signal_client_from_config(cfg.api)
    store = DataStore(DATA_DIR / "alpha_cache.db", client)

    tradeable = ["BTC", "ETH", "SOL"]
    price_sigs = {a: price_signal(a) for a in tradeable}

    # Load data
    features = build_feature_list("BTC")
    matrix = store.get_matrix(features, end=date.today().isoformat())
    data = {col: matrix[col].values for col in matrix.columns}
    print(f"Data: {len(matrix)} rows, {len(data)} features")

    # Find common date range where all 3 assets have prices
    starts = []
    for asset in tradeable:
        ps = price_sigs[asset]
        arr = data.get(ps)
        if arr is None:
            print(f"  {asset}: NO DATA")
            return
        first = np.argmax(np.isfinite(arr))
        starts.append(first)
        print(f"  {asset} ({ps}): first valid at index {first}")
    common_start = max(starts)
    T = len(matrix) - common_start
    print(f"Common range: index {common_start} to {len(matrix)} ({T} days)")

    # Load deployed alphas
    adir = asset_data_dir("BTC")
    reg = ManagedAlphaStore(db_path=adir / "alpha_registry.db")
    deployed_ids = reg.deployed_alpha_ids()
    print(f"Deployed alphas: {len(deployed_ids)}")

    # Parse and pre-evaluate all alphas
    alpha_signals = {}  # alpha_id -> full signal array
    for aid in deployed_ids:
        record = reg.get(aid)
        if not record:
            continue
        try:
            expr = parse(record.expression)
            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(matrix), float(sig))
            alpha_signals[aid] = normalize_signal(sig)
        except Exception:
            continue
    reg.close()
    print(f"Evaluable alphas: {len(alpha_signals)}")

    # Benchmark
    bm = build_benchmark_returns(data, cfg.backtest.benchmark_assets)

    # Simulate day-by-day
    lookback = 252  # 1 year for TC computation
    start_idx = common_start + lookback  # need lookback for TC
    n_days_sim = len(matrix) - start_idx

    if n_days_sim < 60:
        print(f"Not enough simulation days: {n_days_sim}")
        return

    print(f"\nSimulating {n_days_sim} days (from index {start_idx})...")

    # Per-asset daily returns
    asset_returns = {}
    for asset in tradeable:
        ps = price_sigs[asset]
        p = data[ps]
        rets = np.diff(p) / p[:-1]
        asset_returns[asset] = rets

    # Portfolio simulation
    initial_capital = 10000.0
    portfolio_value = initial_capital
    daily_returns = []
    daily_signals = {a: [] for a in tradeable}
    max_per_asset_pct = 0.5

    for t in range(start_idx, len(matrix) - 1):
        # Get alpha signals at time t (use t-1 as "yesterday")
        day_alpha_signals = {}
        day_alpha_arrays = {}
        for aid, sig_arr in alpha_signals.items():
            val = float(sig_arr[t - 1])
            if np.isfinite(val):
                day_alpha_signals[aid] = val
                day_alpha_arrays[aid] = sig_arr[:t]

        if not day_alpha_signals:
            daily_returns.append(0.0)
            continue

        # Per-asset TC-weighted signal
        per_asset = {}
        for asset in tradeable:
            ps = price_sigs[asset]
            a_rets = asset_returns[asset][t - lookback:t]
            finite = np.isfinite(a_rets)
            clean_rets = a_rets[finite]
            if len(clean_rets) < 20:
                per_asset[asset] = 0.0
                continue

            trimmed_arrays = {}
            for aid, arr in day_alpha_arrays.items():
                chunk = arr[t - lookback:t][finite]
                if len(chunk) == len(clean_rets):
                    trimmed_arrays[aid] = chunk

            if not trimmed_arrays:
                per_asset[asset] = 0.0
                continue

            tc = compute_tc_scores(trimmed_arrays, clean_rets)
            weights = compute_tc_weights(tc)
            combined = weighted_combine_scalar(day_alpha_signals, weights)
            per_asset[asset] = combined

        # Neutralize
        neutralized = cross_asset_neutralize(per_asset)

        # Allocate and compute return
        total_abs = sum(abs(v) for v in neutralized.values())
        day_return = 0.0
        for asset in tradeable:
            sig = neutralized.get(asset, 0.0)
            if not np.isfinite(sig) or abs(sig) < 1e-6:
                continue

            weight = abs(sig) / total_abs if total_abs > 0 else 1.0 / len(tradeable)
            weight = min(weight, max_per_asset_pct)
            direction = np.sign(sig)

            # Spot long only: can't short
            if direction < 0:
                direction = 0.0

            position = direction * weight
            asset_ret = asset_returns[asset][t] if t < len(asset_returns[asset]) else 0.0
            if np.isfinite(asset_ret):
                day_return += position * asset_ret

            daily_signals[asset].append(float(neutralized.get(asset, 0.0)))

        daily_returns.append(day_return)

    daily_returns = np.array(daily_returns)
    daily_returns = daily_returns[np.isfinite(daily_returns)]

    # Results
    print(f"\n{'='*60}")
    print("Cross-Sectional Strategy Backtest Results")
    print(f"{'='*60}")
    print(f"  Period:          {n_days_sim} trading days")
    print(f"  Sharpe:          {metrics.sharpe_ratio(daily_returns):+.3f}")
    print(f"  Annual Return:   {metrics.annual_return(daily_returns):+.2%}")
    print(f"  Annual Vol:      {metrics.annual_volatility(daily_returns):.2%}")
    print(f"  Max Drawdown:    {metrics.max_drawdown(daily_returns):.2%}")
    print(f"  Sortino:         {metrics.sortino_ratio(daily_returns):+.3f}")
    print(f"  Calmar:          {metrics.calmar_ratio(daily_returns):+.3f}")
    print(f"  Log Growth:      {metrics.expected_log_growth(daily_returns):+.4f}")

    # Compare vs buy-and-hold BTC
    btc_rets = asset_returns["BTC"][start_idx:start_idx + len(daily_returns)]
    btc_rets = btc_rets[np.isfinite(btc_rets)]
    print(f"\n  --- vs Buy & Hold BTC ---")
    print(f"  BTC Sharpe:      {metrics.sharpe_ratio(btc_rets):+.3f}")
    print(f"  BTC Ann Return:  {metrics.annual_return(btc_rets):+.2%}")
    print(f"  BTC Max DD:      {metrics.max_drawdown(btc_rets):.2%}")

    # Monthly breakdown
    n_months = len(daily_returns) // 21
    if n_months > 0:
        print(f"\n  Monthly returns (last 12):")
        for m in range(max(0, n_months - 12), n_months):
            start = m * 21
            end = min(start + 21, len(daily_returns))
            monthly = np.prod(1 + daily_returns[start:end]) - 1
            print(f"    Month {m+1:2d}: {monthly:+.2%}")

    # Signal distribution
    print(f"\n  Signal distribution (mean neutralized):")
    for asset in tradeable:
        sigs = daily_signals[asset]
        if sigs:
            print(f"    {asset}: mean={np.mean(sigs):+.4f} std={np.std(sigs):.4f} long%={sum(1 for s in sigs if s > 0)/len(sigs):.0%}")


if __name__ == "__main__":
    run_backtest()
