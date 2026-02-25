#!/usr/bin/env python3
"""E2E test: pipeline → registry → forward test.

Generates synthetic market data, runs the full pipeline to produce ACTIVE alphas
in the registry, then splits data into "known" (training) and "future" (unseen)
periods to simulate forward testing.
"""
from __future__ import annotations

import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_os.alpha.lifecycle import AlphaLifecycle, LifecycleConfig
from alpha_os.alpha.monitor import AlphaMonitor, MonitorConfig
from alpha_os.alpha.registry import AlphaRegistry, AlphaState
from alpha_os.dsl import parse, to_string
from alpha_os.evolution.gp import GPConfig
from alpha_os.forward.tracker import ForwardTracker
from alpha_os.governance.gates import GateConfig
from alpha_os.pipeline.runner import PipelineConfig, PipelineRunner


# ---------------------------------------------------------------------------
# Data generation (same as e2e_test.py)
# ---------------------------------------------------------------------------

def generate_market_data(n_days: int = 750, seed: int = 42) -> tuple[dict[str, np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    market_returns = rng.normal(0.0003, 0.012, n_days)
    vol = np.ones(n_days) * 0.012
    for t in range(1, n_days):
        vol[t] = 0.94 * vol[t - 1] + 0.06 * abs(market_returns[t - 1])
        market_returns[t] = rng.normal(0.0003, max(vol[t], 0.005))

    stocks = {
        "nvda": (1.8, 0.001), "aapl": (1.0, 0.0005),
        "msft": (1.0, 0.0004), "amd": (1.5, 0.0006), "tsla": (2.0, 0.0002),
    }
    for name, (beta, drift) in stocks.items():
        idio = rng.normal(0, 0.015, n_days)
        data[name] = 100.0 * np.cumprod(1 + drift + beta * market_returns + idio)

    data["sp500"] = 4500.0 * np.cumprod(1 + market_returns)
    data["nasdaq"] = 14000.0 * np.cumprod(1 + market_returns * 1.2 + rng.normal(0, 0.005, n_days))
    data["russell2000"] = 2000.0 * np.cumprod(1 + market_returns * 0.9 + rng.normal(0, 0.008, n_days))

    vix = np.ones(n_days) * 18.0
    for t in range(1, n_days):
        shock = -40 * market_returns[t] + rng.normal(0, 2)
        vix[t] = max(10, vix[t - 1] + 0.05 * (18 - vix[t - 1]) + shock * 0.3)
    data["vix_close"] = vix
    data["fear_greed"] = np.clip(100 - vix * 2.5 + rng.normal(0, 10, n_days), 0, 100)

    tsy10 = np.ones(n_days) * 4.0
    tsy2 = np.ones(n_days) * 4.5
    for t in range(1, n_days):
        tsy10[t] = tsy10[t - 1] + 0.01 * (4.0 - tsy10[t - 1]) + rng.normal(0, 0.03)
        tsy2[t] = tsy2[t - 1] + 0.01 * (4.5 - tsy2[t - 1]) + rng.normal(0, 0.04)
    data["tsy_yield_10y"] = tsy10
    data["tsy_yield_2y"] = tsy2

    data["dxy"] = 104.0 * np.cumprod(1 + rng.normal(0.0001, 0.003, n_days))
    data["gold"] = 2000.0 * np.cumprod(1 + rng.normal(0.0002, 0.008, n_days))
    data["oil_wti"] = 75.0 * np.cumprod(1 + rng.normal(0, 0.015, n_days))

    return data, list(data.keys())


def seed_datastore_sqlite(db_path: Path, data: dict[str, np.ndarray], base_date: str = "2023-01-01"):
    """Write numpy arrays into a DataStore-compatible SQLite DB."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS signals ("
        "  name TEXT, date TEXT, value REAL,"
        "  PRIMARY KEY (name, date)"
        ")"
    )
    dates = pd.date_range(base_date, periods=len(next(iter(data.values()))), freq="B")
    for name, values in data.items():
        rows = [(name, d.strftime("%Y-%m-%d"), float(v)) for d, v in zip(dates, values)]
        conn.executemany(
            "INSERT OR REPLACE INTO signals (name, date, value) VALUES (?, ?, ?)",
            rows,
        )
    conn.commit()
    conn.close()
    return [d.strftime("%Y-%m-%d") for d in dates]


def main():
    tmp_dir = Path("/tmp/alpha_os_e2e_forward")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALPHA-OS E2E FORWARD TEST")
    print("  Pipeline → Registry → Forward Test (unseen data)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Generate data: 750 days total, split 500 train / 250 forward
    # -------------------------------------------------------------------------
    n_total = 750
    n_train = 500
    n_forward = n_total - n_train

    print(f"\n[1/5] Generating {n_total}-day data ({n_train} train + {n_forward} forward)...")
    t0 = time.perf_counter()
    data_full, features = generate_market_data(n_days=n_total, seed=42)

    data_train = {k: v[:n_train] for k, v in data_full.items()}
    data_forward = {k: v[n_train:] for k, v in data_full.items()}
    prices_train = data_train["nvda"]
    print(f"  {len(features)} signals, generated in {time.perf_counter() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # 2. Run pipeline on training data → register ACTIVE alphas
    # -------------------------------------------------------------------------
    print(f"\n[2/5] Running pipeline on {n_train}-day training data...")
    reg_path = tmp_dir / "registry.db"
    reg_path.unlink(missing_ok=True)
    registry = AlphaRegistry(db_path=reg_path)

    pipe_cfg = PipelineConfig(
        gp=GPConfig(pop_size=200, n_generations=20, max_depth=3),
        gate=GateConfig(
            oos_sharpe_min=1.0, pbo_max=1.0, dsr_pvalue_max=0.05,
            fdr_pass_required=False, max_correlation=0.5, min_n_days=200,
        ),
        commission_pct=0.10, slippage_pct=0.05,
    )

    runner = PipelineRunner(
        features, data_train, prices_train,
        config=pipe_cfg, registry=registry, seed=42,
    )
    result = runner.run()

    active_alphas = registry.list_active()
    print(f"  Generated: {result.n_generated} | Validated: {result.n_validated} "
          f"| Adopted: {result.n_adopted}")
    print(f"  ACTIVE alphas in registry: {len(active_alphas)}")
    if not active_alphas:
        print("\n  ERROR: No ACTIVE alphas — cannot proceed with forward test.")
        return

    print(f"\n  Top 5 ACTIVE alphas:")
    for i, rec in enumerate(active_alphas[:5]):
        print(f"    {i+1}. Sharpe={rec.oos_sharpe:.3f}  {rec.expression}")

    # -------------------------------------------------------------------------
    # 3. Seed DataStore with forward (unseen) data
    # -------------------------------------------------------------------------
    print(f"\n[3/5] Seeding DataStore with {n_forward}-day unseen forward data...")
    ds_path = tmp_dir / "forward_cache.db"
    ds_path.unlink(missing_ok=True)

    # Write forward data starting from the day after training ends
    forward_dates = seed_datastore_sqlite(ds_path, data_forward, base_date="2025-01-01")
    print(f"  Forward period: {forward_dates[0]} to {forward_dates[-1]} ({len(forward_dates)} days)")

    # -------------------------------------------------------------------------
    # 4. Run forward test day by day
    # -------------------------------------------------------------------------
    print(f"\n[4/5] Running forward test on {len(active_alphas)} ACTIVE alphas...")
    t1 = time.perf_counter()

    fwd_db_path = tmp_dir / "forward_returns.db"
    fwd_db_path.unlink(missing_ok=True)
    tracker = ForwardTracker(db_path=fwd_db_path)
    monitor = AlphaMonitor(config=MonitorConfig(rolling_window=63, min_observations=20))
    lifecycle = AlphaLifecycle(registry, config=LifecycleConfig(
        oos_sharpe_min=0.3, probation_sharpe_min=0.0,
    ))

    # Read forward data from SQLite
    conn = sqlite3.connect(str(ds_path))
    df_all = pd.read_sql_query(
        "SELECT name, date, value FROM signals ORDER BY date", conn,
    )
    conn.close()
    matrix = df_all.pivot(index="date", columns="name", values="value").ffill()

    n_evaluated_total = 0
    n_degraded_total = 0
    n_retired_total = 0
    n_dormant_total = 0

    for alpha_rec in active_alphas:
        aid = alpha_rec.alpha_id
        tracker.register_alpha(aid, forward_dates[0])

        try:
            expr = parse(alpha_rec.expression)
        except Exception as e:
            print(f"  SKIP {aid}: parse error: {e}")
            continue

        # Evaluate day by day on expanding window
        for day_idx in range(1, len(matrix)):
            window = matrix.iloc[:day_idx + 1]
            data_slice = {col: window[col].values for col in window.columns}
            today = window.index[day_idx]

            try:
                signal = expr.evaluate(data_slice)
                signal = np.nan_to_num(np.asarray(signal, dtype=float), nan=0.0)
                if signal.ndim == 0:
                    signal = np.full(len(window), float(signal))

                prices = data_slice["nvda"]
                if len(prices) < 2:
                    continue
                price_return = (prices[-1] - prices[-2]) / prices[-2]

                std = signal.std()
                if std > 0:
                    sig_norm = np.clip(signal / std, -1, 1)
                else:
                    sig_norm = np.clip(np.sign(signal), -1, 1)

                daily_ret = float(sig_norm[-2]) * price_return
                tracker.record(aid, today, float(sig_norm[-2]), daily_ret)
                n_evaluated_total += 1
            except Exception:
                continue

        # Monitor after all days
        all_returns = tracker.get_returns(aid)
        if all_returns:
            monitor.clear(aid)
            monitor.record_batch(aid, all_returns)
            status = monitor.check(aid)

            current_state = registry.get(aid).state
            if current_state == AlphaState.ACTIVE:
                new_state = lifecycle.evaluate_active(aid, status.rolling_sharpe)
            elif current_state == AlphaState.PROBATION:
                new_state = lifecycle.evaluate_probation(aid, status.rolling_sharpe)
            elif current_state == AlphaState.DORMANT:
                new_state = lifecycle.evaluate_dormant(aid, status.rolling_sharpe)
            else:
                new_state = current_state

            if new_state == AlphaState.PROBATION and current_state == AlphaState.ACTIVE:
                n_degraded_total += 1
            elif new_state == AlphaState.DORMANT:
                n_dormant_total += 1
            elif new_state == AlphaState.RETIRED:
                n_retired_total += 1

    fwd_time = time.perf_counter() - t1
    print(f"  Forward test: {n_evaluated_total} evaluations in {fwd_time:.2f}s")

    # -------------------------------------------------------------------------
    # 5. Report
    # -------------------------------------------------------------------------
    print(f"\n[5/5] Forward Test Results")
    print("=" * 85)

    alive = registry.list_by_state(AlphaState.ACTIVE)
    probation = registry.list_by_state(AlphaState.PROBATION)
    dormant_list = registry.list_by_state(AlphaState.DORMANT)
    retired = registry.list_by_state(AlphaState.RETIRED)

    print(f"  ACTIVE: {len(alive)} | PROBATION: {len(probation)} | DORMANT: {len(dormant_list)} | RETIRED: {len(retired)}")
    print(f"  Degraded: {n_degraded_total} | Dormant: {n_dormant_total} | Retired: {n_retired_total}")

    tracked_ids = tracker.tracked_alpha_ids()
    if tracked_ids:
        # Collect summaries
        summaries = []
        for aid in tracked_ids:
            s = tracker.summary(aid)
            if s and s.n_days > 0:
                summaries.append(s)

        summaries.sort(key=lambda x: x.sharpe, reverse=True)

        print(f"\n  {'Alpha':>20}  {'Days':>5}  {'Return':>8}  {'Sharpe':>8}  "
              f"{'MaxDD':>8}  {'State':>10}")
        print("  " + "-" * 75)

        for s in summaries[:20]:
            rec = registry.get(s.alpha_id)
            state = rec.state if rec else "?"
            print(
                f"  {s.alpha_id:>20}  {s.n_days:>5}  "
                f"{s.total_return:>7.2%}  {s.sharpe:>8.3f}  "
                f"{s.max_dd:>7.2%}  {state:>10}"
            )

        # Aggregate stats
        if summaries:
            sharpes = [s.sharpe for s in summaries]
            returns = [s.total_return for s in summaries]
            print(f"\n  Aggregate ({len(summaries)} alphas):")
            print(f"    Median Forward Sharpe:  {np.median(sharpes):.3f}")
            print(f"    Mean Forward Sharpe:    {np.mean(sharpes):.3f}")
            print(f"    Best Forward Sharpe:    {max(sharpes):.3f}")
            print(f"    Worst Forward Sharpe:   {min(sharpes):.3f}")
            print(f"    Median Forward Return:  {np.median(returns):.2%}")
            print(f"    % Profitable:           {sum(1 for r in returns if r > 0) / len(returns):.1%}")

        # Compare backtest Sharpe vs forward Sharpe
        bt_vs_fwd = []
        for s in summaries:
            rec = registry.get(s.alpha_id)
            if rec:
                bt_vs_fwd.append((rec.oos_sharpe, s.sharpe))

        if bt_vs_fwd:
            bt_sharpes = [x[0] for x in bt_vs_fwd]
            fwd_sharpes = [x[1] for x in bt_vs_fwd]
            corr = np.corrcoef(bt_sharpes, fwd_sharpes)[0, 1]
            decay = np.mean(fwd_sharpes) / np.mean(bt_sharpes) if np.mean(bt_sharpes) != 0 else 0
            print(f"\n  Backtest vs Forward:")
            print(f"    Mean BT Sharpe:       {np.mean(bt_sharpes):.3f}")
            print(f"    Mean Fwd Sharpe:      {np.mean(fwd_sharpes):.3f}")
            print(f"    Sharpe Decay Ratio:   {decay:.2f}  (1.0 = no decay)")
            print(f"    BT-Fwd Correlation:   {corr:.3f}")

    total_time = time.perf_counter() - t0
    print(f"\n  Total time: {total_time:.1f}s")
    print("=" * 80)

    # Cleanup
    tracker.close()
    registry.close()


if __name__ == "__main__":
    main()
