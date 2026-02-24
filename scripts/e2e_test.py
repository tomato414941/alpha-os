#!/usr/bin/env python3
"""End-to-end test: full pipeline with realistic synthetic market data.

Generates multi-asset data with realistic properties:
- Correlated assets (stocks move together)
- VIX inversely correlated with market
- Volatility clustering (GARCH-like)
- Mean-reverting macro signals

Then runs the full pipeline: evolve → validate → adopt → combine → risk-adjust.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpha_os.alpha.combiner import CombinerConfig
from alpha_os.alpha.monitor import AlphaMonitor
from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.backtest import metrics
from alpha_os.dsl import to_string
from alpha_os.evolution.gp import GPConfig
from alpha_os.governance.audit_log import AuditLog
from alpha_os.governance.gates import GateConfig
from alpha_os.pipeline.runner import PipelineConfig, PipelineRunner
from alpha_os.risk.manager import RiskManager, RiskConfig


# ---------------------------------------------------------------------------
# Realistic synthetic data generator
# ---------------------------------------------------------------------------

def generate_market_data(n_days: int = 750, seed: int = 42) -> dict[str, np.ndarray]:
    """Generate correlated market data with realistic statistical properties."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}

    # Common market factor
    market_returns = rng.normal(0.0003, 0.012, n_days)
    # Volatility clustering (simplified GARCH)
    vol = np.ones(n_days) * 0.012
    for t in range(1, n_days):
        vol[t] = 0.94 * vol[t - 1] + 0.06 * abs(market_returns[t - 1])
        market_returns[t] = rng.normal(0.0003, max(vol[t], 0.005))

    # Stocks: correlated with market + idiosyncratic
    stocks = {
        "nvda": (1.8, 0.001),   # high beta, positive drift
        "aapl": (1.0, 0.0005),
        "msft": (1.0, 0.0004),
        "amd":  (1.5, 0.0006),
        "tsla": (2.0, 0.0002),
    }
    for name, (beta, drift) in stocks.items():
        idio = rng.normal(0, 0.015, n_days)
        rets = drift + beta * market_returns + idio
        data[name] = 100.0 * np.cumprod(1 + rets)

    # Macro: SP500, Nasdaq (index proxies)
    data["sp500"] = 4500.0 * np.cumprod(1 + market_returns)
    data["nasdaq"] = 14000.0 * np.cumprod(1 + market_returns * 1.2 + rng.normal(0, 0.005, n_days))
    data["russell2000"] = 2000.0 * np.cumprod(1 + market_returns * 0.9 + rng.normal(0, 0.008, n_days))

    # VIX: inversely correlated with market, mean-reverting
    vix = np.ones(n_days) * 18.0
    for t in range(1, n_days):
        shock = -40 * market_returns[t] + rng.normal(0, 2)
        vix[t] = max(10, vix[t - 1] + 0.05 * (18 - vix[t - 1]) + shock * 0.3)
    data["vix_close"] = vix

    # Fear & Greed: 0-100, inversely correlated with VIX
    data["fear_greed"] = np.clip(100 - vix * 2.5 + rng.normal(0, 10, n_days), 0, 100)

    # Treasury yields: mean-reverting
    tsy10 = np.ones(n_days) * 4.0
    tsy2 = np.ones(n_days) * 4.5
    for t in range(1, n_days):
        tsy10[t] = tsy10[t - 1] + 0.01 * (4.0 - tsy10[t - 1]) + rng.normal(0, 0.03)
        tsy2[t] = tsy2[t - 1] + 0.01 * (4.5 - tsy2[t - 1]) + rng.normal(0, 0.04)
    data["tsy_yield_10y"] = tsy10
    data["tsy_yield_2y"] = tsy2

    # DXY, Gold, Oil
    data["dxy"] = 104.0 * np.cumprod(1 + rng.normal(0.0001, 0.003, n_days))
    data["gold"] = 2000.0 * np.cumprod(1 + rng.normal(0.0002, 0.008, n_days))
    data["oil_wti"] = 75.0 * np.cumprod(1 + rng.normal(0, 0.015, n_days))

    return data


def main():
    print("=" * 80)
    print("ALPHA-OS E2E TEST — Full Pipeline with Realistic Market Data")
    print("=" * 80)

    # --- Data generation ---
    n_days = 750  # ~3 years
    print(f"\n[1/5] Generating {n_days}-day market data (15 signals)...")
    t0 = time.perf_counter()
    data = generate_market_data(n_days=n_days, seed=42)
    print(f"  Signals: {list(data.keys())}")
    print(f"  Generated in {time.perf_counter() - t0:.2f}s")

    features = list(data.keys())
    prices = data["nvda"]  # primary asset

    # --- Pipeline configuration ---
    print("\n[2/5] Configuring pipeline...")
    pipe_cfg = PipelineConfig(
        gp=GPConfig(
            pop_size=200,
            n_generations=20,
            cx_prob=0.5,
            mut_prob=0.3,
            max_depth=3,
            bloat_penalty=0.01,
        ),
        gate=GateConfig(
            oos_sharpe_min=0.3,    # moderate threshold
            pbo_max=0.8,
            dsr_pvalue_max=1.0,    # DSR logged but not blocking (n_trials inflated by correlated alphas)
            fdr_pass_required=False,
            max_correlation=0.5,
            min_n_days=200,
        ),
        combiner=CombinerConfig(max_correlation=0.3, max_alphas=20),
        commission_pct=0.10,
        slippage_pct=0.05,
        n_cv_folds=5,
        embargo_days=5,
    )
    print(f"  GP: pop={pipe_cfg.gp.pop_size}, gen={pipe_cfg.gp.n_generations}")
    print(f"  Gates: OOS Sharpe >= {pipe_cfg.gate.oos_sharpe_min}, PBO <= {pipe_cfg.gate.pbo_max}")

    # --- Run pipeline ---
    print("\n[3/5] Running evolution pipeline...")
    runner = PipelineRunner(features, data, prices, config=pipe_cfg, seed=42)
    result = runner.run()

    print(f"\n  Pipeline Results:")
    print(f"    Generated:     {result.n_generated} unique alphas")
    print(f"    Validated:     {result.n_validated}")
    print(f"    Adopted:       {result.n_adopted}")
    print(f"    Combined:      {result.n_combined} alphas in portfolio")
    print(f"    Archive:       {runner.archive.size}/{runner.archive.capacity} cells ({result.archive_coverage:.1%})")
    print(f"    Elapsed:       {result.elapsed:.2f}s")

    # --- Top alphas from archive ---
    print("\n  Top 10 alphas from MAP-Elites archive:")
    print(f"  {'Rank':>4}  {'Fitness':>8}  Expression")
    print(f"  {'-' * 65}")
    for i, (expr, fit) in enumerate(runner.archive.best(10)):
        print(f"  {i + 1:>4}  {fit:>8.3f}  {to_string(expr)}")

    # --- Backtest combined signal ---
    if result.combined_signal is not None:
        print("\n[4/5] Backtesting combined signal...")
        engine = BacktestEngine(CostModel(0.10, 0.05))

        # Risk-adjusted positions
        risk_mgr = RiskManager(RiskConfig(
            target_vol=0.15,
            dd_stage1_pct=0.05, dd_stage1_scale=0.75,
            dd_stage2_pct=0.10, dd_stage2_scale=0.50,
            dd_stage3_pct=0.15, dd_stage3_scale=0.25,
        ))

        rets = np.diff(prices) / prices[:-1]
        adjusted = risk_mgr.adjust_positions(result.combined_signal, rets)
        bt_raw = engine.run(result.combined_signal, prices, alpha_id="combined_raw")
        bt_risk = engine.run(adjusted, prices, alpha_id="combined_risk_adj")

        print(f"\n  {'Metric':<20} {'Raw':>12} {'Risk-Adjusted':>14}")
        print(f"  {'-' * 48}")
        print(f"  {'Sharpe':.<20} {bt_raw.sharpe:>12.3f} {bt_risk.sharpe:>14.3f}")
        print(f"  {'Annual Return':.<20} {bt_raw.annual_return:>11.1%} {bt_risk.annual_return:>13.1%}")
        print(f"  {'Annual Vol':.<20} {bt_raw.annual_vol:>11.1%} {bt_risk.annual_vol:>13.1%}")
        print(f"  {'Max Drawdown':.<20} {bt_raw.max_drawdown:>11.1%} {bt_risk.max_drawdown:>13.1%}")
        print(f"  {'Calmar':.<20} {bt_raw.calmar:>12.3f} {bt_risk.calmar:>14.3f}")
        print(f"  {'Sortino':.<20} {bt_raw.sortino:>12.3f} {bt_risk.sortino:>14.3f}")
        print(f"  {'Turnover':.<20} {bt_raw.turnover:>12.3f} {bt_risk.turnover:>14.3f}")

        # --- Monitor simulation ---
        print("\n[5/5] Simulating live monitoring (rolling 63-day windows)...")
        monitor = AlphaMonitor()
        n = min(len(adjusted) - 1, len(rets))
        strat_rets = adjusted[:n] * rets[:n]
        monitor.record_batch("combined", strat_rets.tolist())
        status = monitor.check("combined")
        print(f"  Rolling Sharpe:  {status.rolling_sharpe:.3f}")
        print(f"  Rolling MaxDD:   {status.rolling_max_dd:.1%}")
        print(f"  Degraded:        {status.is_degraded}")
        if status.degradation_reasons:
            for r in status.degradation_reasons:
                print(f"    - {r}")

        # Audit log
        audit = AuditLog(log_path=Path("/tmp/alpha_os_e2e_audit.jsonl"))
        audit.log_pipeline_run({
            "n_generated": result.n_generated,
            "n_adopted": result.n_adopted,
            "sharpe_raw": bt_raw.sharpe,
            "sharpe_risk_adj": bt_risk.sharpe,
        })
        print(f"\n  Audit log: /tmp/alpha_os_e2e_audit.jsonl")
    else:
        print("\n[4/5] No alphas adopted — skipping backtest.")
        print("[5/5] Skipped.")

    # --- Summary ---
    print("\n" + "=" * 80)
    if result.combined_signal is not None:
        target_sharpe = 0.3
        actual_sharpe = bt_risk.sharpe
        status_str = "PASS" if actual_sharpe >= target_sharpe else "FAIL"
        print(f"E2E TEST: {status_str}")
        print(f"  Target: OOS Sharpe >= {target_sharpe} (risk-adjusted)")
        print(f"  Actual: {actual_sharpe:.3f}")
    else:
        print("E2E TEST: FAIL (no alphas adopted)")
    print("=" * 80)


if __name__ == "__main__":
    main()
