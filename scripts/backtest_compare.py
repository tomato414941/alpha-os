"""Compare raw vs residual fitness for DSL-backed capital-backed hypotheses."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

from alpha_os_recovery.backtest.benchmark import build_benchmark_returns
from alpha_os_recovery.backtest.cost_model import CostModel
from alpha_os_recovery.backtest.engine import BacktestEngine
from alpha_os_recovery.config import Config, HYPOTHESES_DB, SIGNAL_CACHE_DB
from alpha_os_recovery.data.signal_client import build_signal_client_from_config
from alpha_os_recovery.data.store import DataStore
from alpha_os_recovery.data.universe import build_feature_list
from alpha_os_recovery.dsl import parse
from alpha_os_recovery.dsl.evaluator import sanitize_signal
from alpha_os_recovery.hypotheses.store import HypothesisKind, HypothesisStore


def load_runtime_dsl_hypotheses(*, asset: str) -> list:
    store = HypothesisStore(HYPOTHESES_DB)
    try:
        records = store.list_capital_backed(asset=asset)
    finally:
        store.close()
    return [
        record
        for record in records
        if record.kind == HypothesisKind.DSL and record.expression
    ]


def main() -> None:
    cfg = Config.load(Path("/home/dev/.config/alpha-os/prod.toml"))
    client = build_signal_client_from_config(cfg.api)
    store = DataStore(SIGNAL_CACHE_DB, client)
    features = build_feature_list("BTC")
    matrix = store.get_matrix(features, end=date.today().isoformat())
    store.close()
    data = {col: matrix[col].values for col in matrix.columns}
    print(f"Data: {len(matrix)} rows, {len(data)} features")

    bm = build_benchmark_returns(data, cfg.backtest.benchmark_assets)
    print(f"Benchmark: {len(bm)} returns")

    records = load_runtime_dsl_hypotheses(asset="BTC")
    print(f"Capital-backed DSL hypotheses: {len(records)}")

    engine = BacktestEngine(
        CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct),
        allow_short=True,
    )

    raw_prices = data["btc_ohlcv"]
    finite_mask = np.isfinite(raw_prices)
    first_valid = int(np.argmax(finite_mask))
    prices = raw_prices[first_valid:]
    bm_trimmed = bm[first_valid:] if len(bm) > first_valid else bm
    print(f"BTC prices: {len(prices)} rows (from index {first_valid})")

    results = []
    skipped = 0
    for record in records:
        try:
            expr = parse(record.expression)
            sig = sanitize_signal(expr.evaluate(data))
            if sig.ndim == 0:
                sig = np.full(len(raw_prices), float(sig))
            sig = sig[first_valid:]
            raw = engine.run(sig, prices, alpha_id=record.hypothesis_id)
            res = engine.run(
                sig,
                prices,
                alpha_id=record.hypothesis_id,
                benchmark_returns=bm_trimmed,
            )
            results.append(
                (
                    record.hypothesis_id[:12],
                    record.expression[:55],
                    raw.sharpe,
                    res.sharpe,
                )
            )
        except Exception:
            skipped += 1

    print(f"Evaluable hypotheses: {len(results)} (skipped={skipped})")

    print("\n--- Top 20 by Residual Sharpe ---")
    print(f"  {'ID':>12}  {'Raw':>8}  {'Resid':>8}  {'Diff':>7}  Expr")
    print("  " + "-" * 95)
    for hypothesis_id, expr, raw_s, res_s in sorted(
        results,
        key=lambda row: row[3],
        reverse=True,
    )[:20]:
        print(
            f"  {hypothesis_id:>12}  {raw_s:>+8.3f}  "
            f"{res_s:>+8.3f}  {res_s - raw_s:>+7.3f}  {expr}"
        )

    print("\n--- Bottom 10 by Residual Sharpe ---")
    for hypothesis_id, expr, raw_s, res_s in sorted(results, key=lambda row: row[3])[:10]:
        print(
            f"  {hypothesis_id:>12}  {raw_s:>+8.3f}  "
            f"{res_s:>+8.3f}  {res_s - raw_s:>+7.3f}  {expr}"
        )

    raw_vals = [row[2] for row in results]
    res_vals = [row[3] for row in results]
    print(f"\nSummary ({len(results)} runtime hypotheses):")
    print(f"  Raw Sharpe:      mean={np.mean(raw_vals):+.3f}  std={np.std(raw_vals):.3f}")
    print(f"  Residual Sharpe: mean={np.mean(res_vals):+.3f}  std={np.std(res_vals):.3f}")
    print(f"  Positive raw:    {sum(1 for value in raw_vals if value > 0)}/{len(raw_vals)}")
    print(f"  Positive resid:  {sum(1 for value in res_vals if value > 0)}/{len(res_vals)}")


if __name__ == "__main__":
    main()
