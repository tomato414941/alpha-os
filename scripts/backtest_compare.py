"""Compare raw vs residual fitness for all deployed alphas."""
import numpy as np
from pathlib import Path
from alpha_os.config import Config, SIGNAL_CACHE_DB, asset_data_dir
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.store import DataStore
from alpha_os.data.universe import build_feature_list
from alpha_os.backtest.engine import BacktestEngine
from alpha_os.backtest.cost_model import CostModel
from alpha_os.backtest.benchmark import build_benchmark_returns
from alpha_os.dsl import parse
from alpha_os.dsl.evaluator import sanitize_signal
from alpha_os.legacy.managed_alphas import ManagedAlphaStore
from datetime import date

cfg = Config.load(Path("/home/dev/.config/alpha-os/prod.toml"))
client = build_signal_client_from_config(cfg.api)
store = DataStore(SIGNAL_CACHE_DB, client)
features = build_feature_list("BTC")
matrix = store.get_matrix(features, end=date.today().isoformat())
data = {col: matrix[col].values for col in matrix.columns}
print(f"Data: {len(matrix)} rows, {len(data)} features")

bm = build_benchmark_returns(data, cfg.backtest.benchmark_assets)
print(f"Benchmark: {len(bm)} returns")

adir = asset_data_dir("BTC")
reg = ManagedAlphaStore(db_path=adir / "alpha_registry.db")
deployed_ids = reg.deployed_alpha_ids()
print(f"Deployed: {len(deployed_ids)}")

engine = BacktestEngine(
    CostModel(cfg.backtest.commission_pct, cfg.backtest.slippage_pct),
    allow_short=True,
)

raw_prices = data["btc_ohlcv"]
# Trim to finite price range only
finite_mask = np.isfinite(raw_prices)
first_valid = np.argmax(finite_mask)
prices = raw_prices[first_valid:]
bm_trimmed = bm[first_valid:] if len(bm) > first_valid else bm
print(f"BTC prices: {len(prices)} rows (from index {first_valid})")

results = []
for aid in deployed_ids:
    record = reg.get(aid)
    if not record:
        continue
    try:
        expr = parse(record.expression)
        sig = sanitize_signal(expr.evaluate(data))
        if sig.ndim == 0:
            sig = np.full(len(raw_prices), float(sig))
        sig = sig[first_valid:]  # trim to match prices
        raw = engine.run(sig, prices, alpha_id=aid)
        res = engine.run(sig, prices, alpha_id=aid, benchmark_returns=bm_trimmed)
        results.append((aid[:12], record.expression[:55], raw.sharpe, res.sharpe))
    except Exception:
        continue

reg.close()

print("\n--- Top 20 by Residual Sharpe ---")
print(f"  {'ID':>12}  {'Raw':>8}  {'Resid':>8}  {'Diff':>7}  Expr")
print("  " + "-" * 95)
for aid, expr, raw_s, res_s in sorted(results, key=lambda x: x[3], reverse=True)[:20]:
    print(f"  {aid:>12}  {raw_s:>+8.3f}  {res_s:>+8.3f}  {res_s - raw_s:>+7.3f}  {expr}")

print("\n--- Bottom 10 by Residual Sharpe ---")
for aid, expr, raw_s, res_s in sorted(results, key=lambda x: x[3])[:10]:
    print(f"  {aid:>12}  {raw_s:>+8.3f}  {res_s:>+8.3f}  {res_s - raw_s:>+7.3f}  {expr}")

raw_vals = [r[2] for r in results]
res_vals = [r[3] for r in results]
print(f"\nSummary ({len(results)} deployed alphas):")
print(f"  Raw Sharpe:      mean={np.mean(raw_vals):+.3f}  std={np.std(raw_vals):.3f}")
print(f"  Residual Sharpe: mean={np.mean(res_vals):+.3f}  std={np.std(res_vals):.3f}")
print(f"  Positive raw:    {sum(1 for v in raw_vals if v > 0)}/{len(raw_vals)}")
print(f"  Positive resid:  {sum(1 for v in res_vals if v > 0)}/{len(res_vals)}")
