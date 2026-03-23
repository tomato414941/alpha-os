"""Diagnose universe coverage: what data do we have vs what's missing."""
import numpy as np
from pathlib import Path
from datetime import date
from alpha_os.config import Config, SIGNAL_CACHE_DB
from alpha_os.data.store import DataStore
from alpha_os.data.signal_client import build_signal_client_from_config
from alpha_os.data.universe import build_feature_list, CROSS_ASSET_UNIVERSE

cfg = Config.load(Path("/home/dev/.config/alpha-os/prod.toml"))
client = build_signal_client_from_config(cfg.api)
store = DataStore(SIGNAL_CACHE_DB, client)
features = build_feature_list("BTC")
matrix = store.get_matrix(features, end=date.today().isoformat())
data = {col: matrix[col].values for col in matrix.columns}

print(f"CROSS_ASSET_UNIVERSE: {len(CROSS_ASSET_UNIVERSE)} entries")
print(f"Data matrix: {len(matrix)} rows, {len(data)} features")

# Check key non-stock assets
key_assets = [
    "btc_ohlcv", "eth_btc", "sol_usdt", "bnb_usdt",
    "sp500", "nasdaq", "gold", "oil_wti", "tlt", "dxy",
    "russell2000", "silver", "eur_usd", "vix_close",
    "tsy_yield_10y", "fear_greed",
]
print("\nKey assets:")
for a in key_assets:
    in_univ = a in CROSS_ASSET_UNIVERSE
    in_data = a in data
    n = int(np.isfinite(data[a]).sum()) if in_data else 0
    status = "OK" if in_univ and n >= 500 else "MISSING" if not in_univ else f"SHORT({n}d)"
    print(f"  {a:20s}  universe={str(in_univ):5s}  days={n:5d}  {status}")

# Universe coverage stats
in_data_count = 0
has_500 = 0
missing = []
short = []
for sig in CROSS_ASSET_UNIVERSE:
    arr = data.get(sig)
    if arr is None:
        missing.append(sig)
        continue
    in_data_count += 1
    n = int(np.isfinite(arr).sum())
    if n >= 500:
        has_500 += 1
    else:
        short.append((sig, n))

print("\nUniverse coverage:")
print(f"  Total:      {len(CROSS_ASSET_UNIVERSE)}")
print(f"  In data:    {in_data_count}")
print(f"  Missing:    {len(missing)}")
print(f"  >= 500d:    {has_500}")
print(f"  < 500d:     {len(short)}")

# What's missing?
if missing:
    # Categorize missing
    crypto_m = [m for m in missing if m.startswith("yf_crypto_")]
    etf_m = [m for m in missing if m.startswith("etf_")]
    stock_m = [m for m in missing if not m.startswith("yf_crypto_") and not m.startswith("etf_")]
    print("\nMissing breakdown:")
    print(f"  Crypto: {len(crypto_m)} (Yahoo format, may need different ticker)")
    print(f"  ETFs:   {len(etf_m)}")
    print(f"  Stocks: {len(stock_m)}")
    if crypto_m[:5]:
        print(f"    crypto examples: {crypto_m[:5]}")
    if etf_m[:5]:
        print(f"    etf examples: {etf_m[:5]}")

# Data we HAVE but NOT in CROSS_ASSET_UNIVERSE
in_data_not_universe = []
for col in data.keys():
    n = int(np.isfinite(data[col]).sum())
    if n >= 500 and col not in CROSS_ASSET_UNIVERSE:
        in_data_not_universe.append((col, n))
in_data_not_universe.sort(key=lambda x: -x[1])

print(f"\nSignals with 500+ days NOT in CROSS_ASSET_UNIVERSE: {len(in_data_not_universe)}")
# Show important ones
important = [
    (name, days) for name, days in in_data_not_universe
    if any(name.startswith(p) for p in ["btc_", "eth_", "sol_", "sp", "gold", "oil",
           "tlt", "dxy", "nasdaq", "russell", "silver", "vix", "tsy", "fear"])
    or name in ["btc_ohlcv", "sp500", "gold", "oil_wti", "dxy"]
]
if important:
    print("  Important ones missing from universe:")
    for name, days in important[:20]:
        print(f"    {name:30s}: {days:5d} days")
