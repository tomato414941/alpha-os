# Cross-Asset Rotation

Cross-asset daily close strategy candidates.

Universe:

```text
SPY QQQ GLD TLT BTCUSDT ETHUSDT
```

Prepare local data from the existing ETF and crypto local datasets:

```text
uv run python -m strategies.cross_asset_rotation.prepare_market_data
```

Run:

```text
uv run python -m strategies.cross_asset_rotation.backtest
```
