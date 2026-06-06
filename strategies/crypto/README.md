# Crypto

This directory contains concrete profit-seeking strategy candidates for crypto
markets.

## Current Hypothesis

BTC/ETH daily close-to-close momentum can produce useful long-or-cash strategy
variants.

## Data

Uses the checked-in historical dataset:

```text
experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/
```

The current strategy uses only `timestamp` and `close`.

To fetch fresh local Binance spot daily data:

```text
uv run python -m strategies.crypto.fetch_market_data
```

The default fresh-data universe is:

```text
BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT
```

That writes uncommitted local CSV files under:

```text
strategies/crypto/market_data/binance_spot_daily/
```

To use those files for the current manual target:

```text
uv run python -m strategies.crypto.latest_target --dataset-dir strategies/crypto/market_data/binance_spot_daily
```

For the expanded universe:

```text
uv run python -m strategies.crypto.latest_target --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT
```

To backtest those files:

```text
uv run python -m strategies.crypto.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily
```

For the expanded universe:

```text
uv run python -m strategies.crypto.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT
```

To run the local robustness check on those files:

```text
uv run python -m strategies.crypto.robustness --dataset-dir strategies/crypto/market_data/binance_spot_daily
```

For the expanded universe:

```text
uv run python -m strategies.crypto.robustness --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT
```

To inspect symbol-level gross contribution:

```text
uv run python -m strategies.crypto.contribution --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT --variant 7d_momentum_30d_trend
```

## Current Variant

- Compute each symbol's 7 day and 30 day close-to-close returns.
- Hold long if both returns are positive.
- Hold cash for symbols that do not pass both filters.
- Equal weight across active symbols.
- Rebalance daily.

## Local Files

- `data.py`: local CSV loading
- `fetch_market_data.py`: local Binance spot daily data fetch
- `allocation.py`: local portfolio allocators used inside strategy variants
- `momentum.py`: momentum strategy and strategy-specific input/output shapes
- `variants.py`: maintained local variant registry
- `backtest.py`: local historical backtest path
- `contribution.py`: symbol-level gross contribution analysis
- `latest_target.py`: latest target weights from the available data
- `results.md`: latest result notes
- `paper_log.md`: manual paper decision notes

Shared code should stay local until another strategy needs the same shape.
