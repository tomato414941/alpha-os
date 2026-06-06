# Crypto Momentum

This is the first concrete profit-seeking strategy candidate in this repository.

## Hypothesis

BTC/ETH daily close-to-close momentum can produce useful long-or-cash strategy
variants.

## Data

Uses the checked-in historical dataset:

```text
experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/
```

The current strategy uses only `timestamp` and `close`.

## Current Variant

- Compute each symbol's 7 day and 30 day close-to-close returns.
- Hold long if both returns are positive.
- Hold cash for symbols that do not pass both filters.
- Equal weight across active symbols.
- Rebalance daily.

## Local Files

- `data.py`: local CSV loading
- `strategy.py`: concrete strategy and strategy-specific input/output shapes
- `backtest.py`: local historical backtest path
- `latest_target.py`: latest target weights from the available data
- `results.md`: latest result notes
- `paper_log.md`: manual paper decision notes

Shared code should stay local until another strategy needs the same shape.
