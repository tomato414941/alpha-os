# Crypto Momentum Baseline

This is the first concrete profit-seeking strategy candidate in this repository.

## Hypothesis

BTC/ETH daily close-to-close momentum can produce a useful long-or-cash baseline.

## Data

Uses the checked-in historical dataset:

```text
experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/
```

The current strategy uses only `timestamp` and `close`.

## Strategy

- Compute each symbol's 7 day close-to-close return.
- Hold long if the 7 day return is positive.
- Hold cash for symbols whose 7 day return is not positive.
- Equal weight across active symbols.
- Rebalance daily.

## Local Files

- `data.py`: local CSV loading
- `strategy.py`: concrete strategy and strategy-specific input/output shapes
- `backtest.py`: local historical backtest path
- `results.md`: latest result notes

Shared code should stay local until another strategy needs the same shape.
