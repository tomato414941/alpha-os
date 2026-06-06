# Results

## Initial Backtest

Data:

- `experiments/datasets/ds_crypto_btc_eth_daily_2024_2025/`
- BTCUSDT and ETHUSDT daily closes
- 2024-01-01 through 2025-12-31

Strategy:

- 7 day close-to-close momentum
- long if momentum is positive
- equal weight active symbols
- daily rebalance
- transaction cost rate: 0.001 per unit turnover

Result:

```text
steps=723
total_return=0.019927
annualized_return=0.010011
annualized_volatility=0.406290
sharpe=0.225859
max_drawdown=-0.496456
mean_daily_turnover=0.298755
```

Interpretation:

This is not good enough as a standalone strategy candidate. It is useful as a
baseline because it is simple, runs on real checked-in data, and exposes the
first full implementation-side path from data to backtest result.
