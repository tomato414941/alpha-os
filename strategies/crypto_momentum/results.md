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

## Candidate: 7 Day Momentum + 30 Day Trend

Strategy:

- 7 day close-to-close momentum must be positive.
- 30 day close-to-close trend must also be positive.
- Equal weight active symbols.
- Hold cash when no symbol passes both filters.
- Daily rebalance.
- transaction cost rate: 0.001 per unit turnover

Result:

```text
steps=700
total_return=0.855439
annualized_return=0.380307
annualized_volatility=0.341363
sharpe=1.112134
max_drawdown=-0.295274
mean_daily_turnover=0.215714
```

Interpretation:

This candidate materially improves the baseline in this first check. It has
higher return, lower volatility, lower drawdown, higher Sharpe, and lower
turnover. It still needs robustness checks before being treated as a serious
live or paper candidate.
