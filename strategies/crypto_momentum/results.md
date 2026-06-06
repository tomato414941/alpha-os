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

## Robustness Check: Momentum + Trend Lookbacks

Command:

```text
uv run python -m strategies.crypto_momentum.robustness
```

Summary with transaction cost rate 0.001:

```text
sample  lookbacks  total_return  sharpe   max_drawdown  turnover
all     3/20       0.891493      1.133090 -0.332566     0.295775
all     7/20       0.551801      0.811778 -0.404886     0.223944
all     7/30       0.855439      1.112134 -0.295274     0.215714
all     14/30      1.110383      1.226070 -0.334587     0.154286
all     14/60      0.250844      0.519533 -0.377171     0.168657

2024    3/20       0.598795      1.678855 -0.201809     0.318841
2024    7/20       0.387250      1.126295 -0.404886     0.228986
2024    7/30       0.743533      1.833679 -0.220765     0.220896
2024    14/30      0.619157      1.599208 -0.334587     0.152239
2024    14/60      0.086591      0.458921 -0.377171     0.193443

2025    3/20       0.220978      0.772964 -0.226829     0.276163
2025    7/20       0.107297      0.483548 -0.292832     0.223837
2025    7/30       0.170200      0.696910 -0.295274     0.194611
2025    14/30      0.467614      1.279074 -0.186407     0.146707
2025    14/60      0.160151      0.663382 -0.248835     0.134868
```

Interpretation:

The trend filter is not a single-parameter accident in this checked-in dataset.
Several neighboring lookback pairs remain profitable after a simple turnover
cost. The result is still stronger in 2024 than 2025, so this should be treated
as a useful candidate family rather than a proven live strategy. The next useful
question is whether the same idea survives a wider universe or a newer data
source.
