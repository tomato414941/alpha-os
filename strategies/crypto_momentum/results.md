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

## Fresh Binance Spot Data Check

Data:

- `strategies/crypto_momentum/market_data/binance_spot_daily/`
- BTCUSDT and ETHUSDT daily closes fetched from Binance public market data
- 2024-01-01 through 2026-06-05

Command:

```text
uv run python -m strategies.crypto_momentum.backtest --dataset-dir strategies/crypto_momentum/market_data/binance_spot_daily
```

Result:

```text
variant=7d_momentum
steps=879
total_return=-0.154176
annualized_return=-0.067168
annualized_volatility=0.395246
sharpe=0.020135
max_drawdown=-0.502731
mean_daily_turnover=0.296928

variant=7d_momentum_30d_trend
steps=856
total_return=0.579415
annualized_return=0.215176
annualized_volatility=0.335170
sharpe=0.746537
max_drawdown=-0.434133
mean_daily_turnover=0.216121
```

Robustness summary with transaction cost rate 0.001:

```text
sample  lookbacks  total_return  sharpe   max_drawdown  turnover
all     3/20       0.603932      0.770875 -0.361147     0.288684
all     7/20       0.196199      0.389398 -0.470330     0.229792
all     7/30       0.579415      0.746537 -0.434133     0.216121
all     14/30      1.281776      1.153371 -0.339487     0.161215
all     14/60      0.284077      0.498102 -0.372864     0.150121

2026    3/20      -0.082531     -0.644363 -0.163481     0.266667
2026    7/20      -0.111941     -0.946391 -0.154052     0.244444
2026    7/30      -0.033113     -0.208130 -0.139447     0.208000
2026    14/30      0.001914      0.163385 -0.140391     0.176000
2026    14/60      0.016661      0.383470 -0.068731     0.115789
```

Interpretation:

The current variant remains profitable when the sample is extended into 2026,
but the result weakens materially. The simple 7 day momentum variant turns
negative on the fresh spot data. The 14/30 pair is strongest over the full
sample, but its 2026-only result is close to flat after turnover cost. This
suggests the family is still worth investigating, but not yet strong enough for
serious live capital. The next question should be universe breadth, not more
fine-tuning on BTC/ETH.

## Skfolio Allocator Check

The `skfolio` allocator is used inside the trading strategy as a portfolio
allocator. It does not change the external `TradingStrategy` boundary.

Compared variants:

- `7d_momentum_30d_trend`
  - active symbols get equal weights
- `7d_momentum_30d_trend_skfolio_max_ratio`
  - the same active symbols are passed to `skfolio.optimization.MeanRisk`
  - objective: maximize ratio
  - long-only, fully invested among active symbols

Checked-in dataset result:

```text
variant=7d_momentum_30d_trend
steps=700
total_return=0.855439
annualized_return=0.380307
annualized_volatility=0.341363
sharpe=1.112134
max_drawdown=-0.295274
mean_daily_turnover=0.215714

variant=7d_momentum_30d_trend_skfolio_max_ratio
steps=700
total_return=0.669436
annualized_return=0.306334
annualized_volatility=0.321048
sharpe=0.991404
max_drawdown=-0.298318
mean_daily_turnover=0.193678
```

Fresh Binance spot data result:

```text
variant=7d_momentum_30d_trend
steps=856
total_return=0.579415
annualized_return=0.215176
annualized_volatility=0.335170
sharpe=0.746537
max_drawdown=-0.434133
mean_daily_turnover=0.216121

variant=7d_momentum_30d_trend_skfolio_max_ratio
steps=856
total_return=0.387392
annualized_return=0.149831
annualized_volatility=0.318201
sharpe=0.596450
max_drawdown=-0.478032
mean_daily_turnover=0.194265
```

Interpretation:

On the current BTC/ETH universe, this `skfolio` allocator reduces turnover and
volatility slightly, but it does not improve return, Sharpe, or drawdown. If
`skfolio` helped in earlier experiments, the likely reason was a broader
universe or a different allocation objective, not this two-asset BTC/ETH setup.
The next useful check is to expand the universe before judging whether
optimizer-backed allocation is useful.
