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
uv run python -m strategies.crypto.robustness
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

- `strategies/crypto/market_data/binance_spot_daily/`
- BTCUSDT and ETHUSDT daily closes fetched from Binance public market data
- 2024-01-01 through 2026-06-05

Command:

```text
uv run python -m strategies.crypto.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily
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

## Expanded Universe Check

Fresh data was expanded to:

```text
BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT
```

The data loader is availability-aware: each date uses the symbols available on
that date. `TONUSDT` starts on 2024-08-08, so it joins the universe from that
date rather than truncating the whole sample.

### 10 Symbols

Data range:

```text
2024-01-01 through 2026-06-05
```

Command:

```text
uv run python -m strategies.crypto.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT
```

Result:

```text
variant=7d_momentum
steps=879
total_return=-0.821665
sharpe=-0.938240
max_drawdown=-0.869059
mean_daily_turnover=0.543858

variant=7d_momentum_30d_trend
steps=856
total_return=-0.322751
sharpe=-0.024984
max_drawdown=-0.712941
mean_daily_turnover=0.458219

variant=7d_momentum_30d_trend_skfolio_max_ratio
steps=856
total_return=0.217615
sharpe=0.444806
max_drawdown=-0.704457
mean_daily_turnover=0.389673

variant=7d_momentum_30d_trend_skfolio_hrp
steps=856
total_return=-0.393749
sharpe=-0.158911
max_drawdown=-0.685957
mean_daily_turnover=0.485561

variant=7d_momentum_30d_trend_skfolio_min_variance
steps=856
total_return=-0.277097
sharpe=-0.043475
max_drawdown=-0.583201
mean_daily_turnover=0.485245
```

### 9 Symbols Without TON

Data intersection:

```text
2024-01-01 through 2026-06-05
```

Command:

```text
uv run python -m strategies.crypto.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT
```

Result:

```text
variant=7d_momentum
steps=879
total_return=-0.671292
sharpe=-0.540383
max_drawdown=-0.758650
mean_daily_turnover=0.522231

variant=7d_momentum_30d_trend
steps=856
total_return=-0.116927
sharpe=0.167221
max_drawdown=-0.645458
mean_daily_turnover=0.441408

variant=7d_momentum_30d_trend_skfolio_max_ratio
steps=856
total_return=0.665477
sharpe=0.660981
max_drawdown=-0.608666
mean_daily_turnover=0.374399
```

Robustness summary for the 9-symbol equal-weight variants with transaction cost
rate 0.001:

```text
sample  lookbacks  total_return  sharpe    max_drawdown  turnover
all     3/20      -0.240452      0.050125 -0.721345     0.577658
all     7/20      -0.470447     -0.221518 -0.746398     0.467946
all     7/30      -0.116927      0.167221 -0.645458     0.441408
all     14/30      0.224125      0.435924 -0.654288     0.360516
all     14/60     -0.499645     -0.316549 -0.642135     0.354881

2026    3/20      -0.201418     -1.539009 -0.201418     0.462557
2026    7/20      -0.238486     -1.949839 -0.238486     0.477672
2026    7/30      -0.089745     -0.644343 -0.187924     0.458089
2026    14/30     -0.130542     -1.019757 -0.231053     0.385289
2026    14/60     -0.121391     -1.473702 -0.219742     0.297536
```

Interpretation:

The wider universe exposes a weak point in the current equal-weight momentum
rule. It performs well in 2024 but fails badly in 2025 and 2026. The
availability-aware 10-symbol run no longer truncates the sample just because
`TONUSDT` starts late, but the conclusion is still similar: allocation matters
once the universe has more than BTC/ETH. The `skfolio` variant remains positive
over the full 10-symbol sample, but drawdown is still large. The next useful
work is not more lookback tuning; it is understanding which symbols drive the
losses and whether the strategy needs symbol selection, volatility control, or a
smaller eligible universe.

## Symbol Contribution Check

Command:

```text
uv run python -m strategies.crypto.contribution --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT AVAXUSDT TONUSDT --variant 7d_momentum_30d_trend
```

Equal-weight 7/30 gross contribution:

```text
symbol,total_gross_contribution,mean_weight,active_days,max_weight
ADAUSDT,-0.370563,0.035763,189,1.000000
BNBUSDT,-0.231841,0.101788,313,1.000000
TONUSDT,-0.181141,0.047334,123,1.000000
LINKUSDT,-0.041047,0.050095,230,1.000000
XRPUSDT,0.072666,0.062093,224,1.000000
BTCUSDT,0.131538,0.094039,323,1.000000
DOGEUSDT,0.154829,0.050683,241,1.000000
AVAXUSDT,0.194583,0.056630,240,1.000000
ETHUSDT,0.249768,0.061955,260,1.000000
SOLUSDT,0.381354,0.064620,263,1.000000
```

Skfolio max-ratio 7/30 gross contribution:

```text
symbol,total_gross_contribution,mean_weight,active_days,max_weight
ADAUSDT,-0.391299,0.012265,189,1.000000
BNBUSDT,-0.329539,0.116156,313,1.000000
TONUSDT,-0.312187,0.047678,123,1.000000
LINKUSDT,-0.016474,0.035028,230,1.000000
AVAXUSDT,0.077377,0.041496,240,1.000000
SOLUSDT,0.190347,0.065136,263,1.000000
BTCUSDT,0.212163,0.111722,323,1.000000
ETHUSDT,0.226535,0.069873,260,1.000000
DOGEUSDT,0.556496,0.052074,241,1.000000
XRPUSDT,0.767777,0.073572,224,1.000000
```

Interpretation:

The expanded-universe losses are not evenly distributed. `ADAUSDT`, `BNBUSDT`,
and `TONUSDT` are the clearest negative contributors in both allocator variants.
`SOLUSDT`, `ETHUSDT`, `DOGEUSDT`, and `XRPUSDT` contribute positively depending
on the allocator. The `max_weight=1.0` values also show that the strategy can
become fully concentrated in one active symbol on some days. The next useful
change is to test a smaller eligible universe and/or add a concentration limit.

## Pullback Candidate Check

Candidate:

```text
3 day return < 0
30 day return > 0
```

Interpretation:

This is a trend pullback rule: only buy symbols that are still in a longer
uptrend but have pulled back over the short window.

Checked-in BTC/ETH result:

```text
variant=3d_pullback_30d_trend
steps=700
total_return=-0.019583
annualized_return=-0.010259
annualized_volatility=0.286456
sharpe=0.106046
max_drawdown=-0.322667
mean_daily_turnover=0.277143

variant=3d_pullback_30d_trend_skfolio_max_ratio
steps=700
total_return=0.061262
annualized_return=0.031489
annualized_volatility=0.284309
sharpe=0.249835
max_drawdown=-0.313487
mean_daily_turnover=0.283410
```

Fresh 10-symbol result:

```text
variant=3d_pullback_30d_trend
steps=856
total_return=-0.646806
annualized_return=-0.358389
annualized_volatility=0.499052
sharpe=-0.636768
max_drawdown=-0.780534
mean_daily_turnover=0.605721

variant=3d_pullback_30d_trend_skfolio_max_ratio
steps=856
total_return=-0.502391
annualized_return=-0.257404
annualized_volatility=0.519849
sharpe=-0.312923
max_drawdown=-0.710213
mean_daily_turnover=0.633496
```

Interpretation:

The simple pullback candidate is weaker than the momentum candidate. It is not a
good next live or paper candidate in this form. The result suggests that
short-term dip buying in this universe needs more structure, such as a stronger
trend filter, a volatility filter, or symbol eligibility rules.

## Smaller Eligible Universe Check

Manual eligible universe:

```text
BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT AVAXUSDT LINKUSDT
```

This removes the clearest negative contributors from the first expanded
universe check:

```text
ADAUSDT BNBUSDT TONUSDT
```

Command:

```text
uv run python -m strategies.crypto.backtest --dataset-dir strategies/crypto/market_data/binance_spot_daily --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT DOGEUSDT AVAXUSDT LINKUSDT
```

Result:

```text
variant=7d_momentum
steps=879
total_return=-0.616537
annualized_return=-0.328350
annualized_volatility=0.572648
sharpe=-0.406631
max_drawdown=-0.738023
mean_daily_turnover=0.493228

variant=7d_momentum_30d_trend
steps=856
total_return=0.551830
annualized_return=0.206081
annualized_volatility=0.515689
sharpe=0.619942
max_drawdown=-0.466048
mean_daily_turnover=0.395110

variant=7d_momentum_30d_trend_skfolio_max_ratio
steps=856
total_return=1.928955
annualized_return=0.581274
annualized_volatility=0.585786
sharpe=1.069576
max_drawdown=-0.550211
mean_daily_turnover=0.327607

variant=7d_momentum_30d_trend_skfolio_hrp
steps=856
total_return=0.512035
annualized_return=0.192794
annualized_volatility=0.481396
sharpe=0.605613
max_drawdown=-0.432220
mean_daily_turnover=0.411131

variant=7d_momentum_30d_trend_skfolio_min_variance
steps=856
total_return=0.597901
annualized_return=0.221220
annualized_volatility=0.455648
sharpe=0.665000
max_drawdown=-0.440830
mean_daily_turnover=0.377942

variant=3d_pullback_30d_trend
steps=856
total_return=-0.418373
annualized_return=-0.206323
annualized_volatility=0.470323
sharpe=-0.255628
max_drawdown=-0.646598
mean_daily_turnover=0.513257

variant=3d_pullback_30d_trend_skfolio_max_ratio
steps=856
total_return=-0.297884
annualized_return=-0.139980
annualized_volatility=0.492057
sharpe=-0.062793
max_drawdown=-0.599741
mean_daily_turnover=0.522971
```

Robustness summary for equal-weight momentum variants with transaction cost
rate 0.001:

```text
sample  lookbacks  total_return  sharpe    max_drawdown  turnover
all     3/20        0.688544      0.681497 -0.550203     0.490317
all     7/20        0.176351      0.396046 -0.551893     0.411047
all     7/30        0.551830      0.619942 -0.466048     0.395110
all     14/30       1.697555      1.071502 -0.396063     0.305936
all     14/60      -0.307552     -0.054813 -0.568627     0.325130

2025    7/30       -0.105595     -0.024062 -0.313825     0.348717
2026    7/30        0.002737      0.184441 -0.170437     0.403390
```

Skfolio max-ratio 7/30 contribution:

```text
symbol,total_gross_contribution,mean_weight,active_days,max_weight
LINKUSDT,-0.158939,0.040259,230,1.000000
BTCUSDT,0.068385,0.141096,323,1.000000
AVAXUSDT,0.078558,0.057302,240,1.000000
ETHUSDT,0.224569,0.097864,260,1.000000
SOLUSDT,0.338456,0.075853,263,1.000000
XRPUSDT,0.551195,0.085754,224,1.000000
DOGEUSDT,0.647580,0.061451,241,1.000000
```

Interpretation:

Removing the worst contributors materially improves the momentum family. The
`7d_momentum_30d_trend_skfolio_max_ratio` variant is the strongest result so
far on fresh data. HRP is not stronger on return or Sharpe, but it has the
shallowest max drawdown in the smaller eligible universe. Minimum variance is
less aggressive than max-ratio and slightly stronger than equal-weight and HRP
on the smaller eligible universe, while reducing drawdown materially in the
10-symbol universe. The result is still not clean: drawdown remains large, and
the 2025/2026 equal-weight robustness rows are weak. The pullback variants
remain poor. This suggests the next useful work is to turn manual universe
selection into an explicit eligibility rule, or to add a concentration limit
before considering paper trading.
