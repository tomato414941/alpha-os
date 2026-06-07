# Crypto Market Structure Results

Data:

- source: Binance public data archive
- market: USD-M futures
- symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT
- period: 2024-01-01 to 2026-06-06, excluding unavailable current-month rows

## Feature Diagnostics

Pooled next-day returns by cross-sectional feature bucket:

| feature | bucket | count | mean next return | hit rate |
| --- | ---: | ---: | ---: | ---: |
| funding_rate_sum | bottom_20 | 1586 | 0.001281 | 0.500 |
| funding_rate_sum | middle_60 | 4214 | -0.000369 | 0.481 |
| funding_rate_sum | top_20 | 2129 | 0.002311 | 0.504 |
| premium_close | bottom_20 | 1586 | 0.000202 | 0.515 |
| premium_close | middle_60 | 4757 | 0.000074 | 0.478 |
| premium_close | top_20 | 1586 | 0.002979 | 0.508 |
| taker_buy_imbalance | bottom_20 | 1586 | -0.002896 | 0.482 |
| taker_buy_imbalance | middle_60 | 4757 | 0.001116 | 0.489 |
| taker_buy_imbalance | top_20 | 1586 | 0.002950 | 0.507 |
| volume_ratio_20d | bottom_20 | 1586 | -0.001284 | 0.492 |
| volume_ratio_20d | middle_60 | 4757 | 0.001009 | 0.490 |
| volume_ratio_20d | top_20 | 1586 | 0.001660 | 0.493 |

The raw buckets are directionally interesting, but hit rates are weak.

## Candidate Backtests

Transaction cost: 0.1% per one-way weight turnover.

| candidate | steps | total return | sharpe | max drawdown | mean daily turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| funding_premium_flow_top_2 | 861 | -0.712851 | -0.369397 | -0.848676 | 1.250871 |
| funding_premium_flow_top_2_weekly | 861 | -0.640398 | -0.241084 | -0.842787 | 0.205575 |
| flow_top_3 | 861 | -0.410303 | 0.018857 | -0.789391 | 1.092915 |
| flow_top_3_weekly | 861 | -0.297320 | 0.131856 | -0.720282 | 0.197058 |
| premium_funding_top_2 | 861 | -0.766845 | -0.441645 | -0.854244 | 1.152149 |
| premium_funding_top_2_weekly | 861 | -0.584926 | -0.188036 | -0.809807 | 0.196283 |

Same-window buy-and-hold benchmark:

| symbol | total return | sharpe | max drawdown |
| --- | ---: | ---: | ---: |
| XRPUSDT | 1.438609 | 0.858375 | -0.658458 |
| BNBUSDT | 1.232687 | 0.895436 | -0.553580 |
| BTCUSDT | 0.771361 | 0.744807 | -0.495556 |
| DOGEUSDT | 0.174447 | 0.537037 | -0.810576 |
| SOLUSDT | -0.095659 | 0.351779 | -0.702673 |
| ETHUSDT | -0.183438 | 0.220393 | -0.638211 |
| LINKUSDT | -0.406144 | 0.177867 | -0.729774 |
| ADAUSDT | -0.532181 | 0.103347 | -0.810547 |
| AVAXUSDT | -0.723981 | -0.164567 | -0.863599 |

## Interpretation

The first market-structure screen is broader than close-only momentum, but the
simple cross-sectional ranker is not tradeable. Weekly rebalancing reduces
turnover but does not make the candidate competitive with buy-and-hold.

## Predictive Screen

The next screen treats market-structure data as a supervised learning problem:

- input: lagged returns, funding, premium, taker buy imbalance, and volume ratio
- label: next-day return
- model: expanding-window ridge regression
- policy: hold top positive predictions, with daily or weekly rebalance

| candidate | predictions | mean daily rank IC | positive prediction hit rate | total return | sharpe | max drawdown | mean daily turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge_top_1_1d | 6129 | -0.012398 | 0.482 | -0.728203 | -0.452947 | -0.929201 | 1.455213 |
| ridge_top_1_7d | 6129 | -0.012398 | 0.482 | 0.923917 | 0.824879 | -0.694861 | 0.236417 |
| ridge_top_2_1d | 6129 | -0.012398 | 0.482 | -0.684378 | -0.485556 | -0.888755 | 1.267254 |
| ridge_top_2_7d | 6129 | -0.012398 | 0.482 | 0.284916 | 0.542849 | -0.638826 | 0.212922 |
| ridge_top_3_1d | 6129 | -0.012398 | 0.482 | -0.616890 | -0.349616 | -0.821147 | 1.131180 |
| ridge_top_3_7d | 6129 | -0.012398 | 0.482 | -0.106839 | 0.256388 | -0.674389 | 0.194322 |

Same-window buy-and-hold from the predictive-screen start:

| symbol | total return | sharpe | max drawdown |
| --- | ---: | ---: | ---: |
| XRPUSDT | 1.326235 | 0.942164 | -0.658458 |
| BNBUSDT | 0.197645 | 0.444733 | -0.553580 |
| BTCUSDT | 0.105441 | 0.346262 | -0.495556 |
| DOGEUSDT | -0.199585 | 0.318993 | -0.810576 |
| LINKUSDT | -0.350224 | 0.187904 | -0.729774 |
| ADAUSDT | -0.463309 | 0.128552 | -0.810547 |
| ETHUSDT | -0.427125 | -0.066371 | -0.632503 |
| SOLUSDT | -0.512670 | -0.087009 | -0.702673 |
| AVAXUSDT | -0.680972 | -0.244222 | -0.846753 |

This is the first result that is meaningfully better than BTC/BNB same-window
buy-and-hold, but it still loses to XRP and has a large drawdown. The weak rank
IC and hit rate mean this is not yet a robust edge. It is a better research
direction than hand-written z-score sums, not a tradeable strategy.

Next useful work is not another small parameter tweak. It should move to a
different model shape:

- learn feature interactions instead of hand-summing z-scores
- separate prediction quality from policy/backtest quality
- test intraday order-flow features where data volume is enough
- add benchmark-aware candidate rejection early
