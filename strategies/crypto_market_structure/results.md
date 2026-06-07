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

## Predictive Exposure Audit

Audit target: `ridge_top_1_7d`.

| symbol | mean weight | held days | gross contribution | mean return when held |
| --- | ---: | ---: | ---: | ---: |
| BNBUSDT | 0.185022 | 126 | -0.150899 | -0.001198 |
| BTCUSDT | 0.164464 | 112 | -0.085470 | -0.000763 |
| LINKUSDT | 0.133627 | 91 | 0.463938 | 0.005098 |
| ADAUSDT | 0.123348 | 84 | 0.694624 | 0.008269 |
| XRPUSDT | 0.092511 | 63 | 1.077518 | 0.017103 |
| DOGEUSDT | 0.082232 | 56 | -0.013912 | -0.000248 |
| SOLUSDT | 0.071953 | 49 | 0.142959 | 0.002918 |
| AVAXUSDT | 0.071953 | 49 | -0.363769 | -0.007424 |
| ETHUSDT | 0.064611 | 44 | -0.298622 | -0.006787 |

The weekly ridge result is not just an XRP exposure artifact. XRP is only the
fifth-largest average exposure. The result comes from intermittent correct
selection of XRP, ADA, and LINK while also carrying negative BNB/BTC/AVAX/ETH
episodes. This supports model-shape exploration, but it also shows that the
current policy is still noisy and drawdown-prone.

## Broad Model Screen

This screen expands the search space without introducing heavy dependencies:

- feature sets: all, momentum, structure, flow, funding/premium
- model labels: return ridge, sign ridge, cross-sectional rank ridge, contrarian return ridge
- portfolio rules: top 1/2/3 positive scores, rebalance every 1/3/7/14 days

Top candidates by Sharpe:

| candidate | predictions | mean daily rank IC | hit rate | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| funding_premium_contrarian_return_ridge_top_1_14d | 5499 | 0.040022 | 0.506 | 3.317314 | 1.488873 | -0.541209 | 0.096563 |
| flow_return_ridge_top_1_7d | 5499 | -0.011401 | 0.485 | 3.839812 | 1.480969 | -0.593317 | 0.240589 |
| flow_return_ridge_top_1_14d | 5499 | -0.011401 | 0.485 | 2.272810 | 1.237376 | -0.639185 | 0.130933 |
| funding_premium_contrarian_return_ridge_top_2_14d | 5499 | 0.040022 | 0.506 | 1.743855 | 1.192705 | -0.502153 | 0.086743 |
| flow_return_ridge_top_2_7d | 5499 | -0.011401 | 0.485 | 1.726964 | 1.157009 | -0.542114 | 0.207856 |
| funding_premium_contrarian_return_ridge_top_3_14d | 5499 | 0.040022 | 0.506 | 1.413867 | 1.106731 | -0.501502 | 0.088380 |
| flow_sign_ridge_top_1_7d | 5499 | -0.002904 | 0.471 | 1.421618 | 1.060776 | -0.540737 | 0.240589 |
| all_sign_ridge_top_1_7d | 5499 | -0.001821 | 0.481 | 1.362449 | 1.028480 | -0.580269 | 0.227496 |
| structure_sign_ridge_top_3_7d | 5499 | -0.020187 | 0.471 | 1.308083 | 1.027657 | -0.591124 | 0.175668 |
| funding_premium_rank_ridge_top_2_14d | 5499 | 0.034844 | 0.487 | 1.179571 | 1.000151 | -0.583573 | 0.093290 |

Same-window buy-and-hold from the broad-screen start:

| symbol | total return | sharpe | max drawdown |
| --- | ---: | ---: | ---: |
| XRPUSDT | 1.264571 | 0.980934 | -0.658458 |
| BNBUSDT | 0.170550 | 0.439633 | -0.553580 |
| BTCUSDT | 0.120208 | 0.375440 | -0.495556 |
| DOGEUSDT | -0.187075 | 0.324394 | -0.810576 |
| LINKUSDT | -0.278203 | 0.236316 | -0.729774 |
| ADAUSDT | -0.413742 | 0.165600 | -0.810547 |
| ETHUSDT | -0.255137 | 0.107280 | -0.632503 |
| SOLUSDT | -0.477191 | -0.088024 | -0.702673 |
| AVAXUSDT | -0.704838 | -0.375462 | -0.846753 |

This is the first broad screen where several candidates beat the strongest
same-window single-asset benchmark. The strongest cluster is not "more complex
ML"; it is a slower rebalance around funding/premium contrarian behavior and
flow-based selection. The next step should stress-test this cluster across
start dates, costs, refit cadence, and symbol subsets.

Next useful work is not another small parameter tweak. It should move to a
different model shape:

- learn feature interactions instead of hand-summing z-scores
- separate prediction quality from policy/backtest quality
- test intraday order-flow features where data volume is enough
- add benchmark-aware candidate rejection early
