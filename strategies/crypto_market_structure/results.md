# Crypto Market Structure Results

Data:

- source: Binance public data archive
- market: USD-M futures
- symbols: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, ADAUSDT, DOGEUSDT, LINKUSDT, AVAXUSDT
- period: 2024-01-01 to 2026-06-06, excluding unavailable current-month rows

Note: the latest broad-universe rerun uses the expanded spot/perp common
universe summarized at the end of this file. Earlier diagnostic sections are
kept as historical screens.

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

## Funding Carry

This lane is closer to a real profit source than directional prediction:

- decision: hold symbols with positive funding
- intended trade approximation: long spot / short perpetual
- reward approximation: funding received minus premium change and turnover cost
- default one-way turnover cost: 0.04%

Top default candidates:

| candidate | steps | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| positive_funding_carry_top_3_14d | 881 | 0.159194 | 7.277638 | -0.002795 | 0.077563 |
| positive_funding_carry_top_2_14d | 881 | 0.156255 | 6.906684 | -0.002388 | 0.085131 |
| positive_funding_carry_top_3_7d | 881 | 0.136744 | 6.225933 | -0.006029 | 0.139992 |
| positive_funding_carry_top_1_14d | 881 | 0.153267 | 5.805652 | -0.006250 | 0.088536 |
| positive_funding_carry_top_2_7d | 881 | 0.128435 | 5.620573 | -0.005422 | 0.156640 |

Cost stress, best candidate by Sharpe:

| one-way turnover cost | candidate | total return | sharpe | max drawdown |
| ---: | --- | ---: | ---: | ---: |
| 0.04% | positive_funding_carry_top_3_14d | 0.159194 | 7.277638 | -0.002795 |
| 0.10% | positive_funding_carry_top_3_14d | 0.112597 | 4.435114 | -0.008086 |
| 0.20% | positive_funding_carry_top_3_14d | 0.038980 | 1.110018 | -0.036655 |
| 0.50% | positive_funding_carry_top_3_14d | -0.154393 | -2.216909 | -0.188430 |

This is not a finished live strategy. The approximation still omits spot/perp
venue separation, borrow and margin treatment, liquidation constraints, order
book depth, failed execution, and exchange-specific fee schedules. Still, it is
more profit-adjacent than another directional predictor because it tests a
specific trade construction rather than only predicting price direction.

## Spot/Perp Carry

This version uses separate spot and perpetual closes instead of premium-index
change as the basis proxy:

- pair PnL: spot return - perp return + funding received
- capital assumption: `capital_per_notional = 2.0`
- paired-leg cost: applied to both spot and perp legs on turnover

Top default candidates:

| candidate | steps | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| spot_perp_positive_funding_top_3_14d | 881 | 0.057046 | 3.038431 | -0.015078 | 0.115399 |
| spot_perp_positive_funding_top_2_14d | 881 | 0.060508 | 2.922381 | -0.016324 | 0.119183 |
| spot_perp_positive_funding_top_1_14d | 881 | 0.065458 | 2.505911 | -0.012568 | 0.121453 |
| spot_perp_positive_funding_top_2_7d | 881 | 0.028031 | 1.318138 | -0.028262 | 0.239501 |
| spot_perp_positive_funding_top_3_7d | 881 | 0.025169 | 1.279093 | -0.027635 | 0.236095 |

Cost stress, best candidate by Sharpe:

| paired-leg cost | candidate | total return | sharpe | max drawdown |
| ---: | --- | ---: | ---: | ---: |
| 0.04% | spot_perp_positive_funding_top_3_14d | 0.057046 | 3.038431 | -0.015078 |
| 0.10% | spot_perp_positive_funding_top_1_14d | -0.000872 | -0.020601 | -0.045647 |
| 0.20% | spot_perp_positive_funding_top_1_14d | -0.102520 | -2.196145 | -0.130495 |
| 0.50% | spot_perp_positive_funding_top_1_14d | -0.350327 | -3.897055 | -0.349762 |

The broader 30-symbol universe is a useful correction against the earlier
9-symbol result. The edge still exists, but it is weaker: Sharpe falls, drawdown
increases, and the trade becomes more sensitive to execution cost.

Next useful work should validate whether this carry survives exchange-specific
fees, maker/taker routing, order book depth, margin requirements, and actual
spot/perp availability. More generic predictors are lower priority.

## Spot/Perp Carry Fee Ceiling

Run:

```bash
uv run python -m strategies.crypto_market_structure.spot_perp_carry_fee_ceiling
```

This estimates the maximum paired-leg cost before each spot/perp carry candidate
loses positive total return. It uses the same historical spot/perp approximation
as `spot_perp_carry.py`.

| candidate | max paired-leg cost bps | zero-cost total | zero-cost sharpe | default total | default sharpe | drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spot_perp_positive_funding_top_1_14d | 9.918648 | 0.112061 | 4.439976 | 0.065458 | 2.505911 | -0.012568 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | 9.589022 | 0.106008 | 5.419287 | 0.060508 | 2.922381 | -0.016324 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | 9.451312 | 0.100927 | 5.746814 | 0.057046 | 3.038431 | -0.015078 | 0.115399 |
| spot_perp_positive_funding_top_1_7d | 5.324513 | 0.127317 | 5.068643 | 0.030272 | 1.146710 | -0.029210 | 0.255392 |
| spot_perp_positive_funding_top_2_7d | 5.309325 | 0.118583 | 6.170666 | 0.028031 | 1.318138 | -0.028262 | 0.239501 |
| spot_perp_positive_funding_top_3_7d | 5.194306 | 0.114128 | 6.476165 | 0.025169 | 1.279093 | -0.027635 | 0.236095 |
| spot_perp_positive_funding_top_1_3d | 2.708906 | 0.145363 | 6.212902 | -0.062662 | -2.483362 | -0.095473 | 0.568672 |
| spot_perp_positive_funding_top_2_3d | 2.520025 | 0.128019 | 6.688512 | -0.068325 | -3.097790 | -0.095937 | 0.542565 |

Interpretation:

- The viable cluster is the low-turnover 14-day family.
- `top_1_14d`, `top_2_14d`, and `top_3_14d` survive only to roughly
  `9.45-9.92 bps` paired-leg cost. This is materially lower than the earlier
  9-symbol screen.
- 7-day variants have little room and only remain plausible under very low-cost
  execution.
- 3-day and 1-day variants are not worth promoting under the default cost
  because turnover consumes the edge.
- The next hard gate is exchange-specific execution: actual spot/perp fees,
  maker/taker routing, margin requirements, borrow constraints, and book depth.

## Spot/Perp Carry Execution Gate

Run:

```bash
uv run python -m strategies.crypto_market_structure.spot_perp_carry_execution_gate
```

This compares each candidate's fee ceiling with simple execution scenarios.
The scenarios are assumptions, not exchange fee schedules.

| candidate | scenario | ceiling bps | scenario bps | headroom bps | pass | default sharpe | turnover |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| spot_perp_positive_funding_top_1_14d | low_slippage_maker_like | 9.918648 | 6.000000 | 3.918648 | True | 2.505911 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | low_slippage_maker_like | 9.589022 | 6.000000 | 3.589022 | True | 2.922381 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | low_slippage_maker_like | 9.451312 | 6.000000 | 3.451312 | True | 3.038431 | 0.115399 |
| spot_perp_positive_funding_top_1_14d | low_slippage_taker_like | 9.918648 | 7.500000 | 2.418648 | True | 2.505911 | 0.121453 |
| spot_perp_positive_funding_top_2_14d | low_slippage_taker_like | 9.589022 | 7.500000 | 2.089022 | True | 2.922381 | 0.119183 |
| spot_perp_positive_funding_top_3_14d | low_slippage_taker_like | 9.451312 | 7.500000 | 1.951312 | True | 3.038431 | 0.115399 |
| spot_perp_positive_funding_top_1_7d | low_slippage_maker_like | 5.324513 | 6.000000 | -0.675487 | False | 1.146710 | 0.255392 |
| spot_perp_positive_funding_top_2_7d | low_slippage_maker_like | 5.309325 | 6.000000 | -0.690675 | False | 1.318138 | 0.239501 |

Interpretation:

- The 14-day cluster passes only the low-slippage maker-like and taker-like
  scenarios.
- It no longer passes the retail taker plus slippage scenario after broadening
  the universe.
- 7-day variants no longer pass even the low-slippage maker-like scenario. 3-day
  and 1-day variants do not deserve promotion.
- The next hard evidence should be venue-specific: actual account fees, symbol
  availability, margin requirements, and book depth at the intended order size.

## Broad Universe Rerun

Run:

```bash
uv run python -m strategies.crypto.fetch_market_data --start-date 2024-01-01 --end-date 2026-06-07
uv run python -m strategies.crypto_market_structure.fetch_market_data --start-date 2024-01-01 --end-date 2026-06-07
uv run python -m strategies.crypto_market_structure.diagnostics
uv run python -m strategies.crypto_market_structure.backtest
uv run python -m strategies.crypto_market_structure.predictive_screen
uv run python -m strategies.crypto_market_structure.broad_model_screen
uv run python -m strategies.crypto_market_structure.funding_carry
uv run python -m strategies.crypto_market_structure.spot_perp_carry
uv run python -m strategies.crypto_market_structure.spot_perp_carry_fee_ceiling
uv run python -m strategies.crypto_market_structure.spot_perp_carry_execution_gate
```

Expanded universe:

- spot archive wrote 30 symbols
- USD-M futures archive wrote 29 symbols
- `PEPEUSDT` spot exists but was not present in the futures output for this run
- spot/perp backtest now uses each day's available symbols instead of requiring
  every symbol to have every date

Important changes versus the 9-symbol screen:

- Directional market-structure ranking is still not a tradable candidate.
- Broad ML screens find some high-return candidates, but drawdowns remain large
  and hit rates are weak.
- Premium-index funding carry remains strong, but it is an optimistic proxy.
- Spot/perp carry survives as the best current profit-adjacent lane, but its
  execution headroom is materially smaller in the broader universe.

Broad feature diagnostics:

| feature | bucket | count | mean next return | hit rate |
| --- | ---: | ---: | ---: | ---: |
| funding_rate_sum | bottom_20 | 5107 | 0.002070 | 0.491 |
| funding_rate_sum | middle_60 | 12787 | -0.000457 | 0.476 |
| funding_rate_sum | top_20 | 7638 | 0.000052 | 0.477 |
| premium_close | bottom_20 | 5107 | 0.000281 | 0.491 |
| premium_close | middle_60 | 13380 | -0.000776 | 0.473 |
| premium_close | top_20 | 7045 | 0.001997 | 0.483 |
| taker_buy_imbalance | bottom_20 | 5107 | -0.003772 | 0.464 |
| taker_buy_imbalance | top_20 | 5107 | 0.001486 | 0.478 |
| volume_ratio_20d | bottom_20 | 5107 | -0.001791 | 0.478 |
| volume_ratio_20d | top_20 | 5107 | 0.002349 | 0.478 |

Broad hand-written directional candidates:

| candidate | steps | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| flow_top_3_weekly | 844 | 0.294361 | 0.525310 | -0.641738 | 0.233412 |
| premium_funding_top_2 | 844 | -0.408981 | 0.245358 | -0.948070 | 1.444313 |
| premium_funding_top_2_weekly | 844 | -0.189686 | 0.375176 | -0.873991 | 0.226303 |

Broad predictive screen:

| candidate | predictions | mean daily rank IC | hit rate | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ridge_top_1_7d | 19749 | -0.010632 | 0.469 | 1.159903 | 0.894820 | -0.741394 | 0.256975 |
| ridge_top_2_7d | 19749 | -0.010632 | 0.469 | -0.143790 | 0.279005 | -0.730647 | 0.264317 |
| ridge_top_3_7d | 19749 | -0.010632 | 0.469 | -0.184596 | 0.221029 | -0.729474 | 0.252080 |

Broad model screen top candidates:

| candidate | predictions | mean daily rank IC | hit rate | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| momentum_contrarian_return_ridge_top_1_1d | 17719 | 0.003080 | 0.476 | 3.012048 | 1.292496 | -0.499313 | 0.878887 |
| structure_contrarian_return_ridge_top_1_3d | 17719 | 0.005023 | 0.472 | 1.963273 | 1.114887 | -0.663075 | 0.587561 |
| all_rank_ridge_top_1_3d | 17719 | 0.054126 | 0.484 | 1.159405 | 1.106520 | -0.427930 | 0.446809 |
| flow_return_ridge_top_2_14d | 17719 | 0.004186 | 0.474 | 1.440758 | 1.038248 | -0.651314 | 0.122750 |

Broad funding-carry proxy:

| candidate | steps | total return | sharpe | max drawdown | turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| positive_funding_carry_top_3_14d | 864 | 0.218605 | 5.542733 | -0.015285 | 0.115741 |
| positive_funding_carry_top_2_14d | 864 | 0.251779 | 5.189345 | -0.020496 | 0.120370 |
| positive_funding_carry_top_3_7d | 864 | 0.203238 | 5.160595 | -0.013892 | 0.226080 |
| positive_funding_carry_top_1_14d | 864 | 0.301011 | 4.780717 | -0.021675 | 0.126157 |

Current interpretation:

- The strongest actionable lane remains spot/perp funding carry, not directional
  return prediction.
- Broadening the universe made the carry result less pretty, which is good
  evidence discipline: it reduces the chance that the candidate is a narrow
  universe artifact.
- The candidate still needs live feasibility checks before paper trading:
  actual fees, spot/perp availability, margin, order-book depth, and execution
  failure handling.

## Spot/Perp Carry Symbol Audit

Run:

```bash
uv run python -m strategies.crypto_market_structure.spot_perp_carry_symbol_audit
```

This decomposes 14-day spot/perp carry candidates by symbol. Gross contribution
excludes transaction costs. Funding contribution and basis contribution split
the return into funding received versus spot/perp price divergence.

Top symbols for `top_1_14d`:

| symbol | held steps | gross contribution | funding contribution | basis contribution | mean funding |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIFUSDT | 112 | 0.033175 | 0.032088 | 0.001087 | 0.000573 |
| INJUSDT | 42 | 0.018149 | 0.017183 | 0.000967 | 0.000818 |
| APTUSDT | 84 | 0.010762 | 0.012280 | -0.001518 | 0.000292 |
| TRXUSDT | 56 | 0.008866 | 0.007530 | 0.001336 | 0.000269 |
| OPUSDT | 41 | 0.007913 | 0.007408 | 0.000506 | 0.000361 |

Top symbols for `top_3_14d`:

| symbol | held steps | gross contribution | funding contribution | basis contribution | mean funding |
| --- | ---: | ---: | ---: | ---: | ---: |
| WIFUSDT | 168 | 0.013639 | 0.012931 | 0.000708 | 0.000459 |
| FETUSDT | 168 | 0.010539 | 0.011390 | -0.000851 | 0.000360 |
| INJUSDT | 98 | 0.010354 | 0.009359 | 0.000995 | 0.000573 |
| APTUSDT | 168 | 0.008838 | 0.009452 | -0.000614 | 0.000338 |
| LINKUSDT | 84 | 0.005217 | 0.004858 | 0.000359 | 0.000339 |

Interpretation:

- `WIFUSDT` is the strongest single symbol in the current carry screen.
- `INJUSDT`, `FETUSDT`, and `APTUSDT` are the next useful follow-up symbols.
- The top contributors are mostly funding-driven, not merely favorable basis
  movement. That makes them better live-feasibility candidates than symbols
  whose gross contribution comes mainly from basis luck.
- `FILUSDT` and `SEIUSDT` are warning cases: the strategy selected them, but
  their net contribution was negative in the broad run.
