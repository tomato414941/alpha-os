# P0 Parallel Results

Generated on 2026-06-07 UTC.

## Status

This pass moves P0 from planning into parallel execution and broadens the
first derivatives-history screen:

- data reachability checks across derivatives, funding, L2, and attention/liquidity
- 90-day, 20-symbol Binance derivatives history screen
- funding-rate history added to the same labeled derivatives table
- funding carry proxy with premium-change and rough round-trip cost
- short Hyperliquid L2 burst for a first adverse-selection label
- paper/manual ticket for the most feasible current funding-spread candidate

The outputs are still first-pass probes. They do not prove an edge.

## Data Reachability

| lane | source | status | history kind | note |
| --- | --- | --- | --- | --- |
| liquidation/OI/funding | Hyperliquid `metaAndAssetCtxs` | reachable | current snapshot | OI/funding/volume/premium context, not history |
| liquidation/OI/funding | Binance USD-M daily metrics | reachable | daily historical file | candidate route for OI/derivatives metrics |
| funding/basis | Binance USD-M premium index klines | reachable | minute historical file | candidate route for premium/index history |
| funding/basis | Hyperliquid predicted fundings | reachable | current snapshot | multi-venue predicted funding, not history |
| L2/fill | Hyperliquid L2 book | reachable | current snapshot | top 20 levels per side |
| L2/fill | Hyperliquid recent trades | reachable | recent snapshot | can pair with L2 bursts |
| attention/liquidity | DeFiLlama stablecoins | reachable | current plus previous period fields | stablecoin supply and peg context |
| attention/liquidity | Alternative.me Fear & Greed | reachable | short history | market-level sentiment |
| attention/liquidity | CoinGecko trending | reachable | current snapshot | attention proxy |

## L2 Burst

Run:

```bash
uv run python -m strategies.p0_parallel.l2_burst_probe --samples 8 --delay-seconds 1
```

| asset | samples | mean spread bps | mean abs imbalance 10 bps | next return after positive imbalance | next return after negative imbalance |
| --- | ---: | ---: | ---: | ---: | ---: |
| BTC | 8 | 0.16167659 | 0.56201889 | 0.000000000000 | -0.000053889168 |
| ETH | 8 | 0.62665968 | 0.11328262 | -0.000050133167 | 0.000156695603 |
| HYPE | 8 | 0.73904222 | 0.26222198 | -0.000042209724 | 0.000019723905 |
| SOL | 8 | 0.15621340 | 0.44664464 | -0.000169212325 | 0.000437496582 |

This is far too small to conclude anything. Its value is that the project now
has a first fill/adverse-selection label shape instead of only a static book
snapshot.

## Binance Derivatives History

Run:

```bash
uv run python -m strategies.p0_parallel.binance_derivatives_history_probe
```

Sample:

- symbols: BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, BNBUSDT, DOGEUSDT,
  ADAUSDT, AVAXUSDT, LINKUSDT, LTCUSDT, BCHUSDT, DOTUSDT, UNIUSDT,
  ETCUSDT, FILUSDT, NEARUSDT, APTUSDT, OPUSDT, ARBUSDT, INJUSDT
- window: 2024-01-01 through 2024-03-30
- metrics rows: 288 per symbol/day
- premium rows: 1440 per symbol/day
- funding rows: 3 per symbol/day
- labeled observations: 1779, excluding the final day per symbol with no
  next-day return

Schema confirmed:

- metrics: `create_time`, `symbol`, `sum_open_interest`,
  `sum_open_interest_value`, `count_toptrader_long_short_ratio`,
  `sum_toptrader_long_short_ratio`, `count_long_short_ratio`,
  `sum_taker_long_short_vol_ratio`
- premium index klines: standard 1m kline fields where `close` is the premium
  index close
- funding rate: `calc_time`, `funding_interval_hours`, `last_funding_rate`

First signal summary:

| feature | observations | corr to next return | low bucket mean next return | high bucket mean next return | high bucket hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean_premium_close | 1779 | 0.03747237 | 0.00687146 | 0.00779112 | 0.50561798 |
| max_abs_premium_close | 1779 | 0.10633370 | 0.00274931 | 0.01035695 | 0.50561798 |
| mean_funding_rate | 1779 | 0.05850243 | 0.00203549 | 0.00830219 | 0.50561798 |
| sum_funding_rate | 1779 | 0.05850243 | 0.00203549 | 0.00830219 | 0.50561798 |
| oi_value_change | 1779 | -0.02697215 | 0.00438935 | 0.00255583 | 0.46067416 |
| mean_count_top_long_short_ratio | 1779 | -0.03189207 | 0.00619534 | 0.00485732 | 0.51910112 |
| mean_sum_top_long_short_ratio | 1779 | 0.01120125 | 0.00289859 | 0.00506707 | 0.48764045 |
| mean_count_long_short_ratio | 1779 | -0.03784357 | 0.00619916 | 0.00292366 | 0.50337079 |
| mean_sum_taker_long_short_vol_ratio | 1779 | 0.02735735 | 0.00634883 | 0.00980937 | 0.57752809 |

The broader pass weakens the earlier 30-day reversal story. Premium and funding
are mildly positive in this 2024Q1 panel, OI value change is weakly negative,
and taker long/short volume has the best high-bucket hit rate. This is still a
coarse daily label, not a deployable signal. The next useful step is regime
splitting and cost-aware carry/reversal labels rather than treating any one
feature as standalone alpha.

## Binance Symbol-Feature Queue

Run:

```bash
uv run python -m strategies.p0_parallel.binance_derivatives_symbol_feature_candidates
```

This turns the same Binance USD-M history into a per-symbol research queue.
It is not a current trade list; it identifies which symbol-feature pairs should
be rerun on recent windows before joining to execution gates.

Top current queue rows:

| symbol | feature | bucket | score | note |
| --- | --- | --- | ---: | --- |
| UNIUSDT | mean_sum_top_long_short_ratio | low | 351.7328 | low bucket had materially higher next-day mean |
| NEARUSDT | mean_funding_rate | high_mean_only | 340.6412 | high funding bucket had stronger mean but weaker hit rate |
| UNIUSDT | oi_value_change | high | 328.9493 | high OI-change bucket had stronger next-day mean |
| DOGEUSDT | mean_premium_close | high | 311.2457 | high premium bucket had stronger mean and hit rate |
| BCHUSDT | mean_premium_close | high | 289.2663 | high premium bucket had stronger mean and hit rate |

The next useful step is to rerun these on recent windows and split by regime.

## Recent Binance Derivatives Window

Run:

```bash
uv run python -m strategies.p0_parallel.binance_derivatives_history_probe \
  --start-date 2026-04-01 \
  --days 67 \
  --max-workers 16 \
  --output-path strategies/p0_parallel/binance_derivatives_recent_history.csv \
  --signal-output-path strategies/p0_parallel/binance_derivatives_recent_signal_summary.csv \
  --schema-output-path strategies/p0_parallel/binance_derivatives_recent_schema.md

uv run python -m strategies.p0_parallel.binance_derivatives_symbol_feature_candidates \
  --history-path strategies/p0_parallel/binance_derivatives_recent_history.csv \
  --output-path strategies/p0_parallel/binance_derivatives_recent_symbol_feature_candidates.csv \
  --markdown-output-path strategies/p0_parallel/binance_derivatives_recent_symbol_feature_candidates.md
```

Recent aggregate feature signals:

| feature | observations | corr | low mean | high mean | high hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| oi_value_change | 1313 | 0.18805816 | -0.01078456 | 0.00570641 | 0.54711246 |
| mean_sum_taker_long_short_vol_ratio | 1313 | 0.08087944 | -0.01053962 | 0.00195517 | 0.47416413 |
| mean_premium_close | 1313 | 0.03739382 | -0.00027887 | 0.00425150 | 0.51671733 |

The recent panel points more toward OI expansion and taker-flow context than
the older 2024Q1 aggregate view.

## Historical vs Recent Feature Regime Compare

Run:

```bash
uv run python -m strategies.p0_parallel.binance_derivatives_feature_regime_compare
```

Top current regime comparison rows:

| symbol | feature | status | historical score | recent score | combined score |
| --- | --- | --- | ---: | ---: | ---: |
| ARBUSDT | mean_sum_top_long_short_ratio | persistent_symbol_feature | 268.2077 | 428.4488 | 472.3644 |
| ARBUSDT | oi_value_change | recent_symbol_feature_priority | 15.2622 | 546.0359 | 425.2651 |
| NEARUSDT | mean_funding_rate | persistent_symbol_feature | 340.6412 | 305.4067 | 417.7388 |
| BCHUSDT | mean_sum_top_long_short_ratio | persistent_symbol_feature | 283.1930 | 305.4253 | 397.6440 |
| OPUSDT | oi_value_change | recent_symbol_feature_priority | 95.7761 | 426.6602 | 375.8508 |

These rows are now surfaced into the current alpha stack as derivatives
symbol-feature priors, not as trade instructions.

## Binance Intraday Feature Labels

Run:

```bash
uv run python -m strategies.p0_parallel.binance_derivatives_intraday_feature_labels
```

This joins Binance USD-M 5m metrics, 5m price klines, and 5m premium-index
klines for ARB, NEAR, BCH, OP, UNI, DOGE, SOL, and ADA. It tests whether each
current derivatives feature separates the next 1h return.

Top current rows:

| symbol | feature | bucket | observations | low mean 1h | high mean 1h | score |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| NEARUSDT | sum_top_long_short_ratio | high | 4885 | -0.002278 | 0.001529 | 399.6268 |
| OPUSDT | count_long_short_ratio | low | 4728 | 0.000868 | -0.002618 | 367.0815 |
| ARBUSDT | count_long_short_ratio | low | 4676 | 0.000322 | -0.003111 | 363.7439 |
| NEARUSDT | count_long_short_ratio | high | 4885 | -0.001067 | 0.001734 | 298.4522 |
| OPUSDT | count_top_long_short_ratio | low | 4728 | 0.000723 | -0.001973 | 287.4978 |

These are intraday research labels, not trade instructions. The next check is
fresh-window repetition plus fees, spread, fill probability, funding PnL, and
position-sizing assumptions.

## Binance Intraday Repeat Compare

Run:

```bash
uv run python -m strategies.p0_parallel.binance_derivatives_intraday_feature_labels \
  --start-date 2026-05-02 \
  --days 18 \
  --max-workers 12 \
  --labels-output-path strategies/p0_parallel/binance_derivatives_intraday_prior_feature_labels.csv \
  --candidates-output-path strategies/p0_parallel/binance_derivatives_intraday_prior_feature_candidates.csv \
  --markdown-output-path strategies/p0_parallel/binance_derivatives_intraday_prior_feature_candidates.md

uv run python -m strategies.p0_parallel.binance_derivatives_intraday_repeat_compare
```

This compares the prior 2026-05-02 through 2026-05-19 window against the recent
2026-05-20 through 2026-06-06 window. The useful rows are not necessarily the
highest single-window scores; they are the rows where the same symbol, feature,
and preferred bucket repeat across non-overlapping windows.

Top repeat/current rows:

| symbol | feature | status | prior bucket | recent bucket | prior score | recent score | combined score |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| ARBUSDT | count_long_short_ratio | intraday_repeat_watch | low | low | 135.9200 | 363.7439 | 365.8527 |
| ARBUSDT | count_top_long_short_ratio | intraday_repeat_watch | low | low | 115.7972 | 253.4757 | 319.6518 |
| ARBUSDT | sum_top_long_short_ratio | intraday_repeat_watch | low | low | 120.7376 | 187.4846 | 312.3820 |
| ADAUSDT | count_long_short_ratio | intraday_repeat_watch | low | low | 96.0980 | 201.6159 | 285.6408 |
| UNIUSDT | abs_premium_close | intraday_repeat_watch | high | high | 107.5477 | 101.4635 | 273.2658 |
| UNIUSDT | premium_close | intraday_repeat_watch | low | low | 107.1757 | 95.2261 | 265.7064 |
| OPUSDT | count_top_long_short_ratio | intraday_bucket_shift | high | low | 239.1800 | 287.4978 | 264.5156 |

ARB, ADA, UNI, and DOGE produce the cleanest repeated buckets in this two-window
check. OP and NEAR remain interesting but show bucket shifts, so they need a
regime explanation before any promotion.

## Funding Carry Proxy

Run:

```bash
uv run python -m strategies.p0_parallel.funding_carry_proxy
```

Assumption:

- if daily funding is positive, short the perp and hedge the delta elsewhere
- if daily funding is negative, long the perp and hedge the delta elsewhere
- proxy PnL = absolute funding earned + perp-direction premium change - rough
  round-trip cost
- default rough round-trip cost: 10 bps

Summary:

- observations: 1765
- mean net proxy PnL: -0.00023486
- hit rate: 0.2771
- best net proxy PnL: 0.00376912
- worst net proxy PnL: -0.00158772

Top candidates:

| date | symbol | perp direction | funding pnl | basis pnl | net proxy pnl |
| --- | --- | ---: | ---: | ---: | ---: |
| 2024-03-05 | OPUSDT | -1 | 0.00397320 | 0.00079592 | 0.00376912 |
| 2024-03-05 | AVAXUSDT | -1 | 0.00304341 | 0.00060052 | 0.00264393 |
| 2024-03-05 | SOLUSDT | -1 | 0.00305449 | 0.00054567 | 0.00260016 |
| 2024-03-02 | ARBUSDT | -1 | 0.00317285 | 0.00040789 | 0.00258074 |
| 2024-03-05 | NEARUSDT | -1 | 0.00308739 | 0.00041663 | 0.00250402 |

This does not yet prove a tradable carry edge. It shows the current one-month
funding carry proxy is mostly cost-sensitive: with a conservative 10 bps rough
cost, only a brief high-funding event survives. With 1 bps, the same proxy is
mostly positive, so the next real question is executable cost, hedge venue,
borrow/spot availability, and holding-period design rather than another
directional next-day return label.

## Paper Ticket

The current ticket candidate is:

- Asset: `MANTA`
- Long venue: `BinPerp`
- Short venue: `HlPerp`
- Annualized spread snapshot: `2.18652228`
- Hyperliquid 24h notional volume: `651345.71`
- Hyperliquid impact spread: `0.00228467`

This is not a trade instruction. It is an operational falsification artifact:
if the candidate cannot become venue-specific order details with fees, size,
margin, and risk limits, then the lane is not operational yet.

## Next Parallel Step

- Extend the Binance derivatives screen beyond one month and add more symbols.
- Convert funding carry from proxy PnL into venue-specific execution assumptions.
- Extend L2 burst from seconds to repeated scheduled snapshots.
- Pair Hyperliquid recent trades with each L2 snapshot.
- Convert the MANTA ticket into explicit fee and notional assumptions, then
  reject it if either venue leg is inaccessible.
