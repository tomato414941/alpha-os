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
