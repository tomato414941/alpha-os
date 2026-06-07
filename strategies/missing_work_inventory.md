# Missing Work Inventory

This file records what alpha-os has not done yet. It is intentionally broader
than the current codebase.

The current project has many small probes, but it still does not have a serious
profit-seeking coverage map. A current snapshot is not a strategy. A data route
is not an edge.

## Main Failure

The project is still biased toward what is easy to fetch and script.

That creates hidden constraints:

- public unauthenticated data first
- current snapshots instead of durable history
- crypto perp/funding first
- price/funding before execution reality
- small scripts before an actual trading workflow
- data availability before a clear profit source

These are convenience constraints, not trading constraints.

## Profit Sources Not Covered

| area | missing work | why it matters |
| --- | --- | --- |
| liquidation cascades | OKX forced-liquidation feed, monitor samples, short-window labels, and rough paper gates are connected | forced flow can create short-horizon dislocations; cross-venue coverage, durable history, and fill reality are still missing |
| open-interest regimes | OI history, OI change, funding + OI interaction | crowded positions can unwind violently |
| volatility trading | Deribit BTC/ETH IV, skew, term structure, and recent realized-vol comparison are connected | price direction is not the only tradable edge; execution, hedge PnL, margin, tail risk, and forecasts are still missing |
| basis term structure | dated futures vs perps vs spot | carry can be cleaner than directional prediction |
| cross-exchange execution arb | venue price differences, fees, withdrawal constraints, latency | venue fragmentation can create real edge |
| cross-exchange market making | quote one venue while hedging another | combines spread capture and hedge routing |
| maker rebate capture | fee tiers, rebates, fill probability, adverse selection | spread alone is not market-making profit |
| queue-position edge | order-book diffs, queue estimation, cancels, fills | L2 snapshots are not enough |
| latency edge | timestamps, feed delay, venue-specific order path | impossible to assess with current daily/snapshot data |
| borrow/lending arb | Morpho current rates are connected; history, collateral risk, and execution checks are still missing | funding-like edge can exist outside perps |
| stablecoin depeg/repeg | peg deviation, liquidity, redemption route, issuer risk | depeg stress can be a distinct trade |
| DeFi yield decay | APY persistence, reward emissions, TVL inflow, exit liquidity | high APY alone is not edge |
| bridge/liquidity migration | chain TVL and stablecoin distribution proxies are connected; bridge-fill data is still missing | capital movement can precede price movement |
| protocol revenue | fees, revenue, active users, TVL quality | fundamental on-chain context is absent |
| token unlocks/emissions | unlock calendars, vesting, staking emissions | supply events can drive tradable pressure |
| listing/delisting events | CEX listing, perp listing, delisting, margin changes | event-driven moves are missing |
| ETF/institutional flows | BTC/ETH ETF flows, AUM, premium/discount | macro crypto flow is absent |
| macro liquidity | rates, DXY, yields, liquidity proxies, risk appetite | cross-asset context is still thin |
| sector rotation | L1, meme, AI, DeFi, RWA, exchange tokens | current universe grouping is primitive |
| prediction markets | crypto-related Polymarket/Kalshi odds | event probabilities can become features |
| social reflexivity | X/Reddit/Telegram attention, influencer events | crypto often moves on attention flows |
| news NLP | RSS headline event extraction is connected | no full-text NLP, social firehose, source deduplication, historical attention, or leakage-safe labels |
| RL sizing | direct reward optimization, drawdown-aware policy learning | current sizing is mostly hand-coded |
| RL execution | order slicing, maker/taker choice, inventory control | execution strategy is not learned or simulated |
| sequence models | transformers/RNNs over trades, LOB, funding, OI | current models are shallow ridge-style screens |
| graph models | token/sector/wallet/flow graph relationships | current features are mostly tabular |
| anomaly detection | abnormal flow, spread, OI, funding, APY, peg deviations | early stress detection is missing |

## Data Not Connected

| data | current state | gap |
| --- | --- | --- |
| historical L2 order books | only current Hyperliquid snapshot | no depth history, no queue, no diffs |
| order-book diffs | absent | cannot model queue/fill/adverse selection |
| liquidation feed/history | current OKX liquidation feed, monitor, short-window labels, and paper gates connected | no durable cross-venue liquidation history, no Binance access from this environment, and no actual fill reconciliation |
| OI history | only current Hyperliquid context | no persistence or crowding labels |
| funding history across venues | partial Binance/HL snapshots | no multi-venue historical carry test |
| options IV/skew | Deribit BTC/ETH surface and current paper tickets connected | no full option chain strategy, hedge PnL, margin, execution costs, or realized-vol forecast |
| borrow/lending rates | current Morpho snapshot connected | no history, collateral drawdown, oracle, liquidation, withdrawal, or gas model |
| exchange fee tiers | absent | execution economics are fake |
| maker rebates | absent | market making cannot be evaluated |
| min order size/lot size | absent | trade feasibility unknown |
| withdrawal/transfer status | absent | cross-exchange strategies may be impossible |
| exchange outage/status | absent | operational risk invisible |
| on-chain exchange inflow/outflow | absent | stablecoin supply is only a proxy |
| wallet/entity flows | absent | no whale/entity signal |
| bridge flows | stablecoin chain migration proxy connected | no bridge-fill route, wallet/entity flow, or forward labels |
| chain TVL/revenue/users | absent | DeFi fundamental context missing |
| token unlock calendar | absent | supply overhang unmodeled |
| ETF flows | absent | institutional flow missing |
| news feed | current RSS headline metadata connected | no full-text feed, source deduplication, social firehose, historical attention, or leakage-safe labels |
| social feed | absent | attention data is only shallow trending proxy |
| historical attention | absent | cannot test lead/lag |
| realized volatility | partial via prices only | no vol-target or vol edge lane |
| market regime labels | absent | no robust regime comparison |
| paper/live fills | absent | no live-vs-backtest reconciliation |

## Evaluation Not Done

| evaluation | missing |
| --- | --- |
| persistence | whether a signal survives for minutes/hours/days |
| capacity | how much capital can enter before edge disappears |
| slippage by order size | effect of size on entry/exit |
| fee sensitivity | maker/taker, tiers, rebates |
| fill probability | especially for maker/limit strategies |
| adverse selection | whether fills happen before price moves against us |
| liquidation risk | margin, forced close, funding spikes |
| borrow/margin availability | whether the trade can actually be opened |
| venue availability | symbol exists on both legs, region/account access |
| live/paper reconciliation | actual fills vs intended actions |
| OOS by regime | bull, bear, chop, high-vol, low-liquidity |
| leakage audit | event timestamps, API cache timestamps, delayed data |
| benchmark fairness | benchmark by strategy type, not only buy-and-hold |
| portfolio interaction | correlation and shared failure modes across strategies |
| kill-switch criteria | when to stop a strategy |
| tax/accounting impact | turnover, funding, DeFi income treatment |

## Operational Work Not Done

| operation | missing |
| --- | --- |
| account selection | no chosen CEX/DEX/broker |
| execution mode | manual, alert-driven, paper, or automated not fixed |
| API key path | no exchange secret workflow for trading |
| order ticket generation | no actionable trade ticket |
| paper trading loop | absent |
| position monitor | absent |
| risk monitor | absent |
| funding monitor | absent |
| collateral monitor | absent |
| liquidation monitor | absent |
| alerting | absent |
| execution log | absent |
| post-trade review | absent |
| deployment target | absent |
| custody/capital allocation | absent |

## Existing External Tools Not Evaluated

| tool/source | relevance | status |
| --- | --- | --- |
| NautilusTrader | production-grade backtest/live engine, order/position domain model, live reconciliation | not evaluated |
| Hummingbot | market making, exchange connectors, AMM/CLOB bot framework | not evaluated |
| Freqtrade | crypto bot and backtesting workflow | not evaluated |
| vectorbt / backtesting.py / backtrader | research/backtest engines | not evaluated |
| ccxt / ccxt.pro | exchange connectivity and live market data | dependency present historically but not integrated |
| CoinGlass | OI, funding, liquidation heatmaps, ETF/institutional data | not connected |
| Coinalyze | OI and liquidation metrics | not connected |
| CryptoQuant | exchange flows and on-chain metrics | not connected |
| Glassnode | on-chain and sentiment indicators | not connected |
| Dune | custom on-chain dashboards/queries | not connected |
| DefiLlama Pro/downloads | broader DeFi datasets beyond current free probes | only shallow free routes used |
| CoinGecko/GeckoTerminal | trending, categories, on-chain DEX/pool data | only trending snapshot used |
| Kwery | normalized order books, funding, OI, liquidation, prediction market data | not connected |
| Databento | high-quality traditional market data | not evaluated |
| Polygon.io / Alpaca / Interactive Brokers | equities/options/live execution | not evaluated |
| Polymarket / Kalshi data | prediction market odds | not connected |

## What A Serious Next Pass Should Do

Do not add one more narrow probe.

Build a broad candidate matrix with at least:

- profit source
- required data
- reachable data provider
- paid/free/auth requirement
- history availability
- execution venue
- expected holding period
- capacity concern
- main failure mode
- first falsification test

Then pick multiple lanes for parallel data collection:

- liquidation/OI/funding history
- L2 order-book history and fill simulation
- attention/news events joined to returns
- stablecoin/on-chain flow joined to regimes
- DeFi yield persistence and exit liquidity
- options/volatility if data is reachable
- paper-trading workflow for the most executable candidate
