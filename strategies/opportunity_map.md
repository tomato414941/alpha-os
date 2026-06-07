# Opportunity Map

This file records profit-source candidates without restricting exploration to
current local data, daily bars, linear models, or low compute.

## Durable Constraints

- No lookahead.
- Costs, slippage, execution feasibility, and risk must be measured.
- Results must be reproducible.
- A concrete trading strategy remains an `observation -> action` component.
- Secrets and private account data must not be committed.

## Profit-Source Candidates

| lane | possible edge | required data | current status |
| --- | --- | --- | --- |
| spot/perp carry | funding received after hedging spot and perp price moves | spot, perp, funding, fees, margin, depth | first Binance spot/perp approximation exists |
| cross-exchange basis | basis or funding spread between venues | multi-exchange spot/perp, fees, transfer, borrow, margin | not implemented |
| market making | spread capture and inventory control | L2 book, trades, fees, queue/fill model | not implemented |
| execution edge | better routing, maker/taker choice, order slicing | L2 book, trades, latency, fee tier | not implemented |
| event-flow prediction | short-horizon flow imbalance or liquidation behavior | trades, aggTrades, order book, funding schedule | data path probed |
| DeFi yield | stablecoin or delta-neutral yield | pool APY, TVL, smart-contract and depeg risk | data path probed |
| on-chain flow | wallet, exchange inflow/outflow, liquidation flows | on-chain, CEX deposit/withdraw proxies | not implemented |
| news/SNS | event or sentiment driven moves | news, social, timestamped labels | not implemented |
| directional ML/RL | price movement or direct policy learning | market state, reward, simulator | only basic screens exist |

## Current Probe

`data_source_probe.py` checks public data routes for:

- Binance spot and USD-M futures monthly trades, aggTrades, and 1m klines
- DeFiLlama yield pools
- Coinbase product discovery
- Hyperliquid perpetual market metadata

This is not a recommendation to use only these sources. It is a first inventory
of reachable public data routes.
