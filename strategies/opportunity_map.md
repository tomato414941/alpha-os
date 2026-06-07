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
| spot/perp carry | funding received after hedging spot and perp price moves | spot, perp, funding, fees, margin, depth | Binance spot/perp carry fee ceiling exists; 14-day low-turnover family is current candidate |
| cross-exchange basis | basis or funding spread between venues | multi-exchange spot/perp, fees, transfer, borrow, margin | OKX-Hyperliquid promotion gate exists; ZEC/BTC are current paper candidates under fee/touch assumptions |
| perp market map | carry, crowding, dislocation, and liquidity states | funding, open interest, volume, premium, impact prices | current Hyperliquid snapshot exists |
| market making | spread capture and inventory control | L2 book, trades, fees, queue/fill model | first Hyperliquid L2 snapshot exists |
| execution edge | better routing, maker/taker choice, order slicing | L2 book, trades, latency, fee tier | not implemented |
| event-flow prediction | short-horizon flow imbalance or liquidation behavior | trades, aggTrades, order book, funding schedule | first Binance USD-M aggTrades 5m diagnostic exists |
| DeFi yield | stablecoin or delta-neutral yield | pool APY, TVL, smart-contract and depeg risk | first DeFiLlama stable-yield screen exists |
| stablecoin liquidity | liquidity expansion/contraction or peg stress | stablecoin supply, peg price, chain distribution | first DeFiLlama stablecoin supply snapshot exists |
| on-chain flow | wallet, exchange inflow/outflow, liquidation flows | on-chain, CEX deposit/withdraw proxies | not implemented |
| news/SNS | event or sentiment driven moves | news, social, timestamped labels | first Fear & Greed and trending snapshot exists |
| directional ML/RL | price movement or direct policy learning | market state, reward, simulator | only basic screens exist |
| liquidation cascade | forced liquidations create continuation or reversal pressure | liquidation events, OI, perp price, book depth | not implemented |
| options volatility | implied/realized volatility spread, skew, term structure | options chain, IV, Greeks, realized vol | not implemented |
| borrow/lending arbitrage | borrow/lend spread, collateral return, utilization stress | borrow rates, lending rates, collateral rules | not implemented |
| stablecoin depeg/repeg | peg stress and redemption/liquidity dislocation | stablecoin price, liquidity, issuer/redemption data | not implemented |
| bridge/liquidity migration | cross-chain capital movement precedes market repricing | bridge flows, chain stablecoin supply, TVL | not implemented |
| protocol fundamentals | revenue, fees, active users, TVL quality | protocol metrics, revenue, fees, users | not implemented |
| token unlock pressure | unlocks/emissions cause supply pressure or hedging flow | unlock calendar, float, volume, derivatives | not implemented |
| listing event | exchange listings/delistings and perp launches move attention/liquidity | listing timestamps, venue data, returns | not implemented |
| ETF/institutional flow | ETF creations/redemptions and AUM drive BTC/ETH flow | ETF flows, AUM, premium/discount | not implemented |
| macro liquidity | rates, DXY, yields, liquidity regime affect crypto beta | macro data, cross-asset prices, regime labels | not implemented |
| sector rotation | capital rotates across token sectors | sector taxonomy, returns, volume, attention | not implemented |
| prediction market signal | event odds can inform crypto/macro positioning | Polymarket/Kalshi odds, event labels | not implemented |
| latency/feed edge | faster or cleaner data creates short-horizon execution edge | timestamped feeds, venue latency, order path | not implemented |
| anomaly/stress detection | abnormal funding, spread, flow, APY, peg, or OI can signal stress | multi-source anomaly features | not implemented |

## Current Probe

`data_source_probe.py` checks public data routes for:

- Binance spot and USD-M futures monthly trades, aggTrades, and 1m klines
- DeFiLlama yield pools
- Coinbase product discovery
- Hyperliquid perpetual market metadata
- Hyperliquid predicted funding across venues

This is not a recommendation to use only these sources. It is a first inventory
of reachable public data routes.
