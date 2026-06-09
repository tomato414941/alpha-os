# Current Modern Alpha Web Intake

This is a one-off intake note from current web research. It is not a new
framework, tracker, queue, or data platform.

Created at `2026-06-09T23:14:39Z`.

## Diagnosis

The current candidate process is not modern enough because it mostly turns
price, funding, spread, and simple relative strength into paper candidates. That
misses the data surfaces that current crypto and cross-market traders actually
watch: order flow, liquidation pressure, open interest changes, wallet/entity
flow, attention, prediction-market odds, and options/volatility context.

The repo already contains some local artifacts for microstructure, liquidation,
wallet, attention, and stablecoin flow. The problem is not that the words are
absent. The problem is that they are not mandatory inputs to candidate
generation, so the workflow keeps drifting back to shallow price/funding screens.

## What The Web Changed

| source | observed point | implication for alpha-os |
| --- | --- | --- |
| CoinGlass API guide: https://www.coinglass.com/learn/CoinGlass-API-Full-Guide-en | Derivatives data is broader than funding: open interest, OI-weighted funding, liquidations, long/short ratios, taker buy/sell volume, large orders, options. | A modern crypto candidate should not be opened from funding alone. It needs at least one positioning or flow companion: OI change, liquidation map/history, taker imbalance, or options/skew. |
| CoinGlass pricing/API surface: https://www.coinglass.com/pricing | Commercial data covers futures, spot, options, tick-level L2/L3 order book, liquidation orders, funding, market snapshots, trades, and many exchanges including Hyperliquid. | Our current Hyperliquid-only public snapshot is too narrow for serious discovery. It is acceptable for a paper screen, but not enough to conclude edge. |
| Explainable Patterns in Cryptocurrency Microstructure: https://arxiv.org/html/2602.00776v1 | Engineered order book and trade features can have stable importance across crypto assets; important features include order-flow imbalance, spread, depth, and adverse selection. | Candidate generation should include microstructure state, not just 24h return and funding. |
| Microstructure Alpha, Frontiers 2026: https://www.frontiersin.org/journals/blockchain/articles/10.3389/fbloc.2026.1811716/abstract | Proper leakage controls and realistic fees can destroy apparent microstructure profitability; cross-asset transfer is not guaranteed. | Do not promote microstructure candidates without cost, venue, leakage, and same-asset validation. |
| Deep LOB forecasting guide: https://arxiv.org/html/2403.09267v1 | Predictive accuracy is not the same as actionable trading; evaluation should focus on complete executable transactions. | A model-like signal is not enough. The candidate must specify whether it can actually be filled after queue/adverse-selection/costs. |
| Fed paper on Kalshi macro markets: https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf | Prediction-market contracts can be treated like option-like event probabilities around macro releases. | CPI/FOMC/Hormuz candidates should use prediction-market odds as event state, not narrative text after the fact. |
| Glassnode Awaiting Liquidity: https://research.glassnode.com/the-week-onchain-week-12-2026/ | Market direction depends on ETF flows, sell-side pressure, spot volume, and overhead supply, not price alone. | BTC/ETH beta candidates need on-chain/ETF/liquidity regime context before being treated as standalone alpha. |
| Kaiko liquidity report: https://www.kaiko.com/resources/the-state-of-liquidity-on-korean-crypto-markets-2 | Liquidity is multidimensional: price, size, time, context, spreads, market depth, slippage, and venue structure. | Spread/depth fields are not enough; candidate records should say which venue and size the idea is valid for. |
| Nansen smart-money/onchain posts: https://nansen.ai/post/how-to-identify-smart-money-movements-in-crypto-uncover-profitable-signals-strategies | Wallet/entity flow, exchange inflow/outflow, and smart-money behavior are treated as trading inputs. | Wallet/entity flow should be an input family, but with survivorship, copy-crowding, and timestamp controls. |
| Polymarket/Kaito attention markets coverage: https://www.bankless.com/read/news/polymarket-partners-with-kaito-to-launch-attention-markets | Attention and mindshare are becoming tradable prediction-market primitives. | Attention is not just a narrative filter. It can be a tradable state variable, but only with source and timing controls. |

## Required Change To The Next Candidate Batch

The next broad alpha batch must not admit a crypto directional candidate unless
it has at least one non-price companion input:

- derivatives positioning: OI change, liquidation pressure, long/short ratio, or
  taker buy/sell imbalance;
- microstructure: order-flow imbalance, depth imbalance, adverse-selection risk,
  or executable spread/depth at the intended size;
- on-chain/entity flow: exchange inflow/outflow, smart-wallet movement, bridge or
  stablecoin flow;
- attention/event state: attention change, source-quality checked news, or
  prediction-market odds;
- options/volatility: implied-volatility shift, skew, or event-vol pricing.

If a candidate only has price, funding, and a relative-return story, it should be
marked as `legacy_shallow_screen` and should not be promoted.

## Immediate Candidate Families To Add

| family | concrete next use | rejection condition |
| --- | --- | --- |
| liquidation plus OI pressure | Find assets where forced-liquidation flow is large relative to OI and decide follow vs fade before seeing returns. | Reject if direction is chosen after price move or if depth/cost makes the move untradeable. |
| order-flow imbalance / microstructure | Use book/trade imbalance as state for short-horizon candidates, with explicit taker or maker execution mode. | Reject if only predictive accuracy is shown without executable fills after costs. |
| wallet/entity flow | Use entity-tagged wallet movement or exchange flow as exogenous pressure, not as a strategy internals field. | Reject public-wallet copy trades if source is stale, survivorship-biased, or crowded. |
| attention plus price lag | Use mindshare/sentiment/news attention as a timed state variable, not as narrative justification. | Reject if attention timestamp is not before the entry or if source duplication dominates. |
| prediction-market event odds | Use odds for CPI/FOMC/Hormuz and similar macro events before choosing trade direction. | Reject if the event condition is described after the asset has moved. |
| options/vol surface | Use skew/IV/event-vol context for BTC/ETH risk and macro-event trades. | Reject if direction is inferred from spot price alone. |
| cross-venue dislocation | Compare funding, basis, depth, and liquidation state across venues rather than one venue. | Reject if venue costs/transfer/borrow/margin make the apparent edge non-actionable. |

## Constraint For The Next Step

Do not build infrastructure first. The next step should be one broad candidate
batch that forces these companion inputs into the candidate rows. Missing inputs
should be visible as `missing_required_modern_input`, not silently ignored.
