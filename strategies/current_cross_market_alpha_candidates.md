# Current Cross-Market Alpha Candidates

These are research candidates, not trade instructions. The purpose is to widen
alpha discovery beyond crypto-only screens without adding new tracking
infrastructure.

Observed on 2026-06-09.

## Current Crypto Entry Anchors

These anchors came from the current Hyperliquid snapshot used in this research
pass. They should be refreshed before any real order.

| asset | mark | 24h bps | annualized funding | spread bps | depth10 notional |
| --- | ---: | ---: | ---: | ---: | ---: |
| ZEC | 429.650000000000 | -278.09 | -1.10489179 | 0.23276113 | 80477.35500000 |
| HYPE | 59.011000000000 | -833.81 | 0.10950000 | 0.16945850 | 148842.34599000 |
| ADA | 0.162640000000 | -450.92 | -0.08034847 | 1.84484826 | 62427.89850000 |
| ZRO | 0.821610000000 | -630.20 | 0.10950000 | 3.16532749 | 5457.05304000 |
| BTC | 61028.000000000000 | -431.63 | 0.06916721 | 0.16387935 | 925580.91138000 |
| ETH | 1630.100000000000 | -378.92 | 0.00914982 | 0.61347811 | 7682006.40114000 |
| SOL | 64.044000000000 | -492.00 | -0.23747309 | 0.15616094 | 562686.80440500 |

## Highest Priority

| id | market | candidate | why now | first check |
| --- | --- | --- | --- | --- |
| zec-carry-strength | crypto perp | Long ZEC perp | ZEC has strong 24h relative strength, large Hyperliquid OI/volume, negative funding, and acceptable current spread/depth. | Fix entry and score 15m/1h/4h net of spread, taker fee, funding, and stop/adverse excursion. |
| hype-relative-weakness | crypto perp | Short HYPE vs BTC/ETH basket | The fresh snapshot invalidated the earlier HYPE long idea: HYPE is now materially weaker than BTC/ETH, while short funding is not hostile. | Compare HYPE short to BTC/ETH hedge over 15m/1h/4h after funding and spread. |
| cpi-hot-risk-off | prediction/macro | Hot CPI: short BTC/ETH beta, short QQQ/SMH, long oil/gold or short TLT | CPI/FOMC is the immediate macro catalyst and crypto is already sensitive to rate-cut repricing. | Before CPI, record prediction-market odds and liquid proxies; after release, compare realized move vs implied odds. |
| hormuz-oil-inflation | commodities/macro | Long oil or oil-call proxy if Hormuz disruption persists | Hormuz/shipping risk feeds oil and inflation expectations, which can pressure risk assets. | Track Hormuz normal-traffic odds, Brent/USO move, BTC/QQQ response, and whether oil move leads crypto risk-off. |
| prediction-market-hormuz | prediction market | Trade Hormuz normal-traffic mispricing only if public shipping evidence diverges from odds | The market has concrete resolution criteria and links directly to oil/inflation paths. | Compare odds to PortWatch/shipping evidence; avoid if no independent evidence edge. |

## Secondary Candidates

| id | market | candidate | why now | first check |
| --- | --- | --- | --- | --- |
| ada-carry-strength | crypto perp | Long ADA perp | ADA has negative funding plus positive 24h strength, but liquidity and spread are weaker than ZEC. | Same 15m/1h/4h cost and funding check as ZEC. |
| zro-relative-weakness | crypto perp | Short ZRO relative weakness | ZRO is weak and related to unlock pressure, but the repeat unlock short failed. | Only test as relative weakness, not as unlock thesis; require depth and stop discipline. |
| semis-soft-cpi-bounce | equities/ETF | Long SMH/NVDA/AVGO if CPI is soft | AI/semis sold off into CPI; soft CPI could re-open duration/growth appetite. | Event-only paper check; do not hold if CPI is hot. |
| semis-hot-cpi-short | equities/ETF | Short SMH/QQQ if CPI is hot | Hot CPI would pressure duration-sensitive growth and crypto beta together. | Event-only paper check against SPY and rates proxies. |
| defensive-rotation | equities/ETF | Long defensives/financials vs QQQ if rotation continues | Recent equity rotation away from AI into Dow/healthcare/financials may continue if rates stay firm. | Compare sector ETF relative returns after CPI/FOMC. |
| gold-hot-cpi-geopolitical | commodities/ETF | Long gold on hot CPI or renewed geopolitical stress | Gold is being framed as CPI/FOMC-sensitive and geopolitical hedge. | Check whether GLD/gold rises when TLT falls; avoid if real-yield pressure dominates. |
| tlt-rate-path | rates/ETF | Short TLT on hot CPI, long TLT on soft CPI | Rates are the clean upstream expression of CPI/FOMC repricing. | Event paper check; use as macro signal even if not traded. |
| coin-hood-exchange-beta | equities | Long/short COIN/HOOD as crypto-market-structure beta | Prediction markets, crypto legislation, and exchange activity can move these faster than spot tokens. | Compare against BTC/ETH and QQQ; reject if it is only generic beta. |
| mstr-btc-premium | equities/relative | Short or avoid MSTR if BTC remains weak and treasury premium compresses | Strategy-related BTC flow headlines add idiosyncratic pressure beyond BTC spot. | Track MSTR vs BTC beta; only candidate if residual weakness is visible. |
| rwa-tokenization | crypto/equity | ONDO/HYPE/RWA-linked names as tokenization theme | RWA/tokenized stocks/stablecoin infrastructure remain active 2026 themes. | Require volume/flow confirmation; avoid narrative-only longs. |
| stablecoin-risk-off | crypto macro | Stablecoin share rising as risk-off signal | Stablecoin rotation can mark crypto de-risking rather than opportunity. | Treat as filter: reduce broad long beta unless single-name strength survives. |
| oil-normalization-risk-on | macro relative | Short oil / long crypto beta if Hormuz normalizes | If oil/inflation pressure fades, risk assets can rebound. | Only after prediction odds and shipping evidence confirm normalization. |
| copper-growth-proxy | commodities | Long copper/pro-growth proxy if soft CPI and equities broaden | Copper can express growth rebound differently from crypto. | Compare copper/CPER vs QQQ/BTC after CPI. |

## What Not To Do

- Do not add another queue, ticket, registry, or report framework for these.
- Do not call any macro theme an alpha until it is tied to a concrete instrument,
  entry, exit/invalidation, cost, and first falsification check.
- Do not keep crypto-only if the upstream event is macro, commodity, or
  prediction-market driven.
