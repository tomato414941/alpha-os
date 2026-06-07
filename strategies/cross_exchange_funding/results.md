# Cross-Exchange Funding Results

Data:

- source: Hyperliquid public info endpoint
- request: `predictedFundings`
- output: current predicted funding rates across venues

## Current Funding Spread Snapshot

The spread is normalized to an hourly rate before annualization. The intended
direction is long the lower-funding venue and short the higher-funding venue.

Top snapshot rows:

| asset | long venue | short venue | hourly spread | annualized spread |
| --- | --- | --- | ---: | ---: |
| STABLE | BybitPerp | BinPerp | 0.00031973 | 2.8009 |
| MANTA | BinPerp | HlPerp | 0.00024960 | 2.1865 |
| ORDI | BybitPerp | HlPerp | 0.00017049 | 1.4935 |
| NIL | HlPerp | BybitPerp | 0.00015312 | 1.3413 |
| BSV | HlPerp | BinPerp | 0.00012785 | 1.1200 |
| BABY | HlPerp | BybitPerp | 0.00011884 | 1.0410 |
| UMA | BybitPerp | BinPerp | 0.00010833 | 0.9489 |
| ZORA | HlPerp | BinPerp | 0.00010377 | 0.9090 |
| USTC | BybitPerp | HlPerp | 0.00010349 | 0.9066 |
| AIXBT | HlPerp | BinPerp | 0.00008683 | 0.7606 |

## Hyperliquid Feasibility Overlay

For rows involving `HlPerp`, the feasibility overlay joins Hyperliquid market
context from `metaAndAssetCtxs`: max leverage, open interest, day notional
volume, mark/oracle dislocation, and impact-price spread.

Top overlay rows:

| asset | long venue | short venue | annualized spread | HL day notional volume | HL impact spread | notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| STABLE | BybitPerp | BinPerp | 2.8009 |  |  | Hyperliquid not involved; external venue feasibility still unknown |
| MANTA | BinPerp | HlPerp | 2.1865 | 651345.71 | 0.002285 | Hyperliquid context available |
| ORDI | BybitPerp | HlPerp | 1.4935 | 204175.88 | 0.002365 | Hyperliquid context available |
| NIL | HlPerp | BybitPerp | 1.3413 | 470602.38 | 0.003226 | Hyperliquid context available |
| BSV | HlPerp | BinPerp | 1.1200 | 210661.21 | 0.003071 | Hyperliquid context available |
| BABY | HlPerp | BybitPerp | 1.0410 | 2143712.08 | 0.002889 | Hyperliquid context available |
| UMA | BybitPerp | BinPerp | 0.9489 |  |  | Hyperliquid not involved; external venue feasibility still unknown |
| ZORA | HlPerp | BinPerp | 0.9090 | 268936.98 | 0.003089 | Hyperliquid context available |
| USTC | BybitPerp | HlPerp | 0.9066 |  |  | Hyperliquid involved but market context not found |
| AIXBT | HlPerp | BinPerp | 0.7606 | 279778.67 | 0.003225 | Hyperliquid context available |

Interpretation:

- The opportunity surface is much broader than Binance-only spot/perp carry.
- The largest spreads are mostly not BTC/ETH majors.
- Some large rows are not Hyperliquid-involved, so this overlay cannot validate
  their external venue liquidity.
- Several Hyperliquid-involved high-spread rows have low-to-moderate day
  notional volume, so size and slippage matter before this is actionable.
- This is highly execution-sensitive. The next validation must check venue
  access, fees, order book depth, open-interest caps, and position limits before
  treating any spread as actionable.
