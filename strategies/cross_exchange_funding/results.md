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
| STABLE | BybitPerp | BinPerp | 0.00033756 | 2.9570 |
| MANTA | BinPerp | HlPerp | 0.00022771 | 1.9948 |
| ORDI | BybitPerp | HlPerp | 0.00017671 | 1.5480 |
| NIL | HlPerp | BybitPerp | 0.00016369 | 1.4339 |
| BABY | HlPerp | BybitPerp | 0.00012083 | 1.0585 |
| BSV | HlPerp | BinPerp | 0.00012056 | 1.0561 |
| UMA | BybitPerp | BinPerp | 0.00011405 | 0.9991 |
| ZORA | HlPerp | BinPerp | 0.00010043 | 0.8798 |
| USTC | BybitPerp | HlPerp | 0.00009912 | 0.8683 |
| AIXBT | HlPerp | BinPerp | 0.00009287 | 0.8136 |

Interpretation:

- The opportunity surface is much broader than Binance-only spot/perp carry.
- The largest spreads are mostly not BTC/ETH majors.
- This is highly execution-sensitive. The next validation must check venue
  access, fees, order book depth, open-interest caps, and position limits before
  treating any spread as actionable.
