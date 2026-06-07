# P0 Parallel Results

Generated on 2026-06-07 UTC.

## Status

This pass moves P0 from planning into parallel execution:

- data reachability checks across derivatives, funding, L2, and attention/liquidity
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

- Download and inspect Binance USD-M daily metrics schema.
- Download and inspect Binance premium index kline schema.
- Extend L2 burst from seconds to repeated scheduled snapshots.
- Pair Hyperliquid recent trades with each L2 snapshot.
- Convert the MANTA ticket into explicit fee and notional assumptions, then
  reject it if either venue leg is inaccessible.
