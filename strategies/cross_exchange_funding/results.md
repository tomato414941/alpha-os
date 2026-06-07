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

## Venue Access Probe

Run:

```bash
uv run python -m strategies.cross_exchange_funding.venue_access_probe
```

Current result from this environment:

| venue | endpoint | status | available | notes |
| --- | --- | ---: | --- | --- |
| Binance USD-M | exchangeInfo | 451 | False | location restricted |
| Bybit linear | instruments | 403 | False | blocked or permission required |
| OKX swap | instruments | 200 | True | reachable |
| Hyperliquid | metaAndAssetCtxs | 200 | True | reachable |

This changes the useful next step. Binance/Bybit rows can still be researched
from historical files or another environment, but the currently executable
public-data lane is OKX plus Hyperliquid.

## OKX-Hyperliquid Funding Spread

Run:

```bash
uv run python -m strategies.cross_exchange_funding.current_okx_hl_funding_spread
```

This screen directly joins:

- OKX USDT swap instruments
- OKX current funding rate
- OKX top-50 order book
- Hyperliquid predicted funding
- Hyperliquid day volume and impact spread

Top snapshot rows:

| asset | long venue | short venue | annualized spread | OKX spread | OKX bid notional | OKX ask notional | HL day volume | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| BABY | HlPerp | OkxSwap | 1.8987 | 0.000634 | 99528 | 85897 | 1827968 | OKX and Hyperliquid context available |
| AIXBT | HlPerp | OkxSwap | 0.7980 | 0.000459 | 67123 | 45576 | 237820 | OKX and Hyperliquid context available |
| SNX | HlPerp | OkxSwap | 0.7168 | 0.000412 | 70358 | 63387 | 258632 | OKX and Hyperliquid context available |
| JTO | HlPerp | OkxSwap | 0.5815 | 0.000156 | 80267 | 112713 | 4714457 | OKX and Hyperliquid context available |
| MERL | HlPerp | OkxSwap | 0.5483 | 0.000484 | 76976 | 70714 | 239431 | OKX and Hyperliquid context available |
| MON | OkxSwap | HlPerp | 0.5392 | 0.000435 | 246046 | 200044 | 2427384 | OKX and Hyperliquid context available |
| MEME | HlPerp | OkxSwap | 0.4963 | 0.000158 | 48657 | 35064 | 594655 | OKX and Hyperliquid context available |
| AZTEC | OkxSwap | HlPerp | 0.4807 | 0.001235 | 100262 | 91263 | 235695 | OKX and Hyperliquid context available |
| TURBO | HlPerp | OkxSwap | 0.4436 | 0.000346 | 47965 | 23804 | 90325 | low Hyperliquid day volume |
| IOTA | OkxSwap | HlPerp | 0.4143 | 0.000432 | 45074 | 43388 | 92683 | low Hyperliquid day volume; wide Hyperliquid impact spread |

Interpretation:

- This is more actionable than the previous venue-agnostic spread table because
  both public APIs are reachable from the current environment.
- The top rows are mostly small/mid-cap perps, not majors.
- The largest spread, `BABY`, still has only about 1.8M Hyperliquid day volume
  and sub-100k top-50 OKX book notional, so size must be small until execution
  is measured.
- `TURBO` and `IOTA` show why this cannot be spread-only: funding can be large
  while one venue is too thin or impact-sensitive.
