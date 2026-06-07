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
- rough round-trip cost proxy: OKX spread plus Hyperliquid impact spread
- break-even holding hours and 8h/24h net funding proxy

Rows are ranked by `net_8h_proxy`, not raw annualized spread. This intentionally
penalizes thin books and wide impact before a candidate reaches the top.

Top cost-aware rows:

| asset | long venue | short venue | annualized spread | rough cost | breakeven hours | net 8h | net 24h | capacity proxy | notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BTC | OkxSwap | HlPerp | 0.1333 | 0.00001754 | 1.15 | 0.00010420 | 0.00034767 | 1018953 | OKX and Hyperliquid context available |
| SOL | HlPerp | OkxSwap | 0.1434 | 0.00017073 | 10.43 | -0.00003977 | 0.00022213 | 3031408 | OKX and Hyperliquid context available |
| HYPE | OkxSwap | HlPerp | 0.1891 | 0.00026750 | 12.39 | -0.00009484 | 0.00025048 | 3077060 | OKX and Hyperliquid context available |
| BNB | HlPerp | OkxSwap | 0.0409 | 0.00027963 | 59.83 | -0.00024224 | -0.00016746 | 65544 | OKX and Hyperliquid context available |
| DOGE | OkxSwap | HlPerp | 0.0545 | 0.00031905 | 51.31 | -0.00026931 | -0.00016981 | 93457 | OKX and Hyperliquid context available |
| AI | HlPerp | OkxSwap | 0.1095 | 0.00042909 | 34.33 | -0.00032909 | -0.00012909 | 0 | missing Hyperliquid context |
| TRX | OkxSwap | HlPerp | 0.1548 | 0.00049615 | 28.07 | -0.00035477 | -0.00007200 | 47141 | OKX and Hyperliquid context available |
| LINK | OkxSwap | HlPerp | 0.0998 | 0.00048028 | 42.16 | -0.00038914 | -0.00020687 | 36479 | OKX and Hyperliquid context available |
| SUI | OkxSwap | HlPerp | 0.0377 | 0.00043781 | 101.72 | -0.00040338 | -0.00033451 | 315086 | OKX and Hyperliquid context available |
| AVAX | OkxSwap | HlPerp | 0.0116 | 0.00041535 | 313.81 | -0.00040476 | -0.00038358 | 97783 | OKX and Hyperliquid context available |

Interpretation:

- This is more actionable than the previous venue-agnostic spread table because
  both public APIs are reachable from the current environment.
- Raw annualized spread is misleading. After rough spread/impact cost, the
  small-cap high-spread rows drop below majors.
- `BTC` is the only current top-row candidate with positive 8h net proxy.
- `SOL` and `HYPE` become plausible only if the position can be held long enough
  for the funding spread to amortize entry and exit cost.
- The next test should measure persistence across repeated snapshots. A
  one-shot spread is not enough; the edge must survive time, fees, and actual
  order placement constraints.

## OKX-Hyperliquid Short Persistence Probe

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_persistence_probe
```

Sample:

- snapshots: 3
- delay: 5 seconds
- raw rows: 360
- summarized assets: 124
- ranking: positive 8h net rate, then mean 8h net proxy

Top summary rows:

| asset | long venue | short venue | observations | positive 8h rate | mean net 8h | mean net 24h | breakeven hours | capacity proxy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | OkxSwap | HlPerp | 3 | 1.0000 | 0.00009284 | 0.00033603 | 1.89 | 764337 |
| SOL | HlPerp | OkxSwap | 3 | 0.0000 | -0.00004338 | 0.00020844 | 10.76 | 3010720 |
| DOGE | OkxSwap | HlPerp | 3 | 0.0000 | -0.00016393 | -0.00005786 | 32.71 | 93433 |
| HYPE | OkxSwap | HlPerp | 3 | 0.0000 | -0.00018692 | 0.00015572 | 16.73 | 3282504 |
| TRX | OkxSwap | HlPerp | 3 | 0.0000 | -0.00022865 | 0.00000397 | 23.73 | 47164 |
| BNB | HlPerp | OkxSwap | 3 | 0.0000 | -0.00030628 | -0.00023572 | 77.60 | 65972 |
| AI | HlPerp | OkxSwap | 3 | 0.0000 | -0.00032977 | -0.00012977 | 34.38 | 0 |
| SUI | OkxSwap | HlPerp | 3 | 0.0000 | -0.00037528 | -0.00030927 | 98.99 | 315611 |
| LINK | OkxSwap | HlPerp | 3 | 0.0000 | -0.00037828 | -0.00019732 | 41.45 | 36484 |
| ZEC | OkxSwap | HlPerp | 3 | 0.0000 | -0.00040808 | -0.00017645 | 36.13 | 83556 |

Interpretation:

- The short persistence check keeps `BTC` as the only clearly positive 8h
  candidate in this snapshot window.
- `SOL` and `HYPE` still matter because their 24h proxy remains positive, but
  they require holding through cost amortization rather than a quick funding
  capture.
- The next real test should either collect this probe over a longer schedule or
  make a tiny paper-ticket with explicit OKX and Hyperliquid order assumptions
  for `BTC`.

## OKX-Hyperliquid 1m Persistence Probe

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_persistence_probe \
  --samples 6 \
  --delay-seconds 10 \
  --output-path strategies/cross_exchange_funding/okx_hl_funding_persistence_1m.csv \
  --summary-output-path strategies/cross_exchange_funding/okx_hl_funding_persistence_1m_summary.csv
```

Sample:

- snapshots: 6
- delay: 10 seconds
- raw rows: 720
- summarized assets: 128

Top summary rows:

| asset | long venue | short venue | observations | positive 8h rate | mean net 8h | mean net 24h | breakeven hours | capacity proxy |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | OkxSwap | HlPerp | 6 | 0.5000 | 0.00000433 | 0.00023519 | 7.70 | 493421 |
| SOL | HlPerp | OkxSwap | 6 | 0.0000 | -0.00017195 | 0.00005322 | 20.03 | 3123576 |
| ZEC | OkxSwap | HlPerp | 6 | 0.0000 | -0.00023688 | 0.00039437 | 14.03 | 112828 |
| HYPE | OkxSwap | HlPerp | 6 | 0.0000 | -0.00032010 | -0.00000988 | 24.50 | 3679499 |
| DOGE | OkxSwap | HlPerp | 6 | 0.0000 | -0.00032755 | -0.00011635 | 32.72 | 98216 |
| AI | HlPerp | OkxSwap | 6 | 0.0000 | -0.00033113 | -0.00013113 | 34.49 | 0 |
| BNB | OkxSwap | HlPerp | 6 | 0.0000 | -0.00050133 | -0.00040289 | 90.30 | 66012 |
| AVAX | HlPerp | OkxSwap | 6 | 0.0000 | -0.00052427 | -0.00051019 | 689.07 | 97000 |
| LINK | OkxSwap | HlPerp | 6 | 0.0000 | -0.00057942 | -0.00039877 | 59.29 | 37261 |
| MEW | HlPerp | OkxSwap | 6 | 0.0000 | -0.00064465 | -0.00044465 | 59.57 | 0 |

Interpretation:

- The longer short-window check weakens the BTC 8h case. BTC remains the top
  candidate, but positive 8h rate falls from `1.0000` in the 3-snapshot probe
  to `0.5000`.
- BTC 24h proxy remains positive, but the 8h version is too thin to trust
  without much stronger fee and maker assumptions.
- `ZEC` appears interesting on 24h proxy, but it does not survive 8h and has
  lower capacity than BTC. It should be monitored, not promoted yet.
- This pushes the lane toward longer scheduled monitoring and fee/maker
  validation, not immediate execution.

## OKX-Hyperliquid Candidate Score

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_score
```

This ranks all assets from the 1m persistence summary after simple fee
assumptions.

Top fee-adjusted rows:

| asset | scenario | long venue | short venue | observations | net 8h after fee | net 24h after fee | capacity proxy | survives 8h | survives 24h |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| JTO | very_low_fee | OkxSwap | HlPerp | 6 | -0.00146767 | 0.0004448 | 51289 | False | True |
| BABY | very_low_fee | HlPerp | OkxSwap | 6 | -0.00212432 | 0.00038917 | 18227 | False | True |
| JTO | low_fee | OkxSwap | HlPerp | 6 | -0.00158767 | 0.0003248 | 51289 | False | True |
| ZEC | very_low_fee | OkxSwap | HlPerp | 6 | -0.00031688 | 0.00031437 | 112828 | False | True |
| BABY | low_fee | HlPerp | OkxSwap | 6 | -0.00224432 | 0.00026917 | 18227 | False | True |
| ZEC | low_fee | OkxSwap | HlPerp | 6 | -0.00043688 | 0.00019437 | 112828 | False | True |
| BTC | very_low_fee | OkxSwap | HlPerp | 6 | -0.00007567 | 0.00015519 | 493421 | False | True |
| JTO | one_bps_each | OkxSwap | HlPerp | 6 | -0.00178767 | 0.0001248 | 51289 | False | True |
| BABY | one_bps_each | HlPerp | OkxSwap | 6 | -0.00244432 | 0.00006917 | 18227 | False | True |
| TIA | very_low_fee | OkxSwap | HlPerp | 6 | -0.00107793 | 0.00003533 | 7006 | False | True |
| BTC | low_fee | OkxSwap | HlPerp | 6 | -0.00019567 | 0.00003519 | 493421 | False | True |

Interpretation:

- No asset survives the 8h version after the tested fee assumptions in the 1m
  sample.
- The 24h monitor list broadens beyond BTC. `JTO`, `BABY`, and `ZEC` rise above
  BTC on fee-adjusted 24h proxy, but they have lower capacity and still depend
  on very favorable execution.
- Under one bps per fill on both venues, only `JTO` and `BABY` remain positive
  in this short sample, and both are too thin to promote without longer
  monitoring.
- The lane now has two classes: BTC as the deepest operational candidate, and
  JTO/BABY/ZEC as higher-spread monitoring candidates.

## OKX-Hyperliquid Paper Ticket

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_paper_ticket
```

Selected candidate:

- asset: `BTC`
- long venue: `OkxSwap`
- short venue: `HlPerp`
- persistence observations: `3`
- positive 8h net rate: `1.0000`
- mean 8h net proxy: `0.00009284`
- min 8h net proxy: `0.00007061`
- mean 24h net proxy: `0.00033603`
- mean breakeven holding time: `1.8913` hours
- mean capacity proxy notional: `764336.86`
- paper notional cap: `1000.00` USDT

This is the first OKX-Hyperliquid candidate that has moved from screen output
to a venue-specific paper workflow. It is still not a trade instruction. The
next falsification is order-level: exact instrument IDs, lot sizes, fee tier,
funding timestamp alignment, taker-vs-maker assumption, collateral path, and
kill switch.

## OKX-Hyperliquid Order Constraints

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_order_constraints
```

Paper size:

- asset: `BTC`
- paper notional: `1000` USDT
- long venue: `OkxSwap`
- short venue: `HlPerp`

Public instrument constraints:

| venue | instrument | rounded size | rounded notional | min / lot / decimals | max leverage | valid |
| --- | --- | ---: | ---: | --- | ---: | --- |
| OKX | BTC-USDT-SWAP | 1.59 contracts | 995.58645 USDT | min 0.01, lot 0.01, tick 0.1 | 100 | True |
| Hyperliquid | BTC | 0.01597 BTC | 999.969535 USDT | size decimals 5 | 40 | True |

Interpretation:

- Public constraints allow the 1000 USDT paper order shape on both venues.
- The two rounded notionals are not exactly equal, so a paper workflow should
  track residual delta from rounding.
- Fee tier, actual account access, margin mode, collateral movement, maker/taker
  feasibility, and funding timestamp alignment remain unresolved. These are now
  the main blockers before any real order workflow.

## OKX-Hyperliquid Funding Alignment

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_alignment
```

Current BTC alignment:

- OKX instrument: `BTC-USDT-SWAP`
- OKX funding rate: `-0.000017878722904`
- OKX long expected rate per event: `0.000017878722904`
- OKX first funding time: `2026-06-07T16:00:00+00:00`
- OKX interval: `8h`
- Hyperliquid funding rate: `0.0000125`
- Hyperliquid short expected rate per event: `0.0000125`
- Hyperliquid first funding time: `2026-06-07T13:00:00+00:00`
- Hyperliquid interval: `1h`
- first event gap: `3h`
- events within 8h: OKX `1`, Hyperliquid `8`
- events within 24h: OKX `3`, Hyperliquid `24`

Interpretation:

- The current funding signs match the paper direction: long OKX and short
  Hyperliquid both expect funding income.
- Hyperliquid funds hourly while OKX funds every 8h, so this is not a
  one-for-one event match. The paper workflow must track accumulated funding by
  event schedule, not only a single spread number.
- The remaining uncertainty is persistence: current signs can flip before each
  funding event, so this still needs a longer scheduled monitor before real
  execution is considered.

## OKX-Hyperliquid Fee Sensitivity

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_fee_sensitivity
```

This is not a fee schedule. It checks how much room the BTC paper ticket has
under simple per-fill fee assumptions. Assumption: one entry and one exit on
each venue.

| scenario | round-trip fee rate | 8h after fee | 8h USDT | 24h after fee | 24h USDT | survives 8h | survives 24h |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| very_low_fee | 0.00008 | 0.00001284 | 0.01284 | 0.00025603 | 0.25603 | True | True |
| low_fee | 0.0002 | -0.00010716 | -0.10716 | 0.00013603 | 0.13603 | False | True |
| one_bps_each | 0.0004 | -0.00030716 | -0.30716 | -0.00006397 | -0.06397 | False | False |
| two_bps_each | 0.0008 | -0.00070716 | -0.70716 | -0.00046397 | -0.46397 | False | False |
| five_bps_each | 0.002 | -0.00190716 | -1.90716 | -0.00166397 | -1.66397 | False | False |

Interpretation:

- The BTC ticket is extremely fee-sensitive.
- The 8h version only survives under very low effective execution cost.
- The 24h version survives under the low-fee scenario, but fails at one bps per
  fill on each venue.
- The next highest-value work is not another raw spread screen. It is maker
  feasibility, real fee tier confirmation, and longer scheduled persistence.

## OKX-Hyperliquid Book Depth

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_book_depth
```

Paper taker-depth check:

| venue | side | target notional | top level notional | average fill | slippage bps | levels | full |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| OkxSwap | buy | 995.58645 | 275099.24052 | 62279.1 | 0.00802838 | 1 | True |
| HlPerp | sell | 999.969535 | 288224.46108 | 62292 | 0.08026713 | 1 | True |

- combined visible taker slippage: `0.08829551` bps
- both legs fit inside the top visible level at the 1000 USDT paper size

Interpretation:

- Paper size is not blocked by visible book depth.
- This does not solve fee sensitivity. Even if visible slippage is small, the
  edge can still disappear under normal taker fees.
- The current bottleneck remains effective execution cost and persistence, not
  visible BTC depth at 1000 USDT.

## OKX-Hyperliquid Focused Candidate Monitor

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_funding_persistence_probe --samples 12 --delay-seconds 10 --assets BTC JTO BABY ZEC --output-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus.csv --summary-output-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus_summary.csv
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_score --summary-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus_summary.csv --output-path strategies/cross_exchange_funding/okx_hl_candidate_score_focus.csv --md-output-path strategies/cross_exchange_funding/okx_hl_candidate_score_focus.md
```

Focused persistence summary:

| asset | long | short | obs | positive 8h rate | mean net 8h | mean net 24h | breakeven hours | capacity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | OkxSwap | HlPerp | 12 | 0.83333333 | 0.00003362 | 0.00026010 | 5.6232 | 422448.80855333 |
| ZEC | OkxSwap | HlPerp | 12 | 0 | -0.00031009 | 0.00034900 | 15.5275 | 106210.05564167 |
| JTO | OkxSwap | HlPerp | 12 | 0 | -0.00070625 | 0.00068005 | 16.1661 | 54543.56750198 |
| BABY | HlPerp | OkxSwap | 12 | 0 | -0.00380683 | -0.00118648 | 31.2663 | 18182.63949194 |

Fee-adjusted focused score:

| asset | scenario | net 8h after fee | net 24h after fee | capacity | survives 24h |
| --- | --- | ---: | ---: | ---: | --- |
| JTO | very_low_fee | -0.00078625 | 0.00060005 | 54543.56750198 | True |
| JTO | low_fee | -0.00090625 | 0.00048005 | 54543.56750198 | True |
| JTO | one_bps_each | -0.00110625 | 0.00028005 | 54543.56750198 | True |
| ZEC | very_low_fee | -0.00039009 | 0.000269 | 106210.05564167 | True |
| BTC | very_low_fee | -0.00004638 | 0.0001801 | 422448.80855333 | True |
| ZEC | low_fee | -0.00051009 | 0.000149 | 106210.05564167 | True |
| BTC | low_fee | -0.00016638 | 0.0000601 | 422448.80855333 | True |

Interpretation:

- BTC is the cleanest operational candidate: deeper capacity and 8h proxy mostly
  positive in this short focused sample. It is still too fee-sensitive to trade
  without real fee-tier and maker-fill evidence.
- JTO has the strongest 24h fee-adjusted proxy, even under the one-bps-each
  scenario, but capacity is small. It is a monitoring candidate, not yet a
  scalable baseline.
- ZEC is between BTC and JTO: stronger 24h room than BTC under low fees, but it
  does not survive one bps each and has less capacity.
- BABY fell out of the focused sample. The earlier 1m screen was not enough to
  keep it in the active candidate set.

## OKX-Hyperliquid Execution Cost Score

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_execution_cost_score --summary-path strategies/cross_exchange_funding/okx_hl_funding_persistence_focus_summary.csv --assets BTC JTO ZEC BABY
```

This score starts from gross funding edge and subtracts current public top-book
taker slippage plus simple round-trip fee scenarios. It is a harder execution
check than the raw funding spread, but it still does not prove maker fills,
account fee tier, or persistence into the actual funding events.

| asset | scenario | gross 8h | gross 24h | entry slippage bps | all-in cost | net 8h | net 24h | capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BABY | very_low_fee | 0.00131017 | 0.00393051 | 12.94997144 | 0.00266999 | -0.00135982 | 0.00126052 | 18182.63949194 |
| BABY | low_fee | 0.00131017 | 0.00393051 | 12.94997144 | 0.00278999 | -0.00147982 | 0.00114052 | 18182.63949194 |
| BABY | one_bps_each | 0.00131017 | 0.00393051 | 12.94997144 | 0.00298999 | -0.00167982 | 0.00094052 | 18182.63949194 |
| JTO | very_low_fee | 0.00069315 | 0.00207945 | 7.52849755 | 0.00158570 | -0.00089255 | 0.00049376 | 54543.56750198 |
| JTO | low_fee | 0.00069315 | 0.00207945 | 7.52849755 | 0.00170570 | -0.00101255 | 0.00037376 | 54543.56750198 |
| ZEC | very_low_fee | 0.00032955 | 0.00098864 | 3.14430447 | 0.00070886 | -0.00037931 | 0.00027978 | 106210.05564167 |
| BTC | very_low_fee | 0.00011324 | 0.00033973 | 0.08887218 | 0.00009777 | 0.00001547 | 0.00024195 | 422448.80855333 |
| JTO | one_bps_each | 0.00069315 | 0.00207945 | 7.52849755 | 0.00190570 | -0.00121255 | 0.00017376 | 54543.56750198 |
| ZEC | low_fee | 0.00032955 | 0.00098864 | 3.14430447 | 0.00082886 | -0.00049931 | 0.00015978 | 106210.05564167 |
| BTC | low_fee | 0.00011324 | 0.00033973 | 0.08887218 | 0.00021777 | -0.00010453 | 0.00012195 | 422448.80855333 |

Interpretation:

- BTC remains the cleanest small live candidate: tiny edge, but low current
  slippage and the most capacity among this set.
- JTO still has real 24h room after current taker-depth cost, but it is not a
  scalable candidate yet because capacity is small and 8h is negative.
- BABY reappears under the direct top-book execution-cost check, despite falling
  out under the earlier rough impact proxy. That disagreement is not a buy
  signal; it says execution-cost measurement is now a core part of the research.
- No focused candidate except BTC under very-low-fee assumptions survives the
  8h all-in check. The current alpha is therefore a longer-hold funding-carry
  candidate, not an immediate 8h arbitrage.

## OKX-Hyperliquid Candidate Triage

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_candidate_triage
```

This turns the execution-cost score into research actions. It is not a trade
instruction. This is a smooth-cost triage; use the event-window triage below
when the two disagree.

| asset | action | long | short | obs | capacity | very-low 8h | low-fee 24h | one-bps 24h | max slippage bps |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | paper_8h_candidate | OkxSwap | HlPerp | 12 | 422448.80855333 | 0.00001547 | 0.00012195 | -0.00007805 | 0.08887218 |
| JTO | active_24h_monitor | OkxSwap | HlPerp | 12 | 54543.56750198 | -0.00089255 | 0.00037376 | 0.00017376 | 7.52849755 |
| ZEC | fee_dependent_24h_monitor | OkxSwap | HlPerp | 12 | 106210.05564167 | -0.00037931 | 0.00015978 | -0.00004022 | 3.14430447 |
| BABY | thin_or_unstable_watch | HlPerp | OkxSwap | 12 | 18182.63949194 | -0.00135982 | 0.00114052 | 0.00094052 | 12.94997144 |

Interpretation:

- BTC is the only current 8h paper candidate, and only under very-low-fee
  assumptions. The next proof needed is account-specific fee and maker-fill
  evidence, not another spread table.
- JTO deserves active 24h monitoring because it survives the one-bps-each
  all-in check, but it is capacity-limited.
- ZEC is still useful as a fee-dependent 24h candidate.
- BABY is not promoted despite high 24h score because capacity is low and
  slippage is high. Keep it as a watch item, not an active candidate.

## OKX-Hyperliquid Event Window Score

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_score
```

This score uses the actual current funding event counts inside 8h and 24h
windows. It should override the smooth hourly proxy when deciding which
candidates deserve monitoring.

| asset | action | scenario | long | short | OKX 8h | HL 8h | OKX 24h | HL 24h | gross 8h | gross 24h | cost | net 8h | net 24h | capacity |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | paper_8h_candidate | very_low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00010582 | 0.00031747 | 0.00009777 | 0.00000805 | 0.00021970 | 422448.80855333 |
| BTC | paper_8h_candidate | low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00010582 | 0.00031747 | 0.00021777 | -0.00011195 | 0.00009970 | 422448.80855333 |
| ZEC | fee_dependent_24h_monitor | very_low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00024083 | 0.00072249 | 0.00070886 | -0.00046803 | 0.00001363 | 106210.05564167 |
| BTC | paper_8h_candidate | one_bps_each | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00010582 | 0.00031747 | 0.00041777 | -0.00031195 | -0.00010030 | 422448.80855333 |
| ZEC | fee_dependent_24h_monitor | low_fee | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00024083 | 0.00072249 | 0.00082886 | -0.00058803 | -0.00010637 | 106210.05564167 |
| ZEC | fee_dependent_24h_monitor | one_bps_each | OkxSwap | HlPerp | 1 | 8 | 3 | 24 | 0.00024083 | 0.00072249 | 0.00102886 | -0.00078803 | -0.00030637 | 106210.05564167 |
| JTO | active_24h_monitor | very_low_fee | OkxSwap | HlPerp | 2 | 8 | 6 | 24 | 0.00005877 | 0.00017631 | 0.00158570 | -0.00152693 | -0.00140939 | 54543.56750198 |
| BABY | thin_or_unstable_watch | very_low_fee | HlPerp | OkxSwap | 2 | 8 | 6 | 24 | 0.00036862 | 0.00110587 | 0.00266999 | -0.00230137 | -0.00156412 | 18182.63949194 |

Interpretation:

- BTC remains the only current 8h paper candidate, and only under very-low-fee
  assumptions.
- JTO should be downgraded despite the earlier smooth proxy. Once actual event
  counts are used, it is negative even under very-low-fee assumptions.
- ZEC is barely positive only under very-low fees in this event-window snapshot.
- BABY is negative after event-window cost in this refreshed snapshot.
- The practical next step is no longer broad ranking. It is fee/maker-fill
  evidence for BTC and longer event-window monitoring for BTC/ZEC/BABY.

## OKX-Hyperliquid Event Window Triage

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_triage
```

This turns event-window scores into research actions. It should override the
smooth execution-cost triage when the two disagree.

| asset | event action | previous action | long | short | capacity | very-low 8h | very-low 24h | low-fee 24h | one-bps 24h | max slippage bps |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | paper_8h_candidate | paper_8h_candidate | OkxSwap | HlPerp | 422448.80855333 | 0.00000805 | 0.00021970 | 0.00009970 | -0.00010030 | 0.08887218 |
| ZEC | very_low_fee_24h_watch | fee_dependent_24h_monitor | OkxSwap | HlPerp | 106210.05564167 | -0.00046803 | 0.00001363 | -0.00010637 | -0.00030637 | 3.14430447 |
| JTO | drop_for_now | active_24h_monitor | OkxSwap | HlPerp | 54543.56750198 | -0.00152693 | -0.00140939 | -0.00152939 | -0.00172939 | 7.52849755 |
| BABY | drop_for_now | thin_or_unstable_watch | HlPerp | OkxSwap | 18182.63949194 | -0.00230137 | -0.00156412 | -0.00168412 | -0.00188412 | 12.94997144 |

Interpretation:

- BTC is still the only 8h paper candidate, but it remains fee-sensitive.
- ZEC is downgraded from fee-dependent 24h monitor to very-low-fee watch.
- BABY is dropped in the refreshed event-window snapshot.
- JTO is dropped for now. The smooth proxy promoted it, but actual funding
  events make the current window negative.

## OKX-Hyperliquid Event Window Monitor

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_event_window_monitor --samples 6 --delay-seconds 10
```

This repeats event-window triage to check whether the current candidate
classification is stable. It is not a trade instruction.

| asset | obs | dominant action | paper 8h rate | active 24h rate | watch rate | drop rate | mean very-low 8h | mean low-fee 24h | mean one-bps 24h | capacity |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 6 | paper_8h_candidate | 1.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00001140 | 0.00010974 | -0.00009026 | 422448.80855333 |
| ZEC | 6 | fee_dependent_24h_monitor | 0.00000000 | 0.00000000 | 1.00000000 | 0.00000000 | -0.00041023 | 0.00006702 | -0.00013298 | 106210.05564167 |
| BABY | 6 | drop_for_now | 0.00000000 | 0.00000000 | 0.00000000 | 1.00000000 | -0.00187223 | -0.00039670 | -0.00059670 | 18182.63949194 |
| JTO | 6 | drop_for_now | 0.00000000 | 0.00000000 | 0.00000000 | 1.00000000 | -0.00149631 | -0.00143752 | -0.00163752 | 54543.56750198 |

Interpretation:

- BTC is stable as the only 8h paper candidate across this short monitor.
- ZEC improved from the one-shot very-low-fee watch into a stable low-fee 24h
  watch during the monitor, but it still does not survive one-bps-each.
- BABY and JTO are both stable drops in this run.
- The next actionable proof remains account-specific: BTC fee/maker-fill
  evidence first, then longer scheduled monitoring for BTC and ZEC.

## OKX-Hyperliquid Maker Touch Probe

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_maker_touch_probe --assets BTC ZEC --samples 6 --delay-seconds 10
```

This is a public-book proxy for maker feasibility. It places a virtual quote at
the current best bid for buy legs or best ask for sell legs, then checks whether
the next sampled opposite quote would cross it. It does not prove queue position
or real fills.

| asset | venue | side | obs | touch rate | mean maker edge bps | min edge bps | max edge bps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| ZEC | OkxSwap | buy | 5 | 0.80000000 | 0.12224977 | 0.12205988 | 0.12232865 |
| ZEC | HlPerp | sell | 5 | 0.40000000 | 0.65889862 | 0.12219113 | 2.56125672 |
| BTC | OkxSwap | buy | 5 | 0.60000000 | 0.00807768 | 0.00807012 | 0.00808302 |
| BTC | HlPerp | sell | 5 | 0.20000000 | 0.14529950 | 0.08073175 | 0.32266391 |

Interpretation:

- In this sample, OKX buy touched more often than Hyperliquid sell on both BTC
  and ZEC.
- The maker edge is still tiny on BTC, especially OKX buy at roughly 0.008 bps.
  If real account fees are not very low, waiting as maker may not be enough.
- This is still not fill evidence. Queue position, post-only behavior, and
  account-specific fees remain unresolved.

## OKX-Hyperliquid Maker Touch Pair Summary

This pairs OKX and Hyperliquid maker-touch observations by asset and sample
window. Both legs must touch in the same window for a clean maker-maker entry
proxy.

| asset | obs | both touch rate | either touch rate | OKX only | HL only | no touch | mean OKX edge bps | mean HL edge bps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | 5 | 0.20000000 | 1.00000000 | 0.60000000 | 0.20000000 | 0.00000000 | 0.12224977 | 0.65889862 |
| BTC | 5 | 0.00000000 | 0.80000000 | 0.60000000 | 0.20000000 | 0.20000000 | 0.00807768 | 0.14529950 |

Interpretation:

- BTC had zero same-window maker-maker touches in this short sample. Even though
  one leg often touched, clean two-leg maker entry did not appear.
- ZEC had a 20% both-touch rate, but its event-window edge is weaker and
  fee-dependent.
- The next execution proof should test longer windows and one-leg-cross /
  one-leg-maker variants, not only pure maker-maker entry.

## OKX-Hyperliquid Execution Mode Score

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_execution_mode_score --assets BTC ZEC
```

This compares maker/cross execution modes against the event-window funding edge.
Maker rebates and real queue position are not modeled.

| asset | scenario | mode | gross 8h | gross 24h | entry slippage bps | cost | net 8h | net 24h | both touch | OKX only | HL only | capacity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | very_low_fee | both_maker | 0.00024083 | 0.00072249 | 0 | 0.00008000 | 0.00016083 | 0.00064249 | 0.20000000 | 0.60000000 | 0.20000000 | 106210.05564167 |
| ZEC | very_low_fee | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12340956 | 0.00010468 | 0.00013615 | 0.00061781 | 0.20000000 | 0.60000000 | 0.20000000 | 106210.05564167 |
| ZEC | low_fee | both_maker | 0.00024083 | 0.00072249 | 0 | 0.00020000 | 0.00004083 | 0.00052249 | 0.20000000 | 0.60000000 | 0.20000000 | 106210.05564167 |
| ZEC | low_fee | okx_cross_hl_maker | 0.00024083 | 0.00072249 | 0.12340956 | 0.00022468 | 0.00001615 | 0.00049781 | 0.20000000 | 0.60000000 | 0.20000000 | 106210.05564167 |
| BTC | very_low_fee | both_maker | 0.00010582 | 0.00031747 | 0 | 0.00008000 | 0.00002582 | 0.00023747 | 0 | 0.60000000 | 0.20000000 | 422448.80855333 |
| BTC | very_low_fee | okx_cross_hl_maker | 0.00010582 | 0.00031747 | 0.00809646 | 0.00008162 | 0.00002420 | 0.00023585 | 0 | 0.60000000 | 0.20000000 | 422448.80855333 |
| BTC | low_fee | both_maker | 0.00010582 | 0.00031747 | 0 | 0.00020000 | -0.00009418 | 0.00011747 | 0 | 0.60000000 | 0.20000000 | 422448.80855333 |
| BTC | low_fee | okx_cross_hl_maker | 0.00010582 | 0.00031747 | 0.00809646 | 0.00020162 | -0.00009580 | 0.00011585 | 0 | 0.60000000 | 0.20000000 | 422448.80855333 |

Interpretation:

- BTC still has the cleanest capacity, but its 8h edge survives only under
  very-low fees. `okx_cross_hl_maker` barely changes the score because OKX
  taker slippage is tiny in this sample.
- BTC pure maker-maker remains execution-fragile: pair both-touch was 0% in the
  recent maker-touch probe.
- ZEC has stronger 24h room and, in this refreshed current-book score, can
  tolerate `okx_cross_hl_maker` much better than `okx_maker_hl_cross`.
- For both BTC and ZEC, crossing the OKX leg while waiting as maker on
  Hyperliquid is the least damaging one-leg-cross mode in this sample.

## OKX-Hyperliquid Fee Ceiling

Run:

```bash
uv run python -m strategies.cross_exchange_funding.okx_hl_fee_ceiling
```

This estimates the maximum equal per-fill fee bps each venue can charge before
the event-window edge is erased. It uses the execution-mode slippage already
measured from the public book.

| asset | mode | max fee 8h bps/fill/venue | max fee 24h bps/fill/venue | both touch | OKX only | HL only | capacity |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ZEC | both_maker | 0.602075 | 1.806225 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | okx_cross_hl_maker | 0.540375 | 1.744525 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | okx_maker_hl_cross | 0.108525 | 1.312675 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| ZEC | both_cross | 0.0468 | 1.25095 | 0.2 | 0.6 | 0.2 | 106210.05564167 |
| BTC | both_maker | 0.26455 | 0.793675 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | okx_cross_hl_maker | 0.2605 | 0.789625 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | okx_maker_hl_cross | 0.224075 | 0.7532 | 0 | 0.6 | 0.2 | 422448.80855333 |
| BTC | both_cross | 0.220025 | 0.74915 | 0 | 0.6 | 0.2 | 422448.80855333 |

Interpretation:

- BTC 8h is extremely fee-sensitive: it needs roughly 0.26 bps or less per
  fill per venue even before queue-position and adverse-selection risk.
- BTC 24h has more room at roughly 0.79 bps per fill per venue, but its funding
  event cadence is slower.
- ZEC 24h has the largest fee headroom in this snapshot, including
  `okx_cross_hl_maker`, but its capacity and event stability are weaker than
  BTC.
- The next hard gate is the real account fee tier. Without that, raw funding
  spread is not enough to promote a mode.
