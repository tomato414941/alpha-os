# Market Making Results

Generated on 2026-06-07 UTC.

Run:

```bash
uv run python -m strategies.market_making.hyperliquid_l2_snapshot
uv run python -m strategies.market_making.hyperliquid_l2_snapshot --assets BTC ETH SOL HYPE WLD JTO ONDO AERO ZEC NEAR DOGE LTC --asset-source-path strategies/perp_market_map/current_hyperliquid_snapshot.csv --asset-source-top 20
uv run python -m strategies.market_making.current_l2_imbalance_monitor
uv run python -m strategies.market_making.current_l2_imbalance_forward_labels
uv run python -m strategies.market_making.current_l2_imbalance_paper_gate
```

Interpretation:

- low spread means spread capture is harder unless maker rebates or queue edge
  exist
- shallow 10 bps depth limits capacity
- imbalance can become a short-horizon signal only after repeated snapshots
- this snapshot does not model queue position, fill probability, adverse
  selection, or fees

## Snapshot

| asset | spread bps | bid depth 10 bps | ask depth 10 bps | imbalance 10 bps |
| --- | ---: | ---: | ---: | ---: |
| WLD | 1.2468 | 10196.0000 | 80275.1000 | -0.7746 |
| ZEC | 0.2400 | 74.0000 | 479.5200 | -0.7326 |
| HYPE | 0.1711 | 881.8600 | 3617.3400 | -0.6080 |
| SOL | 0.1537 | 3030.4700 | 12333.5100 | -0.6055 |
| BTC | 0.1607 | 146.7210 | 55.3017 | 0.4525 |
| LIT | 10.2078 | 1988.0000 | 4844.0000 | -0.4180 |
| LTC | 6.8984 | 615.1800 | 1448.2700 | -0.4037 |
| SUI | 1.7469 | 159108.5000 | 336703.4000 | -0.3582 |
| AVAX | 0.1495 | 5658.4700 | 11284.3300 | -0.3321 |
| JTO | 4.6830 | 12207.0000 | 6717.0000 | 0.2901 |

The broad snapshot now covers current candidates and volume-ranked perps, not
only BTC/ETH/SOL/HYPE. `WLD`, `ZEC`, `HYPE`, `SOL`, and `BTC` show the largest
absolute 10 bps imbalances in this snapshot. These are unlabeled until 15m/1h
outcomes mature.

## L2 Imbalance Monitor

This repeats the broad Hyperliquid L2 imbalance snapshot over a short window.
It is a persistence check, not a fill model or trade instruction.

| asset | obs | dir | persistence | mean imbalance | mean abs imbalance | min abs imbalance | spread bps | near depth USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 3 | 1 | 1.0000 | 0.7881 | 0.7881 | 0.6984 | 0.1609 | 1522952 |
| ONDO | 3 | 1 | 1.0000 | 0.5184 | 0.5184 | 0.3427 | 1.8113 | 4925 |
| XPL | 3 | 1 | 1.0000 | 0.4912 | 0.4912 | 0.2641 | 2.2276 | 6832 |
| VVV | 3 | -1 | 1.0000 | -0.4567 | 0.4567 | 0.3465 | 2.9777 | 2741 |
| SOL | 3 | -1 | 1.0000 | -0.3658 | 0.3658 | 0.2138 | 0.1539 | 357804 |
| LIT | 3 | 1 | 1.0000 | 0.3474 | 0.3474 | 0.2705 | 0.9622 | 4244 |
| XLM | 3 | 1 | 1.0000 | 0.3286 | 0.3286 | 0.1045 | 4.2616 | 12707 |
| SUI | 3 | 1 | 1.0000 | 0.1746 | 0.1746 | 0.0782 | 0.8057 | 69152 |
| DOGE | 3 | 1 | 1.0000 | 0.1317 | 0.1317 | 0.0065 | 0.1181 | 174884 |
| ETH | 3 | -1 | 1.0000 | -0.1189 | 0.1189 | 0.0809 | 0.6143 | 10114140 |

Interpretation:

- `BTC` is the cleanest persistent L2 imbalance candidate because persistence,
  imbalance magnitude, spread, and visible depth are all usable.
- `ONDO`, `XPL`, and `VVV` have strong persistent imbalance but shallow visible
  near-depth.
- `SOL` is a useful follow-up because it has persistent imbalance and materially
  better visible depth than the smaller names.
- `DOGE` and `ETH` are lower-imbalance controls with better depth, useful for
  checking whether the signal only appears in thin books.

## L2 Imbalance Forward Labels

This labels whether the visible 10 bps book imbalance matched subsequent
Hyperliquid price direction. It is an imbalance alpha probe, not a market-making
fill model.

| asset | spread bps | imbalance10 | dir | raw 15m | dir 15m | raw 1h | dir 1h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC | 0.1607 | 0.4525 | 1 |  |  |  |  |
| ETH | 0.6139 | 0.1352 | 1 |  |  |  |  |
| SOL | 0.1537 | -0.6055 | -1 |  |  |  |  |
| HYPE | 0.1711 | -0.6080 | -1 |  |  |  |  |
| WLD | 1.2468 | -0.7746 | -1 |  |  |  |  |
| JTO | 4.6830 | 0.2901 | 1 |  |  |  |  |
| ZEC | 0.2400 | -0.7326 | -1 |  |  |  |  |
| LTC | 6.8984 | -0.4037 | -1 |  |  |  |  |

Interpretation:

- The current broad snapshot is intentionally unlabeled because the 15m/1h
  horizons have not elapsed yet.
- `WLD`, `ZEC`, `HYPE`, `SOL`, and `BTC` are the immediate candidates to label
  after the horizons mature.

## L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance
directional label, then checks visible 10 bps depth. It is a directional paper
gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ETH | 100 | 0.1352 | 10.61 |  |  | 11976159 | 0.0000 | wait_for_label |
| BTC | 100 | 0.4525 | 10.16 |  |  | 3441674 | 0.0000 | wait_for_label |
| SOL | 100 | -0.6055 | 10.15 |  |  | 197152 | 0.0005 | wait_for_label |
| DOGE | 100 | -0.2872 | 12.95 |  |  | 175767 | 0.0006 | wait_for_label |
| HYPE | 100 | -0.6080 | 10.17 |  |  | 51561 | 0.0019 | wait_for_label |
| WLD | 100 | -0.7746 | 11.25 |  |  | 4906 | 0.0204 | wait_for_label |

Interpretation:

- The broad paper gate is waiting for 15m labels.
- `WLD` and `ZEC` have the strongest imbalance, but their visible depth and
  spreads are materially weaker than BTC/ETH/SOL.
- The next useful action is to rerun labels and the gate after 15m/1h outcomes
  mature.
