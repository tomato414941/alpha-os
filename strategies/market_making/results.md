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
| JTO | 4.6830 | 0.2901 | 1 | 0.012475 | 0.012475 |  |  |
| XLM | 2.4802 | 0.1395 | 1 | 0.012187 | 0.012187 |  |  |
| NEAR | 1.9243 | -0.2800 | -1 | -0.011584 | 0.011584 |  |  |
| XPL | 2.1805 | 0.0082 | 1 | 0.003014 | 0.003014 |  |  |
| BNB | 1.1832 | 0.0280 | 1 | 0.002262 | 0.002262 |  |  |
| ADA | 3.0814 | 0.1401 | 1 | 0.001540 | 0.001540 |  |  |
| ETH | 0.6139 | 0.1352 | 1 | 0.001045 | 0.001045 |  |  |
| AERO | 7.8666 | 0.0632 | 1 | 0.000636 | 0.000636 |  |  |
| BTC | 0.1607 | 0.4525 | 1 | 0.000145 | 0.000145 |  |  |
| SOL | 0.1537 | -0.6055 | -1 | 0.003294 | -0.003294 |  |  |

Interpretation:

- `JTO`, `XLM`, and `NEAR` have the strongest 15m direction-aware labels from
  this one broad L2 snapshot.
- `BTC` had the cleanest short-window monitor persistence, but its 15m
  directional label is tiny.
- `SOL`, `HYPE`, `WLD`, `ZEC`, and `ONDO` had strong-looking snapshot imbalance
  but failed this first 15m direction-aware label.

## L2 Imbalance Paper Gate

This subtracts taker round-trip fees and current spread from the book-imbalance
directional label, then checks visible 10 bps depth. It is a directional paper
gate, not a maker-fill model.

| asset | size USD | imbalance10 | cost bps | net15 bps | net1h bps | depth USD | depth usage | gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| JTO | 100 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.0240 | small_paper_probe |
| JTO | 250 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.0601 | small_paper_probe |
| JTO | 500 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.1202 | small_paper_probe |
| JTO | 1000 | 0.2901 | 14.68 | 110.06 |  | 4160 | 0.2404 | small_paper_probe |
| XLM | 100 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0047 | small_paper_probe |
| XLM | 250 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0118 | small_paper_probe |
| XLM | 500 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0235 | small_paper_probe |
| XLM | 1000 | 0.1395 | 12.48 | 109.39 |  | 21244 | 0.0471 | small_paper_probe |
| NEAR | 100 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.0046 | small_paper_probe |
| NEAR | 250 | -0.2800 | 11.92 | 103.92 |  | 21602 | 0.0116 | small_paper_probe |

Interpretation:

- `JTO`, `XLM`, and `NEAR` survive the first rough 15m fee/spread/depth gate.
- This is still a directional gate, not a maker-fill model; queue position,
  fill probability, rebates, and adverse selection are unmodeled.
- The next useful action is to repeat the L2 snapshot/label/gate on fresh
  samples and wait for the 1h labels.
